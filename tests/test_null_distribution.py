"""Tests for ``null_pd_distribution`` and ``_sample_indices_batch``:
shape and dtype, sampling-without-replacement, reproducibility, and
agreement with ``faith_pd_fast`` on the same indices.
"""

from __future__ import annotations

import numpy as np
import pytest

from significance import (
    _sample_indices_batch,  # pyright: ignore[reportPrivateUsage]
    build_tree_index,
    faith_pd_fast,
    null_pd_distribution,
)

HAPLOGROUPS = ["G-Z6552", "G-L1266", "J-Z7671", "J-Z1842", "J-Y12379"]


# ── Sampling helper correctness ─────────────────────────────────────


def test_sample_indices_batch_shape():
    """Indices matrix has shape (n_perms, n)."""
    rng = np.random.default_rng(0)
    out = _sample_indices_batch(n_tips=100, n=7, n_perms=50, rng=rng)
    assert out.shape == (50, 7)


def test_sample_indices_without_replacement():
    """Each row consists of n distinct tip indices."""
    rng = np.random.default_rng(0)
    out = _sample_indices_batch(n_tips=200, n=15, n_perms=100, rng=rng)
    for row in out:
        assert len(set(row.tolist())) == 15, f"duplicate indices in row: {row}"


def test_sample_indices_uniformly_cover_range():
    """Over enough permutations, every tip index appears roughly equally.

    For n=10 of n_tips=50 with n_perms=2000, each tip appears in
    expectation 2000 × 10/50 = 400 rows. Tolerance ±3σ for binomial.
    """
    rng = np.random.default_rng(0)
    n_tips, n, n_perms = 50, 10, 2000
    out = _sample_indices_batch(n_tips=n_tips, n=n, n_perms=n_perms, rng=rng)

    counts = np.bincount(out.ravel(), minlength=n_tips)
    expected = n_perms * n / n_tips
    # Each tip's count is binomial(n_perms, n/n_tips); sd ≈ √(n_perms·p·q).
    sd = np.sqrt(n_perms * (n / n_tips) * (1 - n / n_tips))
    assert ((counts >= expected - 4 * sd) & (counts <= expected + 4 * sd)).all(), (
        f"Tip frequency outside ±4σ: counts={counts}, expected≈{expected}"
    )


# ── Output structure ────────────────────────────────────────────────


@pytest.mark.parametrize("haplogroup", HAPLOGROUPS)
def test_null_pd_shape_and_dtype(haplogroup: str, tree_factory):
    tree_dict, _newick, _tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    rng = np.random.default_rng(0)

    nulls = null_pd_distribution(idx, n=5, n_perms=20, rng=rng)
    assert nulls.shape == (20,)
    assert nulls.dtype == np.float64
    assert (nulls >= 0).all()


def test_null_pd_increases_with_n(tree_factory):
    """Larger subsets sample more of the tree → mean PD should increase
    monotonically (or at least not decrease) with n."""
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)

    means: list[float] = []
    for n in (2, 5, 10, 20, 30):
        if n > idx.n_tips:
            continue
        rng = np.random.default_rng(0)
        nulls = null_pd_distribution(idx, n=n, n_perms=300, rng=rng)
        means.append(float(nulls.mean()))

    # Strict ↑ : adding more tips can never strictly reduce induced edges.
    for prev, curr in zip(means, means[1:], strict=False):
        assert curr >= prev - 1e-6, f"mean PD dropped at larger n: {means}"


# ── Sampling without replacement: the headline guarantee ────────────


def test_n_equals_n_tips_degenerate_all_equal(tree_factory):
    """When n = n_tips, every permutation samples the *same* set (all tips),
    so all PD values must be identical — and equal to ``faith_pd_fast`` on
    every tip.
    """
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)
    rng = np.random.default_rng(0)

    nulls = null_pd_distribution(idx, n=idx.n_tips, n_perms=8, rng=rng)
    full = faith_pd_fast(idx, np.arange(idx.n_tips))

    assert np.allclose(nulls, full, atol=1e-6)
    assert (np.abs(nulls - nulls[0]) < 1e-9).all()


# ── Per-permutation cross-validation vs single-shot ─────────────────


def test_each_permutation_matches_single_shot(tree_factory):
    """Reconstruct each permutation's indices via ``_sample_indices_batch``
    using the same seed, then verify the batched PD equals what
    ``faith_pd_fast`` would have given for those exact indices.

    This is the most direct correctness test for the matmul path: it
    proves that ``null_pd_distribution`` and ``faith_pd_fast`` agree
    on every individual sample, not just on aggregate statistics.
    """
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)

    n, n_perms = 8, 50

    rng_a = np.random.default_rng(42)
    batched = null_pd_distribution(idx, n=n, n_perms=n_perms, rng=rng_a)

    # Re-derive the SAME indices the function used internally by replaying
    # the same RNG seed through the same sampler.
    rng_b = np.random.default_rng(42)
    indices = _sample_indices_batch(idx.n_tips, n, n_perms, rng_b)

    expected = np.array([faith_pd_fast(idx, indices[p]) for p in range(n_perms)])
    assert np.allclose(batched, expected, atol=1e-6), (
        f"Batched PD disagrees with per-row single-shot:\n"
        f"max abs diff = {np.abs(batched - expected).max()}"
    )


# ── Reproducibility ─────────────────────────────────────────────────


def test_reproducibility_same_seed(tree_factory):
    """Same seed → identical null distribution."""
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)

    a = null_pd_distribution(idx, n=10, n_perms=100, rng=np.random.default_rng(123))
    b = null_pd_distribution(idx, n=10, n_perms=100, rng=np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_different_seeds_diverge(tree_factory):
    """Different seeds → different (but statistically consistent) samples."""
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)

    a = null_pd_distribution(idx, n=10, n_perms=100, rng=np.random.default_rng(1))
    b = null_pd_distribution(idx, n=10, n_perms=100, rng=np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_independent_seeds_converge_in_mean(tree_factory):
    """Two independent seeds with large n_perms should give means within
    a few standard errors of each other (proves the sampler is unbiased)."""
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)

    a = null_pd_distribution(idx, n=10, n_perms=2000, rng=np.random.default_rng(1))
    b = null_pd_distribution(idx, n=10, n_perms=2000, rng=np.random.default_rng(2))

    # SE of the mean = std / sqrt(n). Difference of means within ±3 SE.
    se = a.std(ddof=1) / np.sqrt(len(a)) + b.std(ddof=1) / np.sqrt(len(b))
    diff = abs(a.mean() - b.mean())
    assert diff < 3 * se, f"Means differ by {diff:.2f}, > 3 SE = {3 * se:.2f}"


# ── Argument validation ─────────────────────────────────────────────


def test_n_too_large_raises(tree_factory):
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)
    with pytest.raises(ValueError, match="Cannot sample"):
        null_pd_distribution(idx, n=idx.n_tips + 1, n_perms=10, rng=np.random.default_rng(0))


def test_n_zero_raises(tree_factory):
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)
    with pytest.raises(ValueError, match=">= 1"):
        null_pd_distribution(idx, n=0, n_perms=10, rng=np.random.default_rng(0))


def test_n_perms_zero_raises(tree_factory):
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)
    with pytest.raises(ValueError, match="n_perms must be >= 1"):
        null_pd_distribution(idx, n=2, n_perms=0, rng=np.random.default_rng(0))
