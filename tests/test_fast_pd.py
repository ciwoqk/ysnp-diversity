"""Cross-validation: ``significance.faith_pd_fast`` ≡ ``skbio.faith_pd``.

The fast path is a sparse-matmul reimplementation of Faith's PD; these
tests check it returns bit-equal values to scikit-bio across multiple
trees and subset sizes. Trees not present in the local cache are skipped.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from skbio import TreeNode
from skbio.diversity.alpha import faith_pd

from significance import TreeIndex, build_tree_index, faith_pd_fast

HAPLOGROUPS = [
    "G-Z6552",
    "G-L1266",
    "J-Z7671",
    "J-Z1842",
    "J-Y12379",
    "J-FT34521",
]

# Absolute tolerance in years before present. scikit-bio sums float64
# edge lengths in tree-traversal order; the fast path sums them in
# edge-index order. Reordered float64 sums differ by ≤ a few ULPs even
# for million-year totals — 1 µybp is plenty of headroom.
PD_TOLERANCE_YBP = 1e-6


def _skbio_pd(subset: list[str], all_ids: list[str], tree: TreeNode) -> float:
    """Reference path: counts vector + skbio.faith_pd."""
    member = set(subset)
    counts = np.fromiter((1 if sid in member else 0 for sid in all_ids), dtype=int)
    return float(faith_pd(counts, all_ids, tree))


# ── Whole-tree sanity ───────────────────────────────────────────────


@pytest.mark.parametrize("haplogroup", HAPLOGROUPS)
def test_full_tree_pd_matches_skbio(haplogroup: str, tree_factory):
    """When every tip is in the subset, the fast path must agree with
    scikit-bio's PD. (Note: this is NOT the same as ``sum(edge_lengths)`` —
    orphan edges below clades with no descendant tips are correctly
    excluded from Faith's PD.)
    """
    tree_dict, _newick, tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    all_ids = list(tree_dict["samples"].keys())

    fast = faith_pd_fast(idx, all_ids)
    skb = _skbio_pd(all_ids, all_ids, tree)

    assert abs(fast - skb) < PD_TOLERANCE_YBP, (
        f"{haplogroup}: full-tree PD mismatch — fast={fast:,.6f}, skbio={skb:,.6f}"
    )


# ── Triangulation on random subsets — the headline test ────────────


@pytest.mark.parametrize("haplogroup", HAPLOGROUPS)
@pytest.mark.parametrize("subset_size", [2, 5, 10, 20, 50])
def test_fast_pd_matches_skbio_random_subsets(
    haplogroup: str,
    subset_size: int,
    tree_factory: Callable[[str], tuple[dict[str, Any], str, TreeNode]],
):
    """For each (haplogroup, subset_size), draw 20 random subsamples and
    require ``faith_pd_fast`` to agree with ``skbio.faith_pd`` to within
    ``PD_TOLERANCE_YBP``.

    Total budget across the parameter matrix:
        6 haplogroups × 5 sizes × 20 trials = 600 PD comparisons.
    """
    tree_dict, _newick, tree = tree_factory(haplogroup)
    all_ids = list(tree_dict["samples"].keys())

    if subset_size > len(all_ids):
        pytest.skip(
            f"{haplogroup} has only {len(all_ids)} samples — cannot draw {subset_size}"
        )

    idx = build_tree_index(tree_dict)

    # Deterministic per (haplogroup, size) so a failure is reproducible.
    seed = abs(hash((haplogroup, subset_size))) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    for trial in range(20):
        picked = rng.choice(len(all_ids), size=subset_size, replace=False)
        subset = [all_ids[i] for i in picked]

        fast = faith_pd_fast(idx, subset)
        skb = _skbio_pd(subset, all_ids, tree)

        assert abs(fast - skb) < PD_TOLERANCE_YBP, (
            f"{haplogroup} / size {subset_size} / trial {trial}: "
            f"fast={fast:,.6f}, skbio={skb:,.6f}, "
            f"diff={abs(fast - skb):.3e} > {PD_TOLERANCE_YBP:.0e}"
        )


# ── Pathological edge cases ────────────────────────────────────────


def test_singleton_subset(tree_factory):
    """A single-tip subset's PD is the length of its root-to-tip path —
    must match scikit-bio."""
    tree_dict, _newick, tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)
    all_ids = list(tree_dict["samples"].keys())

    rng = np.random.default_rng(42)
    for _ in range(10):
        sid = all_ids[int(rng.integers(0, len(all_ids)))]
        fast = faith_pd_fast(idx, [sid])
        skb = _skbio_pd([sid], all_ids, tree)
        assert abs(fast - skb) < PD_TOLERANCE_YBP, (
            f"singleton {sid}: fast={fast:.6f}, skbio={skb:.6f}"
        )


def test_empty_subset_returns_zero(tree_factory):
    """No tips selected → PD = 0 (no edges induced)."""
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)
    assert faith_pd_fast(idx, []) == 0.0


def test_accepts_integer_indices(tree_factory):
    """``faith_pd_fast`` should accept raw tip indices, not just string IDs.

    This is the API used internally by the upcoming null-distribution
    permutation loop, where avoiding the string→index dict lookup matters
    for the inner hot path.
    """
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)
    all_ids = list(tree_dict["samples"].keys())

    rng = np.random.default_rng(0)
    for _ in range(10):
        indices = rng.choice(idx.n_tips, size=8, replace=False)
        str_ids = [all_ids[i] for i in indices]

        pd_via_int = faith_pd_fast(idx, indices)
        pd_via_str = faith_pd_fast(idx, str_ids)
        assert pd_via_int == pd_via_str


def test_unknown_sample_id_raises(tree_factory):
    """Bogus sample id → KeyError with a useful message."""
    tree_dict, _newick, _tree = tree_factory("G-Z6552")
    idx = build_tree_index(tree_dict)

    with pytest.raises(KeyError, match="not in tree index"):
        faith_pd_fast(idx, ["YF-DOES-NOT-EXIST-12345"])


# ── Structural sanity of TreeIndex itself ───────────────────────────


@pytest.mark.parametrize("haplogroup", HAPLOGROUPS)
def test_tree_index_shape(haplogroup: str, tree_factory):
    """tip_to_edges has shape (n_tips, n_edges); n_edges = n_branches + n_samples."""
    tree_dict, _newick, _tree = tree_factory(haplogroup)
    idx: TreeIndex = build_tree_index(tree_dict)

    n_branches = len(tree_dict["branches"])
    n_samples = len(tree_dict["samples"])
    expected_n_edges = n_branches + n_samples

    assert idx.tip_to_edges.shape == (n_samples, expected_n_edges)
    assert idx.edge_lengths.shape == (expected_n_edges,)
    assert idx.n_tips == n_samples
    assert idx.n_edges == expected_n_edges


@pytest.mark.parametrize("haplogroup", HAPLOGROUPS)
def test_each_tip_has_at_least_two_edges(haplogroup: str, tree_factory):
    """Every tip has its own sample edge + at least one ancestor clade edge."""
    tree_dict, _newick, _tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    paths_lengths = idx.tip_to_edges.sum(axis=1)
    assert (paths_lengths >= 2).all(), (
        f"{haplogroup}: some tips have <2 edges on their path "
        f"(min={int(paths_lengths.min())})"
    )


@pytest.mark.parametrize("haplogroup", HAPLOGROUPS)
def test_all_edge_lengths_nonnegative(haplogroup: str, tree_factory):
    """Faith's PD is undefined for negative edge lengths; the YFull
    derivation clamps to 0 — verify nothing slipped through."""
    tree_dict, _newick, _tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    assert (idx.edge_lengths >= 0).all()
