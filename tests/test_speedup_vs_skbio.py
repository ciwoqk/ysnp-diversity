"""Wall-clock benchmark: ``faith_pd_fast`` vs ``skbio.faith_pd``.

Measures per-call time on the same random subsets across tree sizes
from a few hundred to ~10 000 tips. Marked ``slow`` — run with::

    uv run pytest tests/test_speedup_vs_skbio.py -m slow -s
"""

from __future__ import annotations

import gc
import io
import time

import numpy as np
import pytest
from skbio import TreeNode
from skbio.diversity.alpha import faith_pd as skbio_faith_pd

from significance import build_tree_index, faith_pd_fast

BENCH_HAPLOGROUPS = ["G-Z6552", "J-Z7671", "R1a", "J1"]

# How many PD calls to time, per scenario. Higher = better median stability.
N_REPETITIONS = 30


def _median_ms(fn, n_reps: int) -> float:
    """Median wall-clock time per call, in milliseconds."""
    samples = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)) * 1000


@pytest.mark.slow
@pytest.mark.parametrize("haplogroup", BENCH_HAPLOGROUPS)
def test_single_pd_speedup(haplogroup: str, tree_factory):
    """Per-call timing on one fixed subset of ~10 tips."""
    tree_dict, _newick, skbio_tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    all_ids = list(tree_dict["samples"].keys())

    rng = np.random.default_rng(0)
    n = min(10, idx.n_tips)
    subset_idx = rng.choice(idx.n_tips, size=n, replace=False)
    subset_ids = [all_ids[i] for i in subset_idx]
    counts = np.zeros(len(all_ids), dtype=int)
    counts[subset_idx] = 1

    # Warm-up — first call may JIT-compile / cache-prime.
    faith_pd_fast(idx, subset_idx)
    skbio_faith_pd(counts, all_ids, skbio_tree)

    fast_ms = _median_ms(lambda: faith_pd_fast(idx, subset_idx), N_REPETITIONS)
    skbio_ms = _median_ms(
        lambda: skbio_faith_pd(counts, all_ids, skbio_tree), N_REPETITIONS
    )

    speedup = skbio_ms / fast_ms

    print(
        f"\n  [{haplogroup:9}] tips={idx.n_tips:6,}  "
        f"fast={fast_ms:7.3f} ms  "
        f"skbio={skbio_ms:7.3f} ms  "
        f"speedup={speedup:6.1f}x"
    )


@pytest.mark.slow
@pytest.mark.parametrize("haplogroup", ["G-Z6552", "J-Z7671"])
def test_batched_999_perms_speedup(haplogroup: str, tree_factory):
    """Realistic null-distribution scenario: 999 PD computes for random size-10
    subsets. ``faith_pd_fast`` is run inside a Python loop (no batching)
    so the comparison is honest — both implementations called the same
    number of times.

    Skipped for the big trees because the naive skbio loop takes minutes
    there. The point is already made on smaller trees.
    """
    tree_dict, _newick, skbio_tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    all_ids = list(tree_dict["samples"].keys())
    n_perms = 999
    n = min(10, idx.n_tips)

    rng = np.random.default_rng(0)
    subsets_idx = [
        rng.choice(idx.n_tips, size=n, replace=False) for _ in range(n_perms)
    ]
    subsets_counts = []
    for s in subsets_idx:
        c = np.zeros(len(all_ids), dtype=int)
        c[s] = 1
        subsets_counts.append(c)

    # ── fast: serial loop of faith_pd_fast calls ──
    gc.collect()
    t0 = time.perf_counter()
    for s in subsets_idx:
        faith_pd_fast(idx, s)
    fast_s = time.perf_counter() - t0

    # ── skbio: serial loop of skbio.faith_pd calls ──
    gc.collect()
    t0 = time.perf_counter()
    for c in subsets_counts:
        skbio_faith_pd(c, all_ids, skbio_tree)
    skbio_s = time.perf_counter() - t0

    speedup = skbio_s / fast_s

    print(
        f"\n  [{haplogroup:9}] tips={idx.n_tips:6,}  "
        f"999 calls:  fast={fast_s:6.2f} s  "
        f"skbio={skbio_s:6.2f} s  "
        f"speedup={speedup:6.1f}x"
    )


@pytest.mark.slow
@pytest.mark.parametrize("haplogroup", ["G-Z6552", "J-Z7671"])
def test_vectorised_null_distribution_speedup(haplogroup: str, tree_factory):
    """The realistic comparison: the vectorised ``null_pd_distribution``
    against a naive scikit-bio loop. This is what actually matters for
    the SES.PD pipeline.
    """
    from significance import null_pd_distribution

    tree_dict, _newick, skbio_tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    all_ids = list(tree_dict["samples"].keys())
    n_perms = 999
    n = min(10, idx.n_tips)

    # ── vectorised: one call to null_pd_distribution ──
    gc.collect()
    t0 = time.perf_counter()
    null_pd_distribution(idx, n=n, n_perms=n_perms, rng=np.random.default_rng(0))
    vec_s = time.perf_counter() - t0

    # ── naive skbio loop: 999 separate calls ──
    rng = np.random.default_rng(0)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(n_perms):
        subset = rng.choice(len(all_ids), size=n, replace=False)
        counts = np.zeros(len(all_ids), dtype=int)
        counts[subset] = 1
        skbio_faith_pd(counts, all_ids, skbio_tree)
    naive_s = time.perf_counter() - t0

    speedup = naive_s / vec_s

    print(
        f"\n  [{haplogroup:9}] tips={idx.n_tips:6,}  "
        f"null(999):  vec={vec_s * 1000:7.1f} ms  "
        f"naive_skbio={naive_s * 1000:9.1f} ms  "
        f"speedup={speedup:7.1f}x"
    )
