"""Performance benchmarks on large YFull trees (J1, J2, R1a).

Each test prints timing and size stats on success. Marked ``slow`` —
run explicitly::

    uv run pytest tests/test_performance.py -m slow -s
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pytest

from significance import (
    build_tree_index,
    compute_significance,
    faith_pd_fast,
    null_pd_distribution,
)

LARGE_HAPLOGROUPS = ["J1", "J2", "R1a"]

# Safety nets, not aspirational targets — slow CI may need higher limits.
BUILD_INDEX_LIMIT_S = 5.0
SINGLE_PD_LIMIT_S = 0.5
NULL_DIST_LIMIT_S = 10.0
FULL_PIPELINE_LIMIT_S = 60.0


def _human_bytes(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


@pytest.mark.slow
@pytest.mark.parametrize("haplogroup", LARGE_HAPLOGROUPS)
def test_build_tree_index_perf(haplogroup: str, tree_factory):
    """Index construction time on a large tree."""
    tree_dict, _newick, _tree = tree_factory(haplogroup)

    gc.collect()
    start = time.perf_counter()
    idx = build_tree_index(tree_dict)
    elapsed = time.perf_counter() - start

    # Sparse matrix memory footprint = data + indices + indptr.
    sparse_bytes = (
        idx.tip_to_edges.data.nbytes
        + idx.tip_to_edges.indices.nbytes
        + idx.tip_to_edges.indptr.nbytes
    )
    edges_bytes = idx.edge_lengths.nbytes
    nnz = idx.tip_to_edges.nnz

    print(
        f"\n  [{haplogroup}] "
        f"tips={idx.n_tips:6,}  edges={idx.n_edges:6,}  "
        f"nnz={nnz:8,}  density={nnz / (idx.n_tips * idx.n_edges) * 100:.3f}%  "
        f"sparse_mem={_human_bytes(sparse_bytes)}  "
        f"edges_mem={_human_bytes(edges_bytes)}  "
        f"build={elapsed * 1000:6.0f} ms"
    )

    assert elapsed < BUILD_INDEX_LIMIT_S, (
        f"{haplogroup}: build_tree_index took {elapsed:.2f}s > {BUILD_INDEX_LIMIT_S}s"
    )


@pytest.mark.slow
@pytest.mark.parametrize("haplogroup", LARGE_HAPLOGROUPS)
def test_single_shot_faith_pd_perf(haplogroup: str, tree_factory):
    """Single-shot PD on a 10-sample subset — used inside compute_significance
    for the observed-PD pass over every group."""
    tree_dict, _newick, _tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    rng = np.random.default_rng(0)
    n = min(10, idx.n_tips)
    subset_indices = rng.choice(idx.n_tips, size=n, replace=False)

    # Warm-up call (first call may trigger SciPy lazy compilation).
    faith_pd_fast(idx, subset_indices)

    # 50-call rolling median to dampen jitter.
    durations = []
    for _ in range(50):
        gc.collect()
        start = time.perf_counter()
        faith_pd_fast(idx, subset_indices)
        durations.append(time.perf_counter() - start)
    median_ms = np.median(durations) * 1000

    print(
        f"\n  [{haplogroup}] tips={idx.n_tips:6,}  "
        f"single_pd_median={median_ms:6.2f} ms"
    )
    assert max(durations) < SINGLE_PD_LIMIT_S, (
        f"{haplogroup}: faith_pd_fast worst-case {max(durations):.2f}s "
        f"> {SINGLE_PD_LIMIT_S}s"
    )


@pytest.mark.slow
@pytest.mark.parametrize("haplogroup", LARGE_HAPLOGROUPS)
def test_null_distribution_perf(haplogroup: str, tree_factory):
    """``n_perms=999`` null distribution for ``n=10`` — the hot path."""
    tree_dict, _newick, _tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    rng = np.random.default_rng(0)

    # Warm-up
    null_pd_distribution(idx, n=10, n_perms=99, rng=rng)

    gc.collect()
    start = time.perf_counter()
    nulls = null_pd_distribution(idx, n=10, n_perms=999, rng=np.random.default_rng(0))
    elapsed = time.perf_counter() - start

    print(
        f"\n  [{haplogroup}] tips={idx.n_tips:6,}  "
        f"null_pd(n=10, 999 perms)={elapsed * 1000:6.0f} ms  "
        f"({elapsed * 1000 / 999:.2f} ms per perm)"
    )

    assert elapsed < NULL_DIST_LIMIT_S, (
        f"{haplogroup}: null_pd_distribution took {elapsed:.2f}s > {NULL_DIST_LIMIT_S}s"
    )
    assert nulls.shape == (999,)
    assert (nulls >= 0).all()


@pytest.mark.slow
@pytest.mark.parametrize("haplogroup", LARGE_HAPLOGROUPS)
def test_full_significance_pipeline_perf(haplogroup: str, tree_factory):
    """End-to-end ``compute_significance`` benchmark on a realistic
    50-group workload.

    Generates synthetic groups of varied sizes (matching the shape of
    real geographic/linguistic groupings: small handful to a few dozen).
    Measures full pipeline including null-distribution caching by size.
    """
    tree_dict, _newick, _tree = tree_factory(haplogroup)
    idx = build_tree_index(tree_dict)
    all_ids = list(tree_dict["samples"].keys())
    rng = np.random.default_rng(0)

    # 50 groups: realistic size distribution: many small (2-5), some
    # medium (6-15), few large (16-30).
    sizes = list(rng.integers(2, 6, size=30))   # 30 small
    sizes += list(rng.integers(6, 16, size=15))  # 15 medium
    sizes += list(rng.integers(16, 31, size=5))  # 5 large
    rng.shuffle(sizes)

    groups: dict[str, list[str]] = {}
    pool = list(rng.permutation(all_ids))
    pos = 0
    for i, s in enumerate(sizes):
        if pos + s > len(pool):
            break
        groups[f"g{i:02d}"] = pool[pos : pos + s]
        pos += int(s)

    unique_sizes = sorted({len(s) for s in groups.values()})

    gc.collect()
    start = time.perf_counter()
    result = compute_significance(idx, groups, n_perms=999, seed=42)
    elapsed = time.perf_counter() - start

    n_with_sig = sum(1 for r in result.values() if r["ses_pd"] is not None)

    print(
        f"\n  [{haplogroup}] tips={idx.n_tips:6,}  "
        f"groups={len(groups):3}  unique_sizes={len(unique_sizes):2}  "
        f"with_sig={n_with_sig:3}  "
        f"full_pipeline={elapsed:5.2f} s "
        f"(~{elapsed / len(unique_sizes):.2f} s per unique size)"
    )

    assert elapsed < FULL_PIPELINE_LIMIT_S, (
        f"{haplogroup}: full pipeline took {elapsed:.1f}s > {FULL_PIPELINE_LIMIT_S}s"
    )
    assert len(result) == len(groups)
