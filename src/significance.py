"""Vectorised Faith's PD + SES.PD + permutation p-value.

NumPy/SciPy implementation that runs the per-permutation null
distribution as a single sparse matmul. Cross-validated bit-equal
against ``scikit-bio.diversity.alpha.faith_pd`` in the pytest suite.

Public surface:
    TreeIndex, build_tree_index, faith_pd_fast,
    null_pd_distribution, compute_significance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from newick import _branch_edge_length, _sample_edge_length  # pyright: ignore[reportPrivateUsage]


@dataclass(frozen=True)
class TreeIndex:
    """Frozen numerical view of a parsed YFull tree.

    ``edge_lengths[j]`` is the length of edge ``j``; clade edges first
    (in ``tree_dict["branches"]`` order), then sample edges.
    ``tip_to_edges[i, j] == 1`` iff edge ``j`` lies on tip ``i``'s
    root-to-tip path; stored sparse so big trees fit in tens of MB.
    """

    edge_lengths: np.ndarray
    tip_to_edges: csr_matrix
    tip_id_to_idx: dict[str, int]

    @property
    def n_tips(self) -> int:
        return self.tip_to_edges.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_lengths.shape[0]


def build_tree_index(tree_dict: dict[str, Any]) -> TreeIndex:
    """Walk each sample's root-to-tip path once, collect COO coordinates,
    materialise as a single CSR matrix."""
    branches: dict[str, dict[str, Any]] = tree_dict["branches"]
    samples: dict[str, dict[str, Any]] = tree_dict["samples"]

    edge_lengths_list: list[float] = []
    branch_to_edge_idx: dict[str, int] = {}
    for branch_name, branch_dict in branches.items():
        branch_to_edge_idx[branch_name] = len(edge_lengths_list)
        edge_lengths_list.append(_branch_edge_length(branch_dict))

    sample_to_edge_idx: dict[str, int] = {}
    for sid, s in samples.items():
        sample_to_edge_idx[sid] = len(edge_lengths_list)
        edge_lengths_list.append(_sample_edge_length(s, branches))

    edge_lengths = np.asarray(edge_lengths_list, dtype=np.float64)

    tip_ids = list(samples.keys())
    rows: list[int] = []
    cols: list[int] = []
    for tip_idx, sid in enumerate(tip_ids):
        rows.append(tip_idx)
        cols.append(sample_to_edge_idx[sid])
        current: str | None = samples[sid]["branch"]
        while current is not None:
            edge_j = branch_to_edge_idx.get(current)
            if edge_j is not None:
                rows.append(tip_idx)
                cols.append(edge_j)
            current = branches[current]["parent"]

    # int32 (not int8): the matmul ``sample_sparse @ tip_to_edges``
    # accumulates column sums up to ``n_tips``; int8 silently overflows
    # at 127 and drops edges from the induced set.
    n_tips = len(tip_ids)
    n_edges = edge_lengths.shape[0]
    tip_to_edges = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(n_tips, n_edges),
        dtype=np.int32,
    )

    return TreeIndex(
        edge_lengths=edge_lengths,
        tip_to_edges=tip_to_edges,
        tip_id_to_idx={sid: i for i, sid in enumerate(tip_ids)},
    )


def faith_pd_fast(idx: TreeIndex, sample_ids: list[str] | np.ndarray) -> float:
    """Faith's PD for one tip subset, via sparse row-sum and dot product.

    Accepts either string IDs or integer tip indices.
    """
    arr = np.asarray(sample_ids)
    if arr.size == 0:
        return 0.0

    if arr.dtype.kind in ("U", "S", "O"):
        try:
            tip_indices = np.fromiter(
                (idx.tip_id_to_idx[s] for s in arr.tolist()),
                dtype=np.intp,
                count=arr.size,
            )
        except KeyError as e:
            msg = f"sample id {e.args[0]!r} not in tree index"
            raise KeyError(msg) from None
    else:
        tip_indices = arr.astype(np.intp, copy=False)

    sub = idx.tip_to_edges[tip_indices]
    counts = np.asarray(sub.sum(axis=0)).ravel()
    induced = (counts > 0).astype(np.float64, copy=False)
    return float(idx.edge_lengths @ induced)


def _sample_indices_batch(
    n_tips: int,
    n: int,
    n_perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``n`` distinct tip indices per row across ``n_perms`` rows.

    argsort of uniform random per row gives a permutation; first ``n``
    columns are unique by construction.
    """
    return np.argsort(rng.random((n_perms, n_tips)), axis=1)[:, :n].astype(np.intp)


def null_pd_distribution(
    idx: TreeIndex,
    n: int,
    n_perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Faith's PD for ``n_perms`` random size-``n`` tip subsets, vectorised."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n > idx.n_tips:
        raise ValueError(f"Cannot sample {n} tips from a tree with {idx.n_tips}")
    if n_perms < 1:
        raise ValueError(f"n_perms must be >= 1, got {n_perms}")

    sample_indices = _sample_indices_batch(idx.n_tips, n, n_perms, rng)

    perm_ids = np.repeat(np.arange(n_perms, dtype=np.intp), n)
    tip_ids_flat = sample_indices.ravel()
    sample_sparse = csr_matrix(
        (np.ones(n_perms * n, dtype=np.int32), (perm_ids, tip_ids_flat)),
        shape=(n_perms, idx.n_tips),
        dtype=np.int32,
    )

    counts = sample_sparse @ idx.tip_to_edges
    induced = (counts > 0).astype(np.float64)
    return np.asarray(induced @ idx.edge_lengths).ravel()


def compute_significance(
    idx: TreeIndex,
    groups: dict[str, list[str]],
    *,
    n_perms: int = 999,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """SES.PD and two-tailed permutation p-value per group (Webb 2002).

    For each unique group sample-size, one null distribution is drawn
    and shared across all groups of that size. Groups with fewer than 2
    samples (or more than ``idx.n_tips``) get ``None`` for SES and
    p-value — a singleton has no permutation interpretation.
    """
    rng = np.random.default_rng(seed)

    unique_sizes = sorted(
        {len(s) for s in groups.values() if 2 <= len(s) <= idx.n_tips}
    )
    null_by_size: dict[int, np.ndarray] = {
        n: null_pd_distribution(idx, n, n_perms, rng) for n in unique_sizes
    }

    results: dict[str, dict[str, Any]] = {}
    for group_name, sample_ids in groups.items():
        n = len(sample_ids)
        observed = faith_pd_fast(idx, sample_ids)

        if n < 2 or n > idx.n_tips:
            results[group_name] = {
                "observed_pd": observed,
                "n_samples": n,
                "expected_pd": None,
                "sd_null_pd": None,
                "ses_pd": None,
                "p_value": None,
            }
            continue

        nulls = null_by_size[n]
        mean = float(nulls.mean())
        sd = float(nulls.std(ddof=1))
        ses = (observed - mean) / sd if sd > 0 else 0.0

        ge = int((nulls >= observed).sum())
        le = int((nulls <= observed).sum())
        p_two_tailed = min(2 * min(ge, le) / (n_perms + 1), 1.0)

        results[group_name] = {
            "observed_pd": observed,
            "n_samples": n,
            "expected_pd": mean,
            "sd_null_pd": sd,
            "ses_pd": float(ses),
            "p_value": float(p_two_tailed),
        }

    return results
