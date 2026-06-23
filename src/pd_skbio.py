"""scikit-bio Faith's PD wrapper.

Renders the parsed YFull tree to Newick and calls
``skbio.diversity.alpha.faith_pd``.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from skbio import TreeNode
from skbio.diversity.alpha import faith_pd

from newick import tree_to_newick


def build_skbio_tree(tree_dict: dict[str, Any]) -> tuple[list[str], TreeNode]:
    """Returns ``(all_ids, tree)``. ``all_ids`` defines the column order
    for the presence vector passed to ``faith_pd_skbio``."""
    all_ids = list(tree_dict["samples"].keys())
    newick = tree_to_newick(tree_dict)
    tree = TreeNode.read(io.StringIO(newick))
    return all_ids, tree


def faith_pd_skbio(
    sample_ids: list[str],
    all_ids: list[str],
    tree: TreeNode,
) -> float:
    if not sample_ids:
        return 0.0
    member = set(sample_ids)
    counts = np.fromiter((1 if sid in member else 0 for sid in all_ids), dtype=int)
    return float(faith_pd(counts, all_ids, tree))
