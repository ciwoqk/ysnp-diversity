"""Shared pytest fixtures for the ysnp-diversity test suite.

Tests rely on the on-disk YFull HTML cache at ``data/cache/`` — re-running
the suite never touches the network. Missing fixtures cause an explicit
``pytest.skip``, never a silent network fetch.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable

import pytest
from skbio import TreeNode

# The test runner sets ``pythonpath = ["src"]`` via pyproject.toml, so we
# can import top-level modules directly here.
from newick import tree_to_newick
from parser import parse_tree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    if not CACHE_DIR.is_dir():
        pytest.skip(f"YFull cache not present at {CACHE_DIR}")
    return CACHE_DIR


@pytest.fixture(scope="session")
def tree_factory(cache_dir: Path) -> Callable[[str], tuple[dict[str, Any], str, TreeNode]]:
    """Return ``(tree_dict, newick_str, skbio_tree)`` for a cached haplogroup.

    Session-scoped + memoised: each haplogroup is parsed and Newick-rendered
    at most once per test run, which keeps the cross-validation suite fast
    even with heavy parametrisation.
    """
    cache: dict[str, tuple[dict[str, Any], str, TreeNode]] = {}

    def make(haplogroup: str) -> tuple[dict[str, Any], str, TreeNode]:
        if haplogroup in cache:
            return cache[haplogroup]
        html_path = cache_dir / f"{haplogroup}.html"
        if not html_path.exists():
            pytest.skip(f"No cached HTML for {haplogroup} at {html_path}")
        tree_dict = parse_tree(html_path.read_text(encoding="utf-8"))
        newick = tree_to_newick(tree_dict)
        skbio_tree = TreeNode.read(io.StringIO(newick))
        cache[haplogroup] = (tree_dict, newick, skbio_tree)
        return cache[haplogroup]

    return make
