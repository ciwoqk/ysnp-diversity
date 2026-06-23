"""YFull parsed tree → Newick string for scikit-bio.

Edge lengths derive from ``formed_ybp`` / ``tmrca_ybp``; the root is
binarised since ``skbio.faith_pd`` requires exactly two children there.
"""

from typing import Any


def tree_to_newick(tree: dict[str, Any]) -> str:
    """Render ``{branches, samples}`` as Newick.

    Branch edge = formed_ybp - tmrca_ybp (0 for star branches).
    Sample edge = effective_tmrca(branch) - age_ybp.
    Names are single-quoted so ``*`` and ``-`` are always safe.
    Polytomous root → tail wrapped under a zero-length synthetic node.
    """
    branches = tree["branches"]
    samples = tree["samples"]
    root = tree["root"]
    root_b = branches[root]

    def render(h: str) -> str:
        b = branches[h]
        parts = [render(c) for c in b.get("children", [])]
        parts.extend(_render_sample(sid, samples, branches) for sid in b.get("samples", []))
        subtree = "(" + ",".join(parts) + ")" if parts else ""
        length = _branch_edge_length(b)
        return f"{subtree}{_quote(h)}:{length:g}"

    root_parts = [render(c) for c in root_b.get("children", [])]
    root_parts.extend(_render_sample(sid, samples, branches) for sid in root_b.get("samples", []))

    if len(root_parts) > 2:
        head = root_parts[0]
        tail = "(" + ",".join(root_parts[1:]) + "):0"
        root_subtree = f"({head},{tail})"
    else:
        root_subtree = "(" + ",".join(root_parts) + ")"

    root_length = _branch_edge_length(root_b)
    return f"{root_subtree}{_quote(root)}:{root_length:g};"


def _render_sample(
    sid: str,
    samples: dict[str, dict[str, Any]],
    branches: dict[str, dict[str, Any]],
) -> str:
    s = samples[sid]
    length = _sample_edge_length(s, branches)
    return f"{_quote(sid)}:{length:g}"


def _quote(name: str) -> str:
    return "'" + name.replace("'", "''") + "'"


def _branch_edge_length(b: dict[str, Any]) -> float:
    if b["formed_ybp"] is None or b["tmrca_ybp"] is None:
        return 0.0
    return max(0.0, float(b["formed_ybp"]) - float(b["tmrca_ybp"]))


def _sample_edge_length(s: dict[str, Any], branches: dict[str, dict[str, Any]]) -> float:
    tmrca = _effective_tmrca(s["branch"], branches)
    if tmrca is None:
        return 0.0
    age = s.get("age_ybp") or 0.0
    return max(0.0, tmrca - float(age))


def _effective_tmrca(
    branch_name: str | None,
    branches: dict[str, dict[str, Any]],
) -> float | None:
    """Nearest non-None TMRCA walking up from ``branch_name`` (skips star nodes)."""
    while branch_name is not None:
        b = branches[branch_name]
        if b["tmrca_ybp"] is not None:
            return float(b["tmrca_ybp"])
        branch_name = b["parent"]
    return None
