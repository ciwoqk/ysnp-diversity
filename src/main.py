#!/usr/bin/env python3
"""CLI: scrape a YFull haplogroup tree and report Faith's PD per group.

Two PD backends are selectable via ``--impl`` (skbio default, or the
vectorised fast path in ``significance.py``). ``--export PATH`` writes
the rendered output as .svg / .png / .html.

Examples
--------
::

    uv run ysnp-diversity R1a  -g language_family
    uv run ysnp-diversity J1   -g country  --impl fast
    uv run ysnp-diversity G2a  -g language_family  --export pd.svg
"""

import argparse
import contextlib
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Callable

if sys.stdout.encoding != "utf-8":
    with contextlib.suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from groups import GROUP_BY_MODES, bucket_samples, load_families
from parser import parse_tree
from scraper import fetch_tree

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
EXPORTS_DIR = PROJECT_ROOT / "exports"


PdFn = Callable[[list[str]], float]


# ── Group colour palette ────────────────────────────────────────────
# Muted R-style hues; assigned per group name via MD5 so the same name
# always lands on the same colour across runs.
PALETTE: tuple[str, ...] = (
    "#d4a857",  # gold
    "#80a48f",  # sage
    "#8fa7c2",  # slate
    "#c87a4c",  # rust
    "#5fa8a8",  # teal
    "#b08868",  # bronze
    "#cfa848",  # amber
    "#88a958",  # moss
    "#c87aa7",  # plum
    "#a888c8",  # lilac
)


def _group_color(name: str) -> str:
    h = hashlib.md5(name.encode("utf-8")).digest()[0]
    return PALETTE[h % len(PALETTE)]


# ── PD backend factories ─────────────────────────────────────────────


def _make_skbio_pd(tree_dict: dict[str, Any]) -> tuple[PdFn, str]:
    from pd_skbio import build_skbio_tree, faith_pd_skbio

    all_ids, tree = build_skbio_tree(tree_dict)
    log.info("skbio TreeNode: tips=%d", len(all_ids))

    def pd_fn(subset: list[str]) -> float:
        return faith_pd_skbio(subset, all_ids, tree)

    return pd_fn, "skbio.diversity.alpha.faith_pd (canonical reference)"


def _make_fast_pd(tree_dict: dict[str, Any]) -> tuple[PdFn, str]:
    import numpy as np

    from significance import build_tree_index, faith_pd_fast

    idx = build_tree_index(tree_dict)
    log.info(
        "Tree index: tips=%d, edges=%d, sparse nnz=%d",
        idx.n_tips,
        idx.n_edges,
        idx.tip_to_edges.nnz,
    )

    def pd_fn(subset: list[str]) -> float:
        return faith_pd_fast(idx, np.asarray(subset) if subset else np.empty(0, dtype=np.intp))

    return pd_fn, "significance.faith_pd_fast (vectorised sparse matmul)"


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(
        description=(
            "Faith's PD per group on YFull haplogroup trees. "
            "Default impl is scikit-bio; --impl fast switches to the "
            "vectorised sparse-matmul path (cross-validated bit-equal)."
        ),
    )
    ap.add_argument("haplogroup", help="Haplogroup ID on YFull (e.g., R1a, J1, G2a)")
    ap.add_argument(
        "-g",
        "--group-by",
        choices=sorted(GROUP_BY_MODES),
        required=True,
        help="Group samples by this attribute",
    )
    ap.add_argument(
        "--impl",
        choices=["skbio", "fast"],
        default="skbio",
        help=(
            "PD implementation: 'skbio' (canonical reference, default) or "
            "'fast' (vectorised sparse-matmul, cross-validated bit-equal)."
        ),
    )
    ap.add_argument(
        "--export",
        type=Path,
        metavar="PATH",
        help=(
            "Save the rendered table as a file. Supported extensions: "
            ".svg (vector, recommended), .png (raster, 2× zoom for retina), "
            ".html (interactive)."
        ),
    )
    ap.add_argument(
        "--title",
        metavar="LABEL",
        help=(
            "Override the haplogroup label shown in the rendered output "
            "and export titles. The actual YFull fetch still uses the "
            "positional argument."
        ),
    )
    ap.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore the on-disk YFull HTML cache and re-fetch.",
    )
    args = ap.parse_args()

    html = fetch_tree(args.haplogroup, CACHE_DIR, args.force_refresh)
    tree_dict = parse_tree(html)
    log.info(
        "Parsed: version=%s, branches=%d, samples=%d",
        tree_dict.get("tree_version"),
        len(tree_dict["branches"]),
        len(tree_dict["samples"]),
    )

    if args.impl == "skbio":
        pd_fn, impl_label = _make_skbio_pd(tree_dict)
    else:
        pd_fn, impl_label = _make_fast_pd(tree_dict)

    families = load_families() if args.group_by in ("language", "language_family") else None
    groups, unmatched = bucket_samples(tree_dict, args.group_by, families)

    all_ids = list(tree_dict["samples"].keys())
    tree_total_pd = pd_fn(all_ids)

    rows: list[dict[str, Any]] = []
    for name, samples in groups.items():
        pd = pd_fn(samples)
        rows.append(
            {
                "group": name,
                "n_samples": len(samples),
                "pd_years": pd,
                "pd_fraction": pd / tree_total_pd if tree_total_pd > 0 else 0.0,
            }
        )
    rows.sort(key=lambda r: r["pd_years"], reverse=True)

    root_branch = tree_dict["branches"][tree_dict["root"]]
    _render(
        title=args.title or tree_dict["root"],
        tree_version=tree_dict.get("tree_version"),
        formed_ybp=root_branch.get("formed_ybp"),
        tmrca_ybp=root_branch.get("tmrca_ybp"),
        group_by=args.group_by,
        impl_label=impl_label,
        tree_total=tree_total_pd,
        n_assigned=sum(len(v) for v in groups.values()),
        n_total=len(tree_dict["samples"]),
        rows=rows,
        unmatched=sorted(unmatched) if unmatched else [],
        export_path=args.export,
    )


# ── Output rendering ─────────────────────────────────────────────────


def _bar(share: float, max_share: float, color: str, width: int = 16) -> Any:
    """Unicode-block bar in ``color``, scaled so the largest row fills ``width``."""
    from rich.text import Text

    if max_share <= 0:
        return Text(" " * width)
    filled = int(round(share / max_share * width))
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * (width - filled), style="grey35")
    return bar


def _render(
    *,
    title: str,
    tree_version: str | None,
    formed_ybp: float | None,
    tmrca_ybp: float | None,
    group_by: str,
    impl_label: str,
    tree_total: float,
    n_assigned: int,
    n_total: int,
    rows: list[dict[str, Any]],
    unmatched: list[str],
    export_path: Path | None = None,
) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    # Fixed width keeps terminal and SVG layouts identical.
    console = Console(record=export_path is not None, width=110)

    console.print()
    console.print(
        Text("Faith's PD for ", style="bold")
        + Text(title, style="bold cyan")
        + Text(f"   ·   group by {group_by}", style="dim")
    )

    meta = Text()
    meta.append(f"YFull tree {tree_version}", style="dim italic")
    if formed_ybp is not None:
        meta.append("   ·   formed ", style="dim")
        meta.append(f"{formed_ybp:,.0f}", style="bold yellow")
        meta.append(" ybp", style="dim yellow")
    if tmrca_ybp is not None:
        meta.append("   ·   TMRCA ", style="dim")
        meta.append(f"{tmrca_ybp:,.0f}", style="bold yellow")
        meta.append(" ybp", style="dim yellow")
    console.print(meta)
    console.print()

    summary = Text()
    summary.append("Tree total PD: ", style="cyan")
    summary.append(f"{tree_total:,.0f}", style="bold yellow")
    summary.append(" ybp", style="dim yellow")
    summary.append("     ")
    summary.append("Assigned: ", style="cyan")
    summary.append(f"{n_assigned}", style="bold")
    summary.append(f" / {n_total} samples", style="dim")
    summary.append("     ")
    summary.append("Groups: ", style="cyan")
    summary.append(str(len(rows)), style="bold")
    console.print(summary)
    console.print(Text(f"Implementation: {impl_label}", style="dim italic"))
    console.print()

    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold",
        pad_edge=False,
        show_edge=False,
        collapse_padding=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("group", no_wrap=True, min_width=26)
    table.add_column("n", justify="right", style="cyan", width=4)
    table.add_column("PD (ybp)", justify="right", width=12)
    table.add_column("share", justify="right", width=7)
    table.add_column("", width=16)

    max_share = max((r["pd_fraction"] for r in rows), default=0.0)
    for i, r in enumerate(rows, 1):
        color = _group_color(r["group"])
        table.add_row(
            str(i),
            Text(r["group"], style=color),
            str(r["n_samples"]),
            Text(f"{r['pd_years']:,.0f}", style="bold yellow"),
            Text(f"{r['pd_fraction'] * 100:>5.1f}%"),
            _bar(r["pd_fraction"], max_share, color),
        )
    console.print(table)

    if unmatched:
        console.print()
        console.print(
            Text(f"  (unmatched languages: {', '.join(unmatched)})", style="dim")
        )
    console.print()

    if export_path is not None:
        export_path = _resolve_export_path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = export_path.suffix.lower()
        if suffix == ".svg":
            console.save_svg(str(export_path), title=f"Faith's PD · {title}")
        elif suffix == ".png":
            _save_png(console, export_path, title=f"Faith's PD · {title}")
        elif suffix == ".html":
            console.save_html(str(export_path))
        else:
            raise SystemExit(
                f"--export expects .svg / .png / .html, got {export_path.name!r}"
            )
        log.info("Rendered output saved to %s", export_path)


def _resolve_export_path(path: Path) -> Path:
    """Bare filename → ``exports/`` at project root; explicit paths kept verbatim."""
    if path.is_absolute() or path.parent != Path("."):
        return path
    return EXPORTS_DIR / path.name


def _save_png(console: Any, path: Path, *, title: str, zoom: float = 2.0) -> None:
    """Render the recorded console as SVG in memory, then rasterise to PNG via resvg."""
    import resvg_py

    svg = console.export_svg(title=title)
    png_bytes = resvg_py.svg_to_bytes(svg_string=svg, zoom=zoom)
    path.write_bytes(bytes(png_bytes))


if __name__ == "__main__":
    main()
