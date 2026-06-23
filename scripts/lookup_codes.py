#!/usr/bin/env python3
"""Look up ISO 639-3 codes in Glottolog's languoid.csv.

Use this when `ysnp-diversity` reports unmatched language codes:
just paste them and see Glottolog's classification chain.

Usage:
    uv run python scripts/lookup_codes.py kat xmf sva lzz

Or pipe a comma-separated list:
    echo "kat,xmf,sva" | uv run python scripts/lookup_codes.py --

Output for each code:
    <code>: <Language> [language] < <Subfamily> [family] < ... < <Top family> [family]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

LANGUOID_CSV = Path(__file__).resolve().parent.parent / "data" / "languoid.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "codes",
        nargs="*",
        help="ISO 639-3 codes (space- or comma-separated). Use '-' to read stdin.",
    )
    args = ap.parse_args()

    raw = " ".join(args.codes) if args.codes != ["-"] else sys.stdin.read()
    codes = [c.strip() for c in re.split(r"[\s,;]+", raw) if c.strip()]
    if not codes:
        ap.error("no codes provided")

    if not LANGUOID_CSV.exists():
        sys.exit(
            f"{LANGUOID_CSV} not found. Download from "
            "https://glottolog.org/meta/downloads (file glottolog_languoid.csv.zip)"
        )

    with open(LANGUOID_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["id"]: r for r in rows}
    by_iso = {r["iso639P3code"]: r for r in rows if r["iso639P3code"]}

    width = max(len(c) for c in codes)
    for code in codes:
        r = by_iso.get(code)
        if not r:
            print(f"{code:<{width}}  NOT FOUND in Glottolog")
            continue

        chain: list[str] = []
        cur: dict[str, str] | None = r
        seen: set[str] = set()
        while cur is not None and cur["id"] not in seen:
            seen.add(cur["id"])
            chain.append(f"{cur['name']} [{cur['level']}]")
            cur = by_id.get(cur.get("parent_id") or "")

        print(f"{code:<{width}}  {' < '.join(chain)}")


if __name__ == "__main__":
    main()
