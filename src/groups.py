"""Sample grouping: country, region, language, language_family."""

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


GROUP_BY_ATTRS: dict[str, str] = {
    "country": "country_iso",
    "region": "country_subdivision",
    "language": "language",
}
GROUP_BY_MODES: tuple[str, ...] = (*GROUP_BY_ATTRS, "language_family")

DEFAULT_FAMILIES_PATH = Path(__file__).resolve().parent.parent / "data" / "language_families.json"


def load_families(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DEFAULT_FAMILIES_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d language families from %s", len(data), path)
    return data


def bucket_samples(
    tree: dict[str, Any],
    group_by: str,
    families: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[str]], set[str]]:
    """Returns ``(group_key -> [sample_ids], unmatched_languages)``.

    Samples missing the requested attribute are dropped. ``unmatched``
    is non-empty only for ``language_family`` mode.
    """
    if group_by not in GROUP_BY_MODES:
        msg = f"Unknown group_by: {group_by!r}. Expected one of {list(GROUP_BY_MODES)}"
        raise ValueError(msg)
    if group_by in ("language", "language_family") and families is None:
        msg = f"{group_by} mode requires a loaded `families` list"
        raise ValueError(msg)

    samples = tree["samples"]
    groups: dict[str, list[str]] = {}
    unmatched: set[str] = set()
    lang_fallback = _country_language_fallback(families or [])

    for sid, s in samples.items():
        if group_by == "language_family":
            key = _match_family(s, families or [])
            if key is None and s.get("language"):
                unmatched.add(s["language"])
        elif group_by == "language":
            lang = s.get("language")
            if not lang:
                country = s.get("country_iso")
                lang = lang_fallback.get(country) if country else None
            key = lang
        else:
            key = s.get(GROUP_BY_ATTRS[group_by])
        if key is not None:
            groups.setdefault(key, []).append(sid)

    return groups, unmatched


def _country_language_fallback(families: list[dict[str, Any]]) -> dict[str, str]:
    fallback: dict[str, str] = {}
    for fam in families:
        mapping = fam.get("countries_if_no_language") or {}
        if isinstance(mapping, dict):
            for country, lang in mapping.items():
                fallback.setdefault(country, lang)
    return fallback


def _match_family(sample: dict[str, Any], families: list[dict[str, Any]]) -> str | None:
    """First family whose ``languages`` / ``regions`` / ``countries`` list contains
    any of the sample's values. Falls back to ``countries_if_no_language`` when
    the sample has no language set.
    """
    lang = sample.get("language")
    region = sample.get("country_subdivision")
    country = sample.get("country_iso")
    for fam in families:
        if lang and lang in fam.get("languages", []):
            return fam["name"]
        if region and region in fam.get("regions", []):
            return fam["name"]
        if country and country in fam.get("countries", []):
            return fam["name"]
        if not lang and country and country in fam.get("countries_if_no_language", []):
            return fam["name"]
    return None
