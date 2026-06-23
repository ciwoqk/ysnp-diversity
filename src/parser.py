"""Parse YFull tree HTML into ``{branches, samples}`` dicts.

HTML structure (reference):
    <ul id="tree">
      <li id="lG-XXXX">                    # branch
        <a>...</a>                          # haplogroup name
        <span class="yf-snpforhg">SNP * SNP</span>
        <span class="yf-plus-snps" title="SNP * SNP * ...">+N SNPs</span>
        <span class="yf-age" title="formed CI ... TMRCA CI ...">formed N ybp, TMRCA N ybp</span>
        <ul id="sG-XXXX|uG-XXXX">...children...</ul>
      </li>
      <li valSampleID="YF...">              # sample
        <span class="yf-s-name [yf-s-vcf|yf-s-adna]">id:XXX</span>
        <b class="yf-geo fl XX" title="Country (region)">CCC [XX-YY]</b>
        <b class="yf-geo yf-lang" title="LangName">lng</b>
        <span class="yf-a-age" title="Ancient DNA | N (lo - hi) ybp">age</span>
        <span class="yf-sinf" title="= G-X G-X* G-Y ...">i</span>
      </li>
    </ul>
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any

from bs4 import BeautifulSoup, Tag

log = logging.getLogger(__name__)


_AGE_RE = re.compile(r"formed\s+(\d+)\s+ybp,\s+TMRCA\s+(\d+)\s+ybp", re.IGNORECASE)
_CI_RE = re.compile(
    r"formed CI 95%\s+(\d+)\s*<->\s*(\d+)\s+ybp,"
    r"\s+TMRCA CI 95%\s+(\d+)\s*<->\s*(\d+)\s+ybp",
    re.IGNORECASE,
)
_ADNA_AGE_RE = re.compile(r"Ancient DNA\s*\|\s*(\d+)\s+\((\d+)\s*-\s*(\d+)\)\s+ybp", re.IGNORECASE)
_SUBDIV_RE = re.compile(r"\[([A-Z]{2}-[A-Z0-9]+)\]")
_VERSION_RE = re.compile(r"v\d+\.\d+\.\d+")


def parse_tree(html: str) -> dict[str, Any]:
    """Parse YFull tree HTML into ``{root, tree_version, branches, samples}``."""
    soup = BeautifulSoup(html, "lxml")

    root_ul = soup.find("ul", id="tree")
    if not isinstance(root_ul, Tag):
        raise ValueError("No <ul id='tree'> found in HTML")

    branches: dict[str, dict[str, Any]] = {}
    samples: dict[str, dict[str, Any]] = {}
    _walk(root_ul, parent=None, branches=branches, samples=samples)

    if not branches:
        raise ValueError("No branches parsed from tree HTML")

    log.info("Parsed: branches=%d, samples=%d", len(branches), len(samples))
    return {
        "root": next(iter(branches)),
        "tree_version": _parse_version(soup),
        "parsed_at": datetime.now(UTC).isoformat(),
        "branches": branches,
        "samples": samples,
    }


def _walk(
    ul: Tag,
    parent: str | None,
    branches: dict[str, dict[str, Any]],
    samples: dict[str, dict[str, Any]],
) -> None:
    for li in ul.find_all("li", recursive=False):
        if not isinstance(li, Tag):
            continue
        if li.has_attr("valsampleid"):
            sample = _parse_sample(li, branch=parent or "")
            if sample and parent:
                sid = sample["sample_id"]
                if sid in samples:
                    log.warning(
                        "Duplicate sample %s: already under %s, also seen under %s — keeping first",
                        sid,
                        samples[sid]["branch"],
                        parent,
                    )
                else:
                    samples[sid] = sample
                    branches[parent]["samples"].append(sid)
            continue

        li_id = str(li.get("id", ""))
        if not li_id.startswith("l"):
            continue

        branch = _parse_branch(li, parent=parent)
        if branch is None:
            continue
        branches[branch["haplogroup"]] = branch
        if parent:
            branches[parent]["children"].append(branch["haplogroup"])

        inner_ul = li.find("ul", recursive=False)
        if isinstance(inner_ul, Tag):
            _walk(inner_ul, parent=branch["haplogroup"], branches=branches, samples=samples)


def _parse_branch(li: Tag, parent: str | None) -> dict[str, Any] | None:
    a = li.find("a", recursive=False)
    if not isinstance(a, Tag):
        return None
    haplogroup = a.get_text(strip=True)
    if not haplogroup:
        return None

    a_classes = _classes(a)
    branch: dict[str, Any] = {
        "haplogroup": haplogroup,
        "parent": parent,
        "is_star": haplogroup.endswith("*"),
        "formed_ybp": None,
        "tmrca_ybp": None,
        "formed_ci95": None,
        "tmrca_ci95": None,
        "snps": [],
        "plus_snps": [],
        "is_new": "yf-node-new1" in a_classes,
        "is_modified": "yf-node-new2" in a_classes,
        "children": [],
        "samples": [],
    }

    snpforhg = li.find("span", class_="yf-snpforhg", recursive=False)
    if isinstance(snpforhg, Tag):
        branch["snps"] = _split_snps(snpforhg.get_text(strip=True))

    plus = li.find("span", class_="yf-plus-snps", recursive=False)
    if isinstance(plus, Tag):
        branch["plus_snps"] = _split_snps(str(plus.get("title", "")))

    age = li.find("span", class_="yf-age", recursive=False)
    if isinstance(age, Tag):
        m = _AGE_RE.search(age.get_text())
        if m:
            branch["formed_ybp"] = float(m.group(1))
            branch["tmrca_ybp"] = float(m.group(2))
        ci = _CI_RE.search(str(age.get("title", "")))
        if ci:
            branch["formed_ci95"] = [float(ci.group(1)), float(ci.group(2))]
            branch["tmrca_ci95"] = [float(ci.group(3)), float(ci.group(4))]

    return branch


def _split_snps(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in text.split("*") if s.strip()]


def _parse_sample(li: Tag, branch: str) -> dict[str, Any] | None:
    sample_id = str(li.get("valsampleid") or "").strip()
    if not sample_id:
        return None

    sample: dict[str, Any] = {
        "sample_id": sample_id,
        "branch": branch,
        "country_iso": None,
        "country_name": None,
        "country_subdivision": None,
        "language": None,
        "is_adna": False,
        "is_vcf": False,
        "age_ybp": None,
        "age_ybp_range": None,
        "compatible_branches": [],
        "is_new": False,
    }

    name = li.find("span", class_="yf-s-name")
    if isinstance(name, Tag):
        classes = _classes(name)
        sample["is_adna"] = "yf-s-adna" in classes
        sample["is_vcf"] = "yf-s-vcf" in classes

    for b in li.find_all("b", class_="yf-geo"):
        b_classes = _classes(b)
        if "yf-lang" in b_classes:
            sample["language"] = b.get_text(strip=True) or None
            continue
        sample["country_iso"] = _extract_iso(b_classes)
        sample["country_name"] = str(b.get("title") or "") or None
        sub = _SUBDIV_RE.search(b.get_text(strip=True))
        if sub:
            sample["country_subdivision"] = sub.group(1)

    age = li.find("span", class_="yf-a-age")
    if isinstance(age, Tag):
        m = _ADNA_AGE_RE.search(str(age.get("title", "")))
        if m:
            sample["age_ybp"] = float(m.group(1))
            sample["age_ybp_range"] = [float(m.group(2)), float(m.group(3))]

    sinf = li.find("span", class_="yf-sinf")
    if isinstance(sinf, Tag):
        title = str(sinf.get("title", "")).lstrip("=").strip()
        sample["compatible_branches"] = title.split() if title else []

    if li.find("span", class_="yf-new") is not None:
        sample["is_new"] = True

    return sample


def _extract_iso(classes: list[str]) -> str | None:
    """ISO country code from a yf-geo class list, e.g. ``['yf-geo', 'fl', 'RU']`` → ``'RU'``."""
    for i, c in enumerate(classes):
        if c == "fl" and i + 1 < len(classes):
            return classes[i + 1]
    return None


def _classes(tag: Tag) -> list[str]:
    cls = tag.get("class")
    if cls is None:
        return []
    if isinstance(cls, str):
        return cls.split()
    return list(cls)


def _parse_version(soup: BeautifulSoup) -> str | None:
    """Extract YTree version like ``v14.02.00`` from page text."""
    m = _VERSION_RE.search(soup.get_text())
    return m.group(0) if m else None
