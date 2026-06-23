"""Fetch YFull tree pages with on-disk caching."""

import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

YFULL_URL = "https://www.yfull.com/tree/{haplogroup}/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0
CACHE_MAX_AGE_DAYS = 3


def fetch_tree(haplogroup: str, cache_dir: Path, force_refresh: bool = False) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_safe(haplogroup)}.html"

    if not force_refresh and path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        if age_days < CACHE_MAX_AGE_DAYS:
            log.info("Cache hit: %s (age %.1f days)", haplogroup, age_days)
            return path.read_text(encoding="utf-8")

    html = _fetch_with_retry(YFULL_URL.format(haplogroup=haplogroup))
    path.write_text(html, encoding="utf-8")
    log.info("Cached: %s -> %s (%d bytes)", haplogroup, path.name, len(html))
    return html


def _fetch_with_retry(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("GET %s (attempt %d/%d)", url, attempt, MAX_RETRIES)
            resp = httpx.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.text
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                backoff = BACKOFF_SECONDS * 2 ** (attempt - 1)
                log.warning("Fetch failed: %s. Retrying in %.1fs", e, backoff)
                time.sleep(backoff)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts: {last_error}")


def _safe(haplogroup: str) -> str:
    return haplogroup.replace("/", "_").replace("*", "_star")
