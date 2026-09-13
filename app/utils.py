"""
app/utils.py - Shared helper utilities.
"""

from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import urljoin, urlparse


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure a clean, consistent log format for the whole application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Ensure a URL has an https scheme and no trailing slash."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


def extract_domain(url: str) -> str:
    """Return the bare hostname (e.g. 'stripe.com') from a URL."""
    parsed = urlparse(normalize_url(url))
    # Strip www. prefix so comparisons are consistent
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def same_domain(url: str, base_domain: str) -> bool:
    """
    Return True if *url* belongs to *base_domain* (or a www. sub-domain of it).
    """
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host == base_domain or host.endswith("." + base_domain)


def is_relevant_path(path: str, keywords: list[str]) -> bool:
    """Return True if the URL path contains any of the relevance keywords."""
    path_lower = path.lower()
    return any(kw in path_lower for kw in keywords)


def clean_url(href: str, base_url: str) -> str | None:
    """
    Resolve a relative href against base_url and return an absolute URL.
    Returns None for anchors, mailto:, tel:, javascript:, and other
    non-HTTP schemes.
    """
    if not href:
        return None
    href = href.strip()
    # Skip non-navigable hrefs
    if href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
        return None
    resolved = urljoin(base_url, href)
    parsed = urlparse(resolved)
    if parsed.scheme not in ("http", "https"):
        return None
    # Drop query strings and fragments for deduplication
    clean = parsed._replace(query="", fragment="").geturl()
    return clean.rstrip("/")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_json(data: dict, path: str) -> None:
    """Write *data* as pretty-printed JSON to *path*, creating dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace / blank lines into single spaces/newlines."""
    # Collapse multiple spaces on the same line
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deduplicate_lines(text: str) -> str:
    """
    Remove consecutive duplicate lines (catches repeated nav/footer snippets).
    Keeps the first occurrence of any line seen more than 3 times.
    """
    counts: dict[str, int] = {}
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        counts[stripped] = counts.get(stripped, 0) + 1
        if counts[stripped] <= 3:
            kept.append(line)
    return "\n".join(kept)
