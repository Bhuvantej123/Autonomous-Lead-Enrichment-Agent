"""
app/extractor.py - Converts raw HTML into clean plain text for LLM consumption.

Why we don't send raw HTML to the LLM:
  - Raw HTML contains thousands of tokens of irrelevant noise (tags, attrs,
    scripts, styles, SVGs, tracking snippets) that waste context window space
    and degrade extraction quality.
  - Clean text is cheaper, faster, and produces more accurate LLM output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment

from app import config
from app.utils import deduplicate_lines, normalize_whitespace

logger = logging.getLogger(__name__)

# Tags whose entire subtree we remove — they carry no visible text value
_REMOVE_TAGS = {
    "script", "style", "svg", "noscript", "iframe",
    "nav", "footer", "header", "aside",
    "form", "button", "input", "select", "textarea",
    "meta", "link", "head",
}


@dataclass
class PageText:
    """Clean text extracted from a single crawled page."""
    url: str
    text: str
    char_count: int


@dataclass
class ExtractedContent:
    """Aggregated extraction result for all pages of a single domain."""
    pages: list[PageText]
    combined_text: str      # Merged context sent to the LLM
    total_chars: int


def extract_text(html: str, url: str) -> PageText:
    """
    Parse *html* and return clean, deduped, whitespace-normalised plain text.

    Steps
    -----
    1. Parse with lxml (fast).
    2. Remove invisible / boilerplate tags.
    3. Remove HTML comments.
    4. Extract visible text.
    5. Normalise whitespace and deduplicate repeated lines.
    6. Truncate to MAX_CHARS_PER_PAGE.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags
    for tag in soup.find_all(_REMOVE_TAGS):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Get visible text
    raw_text = soup.get_text(separator="\n")
    cleaned = normalize_whitespace(raw_text)
    cleaned = deduplicate_lines(cleaned)

    # Truncate per-page
    if len(cleaned) > config.MAX_CHARS_PER_PAGE:
        cleaned = cleaned[: config.MAX_CHARS_PER_PAGE] + "\n[... truncated ...]"

    return PageText(url=url, text=cleaned, char_count=len(cleaned))


def combine_pages(pages: list[PageText]) -> ExtractedContent:
    """
    Merge all page texts into a single context block with URL labels.
    Truncates the combined result to MAX_CHARS_COMBINED.
    """
    sections: list[str] = []
    for page in pages:
        if page.text.strip():
            header = f"\n\n--- SOURCE: {page.url} ---\n"
            sections.append(header + page.text)

    combined = "".join(sections)

    if len(combined) > config.MAX_CHARS_COMBINED:
        combined = combined[: config.MAX_CHARS_COMBINED] + "\n[... combined text truncated ...]"
        logger.debug("Combined text truncated to %d chars", config.MAX_CHARS_COMBINED)

    return ExtractedContent(
        pages=pages,
        combined_text=combined,
        total_chars=len(combined),
    )
