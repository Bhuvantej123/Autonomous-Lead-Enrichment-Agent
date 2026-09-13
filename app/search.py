"""
app/search.py - Lightweight web-search enrichment fallback.

Purpose
-------
When a company's website extraction is missing key signals (leadership,
LinkedIn URLs, contact emails), this module runs targeted web searches
and returns clean evidence text that is appended to the LLM context —
resulting in no additional LLM calls.

Design principles
-----------------
- Website evidence always has priority; search supplements, never overrides.
- Search is disabled by default (SEARCH_ENABLED=false in .env).
- Every failure path (timeout, API error, no results) is caught and logged.
  A search failure never fails a domain or the pipeline.
- Queries are tightly scoped to the specific company — no broad web crawls.
- Anti-hallucination rules are unchanged: the LLM prompt still requires
  literal presence in evidence for emails and LinkedIn URLs.
- Search sources are added to the domain's sources list in output.json.

Provider
--------
Serper.dev (https://serper.dev) — Google Search JSON API.
  - Free tier: 2,500 searches/month (no credit card required).
  - Single POST request, no SDK needed.
  - Set SEARCH_API_KEY in .env to activate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import requests

from app import config

logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"
_REQUEST_TIMEOUT = 10  # seconds per query


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str


@dataclass
class SearchEvidence:
    """Aggregated search evidence for one domain enrichment pass."""
    results: list[SearchResult] = field(default_factory=list)
    queries_run: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def found_anything(self) -> bool:
        return bool(self.results)

    def as_text(self) -> str:
        """
        Format search evidence as clean plain text for appending to the
        LLM's context window.

        Labelled clearly so the LLM knows to treat it as supplementary
        evidence and not override website content.
        """
        if not self.results:
            return ""
        lines = [
            "\n\n--- SUPPLEMENTARY WEB SEARCH EVIDENCE ---",
            "Note: Website content above has priority. Use search evidence only",
            "to fill fields that are absent in the website content.",
            "Only include emails/LinkedIn URLs that appear LITERALLY below.",
        ]
        for result in self.results:
            lines.append(f"\nSource URL : {result.url}")
            lines.append(f"Title      : {result.title}")
            lines.append(f"Snippet    : {result.snippet}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

_LEADERSHIP_TERMS = {
    "ceo", "founder", "co-founder", "chief executive", "president",
    "cto", "coo", "cpo", "vp of", "vice president", "head of",
}


def _detect_missing_signals(text: str) -> dict[str, bool]:
    """
    Scan extracted website text to identify which key signals are absent.

    Returns a dict:  {signal_name: True if MISSING, False if already found}
    """
    text_lower = text.lower()

    has_linkedin = "linkedin.com/in/" in text_lower
    has_email = bool(re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
    ))
    has_leadership = any(term in text_lower for term in _LEADERSHIP_TERMS)

    return {
        "leadership": not has_leadership,
        "linkedin":   not has_linkedin,
        "contact":    not has_email,
    }


# ---------------------------------------------------------------------------
# Query execution & filtering
# ---------------------------------------------------------------------------

def _run_query(query: str, api_key: str, max_results: int) -> list[SearchResult]:
    """
    Execute one Serper search query.
    Returns an empty list on any failure — always non-fatal.
    """
    try:
        resp = requests.post(
            _SERPER_URL,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": max_results},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results: list[SearchResult] = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", "").strip(),
                url=item.get("link", "").strip(),
                snippet=item.get("snippet", "").strip(),
            ))
        return results

    except requests.Timeout:
        logger.warning("Search timed out for query: %r", query)
    except requests.HTTPError as exc:
        logger.warning("Search HTTP error (%s) for query: %r", exc.response.status_code, query)
    except requests.RequestException as exc:
        logger.warning("Search request failed: %s", exc)
    except Exception as exc:
        logger.warning("Unexpected search error: %s", exc)

    return []


def _normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    return url.strip().rstrip("/").lower()


def _is_relevant_result(result: SearchResult, domain: str, company_name: str) -> bool:
    """
    Lightweight, conservative relevance filtering for search results.
    Ensures obviously irrelevant links (directory pages, job boards, off-target results)
    are filtered out before passing evidence to the LLM or adding to sources.
    """
    if not result.url or not result.title:
        return False

    url_lower = result.url.lower()
    title_lower = result.title.lower()
    snippet_lower = result.snippet.lower()
    text_combined = f"{title_lower} {snippet_lower} {url_lower}"

    domain_token = domain.split(".")[0].lower()
    company_token = company_name.lower()

    # 1. LinkedIn URL filtering
    if "linkedin.com" in url_lower:
        # Filter out generic directories, job searches, or generic article feeds
        if any(bad in url_lower for bad in ["/dir/", "/jobs/", "/directory/", "/learning/"]):
            return False

        # For LinkedIn profile links (/in/):
        # Prefer profile URLs whose title, snippet, or URL mentions company context or leadership role
        if "/in/" in url_lower:
            has_company = (company_token in text_combined) or (domain_token in text_combined)
            has_leadership = any(term in text_combined for term in _LEADERSHIP_TERMS)

            if not (has_company or has_leadership):
                return False

    # 2. General sanity check: Must contain brand reference or key search signals
    has_brand = (domain_token in text_combined) or (company_token in text_combined)
    has_signal = (
        any(term in text_combined for term in _LEADERSHIP_TERMS)
        or "@" in text_combined
        or "email" in text_combined
        or "contact" in text_combined
        or "executive" in text_combined
        or "leadership" in text_combined
    )

    if not (has_brand or has_signal):
        return False

    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enrich_with_search(
    domain: str,
    extracted_text: str,
    company_name: str | None = None,
) -> SearchEvidence:
    """
    Run targeted web searches for *domain* and return clean search evidence.

    This function:
    1. Checks whether search is enabled and an API key is available.
    2. Detects which key signals (leadership, LinkedIn, email) are absent
       in the already-extracted website text.
    3. Builds tightly-scoped queries only for missing signals.
    4. Runs the queries, filters/deduplicates results for relevance.
    5. Returns a SearchEvidence object whose .as_text() can be appended to
       the LLM context before the (single) LLM extraction call.

    Always returns a valid SearchEvidence — never raises.

    Parameters
    ----------
    domain:
        Company domain, e.g. "postman.com".
    extracted_text:
        Combined clean text from the Playwright crawl. Used to detect
        which signals are already present (website has priority).
    company_name:
        Optional human-readable company name override. If None, derived
        from the domain (e.g. "postman.com" → "Postman").
    """
    empty = SearchEvidence()

    # Guard: search disabled or no API key
    if not config.SEARCH_ENABLED:
        logger.debug("Search disabled for %s (SEARCH_ENABLED=false)", domain)
        return empty

    if not config.SEARCH_API_KEY:
        logger.warning(
            "SEARCH_ENABLED=true but SEARCH_API_KEY is not set — skipping search for %s",
            domain,
        )
        return empty

    # Derive a short display name from the domain
    if not company_name:
        company_name = domain.split(".")[0].capitalize()

    # Detect what's missing
    missing = _detect_missing_signals(extracted_text)
    missing_signals = [k for k, v in missing.items() if v]

    if not missing_signals:
        logger.info(
            "Search: website content for %s already contains all key signals — skipping",
            domain,
        )
        return empty

    # Build targeted queries — only for missing signals
    queries: list[str] = []

    if missing["leadership"] or missing["linkedin"]:
        queries.append(f'"{company_name}" CEO founder co-founder leadership team')

    if missing["linkedin"]:
        # Scoped LinkedIn query — only returns LinkedIn profile pages
        queries.append(f'site:linkedin.com/in "{company_name}" CEO founder')

    if missing["contact"]:
        queries.append(f'"{domain}" contact email')

    logger.info(
        "Search enrichment for %s: %d quer%s for missing signals: %s",
        domain,
        len(queries),
        "y" if len(queries) == 1 else "ies",
        ", ".join(missing_signals),
    )

    seen_urls: set[str] = set()
    evidence = SearchEvidence()

    for query in queries:
        results = _run_query(query, config.SEARCH_API_KEY, config.SEARCH_MAX_RESULTS)
        evidence.queries_run.append(query)

        for r in results:
            norm_url = _normalize_url(r.url)
            if not norm_url or norm_url in seen_urls:
                continue

            if not _is_relevant_result(r, domain, company_name):
                logger.debug("Search: filtered out irrelevant result %s for %s", r.url, domain)
                continue

            seen_urls.add(norm_url)
            evidence.results.append(r)
            if r.url not in evidence.sources:
                evidence.sources.append(r.url)

    if evidence.found_anything:
        logger.info(
            "Search enrichment for %s: found %d relevant result(s) across %d quer%s",
            domain,
            len(evidence.results),
            len(evidence.queries_run),
            "y" if len(evidence.queries_run) == 1 else "ies",
        )
    else:
        logger.info(
            "Search enrichment for %s: no relevant results returned (queries ran: %d)",
            domain,
            len(evidence.queries_run),
        )

    return evidence

