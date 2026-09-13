"""
app/crawler.py - Async Playwright-based web crawler.

For each domain the crawler:
  1. Opens the homepage.
  2. Discovers all internal <a href> links.
  3. Scores and filters links by relevance keywords.
  4. Crawls up to MAX_PAGES_PER_DOMAIN pages (homepage + subpages).
  5. Returns raw HTML + URL for each successfully fetched page.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeout,
    async_playwright,
)

from app import config
from app.utils import clean_url, extract_domain, is_relevant_path, normalize_url, same_domain

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Raw content of a single crawled page."""
    url: str
    html: str
    status: int


async def _fetch_page(page: Page, url: str, timeout_ms: int) -> PageContent | None:
    """
    Navigate to *url* and return its HTML.
    Returns None on timeout, HTTP error, or navigation failure.
    """
    try:
        response = await page.goto(
            url,
            timeout=timeout_ms,
            wait_until="domcontentloaded",
        )
        if response is None:
            logger.warning("No response for %s", url)
            return None

        status = response.status
        if status == 404:
            logger.debug("404 for %s", url)
            return None
        if status >= 400:
            logger.warning("HTTP %d for %s", status, url)
            return None

        # Extra wait for JS-heavy SPAs (give React/Vue time to render)
        try:
            await page.wait_for_load_state("networkidle", timeout=5_000)
        except PlaywrightTimeout:
            pass  # Acceptable — we already have domcontentloaded

        html = await page.content()
        return PageContent(url=page.url, html=html, status=status)

    except PlaywrightTimeout:
        logger.warning("Timeout loading %s", url)
        return None
    except Exception as exc:
        logger.warning("Error loading %s: %s", url, exc)
        return None


async def _discover_links(page: Page, base_domain: str) -> list[str]:
    """
    Extract all <a href> links from the current page, resolve them, and return
    internal links that are on the same domain.
    """
    base_url = page.url
    try:
        hrefs: list[str] = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.getAttribute('href'))",
        )
    except Exception as exc:
        logger.debug("Could not collect links: %s", exc)
        return []

    seen: set[str] = set()
    links: list[str] = []

    for href in hrefs:
        absolute = clean_url(href, base_url)
        if absolute is None:
            continue
        if not same_domain(absolute, base_domain):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)

    return links


def _score_link(url: str, keywords: list[str]) -> int:
    """
    Return a relevance score for *url*.
    Higher = more relevant to company intelligence.
    """
    path = urlparse(url).path.lower()
    score = 0
    for kw in keywords:
        if kw in path:
            score += 1
    return score


async def crawl_domain(domain: str, browser: Browser) -> list[PageContent]:
    """
    Crawl *domain* and return a list of PageContent objects (homepage + relevant
    subpages), up to config.MAX_PAGES_PER_DOMAIN.

    All errors are caught here — returns an empty list on total failure.
    """
    base_url = normalize_url(domain)
    base_domain = extract_domain(domain)
    pages_collected: list[PageContent] = []
    visited: set[str] = set()

    context: BrowserContext = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        java_script_enabled=True,
        ignore_https_errors=True,
    )
    # Block heavy media assets to speed up crawling
    await context.route(
        "**/*.{png,jpg,jpeg,gif,webp,mp4,woff,woff2,ttf,otf}",
        lambda route, _: route.abort(),
    )

    page: Page = await context.new_page()

    try:
        logger.info("Crawling homepage: %s", base_url)
        homepage = await _fetch_page(page, base_url, config.BROWSER_TIMEOUT_MS)
        if homepage is None:
            logger.error("Failed to load homepage for %s", domain)
            return []

        pages_collected.append(homepage)
        visited.add(homepage.url)
        visited.add(base_url)

        # Discover links from homepage
        all_links = await _discover_links(page, base_domain)
        logger.debug("Discovered %d internal links on %s", len(all_links), domain)

        # Score and sort by relevance
        relevant_links = [
            lnk for lnk in all_links
            if is_relevant_path(urlparse(lnk).path, config.RELEVANT_PATH_KEYWORDS)
            and lnk not in visited
        ]
        relevant_links.sort(
            key=lambda u: _score_link(u, config.RELEVANT_PATH_KEYWORDS),
            reverse=True,
        )

        # Crawl up to MAX_PAGES_PER_DOMAIN - 1 (homepage already counted)
        remaining_slots = config.MAX_PAGES_PER_DOMAIN - 1
        for link in relevant_links[:remaining_slots]:
            if link in visited:
                continue
            visited.add(link)

            logger.info("  Fetching subpage: %s", link)
            result = await _fetch_page(page, link, config.BROWSER_TIMEOUT_MS)
            if result is not None:
                pages_collected.append(result)

            # Small polite delay
            await asyncio.sleep(0.5)

    except Exception as exc:
        logger.error("Unexpected crawler error for %s: %s", domain, exc)
    finally:
        await page.close()
        await context.close()

    logger.info(
        "Crawled %d page(s) for %s", len(pages_collected), domain
    )
    return pages_collected


async def crawl_domains(domains: list[str]) -> dict[str, list[PageContent]]:
    """
    Crawl a list of domains using a single shared browser instance.
    Returns a dict mapping domain → list[PageContent].
    """
    results: dict[str, list[PageContent]] = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            for domain in domains:
                try:
                    pages = await crawl_domain(domain, browser)
                    results[domain] = pages
                except Exception as exc:
                    logger.error("crawl_domain failed for %s: %s", domain, exc)
                    results[domain] = []
        finally:
            await browser.close()

    return results
