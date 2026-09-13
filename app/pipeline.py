"""
app/pipeline.py - Per-domain orchestration and confidence scoring.

Flow for each domain:
  crawl → extract text → call LLM → compute confidence → build result

Errors at any stage are caught and recorded in EnrichedCompany.error
so that one failed domain never terminates the overall run.
"""

from __future__ import annotations

import asyncio
import logging

from app import config
from app.crawler import crawl_domains
from app.extractor import combine_pages, extract_text
from app.llm import extract_company_info
from app.models import EnrichedCompany, PipelineOutput, TokenUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    company_overview: str,
    target_audience: str,
    contact_points: list[str],
    leadership: list,
    pages_crawled: int,
    llm_confidence: float,
) -> float:
    """
    Compute a deterministic confidence score (0.0–1.0) based on objective
    signals about data completeness, blended with the LLM's own confidence.

    Breakdown
    ---------
    | Signal                    | Points |
    |---------------------------|--------|
    | company_overview present  |  0.20  |
    | target_audience present   |  0.15  |
    | contact_points found      |  0.10  |
    | leadership found          |  0.20  |
    | pages_crawled >= 3        |  0.15  |
    | LLM confidence (weighted) |  0.20  |
    |------------------------------------|
    | Total possible            |  1.00  |
    """
    score = 0.0
    if company_overview and len(company_overview) > 20:
        score += 0.20
    if target_audience and len(target_audience) > 10:
        score += 0.15
    if contact_points:
        score += 0.10
    if leadership:
        score += 0.20
    if pages_crawled >= 3:
        score += 0.15
    # Blend LLM confidence (0–1) with weight 0.20
    score += min(max(llm_confidence, 0.0), 1.0) * 0.20

    return round(min(score, 1.0), 2)


# ---------------------------------------------------------------------------
# Per-domain processing
# ---------------------------------------------------------------------------

def process_domain(domain: str, page_contents) -> tuple[EnrichedCompany, TokenUsage | None]:
    """
    Given pre-crawled page contents for *domain*, run extraction → LLM → result.
    Returns (EnrichedCompany, optional TokenUsage).
    """
    if not page_contents:
        return (
            EnrichedCompany(
                domain=domain,
                status="failed",
                error="Crawler returned no pages (site unreachable, blocked, or timed out)",
            ),
            None,
        )

    # Extract clean text from each page
    page_texts = []
    sources = []
    for pc in page_contents:
        pt = extract_text(pc.html, pc.url)
        if pt.text.strip():
            page_texts.append(pt)
            sources.append(pt.url)

    if not page_texts:
        return (
            EnrichedCompany(
                domain=domain,
                status="failed",
                error="All crawled pages produced empty text after extraction",
            ),
            None,
        )

    extracted_content = combine_pages(page_texts)
    logger.info(
        "Extracted %d chars from %d page(s) for %s",
        extracted_content.total_chars,
        len(page_texts),
        domain,
    )

    # LLM extraction
    try:
        company_data, token_usage = extract_company_info(
            domain=domain,
            combined_text=extracted_content.combined_text,
        )
    except Exception as exc:
        logger.error("LLM extraction failed for %s: %s", domain, exc)
        return (
            EnrichedCompany(
                domain=domain,
                status="failed",
                error=f"LLM extraction error: {exc}",
                sources=sources,
            ),
            None,
        )

    # Deterministic confidence override
    confidence = compute_confidence(
        company_overview=company_data.company_overview,
        target_audience=company_data.target_audience,
        contact_points=company_data.contact_points,
        leadership=company_data.leadership,
        pages_crawled=len(page_texts),
        llm_confidence=company_data.confidence_score,
    )

    result = EnrichedCompany(
        domain=domain,
        company_overview=company_data.company_overview,
        target_audience=company_data.target_audience,
        contact_points=company_data.contact_points,
        leadership=company_data.leadership,
        confidence_score=confidence,
        sources=sources,
        status="success",
    )

    return result, token_usage


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(domains: list[str]) -> PipelineOutput:
    """
    Run the full enrichment pipeline for all *domains*.

    Returns a PipelineOutput with all results and aggregated token usage.
    """
    logger.info("=== Pipeline start: %d domain(s) ===", len(domains))

    # Step 1: Crawl all domains (async, in one browser session)
    all_page_contents = asyncio.run(crawl_domains(domains))

    # Step 2: Process each domain synchronously (LLM calls are sync)
    results: list[EnrichedCompany] = []
    total_usage = TokenUsage()

    for domain in domains:
        logger.info("--- Processing: %s ---", domain)
        try:
            page_contents = all_page_contents.get(domain, [])
            enriched, token_usage = process_domain(domain, page_contents)
            results.append(enriched)

            if token_usage:
                total_usage.prompt_tokens += token_usage.prompt_tokens
                total_usage.completion_tokens += token_usage.completion_tokens
                total_usage.total_tokens += token_usage.total_tokens
                total_usage.estimated_cost_usd += token_usage.estimated_cost_usd

        except Exception as exc:
            # Absolute last-resort catch — should never reach here
            logger.error("Unexpected pipeline error for %s: %s", domain, exc)
            results.append(
                EnrichedCompany(
                    domain=domain,
                    status="failed",
                    error=f"Unexpected error: {exc}",
                )
            )

    successful = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")

    logger.info(
        "=== Pipeline complete: %d success, %d failed | %d tokens (~$%.4f) ===",
        successful,
        failed,
        total_usage.total_tokens,
        total_usage.estimated_cost_usd,
    )

    return PipelineOutput(
        results=results,
        total_domains=len(domains),
        successful=successful,
        failed=failed,
        token_usage=total_usage if total_usage.total_tokens > 0 else None,
    )
