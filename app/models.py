"""
app/models.py - Pydantic data models for the enrichment pipeline.

These models define the complete input/output schema and are shared
across the crawler, extractor, LLM, and pipeline layers.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM extraction schema
# ---------------------------------------------------------------------------

class TeamMember(BaseModel):
    """A key leadership or team member discovered from the website."""
    name: str = Field(description="Full name of the person")
    role: str = Field(description="Job title or role at the company")
    linkedin_url: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL, only if found in page content",
    )


class CompanyExtraction(BaseModel):
    """
    Structured company intelligence extracted by the LLM.
    All fields that cannot be found must be left empty/null — never invented.
    """
    company_overview: str = Field(
        description=(
            "A concise 2-sentence summary of what the company does and its "
            "main product or service. Leave empty string if not determinable."
        )
    )
    target_audience: str = Field(
        description=(
            "The company's ideal customer profile or target audience. "
            "Leave empty string if not determinable."
        )
    )
    contact_points: list[str] = Field(
        default_factory=list,
        description=(
            "Public contact emails found verbatim on the website. "
            "Do NOT invent emails. Use empty list if none found."
        ),
    )
    leadership: list[TeamMember] = Field(
        default_factory=list,
        description=(
            "Key leadership or team members with their roles. "
            "Only include people explicitly mentioned on the pages. "
            "Use empty list if none found."
        ),
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence that the extracted information is accurate and "
            "complete, from 0.0 (no useful data) to 1.0 (highly confident)."
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline output schema
# ---------------------------------------------------------------------------

class EnrichedCompany(BaseModel):
    """Final enriched result for a single domain."""
    domain: str
    company_overview: str = ""
    target_audience: str = ""
    contact_points: list[str] = Field(default_factory=list)
    leadership: list[TeamMember] = Field(default_factory=list)
    confidence_score: float = 0.0
    sources: list[str] = Field(default_factory=list)
    status: str = "success"           # "success" | "failed" | "partial"
    error: Optional[str] = None


class TokenUsage(BaseModel):
    """Token and estimated cost tracking for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class PipelineOutput(BaseModel):
    """Top-level output written to output/output.json."""
    results: list[EnrichedCompany]
    total_domains: int
    successful: int
    failed: int
    token_usage: Optional[TokenUsage] = None
