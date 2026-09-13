"""
app/llm.py - Groq LLM integration with structured JSON output.

Uses Groq's JSON mode (response_format={"type": "json_object"}) combined
with a strict schema description in the system prompt, then validates
the response with Pydantic for type safety.
"""

from __future__ import annotations

import json
import logging
import time

import groq as groq_sdk
from groq import Groq

from app import config
from app.models import CompanyExtraction, TeamMember, TokenUsage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing constants (USD per 1 million tokens, Groq on-demand pricing)
# ---------------------------------------------------------------------------
_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 1.79},
    "llama3-70b-8192":         {"input": 0.59, "output": 0.79},
    "llama3-8b-8192":          {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768":      {"input": 0.24, "output": 0.24},
    "gemma2-9b-it":            {"input": 0.20, "output": 0.20},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _PRICING.get(model, {"input": 0.59, "output": 1.79})
    return (
        prompt_tokens * pricing["input"] / 1_000_000
        + completion_tokens * pricing["output"] / 1_000_000
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a company intelligence analyst. You will be given clean text extracted \
from a company's public website pages. You must return a valid json object.

Your task is to extract structured information and return it as valid JSON \
matching EXACTLY this schema:

{
  "company_overview": "<2-sentence summary of what the company does>",
  "target_audience": "<ideal customer profile / target market>",
  "contact_points": ["email@example.com"],
  "leadership": [
    {"name": "Full Name", "role": "Job Title", "linkedin_url": null}
  ],
  "confidence_score": 0.85
}

Rules:
1. NEVER invent or hallucinate information. If you cannot find something, \
   use empty string "" or empty list [].
2. company_overview: Exactly 2 sentences describing what the company does \
   and its main product or service.
3. target_audience: Who the company's ideal customers are. Be specific.
4. contact_points: Only include email addresses that appear LITERALLY in the \
   text. Do NOT guess or construct emails.
5. leadership: Only include people explicitly mentioned by name and role. \
   linkedin_url must appear verbatim in the text or be null.
6. confidence_score: Float 0.0-1.0 rating your confidence in the extraction.
7. Return ONLY valid JSON. No markdown, no code fences, no explanation.
"""


def _build_user_prompt(domain: str, combined_text: str) -> str:
    return (
        f"Company domain: {domain}\n\n"
        f"Website content extracted from {domain} and its subpages:\n\n"
        f"{combined_text}\n\n"
        "Return the extracted company information as JSON."
    )


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_company_info(
    domain: str,
    combined_text: str,
) -> tuple[CompanyExtraction, TokenUsage]:
    """
    Send *combined_text* to Groq and return a validated CompanyExtraction.

    Uses JSON mode + manual Pydantic validation.
    Retries once on rate-limit (429) with a 15-second back-off.

    Returns
    -------
    (CompanyExtraction, TokenUsage)
    """
    client = Groq(api_key=config.GROQ_API_KEY)
    model = config.GROQ_MODEL

    last_exc: Exception | None = None
    completion = None

    # Pass 1: JSON mode (preferred — enforced schema)
    # Pass 2: Fallback without JSON mode, trimmed context (handles json_validate_failed)
    for attempt in range(2):
        use_json_mode = (attempt == 0)
        text_for_attempt = combined_text if attempt == 0 else combined_text[:12_000]
        current_messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(domain, text_for_attempt)},
        ]
        kwargs: dict = dict(
            model=model,
            messages=current_messages,  # type: ignore[arg-type]
            temperature=0.0,
        )
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = client.chat.completions.create(**kwargs)
            last_exc = None
            break
        except groq_sdk.RateLimitError as exc:
            last_exc = exc
            wait = 15
            logger.warning("Rate-limited by Groq. Retrying in %ds...", wait)
            time.sleep(wait)
            # Don't break — let loop continue to attempt 2 which also retries
        except groq_sdk.BadRequestError as exc:
            if "json_validate_failed" in str(exc) or "json" in str(exc).lower():
                logger.warning(
                    "JSON mode failed for %s (attempt %d), retrying without JSON mode...",
                    domain, attempt + 1,
                )
                last_exc = exc
                # Continue to attempt 2 (fallback without JSON mode)
            else:
                raise
        except groq_sdk.APIStatusError as exc:
            logger.error("Groq API error for %s: %s", domain, exc)
            raise

    if completion is None:
        raise last_exc or RuntimeError("All LLM attempts failed")

    usage = completion.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    token_usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
    )

    raw_content = completion.choices[0].message.content or "{}"

    # Parse JSON and validate with Pydantic
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw: {raw_content[:300]}") from exc

    # Coerce leadership list into TeamMember objects if they are plain dicts
    if "leadership" in data and isinstance(data["leadership"], list):
        data["leadership"] = [
            TeamMember(**m) if isinstance(m, dict) else m
            for m in data["leadership"]
        ]

    try:
        extracted = CompanyExtraction(**data)
    except Exception as exc:
        raise ValueError(f"Pydantic validation failed: {exc}") from exc

    logger.info(
        "Groq extraction for %s: %d tokens (~$%.5f)",
        domain,
        token_usage.total_tokens,
        token_usage.estimated_cost_usd,
    )

    return extracted, token_usage
