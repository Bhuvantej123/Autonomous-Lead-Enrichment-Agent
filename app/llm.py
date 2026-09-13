"""
app/llm.py - Groq LLM integration with strict JSON Schema structured outputs.

Structured-output strategy
--------------------------
1. The JSON Schema is derived directly from the Pydantic CompanyExtraction
   model via CompanyExtraction.model_json_schema(), so app/models.py remains
   the single source of truth.
2. The schema is post-processed by _make_strict_schema() to satisfy Groq's
   strict mode requirements (all properties required, additionalProperties:false).
3. The schema is sent to Groq as:
       response_format = {
           "type": "json_schema",
           "json_schema": {
               "name": "company_extraction",
               "strict": True,
               "schema": <processed schema>,
           }
       }
4. Pydantic validation is applied again after parsing as a second enforcement
   layer.
5. If the model returns json_validate_failed (rare on large contexts), a
   fallback pass uses json_object mode with a trimmed context.
"""

from __future__ import annotations

import copy
import json
import logging
import time

import groq as groq_sdk
from groq import Groq

from app import config
from app.models import CompanyExtraction, TeamMember, TokenUsage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing constants (USD per 1 million tokens, Groq on-demand)
# ---------------------------------------------------------------------------
_PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-oss-20b":      {"input": 0.90, "output": 0.90},
    "openai/gpt-oss-120b":     {"input": 0.90, "output": 0.90},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 1.79},
    "llama3-70b-8192":         {"input": 0.59, "output": 0.79},
    "llama3-8b-8192":          {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768":      {"input": 0.24, "output": 0.24},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _PRICING.get(model, {"input": 0.90, "output": 0.90})
    return (
        prompt_tokens * pricing["input"] / 1_000_000
        + completion_tokens * pricing["output"] / 1_000_000
    )


# ---------------------------------------------------------------------------
# JSON Schema generation and strict-mode preprocessing
# ---------------------------------------------------------------------------

def _make_strict_schema(schema: dict) -> dict:
    """
    Post-process a Pydantic-generated JSON Schema to satisfy Groq strict mode:

    Rules applied
    -------------
    - Every object's properties are ALL added to ``required`` (including those
      that have defaults in Pydantic, e.g. Optional fields and list fields).
    - ``additionalProperties: false`` is added to every object.
    - ``default`` values are removed (strict mode ignores/rejects them).
    - Recurses through ``$defs``, array ``items``, and ``anyOf``/``oneOf``/
      ``allOf`` composition keywords.

    The Pydantic models in app/models.py remain the single source of truth.
    This function only reshapes the schema for API transport.
    """
    schema = copy.deepcopy(schema)

    def _process(node: dict) -> dict:
        # Objects: make all props required, forbid extra props
        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties", {})
            node["required"] = list(props.keys())
            node["additionalProperties"] = False
            node.pop("default", None)
            for key in props:
                props[key] = _process(props[key])

        # Recurse into $defs (Pydantic puts nested models here)
        for key in node.get("$defs", {}):
            node["$defs"][key] = _process(node["$defs"][key])

        # Recurse into array items
        if "items" in node:
            node["items"] = _process(node["items"])

        # Recurse into composition keywords (anyOf covers Optional[X])
        for kw in ("anyOf", "oneOf", "allOf"):
            if kw in node:
                node[kw] = [_process(item) for item in node[kw]]

        # Drop default at every level
        node.pop("default", None)

        return node

    return _process(schema)


def _build_response_format_strict() -> dict:
    """
    Build the ``response_format`` dict for Groq strict JSON Schema mode.
    Schema is derived from CompanyExtraction and processed for strict mode.
    Called once at module load — schema is static per process.
    """
    raw_schema = CompanyExtraction.model_json_schema()
    strict_schema = _make_strict_schema(raw_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "company_extraction",
            "strict": True,
            "schema": strict_schema,
        },
    }


# Build once at module load — avoids rebuilding on every LLM call
_RESPONSE_FORMAT_STRICT = _build_response_format_strict()
_RESPONSE_FORMAT_OBJECT = {"type": "json_object"}   # fallback only


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a company intelligence analyst. You will be given clean text extracted \
from a company's public website pages. You must return a valid json object \
matching the provided schema exactly.

Rules:
1. NEVER invent or hallucinate information. If you cannot find something, \
   use empty string "" or empty list [].
2. company_overview: Exactly 2 sentences describing what the company does \
   and its main product or service.
3. target_audience: Who the company's ideal customers are (ICP). Be specific.
4. contact_points: Only include email addresses that appear LITERALLY in the \
   text. Do NOT guess or construct emails.
5. leadership: Only include people explicitly mentioned by name and role. \
   linkedin_url must appear verbatim in the text or be null.
6. confidence_score: Float 0.0-1.0 rating your confidence in the extraction.
"""


def _build_user_prompt(domain: str, combined_text: str) -> str:
    return (
        f"Company domain: {domain}\n\n"
        f"Website content extracted from {domain} and its subpages:\n\n"
        f"{combined_text}\n\n"
        "Return the extracted company information as a JSON object."
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

    Attempt strategy
    ----------------
    Pass 1 — ``json_schema`` strict mode (full context, schema-enforced).
              This is the preferred path and satisfies the structured-output
              requirement with strict Pydantic-derived JSON Schema.
    Pass 2 — ``json_object`` fallback (trimmed 12 k-char context).
              Used when the model returns ``json_validate_failed`` on very
              large inputs. Pydantic still validates the output.

    Validation layers
    -----------------
    Layer 1 (API): Groq enforces the JSON Schema at generation time.
    Layer 2 (code): Pydantic validates the parsed dict before returning.
    """
    client = Groq(api_key=config.GROQ_API_KEY)
    model = config.GROQ_MODEL

    attempt_configs = [
        {
            "label": "json_schema strict",
            "response_format": _RESPONSE_FORMAT_STRICT,
            "text": combined_text,
        },
        {
            "label": "json_object fallback",
            "response_format": _RESPONSE_FORMAT_OBJECT,
            "text": combined_text[:12_000],
        },
    ]

    last_exc: Exception | None = None
    completion = None

    for cfg in attempt_configs:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(domain, cfg["text"])},
        ]
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_format=cfg["response_format"],
                temperature=0.0,
            )
            last_exc = None
            logger.debug("LLM pass succeeded [%s] for %s", cfg["label"], domain)
            break

        except groq_sdk.RateLimitError as exc:
            last_exc = exc
            wait = 15
            logger.warning(
                "Rate-limited by Groq [%s]. Retrying in %ds...", cfg["label"], wait
            )
            time.sleep(wait)
            # Groq SDK has its own internal retry; we also continue to the
            # next attempt config if the rate limit persists

        except groq_sdk.BadRequestError as exc:
            err_str = str(exc)
            if "json_validate_failed" in err_str or "json" in err_str.lower():
                logger.warning(
                    "Schema validation failed [%s] for %s — trying fallback",
                    cfg["label"],
                    domain,
                )
                last_exc = exc
                # Continue to next attempt
            else:
                raise

        except groq_sdk.APIStatusError as exc:
            logger.error("Groq API error for %s: %s", domain, exc)
            raise

    if completion is None:
        raise last_exc or RuntimeError("All LLM attempts exhausted")

    # ── Token / cost tracking ──────────────────────────────────────────────
    usage = completion.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    token_usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=_estimate_cost(model, prompt_tokens, completion_tokens),
    )

    # ── Layer 2: Parse JSON + Pydantic validation ─────────────────────────
    raw_content = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}\nRaw: {raw_content[:300]}"
        ) from exc

    # Coerce leadership list items from raw dicts → TeamMember instances
    if isinstance(data.get("leadership"), list):
        data["leadership"] = [
            TeamMember(**m) if isinstance(m, dict) else m
            for m in data["leadership"]
        ]

    try:
        extracted = CompanyExtraction(**data)
    except Exception as exc:
        raise ValueError(f"Pydantic validation failed: {exc}") from exc

    logger.info(
        "Groq extraction for %s [%s]: %d tokens (~$%.5f)",
        domain,
        model,
        token_usage.total_tokens,
        token_usage.estimated_cost_usd,
    )
    return extracted, token_usage
