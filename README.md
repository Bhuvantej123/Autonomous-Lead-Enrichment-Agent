# Autonomous Lead Enrichment Agent

A production-quality Python pipeline that accepts a list of company domains, autonomously crawls their public websites using a headless Chromium browser, extracts clean structured text, and uses **Groq** with **strict JSON Schema structured outputs** to produce actionable company intelligence in JSON format.

---

## Overview

Given a list of domains such as `postman.com`, `supabase.com`, and `vapi.ai`, the agent:

1. Visits each company's homepage with Playwright (headless Chromium).
2. Discovers and scores internal subpages (`/about`, `/team`, `/contact`, `/pricing`, `/leadership`, etc.).
3. Extracts visible text from the DOM — removing scripts, styles, navigation, and boilerplate.
4. Combines the clean text and sends it to the Groq LLM with a structured JSON Schema extraction prompt.
5. Returns a Pydantic-validated model containing company overview, ICP, contacts, leadership, and a confidence score.
6. Writes all results to `output/output.json`.

---

## Architecture

```
main.py                 CLI entry point (argparse)
app/
  __init__.py           Package marker
  config.py             All settings loaded from .env (single source of truth)
  models.py             Pydantic models: TeamMember, CompanyExtraction, EnrichedCompany, PipelineOutput
  crawler.py            Async Playwright crawler — discovers and fetches relevant subpages
  extractor.py          HTML → clean plain text (BeautifulSoup, dedup, whitespace normalization)
  search.py             Lightweight targeted web-search fallback (Serper API)
  llm.py                Groq structured extraction + token/cost tracking
  pipeline.py           Per-domain orchestration + deterministic confidence scoring
  utils.py              URL helpers, logging setup, JSON writer
output/
  output.json           Written after every run
requirements.txt
.env.example
.gitignore
README.md
```

### Data Flow

```
Domain list
    │
    ▼
[crawler.py]  ── Playwright headless Chromium
    │            Discovers relevant subpages by keyword scoring
    │            Fetches HTML (handles JS, redirects, 404s, timeouts)
    ▼
[extractor.py] ── BeautifulSoup
    │             Strips scripts/styles/nav/footer
    │             Normalises whitespace, deduplicates lines
    │             Truncates to token-safe limits
    ▼
[search.py]  ── Web Search Fallback (Serper.dev API, if SEARCH_ENABLED=true)
    │             Detects missing signals (leadership, LinkedIn URLs, contact emails)
    │             Runs targeted search queries only for missing fields
    │             Appends evidence text to LLM context (0 extra LLM calls)
    ▼
[llm.py]  ── Groq API (openai/gpt-oss-20b)
    │         Strict JSON Schema structured output
    │         Schema derived from CompanyExtraction.model_json_schema()
    │         Pydantic second-layer validation
    ▼
[pipeline.py]  ── Deterministic confidence scoring
    │             Per-domain error isolation
    ▼
output/output.json
```

---

## Features

- **Headless browser crawling** via Playwright — handles JavaScript-rendered pages, redirects, and SPAs.
- **Smart subpage discovery** — scores and prioritises pages by keyword relevance, stays on the same domain.
- **Clean text extraction** — removes all HTML noise before sending to the LLM (see rationale below).
- **Strict JSON Schema structured outputs** — Groq enforces the schema at generation time using the schema derived from the Pydantic `CompanyExtraction` model.
- **Pydantic as second validation layer** — the parsed response is validated by Pydantic after API return.
- **Deterministic confidence scoring** — blends objective completeness signals with LLM confidence.
- **Robust error handling** — every domain is isolated; one failure never crashes the run.
- **Token and cost tracking** — reports total tokens used and estimated USD cost per run.
- **Configurable via `.env`** — no hardcoded keys, models, or limits.

---

## LLM Provider and Structured Outputs

### Provider
**[Groq](https://groq.com)** — fast inference API compatible with OpenAI-style chat completions.

### Model
`openai/gpt-oss-20b` (configurable via `GROQ_MODEL` env var)

### Structured Output Implementation

The assignment requires strict structured outputs. The implementation uses **Groq's `json_schema` response format** with `strict: true`:

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "company_extraction",
        "strict": True,
        "schema": CompanyExtraction.model_json_schema(),  # derived from Pydantic
    }
}
```

The schema is post-processed by `_make_strict_schema()` in `app/llm.py` to satisfy Groq strict mode:
- All object properties are added to `required` (including those with Pydantic defaults).
- `additionalProperties: false` is set on every object.
- `default` values are stripped (not supported in strict mode).
- `TeamMember.linkedin_url` is typed as `anyOf: [string, null]` — explicitly supporting null when not discoverable.

**`app/models.py` is the single source of truth.** The schema is never duplicated or hardcoded in `llm.py`.

### Validation Layers

| Layer | Where | What |
|---|---|---|
| 1 — API | Groq server | Enforces JSON Schema at generation time |
| 2 — Code | `app/llm.py` | Pydantic validates the parsed dict |

### Fallback

If the model returns `json_validate_failed` on very large contexts, the pipeline automatically retries with `json_object` mode and a trimmed 12k-character context. Pydantic validation still applies.

---

## Why Clean Text Instead of Raw HTML?

Raw HTML contains thousands of tokens of irrelevant noise:
- `<script>` and `<style>` blocks
- SVG path data and CSS class names
- Tracking pixels and analytics snippets
- Repeated navigation and footer markup

Sending raw HTML would waste context window, increase cost, degrade extraction quality, and risk hitting token limits. The extractor strips all of the above, normalises whitespace, and deduplicates repeated lines.

---

## Prerequisites

- Python 3.11+
- A Groq API key — free at [console.groq.com](https://console.groq.com)
- Internet access (the crawler visits live websites)

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd autonomous-lead-enrichment-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright's Chromium browser

```bash
playwright install chromium
```

> This downloads the Chromium binary (~170 MB). Only needed once per environment.

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your `GROQ_API_KEY`:

```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
```

---

## Running the Agent

### Default run (3 test domains)

```bash
python main.py
```

### Custom domains

```bash
python main.py --domains stripe.com twilio.com segment.com
```

### Verbose / debug logging

```bash
python main.py --verbose
```

### Custom output path

```bash
python main.py --output results/my_run.json
```

---

## Output Format

### Successful enrichment

```json
{
  "results": [
    {
      "domain": "postman.com",
      "company_overview": "Postman is an API platform for building and using APIs. It simplifies each step of the API lifecycle and streamlines collaboration.",
      "target_audience": "Software developers, engineering teams, and enterprises that build, test, and manage APIs.",
      "contact_points": ["info@postman.com", "help@postman.com"],
      "leadership": [
        {
          "name": "Abhinav Asthana",
          "role": "CEO and co-founder",
          "linkedin_url": null
        }
      ],
      "confidence_score": 0.99,
      "sources": [
        "https://postman.com",
        "https://www.postman.com/company/about-postman/"
      ],
      "status": "success",
      "error": null
    }
  ],
  "total_domains": 3,
  "successful": 3,
  "failed": 0,
  "token_usage": {
    "prompt_tokens": 12805,
    "completion_tokens": 2040,
    "total_tokens": 14845,
    "estimated_cost_usd": 0.0112
  }
}
```

### Failed domain

```json
{
  "domain": "example-unreachable.com",
  "status": "failed",
  "error": "Crawler returned no pages (site unreachable, blocked, or timed out)"
}
```

---

## Confidence Score Methodology

| Signal | Weight |
|---|---|
| `company_overview` present (≥ 20 chars) | +0.20 |
| `target_audience` present (≥ 10 chars) | +0.15 |
| At least one `contact_point` found | +0.10 |
| At least one `leadership` member found | +0.20 |
| 3 or more pages successfully crawled | +0.15 |
| LLM's own confidence (weighted 20%) | +0.20 |
| **Total possible** | **1.00** |

---

## Error Handling

Each domain is wrapped in an independent try/except block. Errors are logged and recorded in the result's `error` field, but the pipeline continues to the next domain.

| Error type | Handling |
|---|---|
| Page load timeout | Playwright TimeoutError caught; page skipped |
| HTTP 404 / 4xx / 5xx | Detected from response status; page skipped |
| Navigation / browser crash | Exception caught; domain marked failed |
| Bot blocker / empty page | Empty extracted text detected; domain marked failed |
| Groq API error | Caught in `llm.py`; domain marked failed |
| Rate limit (429) | 15-second back-off + retry |
| `json_validate_failed` | Retry with trimmed context + `json_object` fallback |
| Pydantic validation failure | Caught; domain marked failed with error message |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model for extraction |
| `MAX_PAGES_PER_DOMAIN` | `8` | Max pages crawled per domain |
| `BROWSER_TIMEOUT_SECONDS` | `20` | Playwright page-load timeout |
| `MAX_CHARS_PER_PAGE` | `4000` | Char limit per page before combining |
| `MAX_CHARS_COMBINED` | `24000` | Total char limit sent to the LLM |
| `OUTPUT_FILE` | `output/output.json` | Path for JSON output |
| `SEARCH_ENABLED` | `false` | Enable web-search enrichment fallback |
| `SEARCH_API_KEY` | `""` | Serper.dev API key for Google search |
| `SEARCH_MAX_RESULTS` | `3` | Max search results per query |

---

## Limitations

- **Bot blockers**: Some Cloudflare-protected sites may return empty pages. Handled gracefully.
- **JavaScript SPAs**: The crawler waits for `networkidle` (5s timeout) as a best-effort measure.
- **Dynamic pricing**: Client-side rendered pricing may not appear in the DOM snapshot.
- **LinkedIn URLs**: Only included when they appear verbatim in the page content or search evidence.
- **Rate limits**: Groq free-tier has per-minute token limits; the pipeline retries automatically.

---

## Optional Bonus Features

### Token / Cost Tracking ✅ (implemented)
Every run reports total tokens consumed and estimated USD cost in the summary and in `output/output.json` under `token_usage`.

### Web Search Fallback Enrichment ✅ (implemented)
When enabled via `SEARCH_ENABLED=true` in `.env` with a `SEARCH_API_KEY` (Serper.dev), the pipeline:
1. Detects missing signals in crawled website text (`leadership`, `linkedin`, `contact`).
2. Runs targeted, company-scoped Google searches ONLY for missing fields.
3. Appends clean search snippets to the LLM context prior to extraction (0 extra LLM calls).
4. Adds search result URLs to domain `sources` in `output/output.json`.
5. Non-fatal: search timeouts or errors never fail a domain or interrupt the pipeline.

---

## Project Tree

```
.
├── main.py
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── crawler.py
│   ├── extractor.py
│   ├── search.py
│   ├── llm.py
│   ├── pipeline.py
│   └── utils.py
├── output/
│   └── output.json
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

