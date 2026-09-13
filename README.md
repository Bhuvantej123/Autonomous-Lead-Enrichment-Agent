# Autonomous Lead Enrichment Agent

A production-quality Python pipeline that accepts a list of company domains, autonomously crawls their public websites using a headless Chromium browser, extracts clean structured text, and uses an LLM (OpenAI) to produce actionable company intelligence in JSON format.

---

## Overview

Given a list of domains such as `postman.com`, `supabase.com`, and `vapi.ai`, the agent:

1. Visits each company's homepage with Playwright (headless Chromium).
2. Discovers and scores internal subpages (`/about`, `/team`, `/contact`, `/pricing`, `/leadership`, etc.).
3. Extracts visible text from the DOM — removing scripts, styles, navigation, and boilerplate.
4. Combines the clean text and sends it to an OpenAI LLM with a structured extraction prompt.
5. Returns a validated Pydantic model containing company overview, ICP, contacts, leadership, and a confidence score.
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
  llm.py                OpenAI structured extraction + token/cost tracking
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
    │            Discovers relevant subpages
    │            Fetches HTML (handles JS, redirects, 404s, timeouts)
    ▼
[extractor.py] ── BeautifulSoup
    │             Strips scripts/styles/nav/footer
    │             Normalises whitespace, deduplicates lines
    │             Truncates to token-safe limits
    ▼
[llm.py]  ── OpenAI beta.chat.completions.parse()
    │         Structured Pydantic output (CompanyExtraction)
    │         Temperature=0 for deterministic extraction
    ▼
[pipeline.py]  ── Computes deterministic confidence score
    │             Wraps result in EnrichedCompany
    ▼
output/output.json
```

---

## Features

- **Headless browser crawling** via Playwright — handles JavaScript-rendered pages, redirects, and SPAs.
- **Smart subpage discovery** — scores and prioritises pages by keyword relevance, stays on the same domain.
- **Clean text extraction** — removes all HTML noise before sending to the LLM (see rationale below).
- **Structured LLM output** — uses `client.beta.chat.completions.parse()` with a Pydantic schema so the output is always machine-parseable.
- **Deterministic confidence scoring** — blends objective completeness signals (fields found, pages crawled) with LLM confidence.
- **Robust error handling** — every domain is isolated; one failure never crashes the run.
- **Token and cost tracking** — reports total tokens used and estimated USD cost per run.
- **Configurable via `.env`** — no hardcoded keys, models, or limits.

---

## Why Clean Text Instead of Raw HTML?

Raw HTML contains thousands of tokens of irrelevant noise:
- `<script>` and `<style>` blocks
- SVG path data
- CSS class names and HTML attributes
- Tracking pixels, analytics snippets
- Repeated navigation and footer markup

Sending raw HTML to an LLM would:
- **Waste context window** — leaving less room for actual content.
- **Increase cost** — more tokens = higher API cost.
- **Degrade extraction quality** — the model attends to noise instead of content.
- **Risk hitting token limits** — large pages can easily exceed 16K tokens of raw HTML.

The extractor strips all of the above, normalises whitespace, and deduplicates repeated lines, giving the LLM only the meaningful visible text.

---

## Prerequisites

- Python 3.11+
- An OpenAI API key with access to `gpt-4o-mini` (or your chosen model)
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

Open `.env` and set your `OPENAI_API_KEY`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
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
      "company_overview": "Postman is an API platform that helps developers design, test, and collaborate on APIs. It provides a collaborative workspace used by millions of developers to build and manage APIs efficiently.",
      "target_audience": "Software developers, API engineers, and engineering teams at companies of all sizes who need to design, test, mock, and document APIs.",
      "contact_points": ["support@postman.com"],
      "leadership": [
        {
          "name": "Abhinav Asthana",
          "role": "Co-founder & CEO",
          "linkedin_url": null
        }
      ],
      "confidence_score": 0.85,
      "sources": [
        "https://postman.com",
        "https://postman.com/about",
        "https://postman.com/company/contact-us"
      ],
      "status": "success",
      "error": null
    }
  ],
  "total_domains": 3,
  "successful": 3,
  "failed": 0,
  "token_usage": {
    "prompt_tokens": 12450,
    "completion_tokens": 890,
    "total_tokens": 13340,
    "estimated_cost_usd": 0.0022
  }
}
```

### Failed domain

```json
{
  "domain": "example-unreachable.com",
  "company_overview": "",
  "target_audience": "",
  "contact_points": [],
  "leadership": [],
  "confidence_score": 0.0,
  "sources": [],
  "status": "failed",
  "error": "Crawler returned no pages (site unreachable, blocked, or timed out)"
}
```

---

## Confidence Score Methodology

The final confidence score (0.0–1.0) is computed deterministically:

| Signal | Weight |
|---|---|
| `company_overview` is non-empty (≥ 20 chars) | +0.20 |
| `target_audience` is non-empty (≥ 10 chars) | +0.15 |
| At least one `contact_point` found | +0.10 |
| At least one `leadership` member found | +0.20 |
| 3 or more pages successfully crawled | +0.15 |
| LLM's own confidence (weighted 20%) | +0.20 |
| **Total possible** | **1.00** |

This makes the score explainable and reproducible — interviewers can see exactly how it was derived.

---

## Error Handling Approach

Each domain is wrapped in an independent try/except block. Errors are logged and recorded in the result's `error` field, but the pipeline continues to the next domain.

| Error type | Handling |
|---|---|
| Page load timeout | Playwright TimeoutError caught; page skipped |
| HTTP 404 | Detected from response status; page skipped |
| HTTP 4xx / 5xx | Logged as warning; page skipped |
| Navigation / browser crash | Exception caught; domain marked failed |
| Bot blocker / empty page | Empty extracted text detected; domain marked failed |
| LLM API error | Caught in `llm.py`; domain marked failed with error message |
| Rate limit | One automatic retry with 60-second back-off |
| Malformed LLM output | Pydantic validation catches it; domain marked failed |
| Missing HTML elements | BeautifulSoup handles gracefully; returns empty string |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for extraction |
| `MAX_PAGES_PER_DOMAIN` | `8` | Max pages crawled per domain |
| `BROWSER_TIMEOUT_SECONDS` | `20` | Playwright page-load timeout |
| `MAX_CHARS_PER_PAGE` | `4000` | Char limit per page before combining |
| `MAX_CHARS_COMBINED` | `24000` | Total char limit sent to the LLM |
| `OUTPUT_FILE` | `output/output.json` | Path for JSON output |

---

## Limitations

- **Bot blockers**: Some sites (e.g. Cloudflare-protected) may return empty or challenge pages. The pipeline handles this gracefully but cannot bypass all anti-bot measures.
- **JavaScript-heavy SPAs**: Most React/Vue apps render after `domcontentloaded`. The crawler waits for `networkidle` (with a 5-second timeout) as a best-effort measure.
- **Dynamic pricing pages**: Some pricing information is loaded client-side and may not be captured in the DOM snapshot.
- **LinkedIn URLs**: Only included when they appear verbatim in the page content. The LLM is instructed never to guess or construct them.
- **Rate limits**: The pipeline processes domains sequentially by default. For large batches, set `MAX_CONCURRENCY` accordingly.

---

## Optional Bonus Features

### Token / Cost Tracking ✅ (implemented)
Every run reports total tokens consumed and estimated USD cost in the console summary and in `output/output.json` under `token_usage`.

### LinkedIn / Search Integration ❌ (not implemented)
The assignment marks this as optional. Core requirements take priority. A future implementation could use the Serper or SerpAPI Google Search API to look up LinkedIn profiles.

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
