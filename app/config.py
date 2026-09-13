"""
app/config.py - Centralised environment-variable configuration.

Every other module imports settings from here.
Never call os.getenv() anywhere else in the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Groq ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# ── Crawler ───────────────────────────────────────────────────────────────────
# Maximum number of pages crawled per domain (homepage + subpages)
MAX_PAGES_PER_DOMAIN: int = int(os.getenv("MAX_PAGES_PER_DOMAIN", "8"))

# Playwright page-load timeout in milliseconds
BROWSER_TIMEOUT_MS: int = int(os.getenv("BROWSER_TIMEOUT_SECONDS", "20")) * 1000

# Domains processed concurrently (1 = sequential, safe default)
MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "1"))

# ── Extractor ─────────────────────────────────────────────────────────────────
# Max characters kept per individual page before combining
MAX_CHARS_PER_PAGE: int = int(os.getenv("MAX_CHARS_PER_PAGE", "4000"))

# Max total combined characters sent to the LLM
MAX_CHARS_COMBINED: int = int(os.getenv("MAX_CHARS_COMBINED", "24000"))

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_FILE: str = os.getenv("OUTPUT_FILE", "output/output.json")

# ── Web Search Enrichment (Fallback) ─────────────────────────────────────────
SEARCH_ENABLED: bool = os.getenv("SEARCH_ENABLED", "false").lower() in ("true", "1", "yes")
SEARCH_API_KEY: str = os.getenv("SEARCH_API_KEY", "")
SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "3"))

# ── URL path keywords that indicate a relevant subpage ───────────────────────
RELEVANT_PATH_KEYWORDS: list[str] = [
    "about",
    "company",
    "team",
    "contact",
    "pricing",
    "leadership",
    "careers",
    "people",
    "founders",
    "mission",
    "story",
    "who-we-are",
]
