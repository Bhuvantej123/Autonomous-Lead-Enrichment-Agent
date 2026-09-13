"""
main.py - Entry point for the Autonomous Lead Enrichment Agent.

Usage
-----
Default (3 test domains):
    python main.py

Custom domains:
    python main.py --domains stripe.com twilio.com segment.com

With verbose logging:
    python main.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

# Ensure stdout can handle any character on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp1252", "ascii"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app import config
from app.models import PipelineOutput
from app.pipeline import run_pipeline
from app.utils import save_json, setup_logging

DEFAULT_DOMAINS = [
    "postman.com",
    "supabase.com",
    "vapi.ai",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Lead Enrichment Agent — crawl company websites and extract structured intelligence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --domains stripe.com twilio.com\n"
            "  python main.py --verbose\n"
        ),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        metavar="DOMAIN",
        help="One or more company domains to enrich (default: postman.com supabase.com vapi.ai)",
    )
    parser.add_argument(
        "--output",
        default=config.OUTPUT_FILE,
        metavar="PATH",
        help=f"Output JSON file path (default: {config.OUTPUT_FILE})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


def print_summary(output: PipelineOutput) -> None:
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"  Domains processed : {output.total_domains}")
    print(f"  Successful        : {output.successful}")
    print(f"  Failed            : {output.failed}")
    if output.token_usage:
        tu = output.token_usage
        print(f"  Total tokens      : {tu.total_tokens:,}")
        print(f"  Estimated cost    : ${tu.estimated_cost_usd:.4f} USD")
    print()
    for result in output.results:
        status_icon = "[OK]  " if result.status == "success" else "[FAIL]"
        print(f"  {status_icon}  {result.domain}")
        if result.status == "success":
            print(f"       confidence   : {result.confidence_score:.2f}")
            print(f"       pages used   : {len(result.sources)}")
            print(f"       leadership   : {len(result.leadership)} member(s)")
            print(f"       contacts     : {len(result.contact_points)}")
        else:
            print(f"       error        : {result.error}")
    print("=" * 60 + "\n")


def main() -> int:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    if not config.GROQ_API_KEY:
        logger.error(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key."
        )
        return 1

    logger.info("Domains to enrich: %s", args.domains)

    output = run_pipeline(args.domains)

    # Write JSON output
    output_dict = output.model_dump(mode="json")
    save_json(output_dict, args.output)
    logger.info("Output written to %s", args.output)

    print_summary(output)
    return 0 if output.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
