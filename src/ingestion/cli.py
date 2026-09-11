"""Command-line entry point for Layer 1.

    python -m src.ingestion.cli AAPL MSFT --years 5
    python -m src.ingestion.cli AAPL --period quarter --no-market-data
    python -m src.ingestion.cli --init-only
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import settings
from src.db.database import init_db
from src.ingestion.ingest_pipeline import IngestionPipeline
from src.ingestion.schemas import IngestionReport, PeriodType
from src.logging_config import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Fetch, validate and store financial statements for one or more tickers.",
    )
    parser.add_argument("tickers", nargs="*", help="Ticker symbols, e.g. AAPL MSFT")
    parser.add_argument(
        "--period",
        choices=[p.value for p in PeriodType],
        default=PeriodType.ANNUAL.value,
        help="Statement periodicity (default: annual)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=settings.ingest_default_years,
        help=f"Number of periods to fetch (default: {settings.ingest_default_years})",
    )
    parser.add_argument(
        "--no-market-data",
        action="store_true",
        help="Skip yfinance price history (Layer 2 will fall back to the Z''-Score)",
    )
    parser.add_argument(
        "--init-only", action="store_true", help="Create the database schema and exit"
    )
    parser.add_argument("--json", action="store_true", help="Print reports as JSON")
    parser.add_argument("--log-level", default=settings.log_level, help="DEBUG/INFO/WARNING/ERROR")
    return parser


def _print_human(report: IngestionReport) -> None:
    icon = {"success": "OK", "partial": "PARTIAL", "failed": "FAILED"}[report.status]
    print(f"\n[{icon}] {report.ticker} ({report.period.value})")
    for table, count in sorted(report.records_written.items()):
        print(f"    {table:<24} {count}")
    if report.critical_missing:
        print("    critical fields missing:")
        for period, fields in sorted(report.critical_missing.items()):
            print(f"      {period}: {', '.join(fields)}")
    if report.missing_summary:
        total = sum(
            len(fields) for groups in report.missing_summary.values() for fields in groups.values()
        )
        print(f"    non-critical missing fields: {total} (see ingestion_runs table)")
    for error in report.errors:
        print(f"    ! {error}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.init_only:
        init_db()
        print("Database schema created.")
        return 0

    if not args.tickers:
        build_parser().print_help()
        return 2

    if not settings.fmp_api_key:
        print(
            "FMP_API_KEY is not set. Copy .env.example to .env and add a free key from\n"
            "https://site.financialmodelingprep.com/developer/docs",
            file=sys.stderr,
        )
        return 1

    init_db()
    pipeline = IngestionPipeline(include_market_data=not args.no_market_data)
    try:
        reports = pipeline.run_many(
            [t.upper() for t in args.tickers],
            period=PeriodType(args.period),
            years=args.years,
        )
    finally:
        pipeline.fmp.close()

    if args.json:
        print(json.dumps([json.loads(r.model_dump_json()) for r in reports], indent=2))
    else:
        for report in reports:
            _print_human(report)
        print(f"\nFMP requests used this run: {pipeline.fmp.request_count}")

    return 0 if all(r.status != "failed" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
