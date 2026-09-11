"""Command-line entry point for Layer 2.

    python -m src.ratios.cli AAPL
    python -m src.ratios.cli AAPL --year 2023 --show-missing
    python -m src.ratios.cli AAPL MSFT --json
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import settings
from src.db.database import init_db, session_scope
from src.logging_config import setup_logging
from src.ratios.base import PeriodAnalysis
from src.ratios.ratio_engine import analyze_ticker, build_report, run_analysis
from src.ratios.red_flags import RedFlagResult

SEVERITY_ICONS = {"High": "[HIGH]  ", "Medium": "[MEDIUM]", "Low": "[LOW]   ", "Info": "[INFO]  "}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ratios",
        description="Compute financial ratios, distress scores and red flags from stored statements.",
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols already ingested by Layer 1")
    parser.add_argument("--period", default="FY", help="FY (default) or Q1-Q4")
    parser.add_argument("--year", type=int, help="Show detail for a single fiscal year")
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="List ratios that could not be calculated, with the reason for each",
    )
    parser.add_argument(
        "--no-store", action="store_true", help="Compute and print without writing to the database"
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    parser.add_argument("--log-level", default=settings.log_level)
    return parser


def _print_ratios(analysis: PeriodAnalysis, show_missing: bool) -> None:
    print(f"\n  {analysis.period_label}")
    by_category: dict[str, list] = {}
    for result in analysis.ratios.values():
        by_category.setdefault(result.category, []).append(result)

    for category in sorted(by_category):
        calculable = [r for r in by_category[category] if r.is_calculable]
        if calculable:
            print(f"    {category}")
            for result in sorted(calculable, key=lambda r: r.name):
                zone = result.details.get("zone")
                suffix = f"   <- {zone} Zone" if zone else ""
                print(f"      {result.name:<34} {result.value:>14,.4f}{suffix}")

    if show_missing:
        missing = analysis.not_calculable()
        if missing:
            print("    not calculable")
            for name, result in sorted(missing.items()):
                print(f"      {name:<34} {result.reason}")


def _print_flags(flags: list[RedFlagResult]) -> None:
    if not flags:
        print("      no flags raised")
        return
    for flag in flags:
        icon = SEVERITY_ICONS.get(flag.severity.value, "")
        print(f"      {icon} {flag.flag_name}")
        print(f"               {flag.explanation}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    init_db()

    payload = []
    exit_code = 0

    with session_scope() as session:
        for ticker in [t.upper() for t in args.tickers]:
            if args.no_store:
                analyses, flags = analyze_ticker(session, ticker, args.period)
                report = build_report(ticker, args.period, analyses, flags)
            else:
                report = run_analysis(session, ticker, args.period)
                analyses, flags = analyze_ticker(session, ticker, args.period)

            if not analyses:
                print(
                    f"\n{ticker}: no {args.period} statements on file. "
                    f"Run: python -m src.ingestion.cli {ticker}",
                    file=sys.stderr,
                )
                exit_code = 1
                continue

            selected = [a for a in analyses if args.year is None or a.fiscal_year == args.year]

            if args.json:
                payload.append(
                    {
                        "ticker": ticker,
                        "period": args.period,
                        "coverage": round(report.coverage, 4),
                        "ratios": {
                            a.period_label: {
                                name: {
                                    "value": r.value,
                                    "is_calculable": r.is_calculable,
                                    "reason": r.reason,
                                    "details": r.details,
                                }
                                for name, r in a.ratios.items()
                            }
                            for a in selected
                        },
                        "red_flags": {
                            str(year): [
                                {
                                    "flag_name": f.flag_name,
                                    "severity": f.severity.value,
                                    "category": f.category,
                                    "explanation": f.explanation,
                                    "source_values": f.source_values,
                                }
                                for f in year_flags
                            ]
                            for year, year_flags in flags.items()
                        },
                    }
                )
                continue

            print(f"\n{'=' * 78}\n{ticker} - {report.summary_line()}\n{'=' * 78}")
            for warning in report.warnings:
                print(f"  ! {warning}")

            for analysis in sorted(selected, key=lambda a: a.fiscal_year, reverse=True):
                _print_ratios(analysis, args.show_missing)
                print(f"    red flags")
                _print_flags(flags.get(analysis.fiscal_year, []))

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
