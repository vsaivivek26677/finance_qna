"""Command-line entry point for Layer 3.

    python -m src.rag.cli index AAPL
    python -m src.rag.cli ask AAPL "What was the net margin in FY2024?"
    python -m src.rag.cli summary AAPL
    python -m src.rag.cli eval AAPL --years 2
    python -m src.rag.cli stats
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import settings
from src.db.database import init_db, session_scope
from src.logging_config import setup_logging
from src.rag import repository
from src.rag.evaluation import run_evaluation
from src.rag.groq_client import GroqError
from src.rag.rag_pipeline import RagAnswer, RagPipeline


def build_parser() -> argparse.ArgumentParser:
    # A shared parent so --log-level works either before or after the
    # subcommand; argparse otherwise only accepts it before, which is a
    # perpetual papercut.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=settings.log_level)

    parser = argparse.ArgumentParser(
        prog="rag",
        parents=[common],
        description="Index, query and evaluate the grounded financial RAG pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser(
        "index", parents=[common], help="Chunk, embed and store a company's data"
    )
    index.add_argument("tickers", nargs="*")
    index.add_argument(
        "--all",
        action="store_true",
        dest="index_all",
        help="Index every company that has computed ratios",
    )
    index.add_argument("--period", default="FY")

    ask = sub.add_parser("ask", parents=[common], help="Ask a grounded question")
    ask.add_argument("ticker")
    ask.add_argument("question")
    ask.add_argument("--year", type=int, help="Restrict retrieval to one fiscal year")
    ask.add_argument("--top-k", type=int, default=settings.rag_top_k)
    ask.add_argument("--show-context", action="store_true", help="Print the retrieved chunks")
    ask.add_argument("--block", action="store_true", help="Withhold answers containing unverified figures")

    summary = sub.add_parser("summary", parents=[common], help="Generate an executive summary")
    summary.add_argument("ticker")
    summary.add_argument("--year", type=int)

    evaluate = sub.add_parser("eval", parents=[common], help="Run the ground-truth evaluation set")
    evaluate.add_argument("ticker")
    evaluate.add_argument("--years", type=int, default=3, help="Fiscal years to cover")
    evaluate.add_argument("--show-failures", action="store_true")
    evaluate.add_argument("--json", action="store_true")

    stats = sub.add_parser("stats", parents=[common], help="Aggregate groundedness over logged queries")
    stats.add_argument("--ticker")

    return parser


def _print_answer(result: RagAnswer, show_context: bool = False) -> None:
    print(f"\n{'=' * 78}\n{result.question}\n{'=' * 78}\n")
    print(result.answer)
    print(f"\n{'-' * 78}")
    print(result.sources_block())
    report = result.verification
    print(f"\nGroundedness: {report.groundedness:.0%}  ({report.summary()})")
    if result.model:
        print(
            f"Model: {result.model} | {result.prompt_tokens}+{result.completion_tokens} tokens "
            f"| {result.latency_seconds:.2f}s"
        )
    for note in report.notes:
        print(f"  ! {note}")

    if show_context:
        print(f"\n{'-' * 78}\nRetrieved context:")
        for index, chunk in enumerate(result.chunks, start=1):
            print(f"\n[{index}] {chunk.citation()} (similarity {chunk.score:.3f})")
            print(f"    {chunk.text[:600]}{'...' if len(chunk.text) > 600 else ''}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    init_db()

    if args.command == "index":
        pipeline = RagPipeline()
        with session_scope() as session:
            tickers = args.tickers
            if args.index_all:
                from sqlalchemy import select

                from src.db.models import Company, Ratio

                tickers = [
                    t
                    for (t,) in session.execute(
                        select(Company.ticker)
                        .join(Ratio, Ratio.company_id == Company.id)
                        .group_by(Company.ticker)
                        .order_by(Company.ticker)
                    ).all()
                ]
                print(f"Indexing {len(tickers)} companies with computed ratios...")
            if not tickers:
                print(
                    "No tickers given. Pass ticker symbols, or --all to index every "
                    "company that has computed ratios.",
                    file=sys.stderr,
                )
                return 2

            total = 0
            for ticker in tickers:
                written = pipeline.index_company(session, ticker, period=args.period)
                total += written
                if written:
                    print(f"  {ticker.upper():<8} {written} chunks")
                else:
                    print(
                        f"  {ticker.upper():<8} nothing to index - run Layer 1 ingestion "
                        f"and Layer 2 ratios first",
                        file=sys.stderr,
                    )
            print()
            print(f"Indexed {total} chunks across {len(tickers)} companies.")
        return 0

    if args.command == "stats":
        with session_scope() as session:
            print(json.dumps(repository.groundedness_summary(session, args.ticker), indent=2))
        return 0

    if not settings.groq_api_key:
        print(
            "GROQ_API_KEY is not set. Add it to .env (free key: https://console.groq.com).",
            file=sys.stderr,
        )
        return 1

    try:
        if args.command == "ask":
            pipeline = RagPipeline(top_k=args.top_k, block_on_unverified=args.block)
            with session_scope() as session:
                result = pipeline.ask(
                    args.question, ticker=args.ticker, fiscal_year=args.year, session=session
                )
            _print_answer(result, show_context=args.show_context)
            return 0

        if args.command == "summary":
            pipeline = RagPipeline()
            with session_scope() as session:
                result = pipeline.summarize(args.ticker, fiscal_year=args.year, session=session)
            _print_answer(result)
            return 0

        if args.command == "eval":
            pipeline = RagPipeline()
            with session_scope() as session:
                report = run_evaluation(session, args.ticker, pipeline, max_years=args.years)

            if args.json:
                print(
                    json.dumps(
                        {
                            "ticker": report.ticker,
                            "cases": report.total,
                            "metrics": report.aggregate(),
                        },
                        indent=2,
                    )
                )
                return 0

            print(f"\n{'=' * 78}\nRAG evaluation - {report.ticker}\n{'=' * 78}")
            print(f"\nCases: {report.total}\n")
            for metric, value in sorted(report.aggregate().items()):
                bar = "#" * int(round(value * 40))
                print(f"  {metric:<22} {value:.3f}  {bar}")

            failures = [r for r in report.results if not r.passed]
            print(f"\n  {report.total - len(failures)}/{report.total} answers correct")
            if failures and args.show_failures:
                print("\nFailed cases:")
                for result in failures:
                    print(f"\n  Q: {result.case.question}")
                    print(f"     expected: {result.case.expected_value or result.case.expected_text}")
                    print(f"     answer:   {result.answer.answer[:220]}")
            return 0
    except GroqError as exc:
        print(f"\nGroq error: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
