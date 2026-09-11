"""Daily ingestion: try a batch of new tickers, score them, commit and push.

Meant to run unattended (Windows Task Scheduler, cron, etc.), independent of
any interactive session. Picks candidates from companies_to_try.txt that are
not already in the database and were not attempted in the last
RETRY_COOLDOWN_DAYS days (tracked in ingest_state.json, so a ticker blocked by
FMP's plan limits isn't retried every single day and doesn't burn quota).

Usage:
    python scripts/daily_ingest.py                # default batch size
    python scripts/daily_ingest.py --max-tickers 30
    python scripts/daily_ingest.py --no-push       # ingest + commit locally, skip git push
    python scripts/daily_ingest.py --dry-run       # show what would be attempted, do nothing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TICKER_LIST = PROJECT_ROOT / "scripts" / "companies_to_try.txt"
STATE_FILE = PROJECT_ROOT / "scripts" / "ingest_state.json"
LOG_DIR = PROJECT_ROOT / "scripts" / "logs"

RETRY_COOLDOWN_DAYS = 14
DEFAULT_MAX_TICKERS = 45  # ~4 FMP requests each; leaves headroom under the 250/day budget


def _read_candidates() -> list[str]:
    tickers: list[str] = []
    for line in TICKER_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.extend(line.split())
    # de-duplicate, keep first-seen order
    seen = set()
    ordered = []
    for t in tickers:
        t = t.upper()
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"failed": {}}  # ticker -> ISO date last failed


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _recently_failed(state: dict, ticker: str) -> bool:
    last = state.get("failed", {}).get(ticker)
    if not last:
        return False
    return date.fromisoformat(last) > date.today() - timedelta(days=RETRY_COOLDOWN_DAYS)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True, text=True, check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tickers", type=int, default=DEFAULT_MAX_TICKERS)
    parser.add_argument("--no-push", action="store_true", help="commit locally but do not push")
    parser.add_argument("--dry-run", action="store_true", help="show the plan, do nothing")
    args = parser.parse_args()

    from src.db.database import init_db, session_scope
    from src.db.models import Company, IncomeStatement
    from src.ingestion.fmp_client import FMPAuthError, FMPError, FMPRateLimitError
    from src.ingestion.ingest_pipeline import IngestionPipeline
    from src.ingestion.schemas import PeriodType
    from src.ratios.ratio_engine import run_analysis
    from src.rag.rag_pipeline import RagPipeline
    from src.prediction import repository as prediction_repository
    from src.prediction.predict import ModelUnavailable, score
    from sqlalchemy import select, func

    init_db()
    state = _load_state()
    candidates = _read_candidates()

    with session_scope() as s:
        existing = {c.ticker for c in s.scalars(select(Company))}

    todo = [t for t in candidates if t not in existing and not _recently_failed(state, t)]
    todo = todo[: args.max_tickers]

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{date.today().isoformat()}.log"
    lines: list[str] = [f"=== daily_ingest {datetime.now().isoformat()} ==="]
    lines.append(f"candidates in list: {len(candidates)}  already in DB: {len(existing)}  "
                 f"cooling down: {sum(1 for t in candidates if _recently_failed(state, t))}  "
                 f"attempting today: {len(todo)}")

    if args.dry_run:
        lines.append("DRY RUN - tickers that would be attempted: " + ", ".join(todo))
        print("\n".join(lines))
        return 0

    if not todo:
        lines.append("Nothing to do - no eligible tickers.")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0

    pipeline = IngestionPipeline(include_market_data=True)
    rag_pipeline = RagPipeline()
    succeeded: list[str] = []
    failed: list[str] = []

    for ticker in todo:
        try:
            report = pipeline.run(ticker, period=PeriodType.ANNUAL, years=5)
        except FMPRateLimitError as exc:
            lines.append(f"[BUDGET] stopping at {ticker}: {exc}")
            break
        except (FMPAuthError, FMPError) as exc:
            lines.append(f"[FAILED] {ticker}: {exc}")
            failed.append(ticker)
            continue
        except Exception as exc:  # noqa: BLE001
            lines.append(f"[ERROR] {ticker}: {type(exc).__name__}: {exc}")
            failed.append(ticker)
            continue

        if report.status == "failed":
            lines.append(f"[FAILED] {ticker}: {'; '.join(report.errors) or 'no data'}")
            failed.append(ticker)
            # defensive cleanup - a Company row may have been created before the
            # statement fetch failed, on versions of ingest_pipeline that don't
            # already roll this back themselves
            with session_scope() as s:
                company = s.scalar(select(Company).where(Company.ticker == ticker))
                if company is not None:
                    has_statements = s.scalar(
                        select(func.count()).select_from(IncomeStatement)
                        .where(IncomeStatement.company_id == company.id)
                    )
                    if not has_statements:
                        s.delete(company)
            continue

        try:
            with session_scope() as s:
                run_analysis(s, ticker, period="FY")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  ratios failed for {ticker}: {exc}")

        try:
            with session_scope() as s:
                rag_pipeline.index_company(s, ticker)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  indexing failed for {ticker}: {exc}")

        try:
            with session_scope() as s:
                rows = prediction_repository.build_feature_rows(s, ticker)
                estimates = [(row, score(row.features)) for row in rows]
                prediction_repository.save_estimates(s, ticker, estimates)
        except ModelUnavailable:
            pass
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  distress scoring failed for {ticker}: {exc}")

        succeeded.append(ticker)
        lines.append(f"[SUCCESS] {ticker}  (fmp requests so far: {pipeline.fmp.request_count})")

    pipeline.fmp.close()

    today_iso = date.today().isoformat()
    for t in failed:
        state.setdefault("failed", {})[t] = today_iso
    for t in succeeded:
        state.get("failed", {}).pop(t, None)
    _save_state(state)

    lines.append(f"\nSUMMARY  succeeded={len(succeeded)} {succeeded}  failed={len(failed)}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if not succeeded:
        print("No new companies landed - skipping commit.")
        return 0

    status = _git("status", "--porcelain")
    if not status.stdout.strip():
        print("No file changes detected - skipping commit.")
        return 0

    _git("add", "-A", "data/financials.db", "data/chroma", "scripts/ingest_state.json")
    commit_msg = f"Daily ingestion: add {len(succeeded)} companies ({', '.join(succeeded)}) - {today_iso}"
    commit = _git("commit", "-m", commit_msg)
    print(commit.stdout, commit.stderr)

    if args.no_push:
        print("--no-push set: committed locally only.")
        return 0

    push = _git("push", "origin", "main")
    print(push.stdout, push.stderr)
    if push.returncode != 0:
        print("Push failed - commit is local, retry the push manually.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
