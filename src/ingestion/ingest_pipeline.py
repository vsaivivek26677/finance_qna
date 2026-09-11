"""Layer 1 orchestration: fetch -> normalize/validate -> store.

The pipeline is deliberately failure-tolerant in one specific direction: it will
happily produce a `partial` run (some statements missing, no market data) and
say so loudly, but it will never invent, interpolate, or zero-fill a number to
make a run look complete. Everything it could not get is recorded in the run's
missing map and in each row's `is_missing_json`, which is what lets Layer 2 mark
ratios `is_calculable = False` with a named reason and Layer 3 chunk a field as
"NOT AVAILABLE" instead of omitting it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.config import settings
from src.db.database import init_db, session_scope
from src.ingestion import repository
from src.ingestion.fmp_client import FMPClient, FMPError, FMPNotFoundError
from src.ingestion.normalizer import build_profile, build_statements
from src.ingestion.schemas import (
    CompanyFinancials,
    CompanyProfile,
    IngestionReport,
    PeriodType,
)
from src.ingestion.yfinance_client import YFinanceClient, YFinanceUnavailableError

logger = logging.getLogger(__name__)

_PERIOD_HINT = {PeriodType.ANNUAL: "FY", PeriodType.QUARTER: "Q1"}


class IngestionPipeline:
    """Runs ingestion for one or more tickers."""

    def __init__(
        self,
        fmp_client: FMPClient | None = None,
        yf_client: YFinanceClient | None = None,
        include_market_data: bool = True,
    ) -> None:
        self.fmp = fmp_client or FMPClient()
        self.include_market_data = include_market_data
        self._yf_client = yf_client
        self._yf_resolved = yf_client is not None

    @property
    def yf(self) -> YFinanceClient | None:
        """Lazily construct the yfinance client so its import cost is optional."""
        if not self._yf_resolved:
            self._yf_resolved = True
            try:
                self._yf_client = YFinanceClient()
            except YFinanceUnavailableError as exc:
                logger.warning("Market data disabled: %s", exc)
                self._yf_client = None
        return self._yf_client

    # --- fetch + normalize -------------------------------------------------

    def fetch(
        self,
        ticker: str,
        period: PeriodType = PeriodType.ANNUAL,
        years: int | None = None,
    ) -> tuple[CompanyFinancials, list[str]]:
        """Fetch and normalize everything for one ticker. Returns (data, errors)."""
        ticker = ticker.strip().upper()
        limit = years or settings.ingest_default_years
        errors: list[str] = []

        # 1. Profile. A failure here is not fatal — statements are the point, and
        #    a minimal placeholder profile keeps the foreign key intact.
        try:
            profile = build_profile(self.fmp.get_profile(ticker), ticker)
        except FMPError as exc:
            errors.append(f"profile: {exc}")
            logger.warning("Profile fetch failed for %s: %s", ticker, exc)
            profile = CompanyProfile(ticker=ticker)

        # 2. Statements.
        try:
            payloads = self.fmp.get_all_statements(ticker, period=period.value, limit=limit)
        except FMPError as exc:
            errors.append(f"statements: {exc}")
            logger.error("Statement fetch failed for %s: %s", ticker, exc)
            payloads = {}

        income, balance, cash_flow, normalize_errors = build_statements(
            payloads, ticker, period_hint=_PERIOD_HINT[period]
        )
        errors.extend(normalize_errors)

        # 3. Market data (yfinance, no quota). Optional by design: without it,
        #    Layer 2 uses the book-value Z''-Score instead of the market-value Z.
        market_data = []
        if self.include_market_data and self.yf is not None:
            try:
                market_data = self.yf.get_fiscal_year_end_prices(ticker, years=limit)
            except YFinanceUnavailableError as exc:
                errors.append(f"market_data: {exc}")
                logger.warning("Market data unavailable for %s: %s", ticker, exc)

        data = CompanyFinancials(
            profile=profile,
            income_statements=income,
            balance_sheets=balance,
            cash_flows=cash_flow,
            market_data=market_data,
        )
        return data, errors

    # --- persist -----------------------------------------------------------

    @staticmethod
    def store(session: Session, data: CompanyFinancials) -> dict[str, int]:
        """Write a normalized bundle to the database. Returns per-table counts."""
        company = repository.upsert_company(session, data.profile)
        counts = {
            "income_statements": repository.upsert_statements(
                session, company.id, data.income_statements, "income_statement"
            ),
            "balance_sheets": repository.upsert_statements(
                session, company.id, data.balance_sheets, "balance_sheet"
            ),
            "cash_flow_statements": repository.upsert_statements(
                session, company.id, data.cash_flows, "cash_flow"
            ),
            "market_data": repository.upsert_market_data(session, company.id, data.market_data),
        }
        return counts

    # --- full run ----------------------------------------------------------

    def run(
        self,
        ticker: str,
        period: PeriodType = PeriodType.ANNUAL,
        years: int | None = None,
    ) -> IngestionReport:
        """Ingest one ticker end to end and return an auditable report."""
        ticker = ticker.strip().upper()
        report = IngestionReport(
            ticker=ticker,
            period=period,
            years_requested=years or settings.ingest_default_years,
        )

        try:
            data, errors = self.fetch(ticker, period=period, years=years)
            report.errors.extend(errors)
        except Exception as exc:  # noqa: BLE001 - one ticker must not kill a batch
            report.status = "failed"
            report.errors.append(str(exc))
            report.finished_at = datetime.now(timezone.utc)
            logger.exception("Ingestion failed for %s", ticker)
            self._persist_report(report)
            return report

        report.missing_summary = data.missing_data_map()
        report.critical_missing = self._collect_critical_missing(data)

        try:
            with session_scope() as session:
                company_existed_before = repository.get_company(session, ticker) is not None
                report.records_written = self.store(session, data)
                self._finalize(report, data)
                if report.status == "failed" and not company_existed_before:
                    # The profile fetch upserted a Company row before we knew the
                    # statement fetch would come up empty (e.g. FMP free-tier 403s).
                    # Without this, every failed run leaves a phantom company with
                    # zero statement rows attached.
                    repository.delete_company(session, ticker)
                repository.record_ingestion_run(session, report)
        except Exception as exc:  # noqa: BLE001
            report.status = "failed"
            report.errors.append(f"persistence: {exc}")
            report.finished_at = datetime.now(timezone.utc)
            logger.exception("Persisting %s failed", ticker)
            return report

        logger.info(report.summary_line())
        if report.critical_missing:
            logger.warning(
                "%s: critical fields missing -> %s", ticker, report.critical_missing
            )
        return report

    def run_many(
        self,
        tickers: list[str],
        period: PeriodType = PeriodType.ANNUAL,
        years: int | None = None,
    ) -> list[IngestionReport]:
        """Ingest several tickers; a failure on one does not stop the rest."""
        reports = []
        for ticker in tickers:
            reports.append(self.run(ticker, period=period, years=years))
        return reports

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _collect_critical_missing(data: CompanyFinancials) -> dict[str, list[str]]:
        """Fields whose absence blocks core ratio analysis, keyed by period."""
        out: dict[str, list[str]] = {}
        for records in (data.income_statements, data.balance_sheets, data.cash_flows):
            for record in records:
                missing = record.missing_critical_fields()
                if missing:
                    out.setdefault(record.period_label, []).extend(sorted(missing))
        return {period: sorted(set(fields)) for period, fields in out.items()}

    def _finalize(self, report: IngestionReport, data: CompanyFinancials) -> IngestionReport:
        """Decide success vs partial based on what actually landed."""
        has_statements = any(
            report.records_written.get(key)
            for key in ("income_statements", "balance_sheets", "cash_flow_statements")
        )
        if not has_statements:
            report.status = "failed"
        elif report.errors or report.critical_missing or not data.market_data:
            report.status = "partial"
        else:
            report.status = "success"
        report.finished_at = datetime.now(timezone.utc)
        return report

    @staticmethod
    def _persist_report(report: IngestionReport) -> None:
        """Record a failed run so the audit trail has no silent gaps."""
        try:
            with session_scope() as session:
                repository.record_ingestion_run(session, report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist failed ingestion run for %s: %s", report.ticker, exc)


def ingest_ticker(
    ticker: str,
    period: PeriodType = PeriodType.ANNUAL,
    years: int | None = None,
    include_market_data: bool = True,
) -> IngestionReport:
    """Convenience entry point: initialise the schema and ingest one ticker."""
    init_db()
    pipeline = IngestionPipeline(include_market_data=include_market_data)
    try:
        return pipeline.run(ticker, period=period, years=years)
    finally:
        pipeline.fmp.close()
