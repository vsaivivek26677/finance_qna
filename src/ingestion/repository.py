"""Persistence for ingested records.

Upserts are keyed on the natural business key (ticker; company + period +
fiscal year; company + date) so re-running ingestion for a ticker refreshes
existing rows instead of duplicating history. Restatements therefore overwrite
the stale figures, which is what an analyst would expect.

Portable UPDATE-or-INSERT is used rather than a dialect-specific ON CONFLICT so
the same code runs on SQLite locally and Postgres in deployment.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import (
    BalanceSheet,
    CashFlowStatement,
    Company,
    IncomeStatement,
    IngestionRun,
    MarketData,
)
from src.ingestion.schemas import (
    BalanceSheetRecord,
    CashFlowRecord,
    CompanyProfile,
    IncomeStatementRecord,
    IngestionReport,
    MarketDataPoint,
    StatementBase,
)

logger = logging.getLogger(__name__)

# Record fields that describe provenance rather than a statement line item.
_NON_COLUMN_FIELDS = {"ticker", "missing_fields", "statement_kind"}

STATEMENT_MODELS: dict[str, type] = {
    "income_statement": IncomeStatement,
    "balance_sheet": BalanceSheet,
    "cash_flow": CashFlowStatement,
}


def _model_columns(model: type) -> set[str]:
    return {column.key for column in model.__table__.columns}


def _record_to_columns(record: Any, model: type) -> dict[str, Any]:
    """Project a pydantic record onto the columns its table actually has."""
    allowed = _model_columns(model)
    payload = {
        key: value
        for key, value in record.model_dump().items()
        if key in allowed and key not in _NON_COLUMN_FIELDS
    }
    payload["is_missing_json"] = record.missing_json()
    return payload


def upsert_company(session: Session, profile: CompanyProfile) -> Company:
    """Insert or refresh the company row; returns the persisted entity."""
    company = session.scalar(select(Company).where(Company.ticker == profile.ticker))
    values = _record_to_columns(profile, Company)
    values["is_financial_sector"] = profile.is_financial_sector

    if company is None:
        company = Company(ticker=profile.ticker, **values)
        session.add(company)
        logger.info("Created company %s", profile.ticker)
    else:
        for key, value in values.items():
            # Never overwrite a known value with a newly-missing one; a partial
            # profile response should not erase good data already on file.
            if value is not None or key in {"is_missing_json", "is_financial_sector"}:
                setattr(company, key, value)
        logger.info("Updated company %s", profile.ticker)

    session.flush()  # assign the primary key for the statement rows that follow
    return company


def delete_company(session: Session, ticker: str) -> bool:
    """Remove a company row (and any statement/market rows cascaded from it).

    Used to roll back a phantom company created by a run whose statement fetch
    never landed anything — see `IngestionPipeline.run`. Returns whether a row
    was actually deleted.
    """
    company = session.scalar(select(Company).where(Company.ticker == ticker.strip().upper()))
    if company is None:
        return False
    session.delete(company)
    session.flush()
    logger.info("Deleted phantom company %s (failed run, no statements landed)", ticker)
    return True


def upsert_statements(
    session: Session,
    company_id: int,
    records: Sequence[StatementBase],
    kind: str,
) -> int:
    """Insert or refresh statement rows. Returns the number written."""
    model = STATEMENT_MODELS[kind]
    written = 0

    for record in records:
        existing = session.scalar(
            select(model).where(
                model.company_id == company_id,
                model.period == record.period,
                model.fiscal_year == record.fiscal_year,
            )
        )
        values = _record_to_columns(record, model)
        if existing is None:
            session.add(model(company_id=company_id, **values))
        else:
            for key, value in values.items():
                setattr(existing, key, value)
        written += 1

    session.flush()
    logger.debug("Upserted %d %s rows for company_id=%s", written, kind, company_id)
    return written


def upsert_market_data(
    session: Session, company_id: int, points: Iterable[MarketDataPoint]
) -> int:
    """Insert or refresh daily market observations. Returns the number written."""
    points = list(points)
    if not points:
        return 0

    existing_dates = {
        row.date: row
        for row in session.scalars(
            select(MarketData).where(
                MarketData.company_id == company_id,
                MarketData.date.in_([p.date for p in points]),
            )
        )
    }

    for point in points:
        values = _record_to_columns(point, MarketData)
        row = existing_dates.get(point.date)
        if row is None:
            session.add(MarketData(company_id=company_id, **values))
        else:
            for key, value in values.items():
                setattr(row, key, value)

    session.flush()
    logger.debug("Upserted %d market data points for company_id=%s", len(points), company_id)
    return len(points)


def record_ingestion_run(session: Session, report: IngestionReport) -> IngestionRun:
    """Persist the audit trail row for one ingestion attempt."""
    run = IngestionRun(
        ticker=report.ticker,
        period=report.period.value,
        status=report.status,
        years_requested=report.years_requested,
        records_written_json=report.records_written,
        missing_summary_json={
            "by_period": report.missing_summary,
            "critical": report.critical_missing,
        },
        errors_json={"errors": report.errors},
        started_at=report.started_at,
        finished_at=report.finished_at,
    )
    session.add(run)
    session.flush()
    return run


# --- Read helpers used by later layers -------------------------------------


def get_company(session: Session, ticker: str) -> Company | None:
    return session.scalar(select(Company).where(Company.ticker == ticker.strip().upper()))


def get_statements(
    session: Session, ticker: str, kind: str, period: str = "FY"
) -> list[Any]:
    """All rows of one statement type for a ticker, newest fiscal year first."""
    model = STATEMENT_MODELS[kind]
    company = get_company(session, ticker)
    if company is None:
        return []
    return list(
        session.scalars(
            select(model)
            .where(model.company_id == company.id, model.period == period)
            .order_by(model.fiscal_year.desc())
        )
    )


def get_latest_market_data(session: Session, ticker: str) -> MarketData | None:
    company = get_company(session, ticker)
    if company is None:
        return None
    return session.scalar(
        select(MarketData)
        .where(MarketData.company_id == company.id)
        .order_by(MarketData.date.desc())
        .limit(1)
    )


__all__ = [
    "upsert_company",
    "delete_company",
    "upsert_statements",
    "upsert_market_data",
    "record_ingestion_run",
    "get_company",
    "get_statements",
    "get_latest_market_data",
    "STATEMENT_MODELS",
]
