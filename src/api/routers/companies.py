"""Company profile and financial statement endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.dependencies import cached, get_company, get_db, normalise_ticker, read_cache
from src.api.schemas import (
    CompanyProfileResponse,
    CompanySummary,
    StatementLine,
    StatementsResponse,
)
from src.db.models import (
    BalanceSheet,
    CashFlowStatement,
    Company,
    IncomeStatement,
)

router = APIRouter(tags=["companies"])

STATEMENT_MODELS = {
    "income": IncomeStatement,
    "balance": BalanceSheet,
    "cash_flow": CashFlowStatement,
}

# Columns that identify a row rather than report a figure.
_META_COLUMNS = {
    "id",
    "company_id",
    "period",
    "fiscal_year",
    "period_end_date",
    "filing_date",
    "reported_currency",
    "data_source",
    "fetched_at",
    "is_missing_json",
}


def _fiscal_years(db: Session, company_id: int, period: str = "FY") -> list[int]:
    rows = db.execute(
        select(IncomeStatement.fiscal_year)
        .where(IncomeStatement.company_id == company_id, IncomeStatement.period == period)
        .order_by(IncomeStatement.fiscal_year.desc())
    ).all()
    return [year for (year,) in rows]


def _to_line(row) -> StatementLine:
    values = {
        column.key: getattr(row, column.key)
        for column in row.__table__.columns
        if column.key not in _META_COLUMNS
    }
    return StatementLine(
        period=row.period,
        fiscal_year=row.fiscal_year,
        period_end_date=row.period_end_date,
        reported_currency=row.reported_currency,
        values=values,
        missing_fields=row.is_missing_json or {},
    )


@router.get("/companies", response_model=list[CompanySummary], summary="List indexed companies")
def list_companies(db: Session = Depends(get_db)) -> list[CompanySummary]:
    """Every company with statements on file, with the years available."""

    def produce() -> list[CompanySummary]:
        rows = db.execute(
            select(
                Company.id,
                Company.ticker,
                Company.name,
                Company.sector,
                Company.industry,
                func.count(IncomeStatement.id),
            )
            .outerjoin(IncomeStatement, IncomeStatement.company_id == Company.id)
            .group_by(Company.id)
            .order_by(Company.ticker)
        ).all()
        return [
            CompanySummary(
                ticker=ticker,
                name=name,
                sector=sector,
                industry=industry,
                fiscal_years=_fiscal_years(db, company_id),
            )
            for company_id, ticker, name, sector, industry, _count in rows
        ]

    return cached(read_cache(), "companies:list", produce)


@router.get(
    "/companies/{ticker}",
    response_model=CompanyProfileResponse,
    summary="Company profile",
)
def get_profile(
    company: Company = Depends(get_company), db: Session = Depends(get_db)
) -> CompanyProfileResponse:
    return CompanyProfileResponse(
        ticker=company.ticker,
        name=company.name,
        sector=company.sector,
        industry=company.industry,
        exchange=company.exchange,
        country=company.country,
        currency=company.currency,
        description=company.description,
        is_financial_sector=bool(company.is_financial_sector),
        fiscal_years=_fiscal_years(db, company.id),
        data_source=company.data_source,
        fetched_at=company.fetched_at,
    )


@router.get(
    "/companies/{ticker}/statements",
    response_model=StatementsResponse,
    summary="Income statement, balance sheet and cash flow",
)
def get_statements(
    ticker: str = Depends(normalise_ticker),
    company: Company = Depends(get_company),
    db: Session = Depends(get_db),
    period: str = Query("FY", description="FY for annual, Q1-Q4 for quarterly"),
    fiscal_year: int | None = Query(None, description="Restrict to one fiscal year"),
) -> StatementsResponse:
    """Statements as filed. Fields the source never reported are listed in
    `missing_fields` rather than being returned as zero."""

    def fetch(model) -> list[StatementLine]:
        query = select(model).where(model.company_id == company.id, model.period == period)
        if fiscal_year is not None:
            query = query.where(model.fiscal_year == fiscal_year)
        rows = db.scalars(query.order_by(model.fiscal_year.desc())).all()
        return [_to_line(row) for row in rows]

    def produce() -> StatementsResponse:
        return StatementsResponse(
            ticker=ticker,
            period=period,
            income_statements=fetch(IncomeStatement),
            balance_sheets=fetch(BalanceSheet),
            cash_flow_statements=fetch(CashFlowStatement),
        )

    return cached(read_cache(), f"statements:{ticker}:{period}:{fiscal_year}", produce)
