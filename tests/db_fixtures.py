"""Shared database seeding for tests that need a populated company.

Used by the Layer 2 engine tests and by every Layer 3 test, so the RAG pipeline
is exercised against the same figures the ratio engine was verified on.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.db.models import (
    BalanceSheet,
    CashFlowStatement,
    Company,
    IncomeStatement,
    MarketData,
)
from tests.ratio_fixtures import BALANCE_FY2023, CASH_FLOW_FY2023, INCOME_FY2023


def seed_company(
    session,
    ticker: str = "AAPL",
    years: tuple[int, ...] = (2022, 2023),
    *,
    name: str | None = None,
    sector: str = "Technology",
    is_financial: bool = False,
    market_dates: dict[int, date] | None = None,
    income_overrides: dict[int, dict] | None = None,
    balance_overrides: dict[int, dict] | None = None,
    missing: dict[str, str] | None = None,
) -> Company:
    """Insert a company with a full set of statements for each year."""
    company = Company(
        ticker=ticker,
        name=name or f"{ticker} Inc.",
        sector=sector,
        industry="Consumer Electronics",
        exchange="NASDAQ",
        currency="USD",
        is_financial_sector=is_financial,
    )
    session.add(company)
    session.flush()

    for year in years:
        period_end = date(year, 9, 30)

        income: dict[str, Any] = dict(INCOME_FY2023)
        income.update((income_overrides or {}).get(year, {}))
        income = {k: v for k, v in income.items() if v is not None}
        session.add(
            IncomeStatement(
                company_id=company.id,
                period="FY",
                fiscal_year=year,
                period_end_date=period_end,
                reported_currency="USD",
                is_missing_json=dict(missing or {}),
                **income,
            )
        )

        balance: dict[str, Any] = dict(BALANCE_FY2023)
        balance.update((balance_overrides or {}).get(year, {}))
        balance = {k: v for k, v in balance.items() if v is not None}
        session.add(
            BalanceSheet(
                company_id=company.id,
                period="FY",
                fiscal_year=year,
                period_end_date=period_end,
                reported_currency="USD",
                is_missing_json={},
                **balance,
            )
        )
        session.add(
            CashFlowStatement(
                company_id=company.id,
                period="FY",
                fiscal_year=year,
                period_end_date=period_end,
                reported_currency="USD",
                is_missing_json={},
                **CASH_FLOW_FY2023,
            )
        )
        session.add(
            MarketData(
                company_id=company.id,
                date=(market_dates or {}).get(year, date(year, 9, 29)),
                close_price=171.21,
                market_cap=2800000,
                shares_outstanding=15812.547,
            )
        )
    session.flush()
    return company


def seed_and_analyze(session, ticker: str = "AAPL", **kwargs) -> Company:
    """Seed a company and run Layer 2, so ratios and red flags exist to chunk."""
    from src.ratios.ratio_engine import run_analysis

    company = seed_company(session, ticker, **kwargs)
    run_analysis(session, ticker)
    return company
