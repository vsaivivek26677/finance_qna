"""Builders for `PeriodBundle` objects in ratio tests.

Figures are Apple's FY2023 filing stated in millions. Scale does not affect any
ratio, and millions keep the absolute measures (working capital, enterprise
value) readable in assertions.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.ratios.base import PeriodBundle

# Sentinel: setting a field to REMOVE deletes it from the statement and records
# it in `is_missing_json`, reproducing exactly what Layer 1 stores when a source
# omits a field.
REMOVE = object()


class StatementRow:
    """Stand-in for an ORM row: any column not set reads as None, as in the DB."""

    def __init__(self, **values: Any) -> None:
        self.is_missing_json: dict[str, str] = values.pop("is_missing_json", None) or {}
        self.__dict__.update(values)

    def __getattr__(self, name: str) -> None:  # only reached for unset columns
        return None


INCOME_FY2023: dict[str, float] = {
    "revenue": 383285,
    "cost_of_revenue": 214137,
    "gross_profit": 169148,
    "selling_general_admin": 24932,
    "research_and_development": 29915,
    "operating_expenses": 54847,
    "operating_income": 114301,
    "ebitda": 125820,
    "depreciation_amortization": 11519,
    "interest_expense": 3933,
    "income_before_tax": 113736,
    "income_tax_expense": 16741,
    "net_income": 96995,
    "eps": 6.16,
    "eps_diluted": 6.13,
    "weighted_average_shares_diluted": 15812.547,
}

BALANCE_FY2023: dict[str, float] = {
    "cash_and_equivalents": 29965,
    "short_term_investments": 31590,
    "cash_and_short_term_investments": 61555,
    "net_receivables": 60985,
    "inventory": 6331,
    "total_current_assets": 143566,
    "property_plant_equipment_net": 43715,
    "total_non_current_assets": 209017,
    "total_assets": 352583,
    "accounts_payable": 62611,
    "short_term_debt": 15807,
    "total_current_liabilities": 145308,
    "long_term_debt": 95281,
    "total_non_current_liabilities": 145129,
    "total_liabilities": 290437,
    "retained_earnings": -214,
    "total_equity": 62146,
    "total_debt": 111088,
    "net_debt": 81123,
}

CASH_FLOW_FY2023: dict[str, float] = {
    "net_income": 96995,
    "depreciation_amortization": 11519,
    "operating_cash_flow": 110543,
    "capital_expenditure": -10959,
    "investing_cash_flow": 3705,
    "financing_cash_flow": -108488,
    "free_cash_flow": 99584,
}

MARKET_FY2023: dict[str, float] = {"close_price": 171.21, "market_cap": 2800000}


def _merge(base: dict[str, Any], overrides: dict[str, Any] | None) -> StatementRow:
    values = dict(base)
    missing: dict[str, str] = {}
    for key, value in (overrides or {}).items():
        if value is REMOVE:
            values.pop(key, None)
            missing[key] = "not_reported"
        else:
            values[key] = value
    return StatementRow(is_missing_json=missing, **values)


def make_bundle(
    fiscal_year: int = 2023,
    *,
    income: dict[str, Any] | None = None,
    balance: dict[str, Any] | None = None,
    cash_flow: dict[str, Any] | None = None,
    market: dict[str, Any] | None = None,
    prior: PeriodBundle | None = None,
    is_financial_sector: bool = False,
    sector: str | None = "Technology",
    period: str = "FY",
    ticker: str = "AAPL",
    with_market: bool = True,
    no_income: bool = False,
) -> PeriodBundle:
    """A complete period, with per-field overrides and REMOVE for absent fields."""
    return PeriodBundle(
        ticker=ticker,
        period=period,
        fiscal_year=fiscal_year,
        period_end_date=date(fiscal_year, 9, 30),
        income=None if no_income else _merge(INCOME_FY2023, income),
        balance=_merge(BALANCE_FY2023, balance),
        cash_flow=_merge(CASH_FLOW_FY2023, cash_flow),
        market=_merge(MARKET_FY2023, market) if with_market else None,
        is_financial_sector=is_financial_sector,
        sector=sector,
        prior=prior,
    )


def make_pair(
    current: dict[str, dict[str, Any]] | None = None,
    prior: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> PeriodBundle:
    """Two consecutive periods, returning the later one with `.prior` attached.

    Each argument is a {statement: {field: value}} mapping of overrides, e.g.
    `make_pair(current={"income": {"revenue": 200}}, prior={"income": {"revenue": 100}})`.
    """
    prior_bundle = make_bundle(
        2022,
        income=(prior or {}).get("income"),
        balance=(prior or {}).get("balance"),
        cash_flow=(prior or {}).get("cash_flow"),
        market=(prior or {}).get("market"),
        **kwargs,
    )
    return make_bundle(
        2023,
        income=(current or {}).get("income"),
        balance=(current or {}).get("balance"),
        cash_flow=(current or {}).get("cash_flow"),
        market=(current or {}).get("market"),
        prior=prior_bundle,
        **kwargs,
    )
