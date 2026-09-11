"""Pydantic models for ingested data.

These are the contract between the ingestion layer and everything downstream,
and they are reused directly as FastAPI response models in Layer 5 — so there is
one schema definition, not two.

Design rule: every numeric field is Optional. A source that omits a number gets
`None` in the field *and* an entry in `missing_fields` explaining why. Nothing is
defaulted to zero, because "zero revenue" and "revenue not reported" mean very
different things to a ratio engine.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ingestion.field_maps import CRITICAL_FIELDS, FINANCIAL_SECTOR_LABELS

PERIOD_PATTERN = re.compile(r"^(FY|Q[1-4])$")


class MissingReason(str, Enum):
    """Why a field has no value. Recorded per field, never inferred later."""

    NOT_REPORTED = "not_reported"  # key absent from the source payload
    NULL_IN_SOURCE = "null_in_source"  # key present but explicitly null
    UNPARSEABLE = "unparseable"  # key present but not coercible to a number


class PeriodType(str, Enum):
    ANNUAL = "annual"
    QUARTER = "quarter"


MissingMap = dict[str, MissingReason]
Ticker = Annotated[str, Field(min_length=1, max_length=16)]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProvenanceBase(BaseModel):
    """Fields shared by everything the ingestion layer produces."""

    model_config = ConfigDict(use_enum_values=False, validate_assignment=True)

    ticker: Ticker
    data_source: str = "fmp"
    fetched_at: datetime = Field(default_factory=_utcnow)
    missing_fields: MissingMap = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.strip().upper()

    def missing_json(self) -> dict[str, str]:
        """Serialize the missing map for the `is_missing_json` DB column."""
        return {k: v.value for k, v in self.missing_fields.items()}


class CompanyProfile(ProvenanceBase):
    """Company identity and classification."""

    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    country: str | None = None
    currency: str | None = None
    cik: str | None = None
    description: str | None = None

    # Market snapshot from the profile endpoint — convenient, but the
    # authoritative price series comes from MarketDataPoint.
    market_cap: float | None = None
    price: float | None = None
    shares_outstanding: float | None = None

    @property
    def is_financial_sector(self) -> bool:
        """Banks/insurers are excluded from the Altman Z-Score in Layer 2."""
        if not self.sector:
            return False
        return self.sector.strip().lower() in FINANCIAL_SECTOR_LABELS


class StatementBase(ProvenanceBase):
    """Identity columns common to the three financial statements."""

    period: str = "FY"
    fiscal_year: int
    period_end_date: date | None = None
    filing_date: date | None = None
    reported_currency: str | None = None

    # Set by subclasses so `missing_critical_fields` knows which rules apply.
    statement_kind: str = "statement"

    @field_validator("period")
    @classmethod
    def _validate_period(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if v in {"ANNUAL", "FY", ""}:
            return "FY"
        if not PERIOD_PATTERN.match(v):
            raise ValueError(f"period must be FY or Q1-Q4, got {v!r}")
        return v

    @field_validator("fiscal_year")
    @classmethod
    def _validate_fiscal_year(cls, v: int) -> int:
        if not 1900 <= v <= 2200:
            raise ValueError(f"implausible fiscal_year: {v}")
        return v

    @property
    def period_label(self) -> str:
        """Human/RAG-friendly label, e.g. 'FY2023' or 'Q3 2024'."""
        return f"{self.period}{self.fiscal_year}" if self.period == "FY" else f"{self.period} {self.fiscal_year}"

    def missing_critical_fields(self) -> set[str]:
        """Critical fields that are absent — surfaced, never swallowed."""
        critical = CRITICAL_FIELDS.get(self.statement_kind, set())
        return {f for f in critical if getattr(self, f, None) is None}


class IncomeStatementRecord(StatementBase):
    statement_kind: str = "income_statement"

    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    research_and_development: float | None = None
    selling_general_admin: float | None = None
    operating_expenses: float | None = None
    operating_income: float | None = None
    ebitda: float | None = None
    depreciation_amortization: float | None = None
    interest_expense: float | None = None
    interest_income: float | None = None
    income_before_tax: float | None = None
    income_tax_expense: float | None = None
    net_income: float | None = None
    eps: float | None = None
    eps_diluted: float | None = None
    weighted_average_shares: float | None = None
    weighted_average_shares_diluted: float | None = None


class BalanceSheetRecord(StatementBase):
    statement_kind: str = "balance_sheet"

    cash_and_equivalents: float | None = None
    short_term_investments: float | None = None
    cash_and_short_term_investments: float | None = None
    net_receivables: float | None = None
    inventory: float | None = None
    other_current_assets: float | None = None
    total_current_assets: float | None = None
    property_plant_equipment_net: float | None = None
    goodwill: float | None = None
    intangible_assets: float | None = None
    long_term_investments: float | None = None
    total_non_current_assets: float | None = None
    total_assets: float | None = None

    accounts_payable: float | None = None
    short_term_debt: float | None = None
    deferred_revenue: float | None = None
    other_current_liabilities: float | None = None
    total_current_liabilities: float | None = None
    long_term_debt: float | None = None
    total_non_current_liabilities: float | None = None
    total_liabilities: float | None = None

    common_stock: float | None = None
    retained_earnings: float | None = None
    total_equity: float | None = None
    total_debt: float | None = None
    net_debt: float | None = None


class CashFlowRecord(StatementBase):
    statement_kind: str = "cash_flow"

    net_income: float | None = None
    depreciation_amortization: float | None = None
    stock_based_compensation: float | None = None
    change_in_working_capital: float | None = None
    accounts_receivable_change: float | None = None
    inventory_change: float | None = None
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    acquisitions_net: float | None = None
    investing_cash_flow: float | None = None
    debt_repayment: float | None = None
    dividends_paid: float | None = None
    common_stock_repurchased: float | None = None
    financing_cash_flow: float | None = None
    net_change_in_cash: float | None = None
    cash_at_end_of_period: float | None = None
    free_cash_flow: float | None = None


class MarketDataPoint(ProvenanceBase):
    """A single daily observation. Feeds valuation ratios and Altman X4."""

    data_source: str = "yfinance"
    date: date
    close_price: float | None = None
    volume: float | None = None
    market_cap: float | None = None
    shares_outstanding: float | None = None


class CompanyFinancials(BaseModel):
    """Everything ingested for one ticker in one run."""

    profile: CompanyProfile
    income_statements: list[IncomeStatementRecord] = Field(default_factory=list)
    balance_sheets: list[BalanceSheetRecord] = Field(default_factory=list)
    cash_flows: list[CashFlowRecord] = Field(default_factory=list)
    market_data: list[MarketDataPoint] = Field(default_factory=list)

    @property
    def fiscal_years(self) -> list[int]:
        years = {s.fiscal_year for s in self.income_statements}
        years |= {s.fiscal_year for s in self.balance_sheets}
        years |= {s.fiscal_year for s in self.cash_flows}
        return sorted(years, reverse=True)

    def missing_data_map(self) -> dict[str, dict[str, list[str]]]:
        """The 'missing data map' every downstream layer consumes.

        Shape: {"FY2023": {"income_statement": ["ebitda", ...], ...}, ...}
        """
        out: dict[str, dict[str, list[str]]] = {}
        for group, records in (
            ("income_statement", self.income_statements),
            ("balance_sheet", self.balance_sheets),
            ("cash_flow", self.cash_flows),
        ):
            for rec in records:
                if not rec.missing_fields:
                    continue
                out.setdefault(rec.period_label, {})[group] = sorted(rec.missing_fields)
        return out


class IngestionReport(BaseModel):
    """Outcome of one ingestion run — persisted to `ingestion_runs`."""

    ticker: str
    period: PeriodType
    status: str = "success"  # success | partial | failed
    years_requested: int | None = None
    records_written: dict[str, int] = Field(default_factory=dict)
    missing_summary: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    critical_missing: dict[str, list[str]] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None

    def summary_line(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.records_written.items()))
        return f"[{self.status.upper()}] {self.ticker} ({self.period.value}): {counts or 'no records'}"
