"""SQLAlchemy ORM models — the persistent schema for every layer.

Layer 1 (ingestion) writes: companies, income_statements, balance_sheets,
cash_flow_statements, market_data. Layer 2 writes ratios and red_flags; Layer 3
writes rag_eval_logs; Layer 4 writes distress_predictions. Declaring them all
here means the schema is created once and later layers add logic, not migrations.

Anti-hallucination invariant: every statement row carries `is_missing_json`, an
explicit map of {field_name: reason} for anything the source did not report. A
NULL column is therefore never ambiguous — downstream layers can tell
"not reported" apart from "genuinely zero".
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all models."""


class ProvenanceMixin:
    """Columns every ingested row carries so any number can be traced back."""

    data_source: Mapped[str] = mapped_column(String(32), nullable=False, default="fmp")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    # {"field_name": "not_reported" | "null_in_source", ...}
    is_missing_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class Company(Base, ProvenanceMixin):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(128), index=True)
    industry: Mapped[str | None] = mapped_column(String(128))
    exchange: Mapped[str | None] = mapped_column(String(64))
    country: Mapped[str | None] = mapped_column(String(64))
    currency: Mapped[str | None] = mapped_column(String(8))
    cik: Mapped[str | None] = mapped_column(String(16))
    description: Mapped[str | None] = mapped_column(Text)
    # True for banks/insurers/etc. — Altman Z-Score is not applicable to them.
    is_financial_sector: Mapped[bool] = mapped_column(Boolean, default=False)

    income_statements: Mapped[list["IncomeStatement"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )
    balance_sheets: Mapped[list["BalanceSheet"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )
    cash_flow_statements: Mapped[list["CashFlowStatement"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )
    market_data: Mapped[list["MarketData"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Company {self.ticker} ({self.name})>"


class StatementMixin(ProvenanceMixin):
    """Shared identity columns for the three financial statements."""

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # "FY" for annual, "Q1".."Q4" for quarterly.
    period: Mapped[str] = mapped_column(String(4), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    period_end_date: Mapped[date | None] = mapped_column(Date)
    filing_date: Mapped[date | None] = mapped_column(Date)
    reported_currency: Mapped[str | None] = mapped_column(String(8))


class IncomeStatement(Base, StatementMixin):
    __tablename__ = "income_statements"
    __table_args__ = (
        UniqueConstraint("company_id", "period", "fiscal_year", name="uq_income_company_period"),
        Index("ix_income_company_year", "company_id", "fiscal_year"),
    )

    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))

    revenue: Mapped[float | None] = mapped_column(Float)
    cost_of_revenue: Mapped[float | None] = mapped_column(Float)
    gross_profit: Mapped[float | None] = mapped_column(Float)
    research_and_development: Mapped[float | None] = mapped_column(Float)
    selling_general_admin: Mapped[float | None] = mapped_column(Float)
    operating_expenses: Mapped[float | None] = mapped_column(Float)
    operating_income: Mapped[float | None] = mapped_column(Float)  # EBIT
    ebitda: Mapped[float | None] = mapped_column(Float)
    depreciation_amortization: Mapped[float | None] = mapped_column(Float)
    interest_expense: Mapped[float | None] = mapped_column(Float)
    interest_income: Mapped[float | None] = mapped_column(Float)
    income_before_tax: Mapped[float | None] = mapped_column(Float)
    income_tax_expense: Mapped[float | None] = mapped_column(Float)
    net_income: Mapped[float | None] = mapped_column(Float)
    eps: Mapped[float | None] = mapped_column(Float)
    eps_diluted: Mapped[float | None] = mapped_column(Float)
    weighted_average_shares: Mapped[float | None] = mapped_column(Float)
    weighted_average_shares_diluted: Mapped[float | None] = mapped_column(Float)

    company: Mapped[Company] = relationship(back_populates="income_statements")


class BalanceSheet(Base, StatementMixin):
    __tablename__ = "balance_sheets"
    __table_args__ = (
        UniqueConstraint("company_id", "period", "fiscal_year", name="uq_balance_company_period"),
        Index("ix_balance_company_year", "company_id", "fiscal_year"),
    )

    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))

    cash_and_equivalents: Mapped[float | None] = mapped_column(Float)
    short_term_investments: Mapped[float | None] = mapped_column(Float)
    cash_and_short_term_investments: Mapped[float | None] = mapped_column(Float)
    net_receivables: Mapped[float | None] = mapped_column(Float)
    inventory: Mapped[float | None] = mapped_column(Float)
    other_current_assets: Mapped[float | None] = mapped_column(Float)
    total_current_assets: Mapped[float | None] = mapped_column(Float)
    property_plant_equipment_net: Mapped[float | None] = mapped_column(Float)
    goodwill: Mapped[float | None] = mapped_column(Float)
    intangible_assets: Mapped[float | None] = mapped_column(Float)
    long_term_investments: Mapped[float | None] = mapped_column(Float)
    total_non_current_assets: Mapped[float | None] = mapped_column(Float)
    total_assets: Mapped[float | None] = mapped_column(Float)

    accounts_payable: Mapped[float | None] = mapped_column(Float)
    short_term_debt: Mapped[float | None] = mapped_column(Float)
    deferred_revenue: Mapped[float | None] = mapped_column(Float)
    other_current_liabilities: Mapped[float | None] = mapped_column(Float)
    total_current_liabilities: Mapped[float | None] = mapped_column(Float)
    long_term_debt: Mapped[float | None] = mapped_column(Float)
    total_non_current_liabilities: Mapped[float | None] = mapped_column(Float)
    total_liabilities: Mapped[float | None] = mapped_column(Float)

    common_stock: Mapped[float | None] = mapped_column(Float)
    retained_earnings: Mapped[float | None] = mapped_column(Float)
    total_equity: Mapped[float | None] = mapped_column(Float)
    total_debt: Mapped[float | None] = mapped_column(Float)
    net_debt: Mapped[float | None] = mapped_column(Float)

    company: Mapped[Company] = relationship(back_populates="balance_sheets")


class CashFlowStatement(Base, StatementMixin):
    __tablename__ = "cash_flow_statements"
    __table_args__ = (
        UniqueConstraint("company_id", "period", "fiscal_year", name="uq_cashflow_company_period"),
        Index("ix_cashflow_company_year", "company_id", "fiscal_year"),
    )

    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))

    net_income: Mapped[float | None] = mapped_column(Float)
    depreciation_amortization: Mapped[float | None] = mapped_column(Float)
    stock_based_compensation: Mapped[float | None] = mapped_column(Float)
    change_in_working_capital: Mapped[float | None] = mapped_column(Float)
    accounts_receivable_change: Mapped[float | None] = mapped_column(Float)
    inventory_change: Mapped[float | None] = mapped_column(Float)
    operating_cash_flow: Mapped[float | None] = mapped_column(Float)
    capital_expenditure: Mapped[float | None] = mapped_column(Float)
    acquisitions_net: Mapped[float | None] = mapped_column(Float)
    investing_cash_flow: Mapped[float | None] = mapped_column(Float)
    debt_repayment: Mapped[float | None] = mapped_column(Float)
    dividends_paid: Mapped[float | None] = mapped_column(Float)
    common_stock_repurchased: Mapped[float | None] = mapped_column(Float)
    financing_cash_flow: Mapped[float | None] = mapped_column(Float)
    net_change_in_cash: Mapped[float | None] = mapped_column(Float)
    cash_at_end_of_period: Mapped[float | None] = mapped_column(Float)
    free_cash_flow: Mapped[float | None] = mapped_column(Float)

    company: Mapped[Company] = relationship(back_populates="cash_flow_statements")


class MarketData(Base, ProvenanceMixin):
    """Daily price / market-cap points — feeds valuation ratios and Altman X4."""

    __tablename__ = "market_data"
    __table_args__ = (
        UniqueConstraint("company_id", "date", name="uq_market_company_date"),
        Index("ix_market_company_date", "company_id", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    close_price: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    market_cap: Mapped[float | None] = mapped_column(Float)
    shares_outstanding: Mapped[float | None] = mapped_column(Float)

    company: Mapped[Company] = relationship(back_populates="market_data")


# ---------------------------------------------------------------------------
# Downstream layers
# ---------------------------------------------------------------------------


class Ratio(Base):
    __tablename__ = "ratios"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "period", "fiscal_year", "ratio_name", name="uq_ratio_identity"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    period: Mapped[str] = mapped_column(String(4), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    ratio_name: Mapped[str] = mapped_column(String(64), nullable=False)
    ratio_value: Mapped[float | None] = mapped_column(Float)
    category: Mapped[str | None] = mapped_column(String(32))
    is_calculable: Mapped[bool] = mapped_column(Boolean, default=True)
    # Names the specific missing input when is_calculable is False.
    reason: Mapped[str | None] = mapped_column(Text)
    # How the ratio was computed when more than one basis is defensible
    # (e.g. average vs closing balance), so two periods are never silently
    # compared on different bases.
    method: Mapped[str | None] = mapped_column(Text)
    # The input values behind the number, and any extra structure such as an
    # Altman zone or the nine Piotroski signals. This is what lets Layer 3
    # narrate a score from pre-computed facts instead of inferring them.
    inputs_json: Mapped[dict] = mapped_column(JSON, default=dict)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RedFlag(Base):
    __tablename__ = "red_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    period: Mapped[str] = mapped_column(String(4), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    flag_name: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)  # Low/Medium/High/Info
    explanation: Mapped[str | None] = mapped_column(Text)
    source_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class DistressPrediction(Base):
    """One financial-distress probability per company-period from Layer 4.

    The estimate comes from a classifier trained on an external labelled
    bankruptcy dataset, not from this company's own history, so it is kept
    apart from the rule-based Layer 2 distress scores. `is_scored = False`
    with a `reason` when the period lacks the inputs the model needs -
    the same "absence is explicit" rule the rest of the schema follows.
    """

    __tablename__ = "distress_predictions"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "period", "fiscal_year", "model_name",
            name="uq_distress_prediction_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    period: Mapped[str] = mapped_column(String(4), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)

    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)

    # Calibrated P(financial distress within the dataset's forecast horizon).
    probability: Mapped[float | None] = mapped_column(Float)
    risk_band: Mapped[str | None] = mapped_column(String(16))  # Low/Moderate/Elevated/High
    is_scored: Mapped[bool] = mapped_column(Boolean, default=True)
    reason: Mapped[str | None] = mapped_column(Text)

    # The exact feature vector scored, and the features that pushed the
    # estimate up most, so the number is auditable rather than opaque.
    features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    factors_json: Mapped[dict] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RagEvalLog(Base):
    """Every RAG query, its retrieved context, its answer and its scores.

    Logged for production questions as well as evaluation runs, so groundedness
    can be tracked over real usage rather than only over a fixed test set.
    """

    __tablename__ = "rag_eval_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id", ondelete="SET NULL"))
    ticker: Mapped[str | None] = mapped_column(String(16), index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(16), default="ask")  # ask/summary/eval
    response: Mapped[str | None] = mapped_column(Text)
    retrieved_chunks_json: Mapped[dict] = mapped_column(JSON, default=dict)
    # Metric name -> score. Named generically rather than after any one library.
    scores_json: Mapped[dict] = mapped_column(JSON, default=dict)
    # Share of numeric claims in the answer traceable to the retrieved context.
    groundedness: Mapped[float | None] = mapped_column(Float)
    unverified_json: Mapped[dict] = mapped_column(JSON, default=dict)
    blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class IngestionRun(Base):
    """Audit trail: one row per ingestion attempt per ticker."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    period: Mapped[str] = mapped_column(String(16), nullable=False)  # annual / quarter
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # success/partial/failed
    years_requested: Mapped[int | None] = mapped_column(Integer)
    records_written_json: Mapped[dict] = mapped_column(JSON, default=dict)
    missing_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    errors_json: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
