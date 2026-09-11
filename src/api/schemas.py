"""Response models for the API.

These are the public contract. They deliberately carry the *absence* of data as
first-class information: `is_calculable` with a `reason`, `missing_fields` on
statements, `groundedness` on generated answers. A client that renders these
faithfully cannot present an estimate as a fact, which is the same guarantee the
lower layers enforce internally.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    database: str
    companies: int
    ratios: int
    rag_indexed: bool
    # Indexing is a separate step from ingestion, so a company can have ratios
    # and still be unanswerable. Reporting both counts makes that gap visible
    # instead of leaving it to be discovered by a failed question.
    companies_analysed: int = 0
    companies_indexed: int = 0
    # Layer 4: whether a trained distress model is on disk.
    distress_model_ready: bool = False
    llm_configured: bool
    version: str


class CompanySummary(BaseModel):
    """One row in the company list."""

    model_config = ConfigDict(from_attributes=True)

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    fiscal_years: list[int] = Field(default_factory=list)


class CompanyProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    country: str | None = None
    currency: str | None = None
    description: str | None = None
    is_financial_sector: bool = False
    fiscal_years: list[int] = Field(default_factory=list)
    data_source: str | None = None
    fetched_at: datetime | None = None


class StatementLine(BaseModel):
    """One statement for one period, with its gaps named."""

    period: str
    fiscal_year: int
    period_end_date: date | None = None
    reported_currency: str | None = None
    values: dict[str, float | None] = Field(default_factory=dict)
    # field name -> why it is absent, straight from ingestion
    missing_fields: dict[str, str] = Field(default_factory=dict)


class StatementsResponse(BaseModel):
    ticker: str
    period: str
    income_statements: list[StatementLine] = Field(default_factory=list)
    balance_sheets: list[StatementLine] = Field(default_factory=list)
    cash_flow_statements: list[StatementLine] = Field(default_factory=list)


class RatioValue(BaseModel):
    """A ratio, or an explicit statement that it could not be computed."""

    name: str
    category: str | None = None
    value: float | None = None
    is_calculable: bool = True
    reason: str | None = None
    method: str | None = None
    inputs: dict = Field(default_factory=dict)
    details: dict = Field(default_factory=dict)


class RatioPeriod(BaseModel):
    fiscal_year: int
    period: str
    ratios: list[RatioValue] = Field(default_factory=list)


class RatiosResponse(BaseModel):
    ticker: str
    periods: list[RatioPeriod] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class RatioSeriesResponse(BaseModel):
    """One ratio across fiscal years - the shape the trend charts consume."""

    ticker: str
    ratio_name: str
    points: dict[int, float | None] = Field(default_factory=dict)


class DistressScore(BaseModel):
    name: str
    value: float | None = None
    is_calculable: bool = True
    reason: str | None = None
    zone: str | None = None
    interpretation: str | None = None
    thresholds: dict = Field(default_factory=dict)
    components: dict = Field(default_factory=dict)


class DistressResponse(BaseModel):
    ticker: str
    fiscal_year: int | None = None
    is_financial_sector: bool = False
    scores: list[DistressScore] = Field(default_factory=list)
    note: str | None = None


class DistressFactor(BaseModel):
    feature: str
    value: float
    training_median: float
    direction: str


class DistressPredictionPeriod(BaseModel):
    fiscal_year: int
    period: str
    is_scored: bool
    probability: float | None = None
    risk_band: str | None = None
    reason: str | None = None
    factors: list[DistressFactor] = Field(default_factory=list)


class DistressModelInfo(BaseModel):
    """What the estimate was produced by, so it is not read as a bare number."""

    name: str
    version: str
    trained_at: str | None = None
    training_rows: int = 0
    training_prevalence: float = 0.0
    cv_roc_auc: float | None = None
    cv_pr_auc: float | None = None
    cv_brier: float | None = None
    baseline_pr_auc: float | None = None
    dataset: str = "American public companies, 1999-2018 (sowide/bankruptcy_dataset)"


class DistressPredictionResponse(BaseModel):
    ticker: str
    available: bool = True
    note: str | None = None
    model: DistressModelInfo | None = None
    periods: list[DistressPredictionPeriod] = Field(default_factory=list)


class RedFlagItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    flag_name: str
    severity: str
    category: str | None = None
    explanation: str | None = None
    fiscal_year: int
    period: str
    source_values: dict = Field(default_factory=dict)


class RedFlagsResponse(BaseModel):
    ticker: str
    flags: list[RedFlagItem] = Field(default_factory=list)
    counts_by_severity: dict[str, int] = Field(default_factory=dict)


class SourceChunk(BaseModel):
    """A retrieved context chunk, so any claim can be audited."""

    citation: str
    chunk_type: str
    fiscal_year: int | None = None
    similarity: float
    excerpt: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    fiscal_year: int | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class AnswerResponse(BaseModel):
    """A generated answer with its verification attached, never bare text."""

    ticker: str
    question: str
    answer: str
    groundedness: float
    verified_claims: int = 0
    total_claims: int = 0
    unverified: list[str] = Field(default_factory=list)
    blocked: bool = False
    notes: list[str] = Field(default_factory=list)
    model: str | None = None
    latency_seconds: float = 0.0
    sources: list[SourceChunk] = Field(default_factory=list)


class IngestRequest(BaseModel):
    years: int = Field(default=5, ge=1, le=20)
    period: str = Field(default="annual", pattern="^(annual|quarter)$")
    include_market_data: bool = True
    run_ratios: bool = True
    index_for_rag: bool = True


class IngestResponse(BaseModel):
    ticker: str
    status: str
    records_written: dict[str, int] = Field(default_factory=dict)
    ratios_computed: int = 0
    red_flags_raised: int = 0
    chunks_indexed: int = 0
    distress_estimates: int = 0
    critical_missing: dict[str, list[str]] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
    hint: str | None = None
