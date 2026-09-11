"""Layer 2 orchestration: load statements -> compute ratios -> detect flags -> store.

Importing this module registers every ratio, since the category modules populate
the registry at import time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import (
    BalanceSheet,
    CashFlowStatement,
    Company,
    IncomeStatement,
    MarketData,
)
from src.ratios.base import (
    RATIO_REGISTRY,
    PeriodAnalysis,
    PeriodBundle,
    RatioResult,
    evaluate,
)
from src.ratios.red_flags import RedFlagResult, detect_all

# Registration side effects — each import appends to RATIO_REGISTRY.
from src.ratios import cash_flow as _cash_flow  # noqa: F401
from src.ratios import distress_scores as _distress  # noqa: F401
from src.ratios import efficiency as _efficiency  # noqa: F401
from src.ratios import leverage as _leverage  # noqa: F401
from src.ratios import liquidity as _liquidity  # noqa: F401
from src.ratios import profitability as _profitability  # noqa: F401
from src.ratios import valuation as _valuation  # noqa: F401

logger = logging.getLogger(__name__)

# How far from a fiscal period end a price observation may sit and still be
# treated as the period-end market value. Wide enough to absorb the monthly
# sampling stored by Layer 1, narrow enough that a 2021 balance sheet is never
# valued at a 2026 share price.
MARKET_DATA_WINDOW_DAYS = 45


@dataclass
class AnalysisReport:
    """Outcome of one Layer 2 run over a ticker."""

    ticker: str
    period: str = "FY"
    fiscal_years: list[int] = field(default_factory=list)
    ratios_computed: int = 0
    ratios_not_calculable: int = 0
    red_flags_raised: int = 0
    flags_by_severity: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of attempted ratios that produced a number."""
        total = self.ratios_computed + self.ratios_not_calculable
        return self.ratios_computed / total if total else 0.0

    def summary_line(self) -> str:
        years = f"{min(self.fiscal_years)}-{max(self.fiscal_years)}" if self.fiscal_years else "none"
        return (
            f"{self.ticker} ({self.period} {years}): {self.ratios_computed} ratios computed, "
            f"{self.ratios_not_calculable} not calculable ({self.coverage:.0%} coverage), "
            f"{self.red_flags_raised} red flags"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _pair_market_data(
    session: Session, company_id: int, period_end: date | None
) -> MarketData | None:
    """The price observation closest to a period end, within the allowed window."""
    if period_end is None:
        return None
    lower = period_end - timedelta(days=MARKET_DATA_WINDOW_DAYS)
    upper = period_end + timedelta(days=MARKET_DATA_WINDOW_DAYS)
    candidates = list(
        session.scalars(
            select(MarketData).where(
                MarketData.company_id == company_id,
                MarketData.date >= lower,
                MarketData.date <= upper,
            )
        )
    )
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs((row.date - period_end).days))


def load_history(
    session: Session, ticker: str, period: str = "FY"
) -> list[PeriodBundle]:
    """Assemble one bundle per fiscal year, oldest first, each linked to its prior.

    Statements are joined on fiscal year rather than assumed to arrive together:
    a company can easily have five income statements and four balance sheets, and
    the ratios that need both should fail loudly for the odd year out.
    """
    company = session.scalar(select(Company).where(Company.ticker == ticker.strip().upper()))
    if company is None:
        return []

    def rows_by_year(model) -> dict[int, object]:
        return {
            row.fiscal_year: row
            for row in session.scalars(
                select(model).where(model.company_id == company.id, model.period == period)
            )
        }

    income = rows_by_year(IncomeStatement)
    balance = rows_by_year(BalanceSheet)
    cash = rows_by_year(CashFlowStatement)

    years = sorted(set(income) | set(balance) | set(cash))
    bundles: list[PeriodBundle] = []
    previous: PeriodBundle | None = None

    for year in years:
        statement = income.get(year) or balance.get(year) or cash.get(year)
        period_end = getattr(statement, "period_end_date", None)
        bundle = PeriodBundle(
            ticker=company.ticker,
            period=period,
            fiscal_year=year,
            period_end_date=period_end,
            income=income.get(year),
            balance=balance.get(year),
            cash_flow=cash.get(year),
            market=_pair_market_data(session, company.id, period_end),
            is_financial_sector=bool(company.is_financial_sector),
            sector=company.sector,
            prior=previous,
        )
        bundles.append(bundle)
        previous = bundle

    return bundles


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_ratios(bundle: PeriodBundle) -> dict[str, RatioResult]:
    """Every registered ratio for one period, calculable or explicitly not."""
    return {spec.name: evaluate(spec, bundle) for spec in RATIO_REGISTRY}


def analyze(bundles: list[PeriodBundle]) -> list[PeriodAnalysis]:
    return [PeriodAnalysis(bundle=b, ratios=compute_ratios(b)) for b in bundles]


def analyze_ticker(
    session: Session, ticker: str, period: str = "FY"
) -> tuple[list[PeriodAnalysis], dict[int, list[RedFlagResult]]]:
    """Load, compute and detect for one ticker without writing anything."""
    bundles = load_history(session, ticker, period)
    analyses = analyze(bundles)
    flags = detect_all(analyses)
    return analyses, flags


def build_report(
    ticker: str,
    period: str,
    analyses: list[PeriodAnalysis],
    flags: dict[int, list[RedFlagResult]],
) -> AnalysisReport:
    report = AnalysisReport(ticker=ticker, period=period)
    report.fiscal_years = [a.fiscal_year for a in analyses]

    for analysis in analyses:
        report.ratios_computed += len(analysis.calculable())
        report.ratios_not_calculable += len(analysis.not_calculable())

    for year_flags in flags.values():
        report.red_flags_raised += len(year_flags)
        for flag in year_flags:
            key = flag.severity.value
            report.flags_by_severity[key] = report.flags_by_severity.get(key, 0) + 1

    if not analyses:
        report.warnings.append(
            f"No {period} statements on file for {ticker}. Run Layer 1 ingestion first."
        )
    elif not any(a.value("altman_z_score") is not None for a in analyses):
        if analyses[-1].bundle.is_financial_sector:
            report.warnings.append(
                f"{ticker} is a financial-sector company; the Altman Z-Score is not applicable "
                "and was skipped for every period."
            )
        else:
            report.warnings.append(
                "No market data was paired to any period, so the market-value Altman Z-Score "
                "is unavailable. The book-value Z''-Score was used instead."
            )
    return report


def run_analysis(
    session: Session,
    ticker: str,
    period: str = "FY",
    persist: bool = True,
) -> AnalysisReport:
    """Compute and (by default) persist every ratio and flag for one ticker."""
    from src.ratios import repository  # local import keeps the module import graph flat

    ticker = ticker.strip().upper()
    analyses, flags = analyze_ticker(session, ticker, period)
    report = build_report(ticker, period, analyses, flags)

    if persist and analyses:
        company = session.scalar(select(Company).where(Company.ticker == ticker))
        if company is None:
            report.warnings.append(f"{ticker} is not in the companies table; nothing was stored.")
        else:
            repository.save_analysis(session, company.id, analyses, flags)

    logger.info(report.summary_line())
    for warning in report.warnings:
        logger.warning("%s: %s", ticker, warning)
    return report
