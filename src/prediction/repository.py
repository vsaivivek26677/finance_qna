"""Build model features from stored statements, and persist the estimates.

The feature builder reads exactly the raw figures `features.py` needs, so a
company-year that is missing, say, retained earnings still produces a row -
with that feature None and the usable count lowered, which `predict.score`
turns into an explicit "not scored".

Company-years are assembled via `ratio_engine.load_history`, the same
function Layer 2 uses. That is a deliberate reuse, not just a convenience: it
gives every period `market_value_to_liabilities` a market-cap figure already
paired to the fiscal year end by Layer 2's own 45-day-window logic, instead of
re-implementing that pairing here and risking the two layers disagreeing about
which observation belongs to which period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Company, DistressPrediction, IncomeStatement
from src.prediction.features import build_features_from_figures
from src.prediction.predict import DistressEstimate
from src.ratios.ratio_engine import load_history

logger = logging.getLogger(__name__)


@dataclass
class FeatureRow:
    fiscal_year: int
    period: str
    features: dict[str, float | None]


def build_feature_rows(session: Session, ticker: str, period: str = "FY") -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    for bundle in load_history(session, ticker, period):
        features = build_features_from_figures(
            total_liabilities=bundle.get("balance", "total_liabilities"),
            net_income=bundle.get("income", "net_income"),
            operating_income=bundle.get("income", "operating_income"),
            ebitda=bundle.get("income", "ebitda"),
            retained_earnings=bundle.get("balance", "retained_earnings"),
            revenue=bundle.get("income", "revenue"),
            market_value=bundle.get("market", "market_cap"),
            gross_profit=bundle.get("income", "gross_profit"),
            operating_expenses=bundle.get("income", "operating_expenses"),
        )
        rows.append(FeatureRow(fiscal_year=bundle.fiscal_year, period=bundle.period, features=features))
    return rows


def save_estimates(
    session: Session,
    ticker: str,
    estimates: list[tuple[FeatureRow, DistressEstimate]],
) -> int:
    """Upsert one company's estimates on (company, period, year, model)."""
    company = session.scalar(select(Company).where(Company.ticker == ticker.strip().upper()))
    if company is None:
        return 0

    written = 0
    for row, estimate in estimates:
        existing = session.scalar(
            select(DistressPrediction).where(
                DistressPrediction.company_id == company.id,
                DistressPrediction.period == row.period,
                DistressPrediction.fiscal_year == row.fiscal_year,
                DistressPrediction.model_name == estimate.model_name,
            )
        )
        values = dict(
            model_version=estimate.model_version,
            probability=estimate.probability,
            risk_band=estimate.risk_band,
            is_scored=estimate.is_scored,
            reason=estimate.reason,
            features_json={k: v for k, v in row.features.items()},
            factors_json={"factors": estimate.factors},
        )
        if existing is None:
            session.add(
                DistressPrediction(
                    company_id=company.id,
                    period=row.period,
                    fiscal_year=row.fiscal_year,
                    model_name=estimate.model_name,
                    **values,
                )
            )
        else:
            for key, value in values.items():
                setattr(existing, key, value)
        written += 1

    session.flush()
    return written


def get_estimates(
    session: Session, ticker: str, period: str = "FY"
) -> list[DistressPrediction]:
    company = session.scalar(select(Company).where(Company.ticker == ticker.strip().upper()))
    if company is None:
        return []
    return list(
        session.scalars(
            select(DistressPrediction)
            .where(
                DistressPrediction.company_id == company.id,
                DistressPrediction.period == period,
            )
            .order_by(DistressPrediction.fiscal_year.desc())
        )
    )


def all_scored_tickers(session: Session) -> list[str]:
    rows = session.execute(
        select(Company.ticker)
        .join(IncomeStatement, IncomeStatement.company_id == Company.id)
        .group_by(Company.ticker)
        .order_by(Company.ticker)
    ).all()
    return [t for (t,) in rows]
