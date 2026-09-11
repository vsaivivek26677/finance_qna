"""Financial-distress probability endpoint (Layer 4).

Serves the stored estimates for a company. If none are stored yet but the model
is trained, the features are scored live so a freshly ingested company is not a
dead end - the response says the estimate has not been persisted.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import cached, get_company, get_db, normalise_ticker, read_cache
from src.api.schemas import (
    DistressFactor,
    DistressModelInfo,
    DistressPredictionPeriod,
    DistressPredictionResponse,
)
from src.db.models import Company
from src.prediction import repository as prediction_repository
from src.prediction.predict import ModelUnavailable, load_model, score

router = APIRouter(tags=["analysis"])


def _model_info() -> DistressModelInfo | None:
    try:
        model = load_model()
    except ModelUnavailable:
        return None
    metrics = model.metrics or {}
    cv = metrics.get("model", {})
    baseline = metrics.get("baseline", {})
    return DistressModelInfo(
        name=model.model_name,
        version=model.model_version,
        trained_at=model.trained_at,
        training_rows=model.n_train,
        training_prevalence=model.prevalence,
        cv_roc_auc=cv.get("roc_auc"),
        cv_pr_auc=cv.get("pr_auc"),
        cv_brier=cv.get("brier"),
        baseline_pr_auc=baseline.get("pr_auc"),
    )


@router.get(
    "/companies/{ticker}/distress-prediction",
    response_model=DistressPredictionResponse,
    summary="Model-estimated probability of financial distress",
)
def get_distress_prediction(
    ticker: str = Depends(normalise_ticker),
    company: Company = Depends(get_company),
    db: Session = Depends(get_db),
    period: str = Query("FY"),
) -> DistressPredictionResponse:
    """A calibrated probability from a classifier trained on ~78.7k labelled
    American-company firm-years, applied to ratios scaled by total liabilities.

    This is distinct from the Altman/Piotroski/Beneish scores: those apply a
    fixed textbook formula, this one is a learned weighting of the same inputs.
    A period without enough inputs comes back `is_scored: false` with a reason.
    """

    def produce() -> DistressPredictionResponse:
        info = _model_info()
        if info is None:
            return DistressPredictionResponse(
                ticker=ticker,
                available=False,
                note=(
                    "No distress model is trained. Run "
                    "`python -m src.prediction.cli train`."
                ),
            )

        stored = prediction_repository.get_estimates(db, ticker, period=period)
        if stored:
            periods = [
                DistressPredictionPeriod(
                    fiscal_year=row.fiscal_year,
                    period=row.period,
                    is_scored=bool(row.is_scored),
                    probability=row.probability,
                    risk_band=row.risk_band,
                    reason=row.reason,
                    factors=[
                        DistressFactor(**factor)
                        for factor in (row.factors_json or {}).get("factors", [])
                    ],
                )
                for row in stored
            ]
            return DistressPredictionResponse(ticker=ticker, model=info, periods=periods)

        # Not persisted yet - score live so the company is still usable.
        rows = prediction_repository.build_feature_rows(db, ticker, period=period)
        periods = []
        for row in sorted(rows, key=lambda r: r.fiscal_year, reverse=True):
            estimate = score(row.features)
            periods.append(
                DistressPredictionPeriod(
                    fiscal_year=row.fiscal_year,
                    period=row.period,
                    is_scored=estimate.is_scored,
                    probability=estimate.probability,
                    risk_band=estimate.risk_band,
                    reason=estimate.reason,
                    factors=[DistressFactor(**factor) for factor in estimate.factors],
                )
            )
        note = (
            None if not periods
            else "Scored live; not yet persisted. Run `python -m src.prediction.cli score`."
        )
        return DistressPredictionResponse(
            ticker=ticker, model=info, periods=periods, note=note
        )

    return cached(read_cache(), f"distress-prediction:{ticker}:{period}", produce)
