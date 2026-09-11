"""Ratio, distress-score and red-flag endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import cached, get_company, get_db, normalise_ticker, read_cache
from src.api.schemas import (
    DistressResponse,
    DistressScore,
    RatioPeriod,
    RatioSeriesResponse,
    RatiosResponse,
    RatioValue,
    RedFlagItem,
    RedFlagsResponse,
)
from src.db.models import Company
from src.ratios import repository as ratio_repository

router = APIRouter(tags=["analysis"])

DISTRESS_SCORES = [
    ("altman_z_score", "Altman Z-Score"),
    ("altman_z_double_prime_score", "Altman Z''-Score (book value)"),
    ("piotroski_f_score", "Piotroski F-Score"),
    ("beneish_m_score", "Beneish M-Score"),
]

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}


@router.get(
    "/companies/{ticker}/ratios",
    response_model=RatiosResponse,
    summary="All computed ratios, grouped by period",
)
def get_ratios(
    ticker: str = Depends(normalise_ticker),
    company: Company = Depends(get_company),
    db: Session = Depends(get_db),
    period: str = Query("FY"),
    fiscal_year: int | None = Query(None),
    category: str | None = Query(None, description="liquidity, profitability, ..."),
    calculable_only: bool = Query(False),
) -> RatiosResponse:
    """Ratios with `is_calculable` and, where false, the reason.

    A ratio that could not be computed is returned as an explicit entry rather
    than omitted, so a client cannot mistake absence for zero.
    """

    def produce() -> RatiosResponse:
        rows = ratio_repository.get_ratios(
            db, ticker, period=period, fiscal_year=fiscal_year, calculable_only=calculable_only
        )
        if category:
            rows = [r for r in rows if r.category == category]

        by_year: dict[int, list[RatioValue]] = {}
        for row in rows:
            by_year.setdefault(row.fiscal_year, []).append(
                RatioValue(
                    name=row.ratio_name,
                    category=row.category,
                    value=row.ratio_value,
                    is_calculable=bool(row.is_calculable),
                    reason=row.reason,
                    method=row.method,
                    inputs=row.inputs_json or {},
                    details=row.details_json or {},
                )
            )

        periods = [
            RatioPeriod(
                fiscal_year=year,
                period=period,
                ratios=sorted(values, key=lambda r: (r.category or "", r.name)),
            )
            for year, values in sorted(by_year.items(), reverse=True)
        ]
        return RatiosResponse(
            ticker=ticker,
            periods=periods,
            categories=sorted({r.category for r in rows if r.category}),
        )

    key = f"ratios:{ticker}:{period}:{fiscal_year}:{category}:{calculable_only}"
    return cached(read_cache(), key, produce)


@router.get(
    "/companies/{ticker}/ratios/{ratio_name}/series",
    response_model=RatioSeriesResponse,
    summary="One ratio across every fiscal year",
)
def get_ratio_series(
    ratio_name: str,
    ticker: str = Depends(normalise_ticker),
    company: Company = Depends(get_company),
    db: Session = Depends(get_db),
    period: str = Query("FY"),
) -> RatioSeriesResponse:
    """The shape the dashboard trend charts consume."""

    def produce() -> RatioSeriesResponse:
        return RatioSeriesResponse(
            ticker=ticker,
            ratio_name=ratio_name,
            points=ratio_repository.get_ratio_series(db, ticker, ratio_name, period=period),
        )

    return cached(read_cache(), f"series:{ticker}:{ratio_name}:{period}", produce)


@router.get(
    "/companies/{ticker}/distress-score",
    response_model=DistressResponse,
    summary="Altman Z, Altman Z'', Piotroski F and Beneish M",
)
def get_distress(
    ticker: str = Depends(normalise_ticker),
    company: Company = Depends(get_company),
    db: Session = Depends(get_db),
    period: str = Query("FY"),
    fiscal_year: int | None = Query(None, description="Defaults to the latest year on file"),
) -> DistressResponse:
    """Distress scores with their zone, thresholds and components.

    Financial-sector companies return the Altman scores as not calculable with
    "Not Applicable - Financial Sector"; the model is not defined for banks and a
    number here would be misleading.
    """

    def produce() -> DistressResponse:
        rows = ratio_repository.get_ratios(db, ticker, period=period, fiscal_year=fiscal_year)
        rows = [r for r in rows if r.category == "distress"]
        if not rows:
            return DistressResponse(
                ticker=ticker,
                is_financial_sector=bool(company.is_financial_sector),
                note="No distress scores on file. Run the ratio engine for this company.",
            )

        year = fiscal_year if fiscal_year is not None else max(r.fiscal_year for r in rows)
        selected = {r.ratio_name: r for r in rows if r.fiscal_year == year}

        scores = []
        for name, label in DISTRESS_SCORES:
            row = selected.get(name)
            if row is None:
                continue
            details = row.details_json or {}
            scores.append(
                DistressScore(
                    name=label,
                    value=row.ratio_value,
                    is_calculable=bool(row.is_calculable),
                    reason=row.reason,
                    zone=details.get("zone"),
                    interpretation=details.get("interpretation"),
                    thresholds=details.get("thresholds", {}),
                    components=row.inputs_json or {},
                )
            )

        note = None
        if company.is_financial_sector:
            note = (
                "This company is in the financial sector. The Altman models are calibrated "
                "on non-financial firms and are reported as Not Applicable rather than "
                "producing a misleading score."
            )
        return DistressResponse(
            ticker=ticker,
            fiscal_year=year,
            is_financial_sector=bool(company.is_financial_sector),
            scores=scores,
            note=note,
        )

    return cached(read_cache(), f"distress:{ticker}:{period}:{fiscal_year}", produce)


@router.get(
    "/companies/{ticker}/red-flags",
    response_model=RedFlagsResponse,
    summary="Rule-based red flags with severity",
)
def get_red_flags(
    ticker: str = Depends(normalise_ticker),
    company: Company = Depends(get_company),
    db: Session = Depends(get_db),
    period: str = Query("FY"),
    fiscal_year: int | None = Query(None),
    min_severity: str | None = Query(None, description="High, Medium, Low or Info"),
) -> RedFlagsResponse:
    """Deterministic rule output - never LLM-generated."""

    def produce() -> RedFlagsResponse:
        rows = ratio_repository.get_red_flags(
            db, ticker, period=period, fiscal_year=fiscal_year, min_severity=min_severity
        )
        flags = [
            RedFlagItem(
                flag_name=row.flag_name,
                severity=row.severity,
                category=(row.source_json or {}).get("category"),
                explanation=row.explanation,
                fiscal_year=row.fiscal_year,
                period=row.period,
                source_values=(row.source_json or {}).get("source_values", {}),
            )
            for row in rows
        ]
        flags.sort(key=lambda f: (-f.fiscal_year, SEVERITY_ORDER.get(f.severity, 9), f.flag_name))

        counts: dict[str, int] = {}
        for flag in flags:
            counts[flag.severity] = counts.get(flag.severity, 0) + 1
        return RedFlagsResponse(ticker=ticker, flags=flags, counts_by_severity=counts)

    key = f"flags:{ticker}:{period}:{fiscal_year}:{min_severity}"
    return cached(read_cache(), key, produce)
