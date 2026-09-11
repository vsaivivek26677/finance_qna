"""Persistence for computed ratios and detected red flags.

Ratios upsert on their natural key (company + period + fiscal year + name).
Red flags are replaced wholesale per period instead: rules can be added, removed
or retuned between runs, and an upsert would leave flags from a previous rule set
sitting in the table with no way to tell they are stale.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.db.models import Company, Ratio, RedFlag
from src.ratios.base import PeriodAnalysis, RatioResult
from src.ratios.red_flags import RedFlagResult

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    """Coerce numpy/decimal scalars and nested containers into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value)


def save_ratios(
    session: Session,
    company_id: int,
    period: str,
    fiscal_year: int,
    results: Sequence[RatioResult],
) -> int:
    """Insert or refresh one period's ratios. Returns the number written."""
    existing = {
        row.ratio_name: row
        for row in session.scalars(
            select(Ratio).where(
                Ratio.company_id == company_id,
                Ratio.period == period,
                Ratio.fiscal_year == fiscal_year,
            )
        )
    }

    for result in results:
        values = {
            "ratio_value": result.value,
            "category": result.category,
            "is_calculable": result.is_calculable,
            "reason": result.reason,
            "method": result.method,
            "inputs_json": _jsonable(result.inputs),
            "details_json": _jsonable(result.details),
        }
        row = existing.get(result.name)
        if row is None:
            session.add(
                Ratio(
                    company_id=company_id,
                    period=period,
                    fiscal_year=fiscal_year,
                    ratio_name=result.name,
                    **values,
                )
            )
        else:
            for key, value in values.items():
                setattr(row, key, value)

    session.flush()
    return len(results)


def save_red_flags(
    session: Session,
    company_id: int,
    period: str,
    fiscal_year: int,
    flags: Sequence[RedFlagResult],
) -> int:
    """Replace one period's flags with the current rule set's output."""
    session.execute(
        delete(RedFlag).where(
            RedFlag.company_id == company_id,
            RedFlag.period == period,
            RedFlag.fiscal_year == fiscal_year,
        )
    )
    for flag in flags:
        session.add(
            RedFlag(
                company_id=company_id,
                period=period,
                fiscal_year=fiscal_year,
                flag_name=flag.flag_name,
                severity=flag.severity.value,
                explanation=flag.explanation,
                source_json=_jsonable(
                    {"category": flag.category, "source_values": flag.source_values}
                ),
            )
        )
    session.flush()
    return len(flags)


def save_analysis(
    session: Session,
    company_id: int,
    analyses: Sequence[PeriodAnalysis],
    flags_by_year: dict[int, list[RedFlagResult]],
) -> tuple[int, int]:
    """Persist every period's ratios and flags. Returns (ratios, flags) counts."""
    ratio_count = flag_count = 0
    for analysis in analyses:
        ratio_count += save_ratios(
            session,
            company_id,
            analysis.period,
            analysis.fiscal_year,
            list(analysis.ratios.values()),
        )
        flag_count += save_red_flags(
            session,
            company_id,
            analysis.period,
            analysis.fiscal_year,
            flags_by_year.get(analysis.fiscal_year, []),
        )
    return ratio_count, flag_count


# --- Read helpers (consumed by Layers 3-5) ---------------------------------


def get_ratios(
    session: Session,
    ticker: str,
    period: str = "FY",
    fiscal_year: int | None = None,
    calculable_only: bool = False,
) -> list[Ratio]:
    company = session.scalar(select(Company).where(Company.ticker == ticker.strip().upper()))
    if company is None:
        return []
    query = select(Ratio).where(Ratio.company_id == company.id, Ratio.period == period)
    if fiscal_year is not None:
        query = query.where(Ratio.fiscal_year == fiscal_year)
    if calculable_only:
        query = query.where(Ratio.is_calculable.is_(True))
    return list(session.scalars(query.order_by(Ratio.fiscal_year.desc(), Ratio.ratio_name)))


def get_red_flags(
    session: Session,
    ticker: str,
    period: str = "FY",
    fiscal_year: int | None = None,
    min_severity: str | None = None,
) -> list[RedFlag]:
    company = session.scalar(select(Company).where(Company.ticker == ticker.strip().upper()))
    if company is None:
        return []
    query = select(RedFlag).where(RedFlag.company_id == company.id, RedFlag.period == period)
    if fiscal_year is not None:
        query = query.where(RedFlag.fiscal_year == fiscal_year)
    rows = list(session.scalars(query.order_by(RedFlag.fiscal_year.desc(), RedFlag.flag_name)))

    if min_severity:
        rank = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
        cutoff = rank.get(min_severity, 3)
        rows = [row for row in rows if rank.get(row.severity, 3) <= cutoff]
    return rows


def get_ratio_series(
    session: Session, ticker: str, ratio_name: str, period: str = "FY"
) -> dict[int, float | None]:
    """One ratio across every year on file — the shape the trend charts want."""
    rows = get_ratios(session, ticker, period)
    return {
        row.fiscal_year: row.ratio_value
        for row in rows
        if row.ratio_name == ratio_name and row.is_calculable
    }
