"""Raw API payload -> validated pydantic record, with an explicit missing map.

This module is deliberately pure (no network, no DB) so it can be unit-tested
against saved API fixtures, and so a second data source can reuse it by adding
aliases in `field_maps.py`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Mapping, TypeVar

from src.ingestion.field_maps import (
    BALANCE_SHEET_ALIASES,
    CASH_FLOW_ALIASES,
    INCOME_STATEMENT_ALIASES,
    PROFILE_ALIASES,
    STATEMENT_META_ALIASES,
)
from src.ingestion.schemas import (
    BalanceSheetRecord,
    CashFlowRecord,
    CompanyProfile,
    IncomeStatementRecord,
    MissingMap,
    MissingReason,
    StatementBase,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=StatementBase)

_SENTINEL = object()


def coerce_number(value: Any) -> float | None:
    """Best-effort numeric coercion. Returns None when the value is not a number.

    Booleans are rejected on purpose: `True` coercing to 1.0 would fabricate a
    financial figure out of a flag.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "")
        if cleaned in {"", "-", "N/A", "NA", "None", "null"}:
            return None
        # (1,234) is accounting notation for a negative number.
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        if negative:
            cleaned = cleaned[1:-1]
        try:
            parsed = float(cleaned)
        except ValueError:
            return None
        return -parsed if negative else parsed
    return None


def coerce_date(value: Any) -> date | None:
    """Parse the date formats FMP uses; None if the value is unusable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[: len(fmt) + 2].strip(), fmt).date()
        except ValueError:
            continue
    logger.debug("Unparseable date value: %r", value)
    return None


def _lookup(raw: Mapping[str, Any], aliases: list[str]) -> tuple[Any, bool]:
    """Return (value, key_was_present) for the first alias found in the payload."""
    present = False
    for alias in aliases:
        if alias in raw:
            present = True
            value = raw[alias]
            if value is not None:
                return value, True
    return (_SENTINEL if not present else None), present


def extract_numeric_fields(
    raw: Mapping[str, Any], aliases: dict[str, list[str]]
) -> tuple[dict[str, float | None], MissingMap]:
    """Map a raw payload onto canonical numeric fields.

    Every canonical field ends up in exactly one of two places: a real value in
    the returned dict, or an entry in the missing map naming why it has none.
    """
    values: dict[str, float | None] = {}
    missing: MissingMap = {}

    for canonical, alias_list in aliases.items():
        value, present = _lookup(raw, alias_list)
        if not present:
            values[canonical] = None
            missing[canonical] = MissingReason.NOT_REPORTED
            continue
        if value is None:
            values[canonical] = None
            missing[canonical] = MissingReason.NULL_IN_SOURCE
            continue
        number = coerce_number(value)
        if number is None:
            values[canonical] = None
            missing[canonical] = MissingReason.UNPARSEABLE
            logger.debug("Unparseable value for %s: %r", canonical, value)
            continue
        values[canonical] = number

    return values, missing


def _extract_meta(raw: Mapping[str, Any], fallback_period: str) -> dict[str, Any]:
    """Pull the identity columns (period, fiscal year, dates) off a statement."""
    period_value, _ = _lookup(raw, STATEMENT_META_ALIASES["period"])
    period = str(period_value).strip().upper() if period_value not in (None, _SENTINEL) else ""
    if not period:
        period = fallback_period

    end_date_value, _ = _lookup(raw, STATEMENT_META_ALIASES["period_end_date"])
    period_end_date = coerce_date(None if end_date_value is _SENTINEL else end_date_value)

    filing_value, _ = _lookup(raw, STATEMENT_META_ALIASES["filing_date"])
    filing_date = coerce_date(None if filing_value is _SENTINEL else filing_value)

    year_value, _ = _lookup(raw, STATEMENT_META_ALIASES["fiscal_year"])
    fiscal_year = coerce_number(None if year_value is _SENTINEL else year_value)
    if fiscal_year is None:
        # Fall back to the statement date rather than dropping the row.
        if period_end_date is None:
            raise ValueError("statement has neither a fiscal year nor a usable date")
        fiscal_year = float(period_end_date.year)

    currency_value, _ = _lookup(raw, STATEMENT_META_ALIASES["reported_currency"])
    currency = None if currency_value in (None, _SENTINEL) else str(currency_value).strip() or None

    return {
        "period": period,
        "fiscal_year": int(fiscal_year),
        "period_end_date": period_end_date,
        "filing_date": filing_date,
        "reported_currency": currency,
    }


def _build_statement(
    model: type[T],
    aliases: dict[str, list[str]],
    raw: Mapping[str, Any],
    ticker: str,
    period_hint: str,
    data_source: str,
) -> T:
    meta = _extract_meta(raw, fallback_period=period_hint)
    values, missing = extract_numeric_fields(raw, aliases)
    return model(
        ticker=ticker,
        data_source=data_source,
        missing_fields=missing,
        **meta,
        **values,
    )


def build_income_statement(
    raw: Mapping[str, Any], ticker: str, period_hint: str = "FY", data_source: str = "fmp"
) -> IncomeStatementRecord:
    return _build_statement(
        IncomeStatementRecord, INCOME_STATEMENT_ALIASES, raw, ticker, period_hint, data_source
    )


def build_balance_sheet(
    raw: Mapping[str, Any], ticker: str, period_hint: str = "FY", data_source: str = "fmp"
) -> BalanceSheetRecord:
    return _build_statement(
        BalanceSheetRecord, BALANCE_SHEET_ALIASES, raw, ticker, period_hint, data_source
    )


def build_cash_flow(
    raw: Mapping[str, Any], ticker: str, period_hint: str = "FY", data_source: str = "fmp"
) -> CashFlowRecord:
    return _build_statement(
        CashFlowRecord, CASH_FLOW_ALIASES, raw, ticker, period_hint, data_source
    )


def build_profile(
    raw: Mapping[str, Any], ticker: str, data_source: str = "fmp"
) -> CompanyProfile:
    """Normalize a company profile payload. Text fields stay strings."""
    numeric_fields = {"market_cap", "price", "shares_outstanding"}
    values: dict[str, Any] = {}
    missing: MissingMap = {}

    for canonical, alias_list in PROFILE_ALIASES.items():
        value, present = _lookup(raw, alias_list)
        if not present:
            values[canonical] = None
            missing[canonical] = MissingReason.NOT_REPORTED
            continue
        if value is None:
            values[canonical] = None
            missing[canonical] = MissingReason.NULL_IN_SOURCE
            continue
        if canonical in numeric_fields:
            number = coerce_number(value)
            if number is None:
                values[canonical] = None
                missing[canonical] = MissingReason.UNPARSEABLE
            else:
                values[canonical] = number
        else:
            text = str(value).strip()
            if text:
                values[canonical] = text
            else:
                values[canonical] = None
                missing[canonical] = MissingReason.NULL_IN_SOURCE

    return CompanyProfile(ticker=ticker, data_source=data_source, missing_fields=missing, **values)


def build_statements(
    payloads: Mapping[str, list[Mapping[str, Any]]],
    ticker: str,
    period_hint: str = "FY",
    data_source: str = "fmp",
) -> tuple[list[IncomeStatementRecord], list[BalanceSheetRecord], list[CashFlowRecord], list[str]]:
    """Normalize all three statement lists, collecting per-row errors.

    A row that cannot be normalized (e.g. no fiscal year and no date) is skipped
    with a recorded error rather than aborting the whole ingestion.
    """
    builders = {
        "income_statement": build_income_statement,
        "balance_sheet": build_balance_sheet,
        "cash_flow": build_cash_flow,
    }
    results: dict[str, list[Any]] = {k: [] for k in builders}
    errors: list[str] = []

    for kind, builder in builders.items():
        for index, raw in enumerate(payloads.get(kind, []) or []):
            try:
                results[kind].append(builder(raw, ticker, period_hint, data_source))
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
                errors.append(f"{kind}[{index}]: {exc}")
                logger.warning("Skipping unnormalizable %s row %d for %s: %s", kind, index, ticker, exc)

    return (
        results["income_statement"],
        results["balance_sheet"],
        results["cash_flow"],
        errors,
    )
