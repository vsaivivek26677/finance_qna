"""Shared machinery for the ratio engine.

The central design rule of Layer 2: a ratio is either computed from real
reported numbers, or it is explicitly *not calculable* with a reason that names
the specific missing or unusable input. There is no third option — no silent
estimate, no zero substituted for an absent field, no ratio quietly omitted.

Ratio functions express that by raising `NotCalculable` with a human-readable
reason; the engine turns it into a `RatioResult(is_calculable=False, reason=...)`
that persists to the database and flows through to the RAG layer verbatim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal

from src.db.models import BalanceSheet, CashFlowStatement, IncomeStatement, MarketData

StatementName = Literal["income", "balance", "cash_flow", "market"]

STATEMENT_LABELS: dict[str, str] = {
    "income": "income statement",
    "balance": "balance sheet",
    "cash_flow": "cash flow statement",
    "market": "market data",
}

# Ratio categories, used for grouping in the dashboard and the RAG chunker.
LIQUIDITY = "liquidity"
PROFITABILITY = "profitability"
LEVERAGE = "leverage"
EFFICIENCY = "efficiency"
VALUATION = "valuation"
CASH_FLOW = "cash_flow"
DISTRESS = "distress"


class NotCalculable(Exception):
    """A ratio cannot be computed, with a reason naming the specific cause."""


@dataclass
class Computed:
    """Return type for ratios that carry more than a bare number."""

    value: float
    method: str | None = None
    inputs: dict[str, float | None] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RatioResult:
    """One ratio for one company-period — calculable or explicitly not."""

    name: str
    category: str
    value: float | None = None
    is_calculable: bool = True
    reason: str | None = None
    method: str | None = None
    inputs: dict[str, float | None] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """One-line rendering — the form Layer 3 chunks into the vector store."""
        if not self.is_calculable:
            return f"{self.name}: NOT AVAILABLE ({self.reason})"
        zone = self.details.get("zone")
        suffix = f" ({zone} Zone)" if zone else ""
        return f"{self.name}: {self.value:,.4f}{suffix}"


@dataclass
class PeriodBundle:
    """Every statement for one company-period, plus the preceding period.

    Ratio functions read only from here, which is what keeps them pure and
    directly unit-testable without a database.
    """

    ticker: str
    period: str
    fiscal_year: int
    period_end_date: date | None = None
    income: IncomeStatement | None = None
    balance: BalanceSheet | None = None
    cash_flow: CashFlowStatement | None = None
    market: MarketData | None = None
    is_financial_sector: bool = False
    sector: str | None = None
    prior: "PeriodBundle | None" = None

    @property
    def period_label(self) -> str:
        return (
            f"FY{self.fiscal_year}"
            if self.period == "FY"
            else f"{self.period} {self.fiscal_year}"
        )

    # --- field access ------------------------------------------------------

    def _row(self, statement: StatementName) -> Any:
        return getattr(self, statement, None)

    def get(self, statement: StatementName, name: str) -> float | None:
        """Value if present, else None. Use when a field is genuinely optional."""
        row = self._row(statement)
        if row is None:
            return None
        value = getattr(row, name, None)
        return float(value) if value is not None else None

    def require(self, statement: StatementName, name: str) -> float:
        """Value, or `NotCalculable` naming the field and why it is unavailable.

        When the ingestion layer recorded *why* a field is missing, that reason
        is quoted here — so "revenue not reported" and "revenue null in source"
        stay distinguishable all the way to the dashboard.
        """
        label = STATEMENT_LABELS[statement]
        row = self._row(statement)
        if row is None:
            raise NotCalculable(f"no {label} on file for {self.period_label}")

        value = getattr(row, name, None)
        if value is None:
            recorded = (getattr(row, "is_missing_json", None) or {}).get(name)
            reason = recorded.replace("_", " ") if recorded else "not reported"
            raise NotCalculable(f"{name} {reason} in {self.period_label} {label}")
        return float(value)

    def require_prior(self, statement: StatementName, name: str) -> float:
        """Same, but from the preceding period — for YoY and change-based scores."""
        if self.prior is None:
            raise NotCalculable(
                f"no prior period on file; {self.period_label} is the earliest year available"
            )
        return self.prior.require(statement, name)

    def average_or_closing(self, statement: StatementName, name: str) -> tuple[float, str]:
        """Average balance across the year when the opening figure is known.

        Flow-over-stock ratios (ROE, asset turnover) are more faithful against an
        average balance. Which basis was used is reported rather than assumed, so
        two periods computed on different bases are never silently compared.
        """
        closing = self.require(statement, name)
        if self.prior is not None:
            opening = self.prior.get(statement, name)
            if opening is not None:
                return (closing + opening) / 2.0, "average of opening and closing balance"
        return closing, "closing balance only (no prior period on file)"

    # --- derived helpers ---------------------------------------------------

    def total_debt(self) -> float:
        """Reported total debt, or short-term + long-term when it is absent."""
        reported = self.get("balance", "total_debt")
        if reported is not None:
            return reported
        short_term = self.get("balance", "short_term_debt")
        long_term = self.get("balance", "long_term_debt")
        if short_term is None and long_term is None:
            raise NotCalculable(
                f"total_debt not reported in {self.period_label} balance sheet, and neither "
                "short_term_debt nor long_term_debt is available to derive it"
            )
        return (short_term or 0.0) + (long_term or 0.0)

    def free_cash_flow(self) -> float:
        """Reported FCF, or operating cash flow less capital expenditure.

        Capex sign convention varies between sources, so its magnitude is used.
        """
        reported = self.get("cash_flow", "free_cash_flow")
        if reported is not None:
            return reported
        operating = self.require("cash_flow", "operating_cash_flow")
        capex = self.require("cash_flow", "capital_expenditure")
        return operating - abs(capex)

    def market_value_of_equity(self) -> float:
        """Market capitalisation at the period end, from the paired price point."""
        if self.market is None:
            raise NotCalculable(
                f"no market price data within range of the {self.period_label} period end"
            )
        cap = self.get("market", "market_cap")
        if cap is not None:
            return cap
        price = self.get("market", "close_price")
        shares = self.get("market", "shares_outstanding")
        if price is None or shares is None:
            raise NotCalculable(
                f"market capitalisation unavailable for {self.period_label} "
                "(needs market_cap, or close_price and shares_outstanding)"
            )
        return price * shares


@dataclass
class PeriodAnalysis:
    """A period's statements together with every ratio computed from them.

    This is the unit the red-flag detector and the RAG chunker both consume.
    """

    bundle: PeriodBundle
    ratios: dict[str, RatioResult] = field(default_factory=dict)

    @property
    def fiscal_year(self) -> int:
        return self.bundle.fiscal_year

    @property
    def period(self) -> str:
        return self.bundle.period

    @property
    def period_label(self) -> str:
        return self.bundle.period_label

    def value(self, name: str) -> float | None:
        """The ratio's value, or None when it was not calculable."""
        result = self.ratios.get(name)
        return result.value if result is not None and result.is_calculable else None

    def result(self, name: str) -> RatioResult | None:
        return self.ratios.get(name)

    def calculable(self) -> dict[str, RatioResult]:
        return {name: r for name, r in self.ratios.items() if r.is_calculable}

    def not_calculable(self) -> dict[str, RatioResult]:
        return {name: r for name, r in self.ratios.items() if not r.is_calculable}


def humanise(name: str) -> str:
    """`return_on_equity` -> `Return On Equity`, with finance acronyms preserved."""
    words = name.replace("_", " ").split()
    acronyms = {"eps", "ebitda", "ebit", "roe", "roa", "roic", "fcf", "ocf", "dso", "dio", "ev"}
    return " ".join(w.upper() if w.lower() in acronyms else w.capitalize() for w in words)


# ---------------------------------------------------------------------------
# Arithmetic guards
# ---------------------------------------------------------------------------


def divide(numerator: float, denominator: float, denominator_label: str) -> float:
    """Division that refuses to produce a number from a zero denominator."""
    if denominator == 0:
        raise NotCalculable(f"{denominator_label} is zero, so the ratio is undefined")
    return numerator / denominator


def require_positive(value: float, label: str) -> float:
    """Guard for ratios that are meaningless over a negative or zero base.

    P/E on negative earnings, or D/E on negative equity, produce numbers that
    look valid and rank nonsensically — better to refuse and say why.
    """
    if value <= 0:
        raise NotCalculable(
            f"{label} is {value:,.0f}; the ratio is not meaningful when it is not positive"
        )
    return value


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RatioFn = Callable[[PeriodBundle], "float | Computed"]


@dataclass(frozen=True)
class RatioSpec:
    name: str
    category: str
    fn: RatioFn
    description: str = ""


RATIO_REGISTRY: list[RatioSpec] = []


def ratio(name: str, category: str, description: str = "") -> Callable[[RatioFn], RatioFn]:
    """Register a ratio function under a stable snake_case name."""

    def decorator(fn: RatioFn) -> RatioFn:
        if any(spec.name == name for spec in RATIO_REGISTRY):
            raise ValueError(f"duplicate ratio name: {name}")
        RATIO_REGISTRY.append(
            RatioSpec(name=name, category=category, fn=fn, description=description or (fn.__doc__ or "").strip())
        )
        return fn

    return decorator


def evaluate(spec: RatioSpec, bundle: PeriodBundle) -> RatioResult:
    """Run one ratio, converting any failure into an explained non-result."""
    try:
        outcome = spec.fn(bundle)
    except NotCalculable as exc:
        return RatioResult(spec.name, spec.category, is_calculable=False, reason=str(exc))
    except ZeroDivisionError:
        return RatioResult(
            spec.name, spec.category, is_calculable=False, reason="division by zero"
        )

    if isinstance(outcome, Computed):
        value, method, inputs, details = (
            outcome.value,
            outcome.method,
            outcome.inputs,
            outcome.details,
        )
    else:
        value, method, inputs, details = float(outcome), None, {}, {}

    if not math.isfinite(value):
        return RatioResult(
            spec.name,
            spec.category,
            is_calculable=False,
            reason="computation produced a non-finite value",
        )

    return RatioResult(
        name=spec.name,
        category=spec.category,
        value=value,
        is_calculable=True,
        method=method,
        inputs=inputs,
        details=details,
    )
