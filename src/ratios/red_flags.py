"""Rule-based red flag detection.

Deterministic, non-LLM, and auditable: every flag names the values that
triggered it. That matters architecturally, not just cosmetically — these flags
are *ground truth facts* handed to the RAG layer, where the LLM narrates
pre-computed findings and is never asked to decide whether a company is
distressed. A rule here is the difference between "the model says the company
looks risky" and "interest coverage was 0.8x in FY2024".

Every threshold lives in `THRESHOLDS` so the rules stay tunable and testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.ingestion.field_maps import CRITICAL_FIELDS
from src.ratios.base import PeriodAnalysis

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"


# Flag categories mirror the risk taxonomy in the project plan.
SOLVENCY = "Solvency / Bankruptcy Risk"
LIQUIDITY_STRESS = "Liquidity Stress"
PROFITABILITY_DETERIORATION = "Profitability Deterioration"
LEVERAGE_CONCERNS = "Leverage Concerns"
EARNINGS_QUALITY = "Earnings Quality / Manipulation Risk"
DATA_INTEGRITY = "Data Integrity"


THRESHOLDS: dict[str, float] = {
    "interest_coverage_min": 1.5,
    "current_ratio_min": 1.0,
    "quick_ratio_min": 0.5,
    "negative_operating_cf_periods": 2,
    "margin_compression_periods": 3,
    "debt_to_equity_yoy_increase": 0.20,
    "piotroski_weak_max": 3.0,
    "beneish_threshold": -1.78,
    # Receivables must outgrow revenue by this margin before it is worth
    # flagging; a couple of points of drift is ordinary collection timing.
    "receivables_growth_gap": 0.10,
    "sector_leverage_multiple": 1.5,
}


@dataclass
class RedFlagResult:
    """One triggered flag, with the values that triggered it."""

    flag_name: str
    severity: Severity
    category: str
    explanation: str
    source_values: dict[str, Any] = field(default_factory=dict)
    fiscal_year: int | None = None
    period: str | None = None

    def describe(self) -> str:
        return f"[{self.severity.value}] {self.flag_name}: {self.explanation}"


@dataclass
class Window:
    """A period plus its predecessors, oldest first, ending at the target period.

    Multi-period rules ("negative cash flow two years running") read backwards
    from `current` through this window.
    """

    history: list[PeriodAnalysis]
    sector_medians: dict[str, float] | None = None

    @property
    def current(self) -> PeriodAnalysis:
        return self.history[-1]

    def at(self, offset: int) -> PeriodAnalysis | None:
        """`offset` periods back from the current one; None if unavailable."""
        index = len(self.history) - 1 - offset
        return self.history[index] if index >= 0 else None

    def value(self, ratio_name: str, offset: int = 0) -> float | None:
        analysis = self.at(offset)
        return analysis.value(ratio_name) if analysis else None

    def series(self, ratio_name: str, count: int) -> list[float | None]:
        """The last `count` values, oldest to newest, padded with None."""
        window = self.history[-count:]
        return [a.value(ratio_name) for a in window]

    def field(self, statement: str, name: str, offset: int = 0) -> float | None:
        analysis = self.at(offset)
        return analysis.bundle.get(statement, name) if analysis else None


RuleFn = Callable[[Window], "RedFlagResult | list[RedFlagResult] | None"]


@dataclass(frozen=True)
class RuleSpec:
    """A registered rule and the company types it is valid for."""

    fn: RuleFn
    applies_to_financials: bool = True

    @property
    def name(self) -> str:
        return self.fn.__name__


RULE_REGISTRY: list[RuleSpec] = []


def rule(fn: RuleFn | None = None, *, applies_to_financials: bool = True):
    """Register a detection rule.

    `applies_to_financials=False` marks a rule whose premise does not hold for
    banks and insurers — for the same reason the Altman models exclude them.
    A bank's interest expense is a cost of funding rather than a debt burden, its
    balance sheet is not classified into current and non-current, and lending
    activity routinely drives operating cash flow negative. Running those rules
    over a bank produces confident false positives, which is precisely the kind
    of unfounded claim this layer exists to keep out of the RAG context.
    """

    def decorator(target: RuleFn) -> RuleFn:
        RULE_REGISTRY.append(RuleSpec(fn=target, applies_to_financials=applies_to_financials))
        return target

    return decorator(fn) if fn is not None else decorator


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Solvency / bankruptcy risk
# ---------------------------------------------------------------------------


@rule
def altman_distress_zone(window: Window) -> RedFlagResult | None:
    """Altman Z (or its book-value fallback) sitting in the distress band."""
    current = window.current
    result = current.result("altman_z_score")
    variant = "Altman Z-Score"
    if result is None or not result.is_calculable:
        result = current.result("altman_z_double_prime_score")
        variant = "Altman Z''-Score (book value)"
    if result is None or not result.is_calculable:
        return None

    zone = result.details.get("zone")
    if zone == "Distress":
        severity, name = Severity.HIGH, "Altman Z-Score in Distress Zone"
        detail = "indicates elevated bankruptcy risk"
    elif zone == "Grey":
        severity, name = Severity.LOW, "Altman Z-Score in Grey Zone"
        detail = "sits between the safe and distress bands"
    else:
        return None

    thresholds = result.details.get("thresholds", {})
    return RedFlagResult(
        flag_name=name,
        severity=severity,
        category=SOLVENCY,
        explanation=(
            f"{variant} of {result.value:.2f} for {current.period_label} {detail} "
            f"(distress below {thresholds.get('distress_below')}, "
            f"safe above {thresholds.get('safe_above')})."
        ),
        source_values={"score": result.value, "zone": zone, "variant": variant, **result.inputs},
    )


@rule
def negative_shareholders_equity(window: Window) -> RedFlagResult | None:
    """Liabilities exceed assets — the company is technically insolvent."""
    equity = window.field("balance", "total_equity")
    if equity is None or equity >= 0:
        return None
    assets = window.field("balance", "total_assets")
    liabilities = window.field("balance", "total_liabilities")
    return RedFlagResult(
        flag_name="Negative Shareholders' Equity",
        severity=Severity.HIGH,
        category=SOLVENCY,
        explanation=(
            f"Shareholders' equity is {equity:,.0f} in {window.current.period_label}: total "
            f"liabilities exceed total assets, leaving no book value for shareholders. Several "
            f"leverage and return ratios are undefined as a result."
        ),
        source_values={
            "total_equity": equity,
            "total_assets": assets,
            "total_liabilities": liabilities,
        },
    )


@rule(applies_to_financials=False)
def weak_interest_coverage(window: Window) -> RedFlagResult | None:
    """Operating profit is thin relative to the interest bill."""
    coverage = window.value("interest_coverage")
    if coverage is None or coverage >= THRESHOLDS["interest_coverage_min"]:
        return None

    result = window.current.result("interest_coverage")
    inputs = result.inputs if result else {}
    if coverage < 1.0:
        explanation = (
            f"Interest coverage is {coverage:.2f}x in {window.current.period_label}: operating "
            f"profit does not cover the interest bill, so debt service depends on cash reserves, "
            f"asset sales or refinancing."
        )
    else:
        explanation = (
            f"Interest coverage is {coverage:.2f}x in {window.current.period_label}, below the "
            f"{THRESHOLDS['interest_coverage_min']}x threshold: little headroom before operating "
            f"profit fails to cover interest."
        )
    return RedFlagResult(
        flag_name="Interest Coverage Danger",
        severity=Severity.HIGH,
        category=SOLVENCY,
        explanation=explanation,
        source_values={"interest_coverage": coverage, **inputs},
    )


# ---------------------------------------------------------------------------
# Liquidity stress
# ---------------------------------------------------------------------------


@rule(applies_to_financials=False)
def current_ratio_below_one(window: Window) -> RedFlagResult | None:
    current_ratio = window.value("current_ratio")
    if current_ratio is None or current_ratio >= THRESHOLDS["current_ratio_min"]:
        return None
    result = window.current.result("current_ratio")
    return RedFlagResult(
        flag_name="Current Ratio Below 1.0",
        severity=Severity.MEDIUM,
        category=LIQUIDITY_STRESS,
        explanation=(
            f"Current ratio is {current_ratio:.2f} in {window.current.period_label}: current "
            f"liabilities exceed current assets, so obligations due within a year are not covered "
            f"by assets convertible within a year."
        ),
        source_values={"current_ratio": current_ratio, **(result.inputs if result else {})},
    )


@rule(applies_to_financials=False)
def quick_ratio_below_threshold(window: Window) -> RedFlagResult | None:
    quick = window.value("quick_ratio")
    if quick is None or quick >= THRESHOLDS["quick_ratio_min"]:
        return None
    result = window.current.result("quick_ratio")
    return RedFlagResult(
        flag_name="Quick Ratio Below 0.5",
        severity=Severity.MEDIUM,
        category=LIQUIDITY_STRESS,
        explanation=(
            f"Quick ratio is {quick:.2f} in {window.current.period_label}: excluding inventory, "
            f"liquid current assets cover less than half of current liabilities."
        ),
        source_values={"quick_ratio": quick, **(result.inputs if result else {})},
    )


@rule(applies_to_financials=False)
def sustained_negative_operating_cash_flow(window: Window) -> RedFlagResult | None:
    """Operations have burned cash for consecutive periods."""
    required = int(THRESHOLDS["negative_operating_cf_periods"])
    values: list[tuple[str, float]] = []
    for offset in range(required):
        analysis = window.at(offset)
        if analysis is None:
            return None
        cash_flow = analysis.bundle.get("cash_flow", "operating_cash_flow")
        if cash_flow is None or cash_flow >= 0:
            return None
        values.append((analysis.period_label, cash_flow))

    periods = ", ".join(f"{label}: {value:,.0f}" for label, value in reversed(values))
    return RedFlagResult(
        flag_name="Negative Operating Cash Flow (Sustained)",
        severity=Severity.HIGH,
        category=LIQUIDITY_STRESS,
        explanation=(
            f"Operating cash flow has been negative for {required} consecutive periods "
            f"({periods}): the core business is consuming cash rather than generating it."
        ),
        source_values={"operating_cash_flow_by_period": dict(values)},
    )


# ---------------------------------------------------------------------------
# Profitability deterioration
# ---------------------------------------------------------------------------


@rule
def margin_compression(window: Window) -> RedFlagResult | None:
    """Operating margin fell in each of the last N period-over-period steps."""
    required = int(THRESHOLDS["margin_compression_periods"])
    margins = window.series("operating_margin", required + 1)
    if len(margins) < required + 1 or any(m is None for m in margins):
        return None

    declines = [margins[i] < margins[i - 1] for i in range(1, len(margins))]
    if not all(declines):
        return None

    labels = [a.period_label for a in window.history[-(required + 1) :]]
    trail = ", ".join(f"{label}: {_pct(m)}" for label, m in zip(labels, margins))
    return RedFlagResult(
        flag_name="Margin Compression",
        severity=Severity.MEDIUM,
        category=PROFITABILITY_DETERIORATION,
        explanation=(
            f"Operating margin declined in {required} consecutive periods ({trail}), a total "
            f"contraction of {_pct(margins[0] - margins[-1])} of revenue."
        ),
        source_values={"operating_margin_by_period": dict(zip(labels, margins))},
    )


@rule
def net_losses(window: Window) -> RedFlagResult | None:
    """A loss this period, escalated when losses are consecutive."""
    net_income = window.field("income", "net_income")
    if net_income is None or net_income >= 0:
        return None

    consecutive, values = 0, {}
    for offset in range(len(window.history)):
        value = window.field("income", "net_income", offset)
        if value is None or value >= 0:
            break
        analysis = window.at(offset)
        values[analysis.period_label] = value
        consecutive += 1

    if consecutive >= 2:
        return RedFlagResult(
            flag_name="Sustained Net Losses",
            severity=Severity.HIGH,
            category=PROFITABILITY_DETERIORATION,
            explanation=(
                f"The company reported a net loss in {consecutive} consecutive periods "
                f"(most recently {net_income:,.0f} in {window.current.period_label}), eroding "
                f"retained earnings and book value."
            ),
            source_values={"net_income_by_period": values},
        )
    return RedFlagResult(
        flag_name="Net Loss",
        severity=Severity.MEDIUM,
        category=PROFITABILITY_DETERIORATION,
        explanation=(
            f"The company reported a net loss of {net_income:,.0f} in "
            f"{window.current.period_label}."
        ),
        source_values={"net_income": net_income},
    )


@rule(applies_to_financials=False)
def negative_fcf_despite_profit(window: Window) -> RedFlagResult | None:
    """Profitable on paper, cash-negative in reality — an accrual quality warning."""
    net_income = window.field("income", "net_income")
    fcf = window.value("free_cash_flow")
    if net_income is None or fcf is None or net_income <= 0 or fcf >= 0:
        return None
    operating = window.field("cash_flow", "operating_cash_flow")
    return RedFlagResult(
        flag_name="Negative Free Cash Flow Despite Positive Net Income",
        severity=Severity.MEDIUM,
        category=PROFITABILITY_DETERIORATION,
        explanation=(
            f"{window.current.period_label} reported net income of {net_income:,.0f} but free "
            f"cash flow of {fcf:,.0f}. Reported profit is not converting into cash, which can "
            f"reflect heavy investment or aggressive revenue recognition."
        ),
        source_values={
            "net_income": net_income,
            "free_cash_flow": fcf,
            "operating_cash_flow": operating,
        },
    )


# ---------------------------------------------------------------------------
# Leverage concerns
# ---------------------------------------------------------------------------


@rule(applies_to_financials=False)
def rising_debt_to_equity(window: Window) -> RedFlagResult | None:
    """Debt-to-equity jumped materially year over year."""
    current = window.value("debt_to_equity")
    prior = window.value("debt_to_equity", offset=1)
    if current is None or prior is None or prior <= 0:
        return None

    change = (current - prior) / prior
    if change <= THRESHOLDS["debt_to_equity_yoy_increase"]:
        return None

    prior_label = window.at(1).period_label
    return RedFlagResult(
        flag_name="Debt-to-Equity Rising Sharply",
        severity=Severity.MEDIUM,
        category=LEVERAGE_CONCERNS,
        explanation=(
            f"Debt-to-equity rose {_pct(change)} year over year, from {prior:.2f} in "
            f"{prior_label} to {current:.2f} in {window.current.period_label}."
        ),
        source_values={
            "debt_to_equity": current,
            "prior_debt_to_equity": prior,
            "change_pct": change,
        },
    )


@rule(applies_to_financials=False)
def leverage_above_sector(window: Window) -> RedFlagResult | None:
    """Debt-to-equity well above the sector median.

    Skipped entirely when no median was supplied — comparing against an invented
    benchmark would be exactly the kind of fabricated fact this layer exists to
    prevent.
    """
    medians = window.sector_medians or {}
    median = medians.get("debt_to_equity")
    current = window.value("debt_to_equity")
    if median is None or current is None or median <= 0:
        return None

    multiple = current / median
    if multiple <= THRESHOLDS["sector_leverage_multiple"]:
        return None

    return RedFlagResult(
        flag_name="Leverage Well Above Sector Median",
        severity=Severity.MEDIUM,
        category=LEVERAGE_CONCERNS,
        explanation=(
            f"Debt-to-equity of {current:.2f} in {window.current.period_label} is {multiple:.1f}x "
            f"the sector median of {median:.2f}."
        ),
        source_values={
            "debt_to_equity": current,
            "sector_median": median,
            "multiple_of_median": multiple,
        },
    )


# ---------------------------------------------------------------------------
# Earnings quality / manipulation risk
# ---------------------------------------------------------------------------


@rule
def beneish_manipulation_risk(window: Window) -> RedFlagResult | None:
    result = window.current.result("beneish_m_score")
    if result is None or not result.is_calculable:
        return None
    if result.value <= THRESHOLDS["beneish_threshold"]:
        return None

    drivers = sorted(result.inputs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    driver_text = ", ".join(f"{name.upper()}={value:.2f}" for name, value in drivers)
    return RedFlagResult(
        flag_name="Beneish M-Score Above Manipulation Threshold",
        severity=Severity.HIGH,
        category=EARNINGS_QUALITY,
        explanation=(
            f"Beneish M-Score of {result.value:.2f} in {window.current.period_label} exceeds the "
            f"{THRESHOLDS['beneish_threshold']} threshold, indicating a statistically elevated "
            f"likelihood of earnings manipulation. Largest contributing indices: {driver_text}. "
            f"This is a screening signal, not evidence of manipulation - rapid legitimate growth "
            f"also lifts several indices."
        ),
        source_values={"m_score": result.value, **result.inputs},
    )


@rule
def weak_piotroski_score(window: Window) -> RedFlagResult | None:
    result = window.current.result("piotroski_f_score")
    if result is None or not result.is_calculable:
        return None
    if result.value > THRESHOLDS["piotroski_weak_max"]:
        return None

    signals = result.details.get("signals", {})
    failed = [name.replace("_", " ") for name, passed in signals.items() if not passed]
    return RedFlagResult(
        flag_name="Weak Piotroski F-Score",
        severity=Severity.MEDIUM,
        category=EARNINGS_QUALITY,
        explanation=(
            f"Piotroski F-Score of {result.value:.0f} out of 9 in {window.current.period_label} "
            f"indicates weak fundamentals. Failed signals: {', '.join(failed) or 'none'}."
        ),
        source_values={"f_score": result.value, "failed_signals": failed},
    )


@rule(applies_to_financials=False)
def receivables_outpacing_revenue(window: Window) -> RedFlagResult | None:
    """Receivables growing faster than sales — revenue booked but not collected."""
    revenue = window.field("income", "revenue")
    prior_revenue = window.field("income", "revenue", offset=1)
    receivables = window.field("balance", "net_receivables")
    prior_receivables = window.field("balance", "net_receivables", offset=1)

    if None in (revenue, prior_revenue, receivables, prior_receivables):
        return None
    if prior_revenue <= 0 or prior_receivables <= 0:
        return None

    revenue_growth = (revenue - prior_revenue) / prior_revenue
    receivables_growth = (receivables - prior_receivables) / prior_receivables
    gap = receivables_growth - revenue_growth
    if gap <= THRESHOLDS["receivables_growth_gap"]:
        return None

    return RedFlagResult(
        flag_name="Receivables Growing Faster Than Revenue",
        severity=Severity.MEDIUM,
        category=EARNINGS_QUALITY,
        explanation=(
            f"Receivables grew {_pct(receivables_growth)} in {window.current.period_label} while "
            f"revenue grew {_pct(revenue_growth)}, a gap of {_pct(gap)}. Sales are being booked "
            f"faster than they are collected, which can signal loosened credit terms or "
            f"premature revenue recognition."
        ),
        source_values={
            "revenue_growth": revenue_growth,
            "receivables_growth": receivables_growth,
            "gap": gap,
            "revenue": revenue,
            "net_receivables": receivables,
        },
    )


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


@rule
def missing_critical_data(window: Window) -> RedFlagResult | None:
    """Absent critical fields are surfaced as a visible flag, never hidden.

    A ratio table with quiet gaps invites the reader to assume the gaps are
    zeros. Naming the missing fields makes the limits of the analysis explicit.
    """
    bundle = window.current.bundle
    statements = {
        "income_statement": ("income", bundle.income),
        "balance_sheet": ("balance", bundle.balance),
        "cash_flow": ("cash_flow", bundle.cash_flow),
    }

    missing: dict[str, list[str]] = {}
    for kind, (attr, row) in statements.items():
        if row is None:
            missing[kind] = ["entire statement not on file"]
            continue
        absent = sorted(
            name for name in CRITICAL_FIELDS.get(kind, set()) if getattr(row, name, None) is None
        )
        if absent:
            missing[kind] = absent

    if not missing:
        return None

    detail = "; ".join(f"{kind}: {', '.join(fields)}" for kind, fields in sorted(missing.items()))
    return RedFlagResult(
        flag_name="Missing Critical Financial Data",
        severity=Severity.INFO,
        category=DATA_INTEGRITY,
        explanation=(
            f"Critical fields are not available for {window.current.period_label} ({detail}). "
            f"Ratios depending on them are reported as not calculable rather than estimated."
        ),
        source_values={"missing_fields": missing},
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def detect_for_period(
    history: list[PeriodAnalysis],
    sector_medians: dict[str, float] | None = None,
) -> list[RedFlagResult]:
    """Run every rule against the last period in `history` (oldest first)."""
    if not history:
        return []

    window = Window(history=history, sector_medians=sector_medians)
    is_financial = window.current.bundle.is_financial_sector
    flags: list[RedFlagResult] = []
    skipped: list[str] = []

    for spec in RULE_REGISTRY:
        if is_financial and not spec.applies_to_financials:
            skipped.append(spec.name)
            continue
        try:
            outcome = spec.fn(window)
        except Exception as exc:  # noqa: BLE001 - one bad rule must not hide the rest
            logger.warning("Red flag rule %s failed: %s", spec.name, exc)
            continue
        if outcome is None:
            continue
        for flag in outcome if isinstance(outcome, list) else [outcome]:
            flag.fiscal_year = window.current.fiscal_year
            flag.period = window.current.period
            flags.append(flag)

    if skipped:
        flags.append(
            RedFlagResult(
                flag_name="Financial-Sector Rules Not Applied",
                severity=Severity.INFO,
                category=DATA_INTEGRITY,
                explanation=(
                    f"{window.current.bundle.ticker} is classified in the financial sector, where "
                    f"{len(skipped)} rules do not hold: interest expense is a cost of funding "
                    f"rather than a debt burden, the balance sheet is not split into current and "
                    f"non-current, and lending activity routinely drives operating cash flow "
                    f"negative. These rules were skipped rather than reported as risks. Bank-"
                    f"specific measures (capital adequacy, net interest margin, loan-loss "
                    f"coverage) are outside this engine's scope."
                ),
                source_values={"skipped_rules": skipped},
                fiscal_year=window.current.fiscal_year,
                period=window.current.period,
            )
        )

    severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
    flags.sort(key=lambda f: (severity_order[f.severity], f.flag_name))
    return flags


def detect_all(
    history: list[PeriodAnalysis],
    sector_medians: dict[str, float] | None = None,
) -> dict[int, list[RedFlagResult]]:
    """Flags for every period, each evaluated against only the data preceding it.

    Each period sees only its own history, never later years, so a flag raised
    for FY2021 is exactly what an analyst could have known at the time.
    """
    ordered = sorted(history, key=lambda a: a.fiscal_year)
    return {
        analysis.fiscal_year: detect_for_period(ordered[: index + 1], sector_medians)
        for index, analysis in enumerate(ordered)
    }
