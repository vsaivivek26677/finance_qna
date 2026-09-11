"""Composite distress and earnings-quality scores.

Altman Z-Score, Altman Z''-Score, Piotroski F-Score and Beneish M-Score. These
are the scores the RAG layer narrates, so three rules are enforced strictly:

* **Financial-sector companies are excluded from the Altman models.** They were
  calibrated on manufacturers; a bank's balance sheet structure produces a
  meaningless number that still looks authoritative. The engine says
  "Not Applicable - Financial Sector" instead.
* **Missing an input means no score**, with the specific input named. A Z-Score
  computed with retained earnings silently treated as zero is worse than no
  Z-Score at all.
* **Composite scores report their components.** Every score persists the
  individual terms that produced it, so the LLM can explain *why* a company sits
  where it does from pre-computed facts rather than guessing.
"""

from __future__ import annotations

from typing import Any

from src.ratios.base import (
    DISTRESS,
    Computed,
    NotCalculable,
    PeriodBundle,
    divide,
    ratio,
)

# --- Altman Z (public manufacturers) ---------------------------------------
Z_COEFFICIENTS = {"x1": 1.2, "x2": 1.4, "x3": 3.3, "x4": 0.6, "x5": 1.0}
Z_SAFE_THRESHOLD = 2.99
Z_DISTRESS_THRESHOLD = 1.81

# --- Altman Z'' (non-manufacturers / no market price) ----------------------
Z_DOUBLE_PRIME_COEFFICIENTS = {"x1": 6.56, "x2": 3.26, "x3": 6.72, "x4": 1.05}
Z_DP_SAFE_THRESHOLD = 2.60
Z_DP_DISTRESS_THRESHOLD = 1.10

# --- Beneish M -------------------------------------------------------------
BENEISH_INTERCEPT = -4.84
BENEISH_COEFFICIENTS = {
    "dsri": 0.920,
    "gmi": 0.528,
    "aqi": 0.404,
    "sgi": 0.892,
    "depi": 0.115,
    "sgai": -0.172,
    "tata": 4.679,
    "lvgi": -0.327,
}
BENEISH_MANIPULATION_THRESHOLD = -1.78

PIOTROSKI_MAX_SCORE = 9

NOT_APPLICABLE_FINANCIAL = (
    "Not Applicable - Financial Sector: the Altman model is calibrated on "
    "non-financial firms and produces a misleading score for banks and insurers"
)


def _reject_financial_sector(ctx: PeriodBundle) -> None:
    if ctx.is_financial_sector:
        raise NotCalculable(NOT_APPLICABLE_FINANCIAL)


def _zone(value: float, safe: float, distress: float) -> str:
    if value > safe:
        return "Safe"
    if value < distress:
        return "Distress"
    return "Grey"


def _altman_common_terms(ctx: PeriodBundle) -> dict[str, float]:
    """X1, X2, X3, X5 — shared by both Altman variants."""
    total_assets = ctx.require("balance", "total_assets")
    if total_assets == 0:
        raise NotCalculable(f"total assets are zero in {ctx.period_label}")

    current_assets = ctx.require("balance", "total_current_assets")
    current_liabilities = ctx.require("balance", "total_current_liabilities")
    retained_earnings = ctx.require("balance", "retained_earnings")
    operating_income = ctx.require("income", "operating_income")
    revenue = ctx.require("income", "revenue")

    return {
        "total_assets": total_assets,
        "x1_working_capital_to_assets": (current_assets - current_liabilities) / total_assets,
        "x2_retained_earnings_to_assets": retained_earnings / total_assets,
        "x3_ebit_to_assets": operating_income / total_assets,
        "x5_revenue_to_assets": revenue / total_assets,
    }


@ratio("altman_z_score", DISTRESS, "Altman Z-Score (market value of equity)")
def altman_z_score(ctx: PeriodBundle) -> Computed:
    """Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5.

    X4 uses the *market* value of equity. Without a paired price point this
    variant is unavailable and `altman_z_double_prime_score` is the fallback.
    """
    _reject_financial_sector(ctx)
    terms = _altman_common_terms(ctx)

    total_liabilities = ctx.require("balance", "total_liabilities")
    market_equity = ctx.market_value_of_equity()
    x4 = divide(market_equity, total_liabilities, "total liabilities")

    score = (
        Z_COEFFICIENTS["x1"] * terms["x1_working_capital_to_assets"]
        + Z_COEFFICIENTS["x2"] * terms["x2_retained_earnings_to_assets"]
        + Z_COEFFICIENTS["x3"] * terms["x3_ebit_to_assets"]
        + Z_COEFFICIENTS["x4"] * x4
        + Z_COEFFICIENTS["x5"] * terms["x5_revenue_to_assets"]
    )
    zone = _zone(score, Z_SAFE_THRESHOLD, Z_DISTRESS_THRESHOLD)

    return Computed(
        value=score,
        method="market-value variant (original Altman 1968 model)",
        inputs={
            "x1_working_capital_to_assets": terms["x1_working_capital_to_assets"],
            "x2_retained_earnings_to_assets": terms["x2_retained_earnings_to_assets"],
            "x3_ebit_to_assets": terms["x3_ebit_to_assets"],
            "x4_market_equity_to_liabilities": x4,
            "x5_revenue_to_assets": terms["x5_revenue_to_assets"],
        },
        details={
            "zone": zone,
            "thresholds": {"safe_above": Z_SAFE_THRESHOLD, "distress_below": Z_DISTRESS_THRESHOLD},
            "market_value_of_equity": market_equity,
        },
    )


@ratio("altman_z_double_prime_score", DISTRESS, "Altman Z''-Score (book value of equity)")
def altman_z_double_prime_score(ctx: PeriodBundle) -> Computed:
    """Z'' = 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4, with X4 on book equity.

    Drops the asset-turnover term and reweights the rest, which makes it the
    appropriate variant for non-manufacturers and for any company whose market
    value is unknown. Its zone boundaries differ from the original model's.
    """
    _reject_financial_sector(ctx)
    terms = _altman_common_terms(ctx)

    total_liabilities = ctx.require("balance", "total_liabilities")
    book_equity = ctx.require("balance", "total_equity")
    x4 = divide(book_equity, total_liabilities, "total liabilities")

    score = (
        Z_DOUBLE_PRIME_COEFFICIENTS["x1"] * terms["x1_working_capital_to_assets"]
        + Z_DOUBLE_PRIME_COEFFICIENTS["x2"] * terms["x2_retained_earnings_to_assets"]
        + Z_DOUBLE_PRIME_COEFFICIENTS["x3"] * terms["x3_ebit_to_assets"]
        + Z_DOUBLE_PRIME_COEFFICIENTS["x4"] * x4
    )
    zone = _zone(score, Z_DP_SAFE_THRESHOLD, Z_DP_DISTRESS_THRESHOLD)

    return Computed(
        value=score,
        method="book-value variant (Altman Z''), used when market value is unavailable",
        inputs={
            "x1_working_capital_to_assets": terms["x1_working_capital_to_assets"],
            "x2_retained_earnings_to_assets": terms["x2_retained_earnings_to_assets"],
            "x3_ebit_to_assets": terms["x3_ebit_to_assets"],
            "x4_book_equity_to_liabilities": x4,
        },
        details={
            "zone": zone,
            "thresholds": {
                "safe_above": Z_DP_SAFE_THRESHOLD,
                "distress_below": Z_DP_DISTRESS_THRESHOLD,
            },
        },
    )


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------


def _closing_ratio(ctx: PeriodBundle, numerator: tuple[str, str], denominator: tuple[str, str]) -> float:
    """A ratio on closing balances — Piotroski compares like with like."""
    num = ctx.require(*numerator)
    den = ctx.require(*denominator)
    return divide(num, den, f"{denominator[1]} in {ctx.period_label}")


@ratio("piotroski_f_score", DISTRESS, "Piotroski F-Score (0-9 fundamental strength)")
def piotroski_f_score(ctx: PeriodBundle) -> Computed:
    """Nine binary tests of profitability, leverage and operating efficiency.

    Six of the nine are year-over-year comparisons, so the score is unavailable
    for the earliest year on file. A partial score is not returned: five out of
    a possible six tests reads as 5/9 and understates the company.
    """
    if ctx.prior is None:
        raise NotCalculable(
            f"no prior period on file; the F-Score needs year-over-year comparisons and "
            f"{ctx.period_label} is the earliest year available"
        )

    prior = ctx.prior
    signals: dict[str, bool] = {}

    # --- Profitability (4 signals) ---
    net_income = ctx.require("income", "net_income")
    operating_cash_flow = ctx.require("cash_flow", "operating_cash_flow")
    signals["positive_net_income"] = net_income > 0
    signals["positive_operating_cash_flow"] = operating_cash_flow > 0

    roa = _closing_ratio(ctx, ("income", "net_income"), ("balance", "total_assets"))
    prior_roa = _closing_ratio(prior, ("income", "net_income"), ("balance", "total_assets"))
    signals["improving_return_on_assets"] = roa > prior_roa

    # Accruals: cash earnings exceeding accounting earnings is the quality signal.
    signals["cash_flow_exceeds_net_income"] = operating_cash_flow > net_income

    # --- Leverage, liquidity and source of funds (3 signals) ---
    leverage = _closing_ratio(ctx, ("balance", "long_term_debt"), ("balance", "total_assets"))
    prior_leverage = _closing_ratio(
        prior, ("balance", "long_term_debt"), ("balance", "total_assets")
    )
    signals["decreasing_long_term_leverage"] = leverage <= prior_leverage

    current_ratio = _closing_ratio(
        ctx, ("balance", "total_current_assets"), ("balance", "total_current_liabilities")
    )
    prior_current_ratio = _closing_ratio(
        prior, ("balance", "total_current_assets"), ("balance", "total_current_liabilities")
    )
    signals["improving_current_ratio"] = current_ratio > prior_current_ratio

    shares = ctx.require("income", "weighted_average_shares_diluted")
    prior_shares = prior.require("income", "weighted_average_shares_diluted")
    signals["no_new_share_issuance"] = shares <= prior_shares

    # --- Operating efficiency (2 signals) ---
    gross_margin = _closing_ratio(ctx, ("income", "gross_profit"), ("income", "revenue"))
    prior_gross_margin = _closing_ratio(prior, ("income", "gross_profit"), ("income", "revenue"))
    signals["improving_gross_margin"] = gross_margin > prior_gross_margin

    asset_turnover = _closing_ratio(ctx, ("income", "revenue"), ("balance", "total_assets"))
    prior_asset_turnover = _closing_ratio(prior, ("income", "revenue"), ("balance", "total_assets"))
    signals["improving_asset_turnover"] = asset_turnover > prior_asset_turnover

    score = float(sum(signals.values()))
    if score >= 8:
        interpretation = "Strong"
    elif score >= 5:
        interpretation = "Moderate"
    else:
        interpretation = "Weak"

    return Computed(
        value=score,
        method=f"{PIOTROSKI_MAX_SCORE} binary signals versus {prior.period_label}",
        inputs={"return_on_assets": roa, "prior_return_on_assets": prior_roa},
        details={
            "signals": signals,
            "max_score": PIOTROSKI_MAX_SCORE,
            "interpretation": interpretation,
            "compared_with": prior.period_label,
        },
    )


# ---------------------------------------------------------------------------
# Beneish M-Score
# ---------------------------------------------------------------------------


def _depreciation(ctx: PeriodBundle) -> float:
    """Depreciation from the income statement, falling back to the cash flow."""
    value = ctx.get("income", "depreciation_amortization")
    if value is None:
        value = ctx.get("cash_flow", "depreciation_amortization")
    if value is None:
        raise NotCalculable(
            f"depreciation_amortization not reported in {ctx.period_label} income statement "
            "or cash flow statement"
        )
    return value


@ratio("beneish_m_score", DISTRESS, "Beneish M-Score (earnings manipulation risk)")
def beneish_m_score(ctx: PeriodBundle) -> Computed:
    """Eight indices comparing this year with last, weighted into one score.

    Above -1.78 flags a statistically elevated likelihood of earnings
    manipulation. It is a screen, not a verdict: legitimate rapid growth also
    lifts several of the indices, which is why the score persists its components.
    """
    if ctx.prior is None:
        raise NotCalculable(
            f"no prior period on file; the M-Score compares consecutive years and "
            f"{ctx.period_label} is the earliest year available"
        )
    prior = ctx.prior

    revenue = ctx.require("income", "revenue")
    prior_revenue = prior.require("income", "revenue")
    if prior_revenue == 0 or revenue == 0:
        raise NotCalculable("revenue is zero in one of the two periods; the indices are undefined")

    receivables = ctx.require("balance", "net_receivables")
    prior_receivables = prior.require("balance", "net_receivables")
    total_assets = ctx.require("balance", "total_assets")
    prior_total_assets = prior.require("balance", "total_assets")
    current_assets = ctx.require("balance", "total_current_assets")
    prior_current_assets = prior.require("balance", "total_current_assets")
    ppe = ctx.require("balance", "property_plant_equipment_net")
    prior_ppe = prior.require("balance", "property_plant_equipment_net")

    # DSRI - days sales in receivables. Receivables outrunning sales is the
    # classic signature of revenue recognised before it is collectible.
    dsri = divide(
        receivables / revenue, prior_receivables / prior_revenue, "prior-year receivables to sales"
    )

    # GMI - gross margin index. Above 1 means margins deteriorated.
    gross_profit = ctx.require("income", "gross_profit")
    prior_gross_profit = prior.require("income", "gross_profit")
    gross_margin = gross_profit / revenue
    prior_gross_margin = prior_gross_profit / prior_revenue
    gmi = divide(prior_gross_margin, gross_margin, "current-year gross margin")

    # AQI - asset quality index: the share of assets that are neither current
    # nor plant, i.e. the soft assets where capitalised costs get parked.
    soft_assets = 1.0 - (current_assets + ppe) / total_assets
    prior_soft_assets = 1.0 - (prior_current_assets + prior_ppe) / prior_total_assets
    aqi = divide(soft_assets, prior_soft_assets, "prior-year non-current non-plant asset share")

    # SGI - sales growth index.
    sgi = revenue / prior_revenue

    # DEPI - depreciation index. Above 1 means the depreciation rate slowed,
    # which lifts reported earnings.
    depreciation = _depreciation(ctx)
    prior_depreciation = _depreciation(prior)
    depreciation_rate = divide(
        depreciation, depreciation + ppe, "depreciation plus net plant in the current year"
    )
    prior_depreciation_rate = divide(
        prior_depreciation, prior_depreciation + prior_ppe, "depreciation plus net plant in the prior year"
    )
    depi = divide(prior_depreciation_rate, depreciation_rate, "current-year depreciation rate")

    # SGAI - SG&A index.
    sga = ctx.require("income", "selling_general_admin")
    prior_sga = prior.require("income", "selling_general_admin")
    sgai = divide(sga / revenue, prior_sga / prior_revenue, "prior-year SG&A to sales")

    # TATA - total accruals to total assets, the single heaviest-weighted term.
    net_income = ctx.require("income", "net_income")
    operating_cash_flow = ctx.require("cash_flow", "operating_cash_flow")
    tata = (net_income - operating_cash_flow) / total_assets

    # LVGI - leverage index.
    current_liabilities = ctx.require("balance", "total_current_liabilities")
    prior_current_liabilities = prior.require("balance", "total_current_liabilities")
    long_term_debt = ctx.require("balance", "long_term_debt")
    prior_long_term_debt = prior.require("balance", "long_term_debt")
    leverage = (current_liabilities + long_term_debt) / total_assets
    prior_leverage = (prior_current_liabilities + prior_long_term_debt) / prior_total_assets
    lvgi = divide(leverage, prior_leverage, "prior-year leverage")

    indices = {
        "dsri": dsri,
        "gmi": gmi,
        "aqi": aqi,
        "sgi": sgi,
        "depi": depi,
        "sgai": sgai,
        "tata": tata,
        "lvgi": lvgi,
    }
    score = BENEISH_INTERCEPT + sum(
        BENEISH_COEFFICIENTS[key] * value for key, value in indices.items()
    )

    return Computed(
        value=score,
        method=f"eight-index model versus {prior.period_label}",
        inputs=indices,
        details={
            "threshold": BENEISH_MANIPULATION_THRESHOLD,
            "flagged": score > BENEISH_MANIPULATION_THRESHOLD,
            "interpretation": (
                "Elevated manipulation risk"
                if score > BENEISH_MANIPULATION_THRESHOLD
                else "Below manipulation threshold"
            ),
            "compared_with": prior.period_label,
        },
    )


def zone_of(result: Any) -> str | None:
    """Zone label from a distress RatioResult, if it has one."""
    if result is None or not getattr(result, "is_calculable", False):
        return None
    return (getattr(result, "details", None) or {}).get("zone")
