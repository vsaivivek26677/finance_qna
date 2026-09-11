"""Liquidity ratios — can the company cover its near-term obligations?"""

from __future__ import annotations

from src.ratios.base import (
    LIQUIDITY,
    Computed,
    NotCalculable,
    PeriodBundle,
    divide,
    ratio,
)


@ratio("current_ratio", LIQUIDITY, "Current assets / current liabilities")
def current_ratio(ctx: PeriodBundle) -> Computed:
    current_assets = ctx.require("balance", "total_current_assets")
    current_liabilities = ctx.require("balance", "total_current_liabilities")
    return Computed(
        value=divide(current_assets, current_liabilities, "total current liabilities"),
        inputs={
            "total_current_assets": current_assets,
            "total_current_liabilities": current_liabilities,
        },
    )


@ratio("quick_ratio", LIQUIDITY, "(Current assets - inventory) / current liabilities")
def quick_ratio(ctx: PeriodBundle) -> Computed:
    """Acid test: liquidity excluding inventory, which may not sell at book value."""
    current_assets = ctx.require("balance", "total_current_assets")
    current_liabilities = ctx.require("balance", "total_current_liabilities")
    inventory = ctx.require("balance", "inventory")
    return Computed(
        value=divide(current_assets - inventory, current_liabilities, "total current liabilities"),
        method="(current assets - inventory) / current liabilities",
        inputs={
            "total_current_assets": current_assets,
            "inventory": inventory,
            "total_current_liabilities": current_liabilities,
        },
    )


@ratio("cash_ratio", LIQUIDITY, "Cash and short-term investments / current liabilities")
def cash_ratio(ctx: PeriodBundle) -> Computed:
    """Strictest liquidity measure: only genuinely cash-like assets count."""
    current_liabilities = ctx.require("balance", "total_current_liabilities")

    combined = ctx.get("balance", "cash_and_short_term_investments")
    if combined is not None:
        cash_like, method = combined, "reported cash and short-term investments"
    else:
        cash = ctx.get("balance", "cash_and_equivalents")
        investments = ctx.get("balance", "short_term_investments")
        if cash is None and investments is None:
            raise NotCalculable(
                f"no cash figure reported in {ctx.period_label} balance sheet "
                "(needs cash_and_short_term_investments, or cash_and_equivalents)"
            )
        cash_like = (cash or 0.0) + (investments or 0.0)
        method = "cash and equivalents plus short-term investments"

    return Computed(
        value=divide(cash_like, current_liabilities, "total current liabilities"),
        method=method,
        inputs={"cash_like_assets": cash_like, "total_current_liabilities": current_liabilities},
    )


@ratio("working_capital", LIQUIDITY, "Current assets - current liabilities")
def working_capital(ctx: PeriodBundle) -> Computed:
    """Absolute buffer, not a ratio — also the numerator of Altman's X1."""
    current_assets = ctx.require("balance", "total_current_assets")
    current_liabilities = ctx.require("balance", "total_current_liabilities")
    return Computed(
        value=current_assets - current_liabilities,
        inputs={
            "total_current_assets": current_assets,
            "total_current_liabilities": current_liabilities,
        },
    )
