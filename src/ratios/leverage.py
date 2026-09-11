"""Leverage and solvency ratios — how much debt, and can it be serviced?"""

from __future__ import annotations

from src.ratios.base import (
    LEVERAGE,
    Computed,
    NotCalculable,
    PeriodBundle,
    divide,
    ratio,
)


@ratio("debt_to_equity", LEVERAGE, "Total debt / shareholders' equity")
def debt_to_equity(ctx: PeriodBundle) -> Computed:
    debt = ctx.total_debt()
    equity = ctx.require("balance", "total_equity")
    if equity <= 0:
        raise NotCalculable(
            f"shareholders' equity is {equity:,.0f} in {ctx.period_label}; debt-to-equity is "
            "undefined when equity is not positive"
        )
    return Computed(
        value=divide(debt, equity, "shareholders' equity"),
        inputs={"total_debt": debt, "total_equity": equity},
    )


@ratio("debt_to_assets", LEVERAGE, "Total debt / total assets")
def debt_to_assets(ctx: PeriodBundle) -> Computed:
    debt = ctx.total_debt()
    assets = ctx.require("balance", "total_assets")
    return Computed(
        value=divide(debt, assets, "total assets"),
        inputs={"total_debt": debt, "total_assets": assets},
    )


@ratio("liabilities_to_assets", LEVERAGE, "Total liabilities / total assets")
def liabilities_to_assets(ctx: PeriodBundle) -> Computed:
    """Broader than debt-to-assets: includes payables, deferred revenue, leases."""
    liabilities = ctx.require("balance", "total_liabilities")
    assets = ctx.require("balance", "total_assets")
    return Computed(
        value=divide(liabilities, assets, "total assets"),
        inputs={"total_liabilities": liabilities, "total_assets": assets},
    )


@ratio("interest_coverage", LEVERAGE, "Operating income (EBIT) / interest expense")
def interest_coverage(ctx: PeriodBundle) -> Computed:
    """How many times over operating profit covers the interest bill.

    Interest expense is taken as a magnitude: sources disagree on whether it is
    reported positive or negative, and a sign flip here would invert the ratio.
    """
    operating_income = ctx.require("income", "operating_income")
    interest_expense = abs(ctx.require("income", "interest_expense"))
    if interest_expense == 0:
        raise NotCalculable(
            f"no interest expense reported in {ctx.period_label}; interest coverage is "
            "undefined (which is not itself a sign of distress)"
        )
    return Computed(
        value=operating_income / interest_expense,
        inputs={"operating_income": operating_income, "interest_expense": interest_expense},
    )


@ratio("net_debt_to_ebitda", LEVERAGE, "Net debt / EBITDA")
def net_debt_to_ebitda(ctx: PeriodBundle) -> Computed:
    """The covenant metric lenders actually test against."""
    net_debt = ctx.get("balance", "net_debt")
    if net_debt is None:
        cash = ctx.get("balance", "cash_and_equivalents")
        if cash is None:
            raise NotCalculable(
                f"net_debt not reported in {ctx.period_label} balance sheet, and "
                "cash_and_equivalents is unavailable to derive it"
            )
        net_debt, method = ctx.total_debt() - cash, "derived as total debt less cash"
    else:
        method = None

    ebitda = ctx.require("income", "ebitda")
    if ebitda <= 0:
        raise NotCalculable(
            f"EBITDA is {ebitda:,.0f} in {ctx.period_label}; leverage multiples are not "
            "meaningful against non-positive earnings"
        )
    return Computed(
        value=net_debt / ebitda,
        method=method,
        inputs={"net_debt": net_debt, "ebitda": ebitda},
    )
