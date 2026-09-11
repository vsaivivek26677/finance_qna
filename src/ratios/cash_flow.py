"""Cash flow ratios — whether reported profit actually converts into cash.

This category carries most of the earnings-quality signal: a company can book
revenue and net income long before cash arrives, and the gap between the two is
what the red-flag detector watches.
"""

from __future__ import annotations

from src.ratios.base import (
    CASH_FLOW,
    Computed,
    PeriodBundle,
    divide,
    ratio,
    require_positive,
)


@ratio("free_cash_flow", CASH_FLOW, "Operating cash flow less capital expenditure")
def free_cash_flow(ctx: PeriodBundle) -> Computed:
    value = ctx.free_cash_flow()
    return Computed(value=value, inputs={"free_cash_flow": value})


@ratio("free_cash_flow_margin", CASH_FLOW, "Free cash flow / revenue")
def free_cash_flow_margin(ctx: PeriodBundle) -> Computed:
    fcf = ctx.free_cash_flow()
    revenue = ctx.require("income", "revenue")
    return Computed(
        value=divide(fcf, revenue, "revenue"),
        inputs={"free_cash_flow": fcf, "revenue": revenue},
    )


@ratio("operating_cash_flow_margin", CASH_FLOW, "Operating cash flow / revenue")
def operating_cash_flow_margin(ctx: PeriodBundle) -> Computed:
    operating = ctx.require("cash_flow", "operating_cash_flow")
    revenue = ctx.require("income", "revenue")
    return Computed(
        value=divide(operating, revenue, "revenue"),
        inputs={"operating_cash_flow": operating, "revenue": revenue},
    )


@ratio("cash_conversion_ratio", CASH_FLOW, "Operating cash flow / net income")
def cash_conversion_ratio(ctx: PeriodBundle) -> Computed:
    """Below 1.0 sustained means accounting profit is outrunning cash collection."""
    operating = ctx.require("cash_flow", "operating_cash_flow")
    net_income = ctx.require("income", "net_income")
    require_positive(net_income, "net income")
    return Computed(
        value=operating / net_income,
        inputs={"operating_cash_flow": operating, "net_income": net_income},
    )


@ratio("capex_to_operating_cash_flow", CASH_FLOW, "Capital expenditure / operating cash flow")
def capex_to_operating_cash_flow(ctx: PeriodBundle) -> Computed:
    """How much of the cash generated is consumed just to keep the assets running."""
    capex = abs(ctx.require("cash_flow", "capital_expenditure"))
    operating = ctx.require("cash_flow", "operating_cash_flow")
    require_positive(operating, "operating cash flow")
    return Computed(
        value=capex / operating,
        inputs={"capital_expenditure": capex, "operating_cash_flow": operating},
    )
