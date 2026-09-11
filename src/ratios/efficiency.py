"""Efficiency ratios — how hard the balance sheet works to produce revenue."""

from __future__ import annotations

from src.ratios.base import (
    EFFICIENCY,
    Computed,
    NotCalculable,
    PeriodBundle,
    divide,
    ratio,
)

DAYS_IN_YEAR = 365.0


@ratio("asset_turnover", EFFICIENCY, "Revenue / total assets")
def asset_turnover(ctx: PeriodBundle) -> Computed:
    revenue = ctx.require("income", "revenue")
    assets, method = ctx.average_or_closing("balance", "total_assets")
    return Computed(
        value=divide(revenue, assets, "total assets"),
        method=method,
        inputs={"revenue": revenue, "total_assets": assets},
    )


@ratio("inventory_turnover", EFFICIENCY, "Cost of revenue / inventory")
def inventory_turnover(ctx: PeriodBundle) -> Computed:
    """Cost of revenue is the correct numerator — inventory is carried at cost."""
    cost_of_revenue = ctx.require("income", "cost_of_revenue")
    inventory, method = ctx.average_or_closing("balance", "inventory")
    if inventory == 0:
        raise NotCalculable(
            f"inventory is zero in {ctx.period_label}; inventory turnover is undefined "
            "(common for companies that carry no inventory)"
        )
    return Computed(
        value=cost_of_revenue / inventory,
        method=method,
        inputs={"cost_of_revenue": cost_of_revenue, "inventory": inventory},
    )


@ratio("receivables_turnover", EFFICIENCY, "Revenue / net receivables")
def receivables_turnover(ctx: PeriodBundle) -> Computed:
    revenue = ctx.require("income", "revenue")
    receivables, method = ctx.average_or_closing("balance", "net_receivables")
    if receivables == 0:
        raise NotCalculable(
            f"net receivables are zero in {ctx.period_label}; receivables turnover is undefined"
        )
    return Computed(
        value=revenue / receivables,
        method=method,
        inputs={"revenue": revenue, "net_receivables": receivables},
    )


@ratio("days_sales_outstanding", EFFICIENCY, "365 / receivables turnover")
def days_sales_outstanding(ctx: PeriodBundle) -> Computed:
    """Average days to collect. Rising DSO is an early earnings-quality warning."""
    revenue = ctx.require("income", "revenue")
    receivables, method = ctx.average_or_closing("balance", "net_receivables")
    if revenue == 0:
        raise NotCalculable(f"revenue is zero in {ctx.period_label}; DSO is undefined")
    return Computed(
        value=DAYS_IN_YEAR * receivables / revenue,
        method=method,
        inputs={"net_receivables": receivables, "revenue": revenue},
    )


@ratio("days_inventory_outstanding", EFFICIENCY, "365 / inventory turnover")
def days_inventory_outstanding(ctx: PeriodBundle) -> Computed:
    cost_of_revenue = ctx.require("income", "cost_of_revenue")
    inventory, method = ctx.average_or_closing("balance", "inventory")
    if cost_of_revenue == 0:
        raise NotCalculable(
            f"cost of revenue is zero in {ctx.period_label}; days inventory is undefined"
        )
    return Computed(
        value=DAYS_IN_YEAR * inventory / cost_of_revenue,
        method=method,
        inputs={"inventory": inventory, "cost_of_revenue": cost_of_revenue},
    )
