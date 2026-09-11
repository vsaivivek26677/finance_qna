"""Profitability ratios — margins and returns on capital."""

from __future__ import annotations

from src.ratios.base import (
    PROFITABILITY,
    Computed,
    NotCalculable,
    PeriodBundle,
    divide,
    ratio,
)


@ratio("gross_margin", PROFITABILITY, "Gross profit / revenue")
def gross_margin(ctx: PeriodBundle) -> Computed:
    revenue = ctx.require("income", "revenue")
    gross_profit = ctx.get("income", "gross_profit")
    if gross_profit is None:
        cost = ctx.get("income", "cost_of_revenue")
        if cost is None:
            raise NotCalculable(
                f"gross_profit not reported in {ctx.period_label} income statement, and "
                "cost_of_revenue is unavailable to derive it"
            )
        gross_profit, method = revenue - cost, "derived as revenue less cost of revenue"
    else:
        method = None
    return Computed(
        value=divide(gross_profit, revenue, "revenue"),
        method=method,
        inputs={"gross_profit": gross_profit, "revenue": revenue},
    )


@ratio("operating_margin", PROFITABILITY, "Operating income (EBIT) / revenue")
def operating_margin(ctx: PeriodBundle) -> Computed:
    operating_income = ctx.require("income", "operating_income")
    revenue = ctx.require("income", "revenue")
    return Computed(
        value=divide(operating_income, revenue, "revenue"),
        inputs={"operating_income": operating_income, "revenue": revenue},
    )


@ratio("net_margin", PROFITABILITY, "Net income / revenue")
def net_margin(ctx: PeriodBundle) -> Computed:
    net_income = ctx.require("income", "net_income")
    revenue = ctx.require("income", "revenue")
    return Computed(
        value=divide(net_income, revenue, "revenue"),
        inputs={"net_income": net_income, "revenue": revenue},
    )


@ratio("return_on_equity", PROFITABILITY, "Net income / shareholders' equity")
def return_on_equity(ctx: PeriodBundle) -> Computed:
    net_income = ctx.require("income", "net_income")
    equity, method = ctx.average_or_closing("balance", "total_equity")
    if equity < 0:
        # A negative denominator flips the sign: a loss-making company would show
        # a positive ROE. The negative equity itself is raised as a red flag.
        raise NotCalculable(
            f"shareholders' equity is negative ({equity:,.0f}) in {ctx.period_label}; "
            "return on equity would be misleading"
        )
    return Computed(
        value=divide(net_income, equity, "shareholders' equity"),
        method=method,
        inputs={"net_income": net_income, "total_equity": equity},
    )


@ratio("return_on_assets", PROFITABILITY, "Net income / total assets")
def return_on_assets(ctx: PeriodBundle) -> Computed:
    net_income = ctx.require("income", "net_income")
    assets, method = ctx.average_or_closing("balance", "total_assets")
    return Computed(
        value=divide(net_income, assets, "total assets"),
        method=method,
        inputs={"net_income": net_income, "total_assets": assets},
    )


@ratio("return_on_invested_capital", PROFITABILITY, "NOPAT / (total debt + equity)")
def return_on_invested_capital(ctx: PeriodBundle) -> Computed:
    """ROIC = after-tax operating profit over the capital funding the business.

    The effective tax rate is taken from the company's own reported tax expense
    rather than a statutory assumption. In a pre-tax loss year that rate is not
    defined, so it is set to zero and the substitution is reported in `method`.
    """
    operating_income = ctx.require("income", "operating_income")
    pretax_income = ctx.require("income", "income_before_tax")
    tax_expense = ctx.require("income", "income_tax_expense")

    if pretax_income > 0:
        tax_rate = min(max(tax_expense / pretax_income, 0.0), 1.0)
        method = f"effective tax rate {tax_rate:.1%} from reported tax expense"
    else:
        tax_rate = 0.0
        method = "effective tax rate set to 0% (pre-tax loss makes the rate undefined)"

    nopat = operating_income * (1.0 - tax_rate)
    equity = ctx.require("balance", "total_equity")
    invested_capital = ctx.total_debt() + equity

    return Computed(
        value=divide(nopat, invested_capital, "invested capital (total debt plus equity)"),
        method=method,
        inputs={
            "operating_income": operating_income,
            "effective_tax_rate": tax_rate,
            "nopat": nopat,
            "invested_capital": invested_capital,
        },
    )
