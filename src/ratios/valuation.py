"""Valuation ratios — priced off the market snapshot nearest the period end.

These are the only ratios that depend on market data. When no price point was
paired to the period (yfinance unavailable, a delisted ticker, a fiscal end
outside the price history), they are reported as not calculable rather than
being computed from a stale or current-day price, which would silently mix a
2026 market value with a 2021 balance sheet.
"""

from __future__ import annotations

from src.ratios.base import (
    VALUATION,
    Computed,
    NotCalculable,
    PeriodBundle,
    divide,
    ratio,
    require_positive,
)


@ratio("price_to_earnings", VALUATION, "Market capitalisation / net income")
def price_to_earnings(ctx: PeriodBundle) -> Computed:
    market_cap = ctx.market_value_of_equity()
    net_income = ctx.require("income", "net_income")
    require_positive(net_income, "net income")
    return Computed(
        value=market_cap / net_income,
        method="market capitalisation at period end / net income",
        inputs={"market_cap": market_cap, "net_income": net_income},
    )


@ratio("price_to_book", VALUATION, "Market capitalisation / book value of equity")
def price_to_book(ctx: PeriodBundle) -> Computed:
    market_cap = ctx.market_value_of_equity()
    equity = ctx.require("balance", "total_equity")
    require_positive(equity, "book value of equity")
    return Computed(
        value=market_cap / equity,
        inputs={"market_cap": market_cap, "total_equity": equity},
    )


@ratio("price_to_sales", VALUATION, "Market capitalisation / revenue")
def price_to_sales(ctx: PeriodBundle) -> Computed:
    """Meaningful even for loss-making companies, unlike P/E."""
    market_cap = ctx.market_value_of_equity()
    revenue = ctx.require("income", "revenue")
    return Computed(
        value=divide(market_cap, revenue, "revenue"),
        inputs={"market_cap": market_cap, "revenue": revenue},
    )


@ratio("enterprise_value", VALUATION, "Market cap + total debt - cash")
def enterprise_value(ctx: PeriodBundle) -> Computed:
    """What an acquirer would pay for the whole business, net of its cash."""
    market_cap = ctx.market_value_of_equity()
    debt = ctx.total_debt()
    cash = ctx.get("balance", "cash_and_equivalents")
    if cash is None:
        cash = ctx.get("balance", "cash_and_short_term_investments")
    if cash is None:
        raise NotCalculable(
            f"no cash figure reported in {ctx.period_label} balance sheet; enterprise value "
            "cannot be netted down"
        )
    return Computed(
        value=market_cap + debt - cash,
        inputs={"market_cap": market_cap, "total_debt": debt, "cash": cash},
    )


@ratio("ev_to_ebitda", VALUATION, "Enterprise value / EBITDA")
def ev_to_ebitda(ctx: PeriodBundle) -> Computed:
    market_cap = ctx.market_value_of_equity()
    debt = ctx.total_debt()
    cash = ctx.get("balance", "cash_and_equivalents")
    if cash is None:
        cash = ctx.get("balance", "cash_and_short_term_investments")
    if cash is None:
        raise NotCalculable(
            f"no cash figure reported in {ctx.period_label} balance sheet; enterprise value "
            "cannot be netted down"
        )
    ev = market_cap + debt - cash

    ebitda = ctx.require("income", "ebitda")
    require_positive(ebitda, "EBITDA")
    return Computed(
        value=ev / ebitda,
        inputs={"enterprise_value": ev, "ebitda": ebitda},
    )


@ratio("earnings_per_share_diluted", VALUATION, "Reported diluted EPS")
def earnings_per_share_diluted(ctx: PeriodBundle) -> Computed:
    """Passed through from the filing rather than recomputed, so it ties out."""
    eps = ctx.get("income", "eps_diluted")
    if eps is None:
        eps = ctx.require("income", "eps")
        method = "basic EPS (diluted not reported)"
    else:
        method = None
    return Computed(value=eps, method=method, inputs={"eps_diluted": eps})
