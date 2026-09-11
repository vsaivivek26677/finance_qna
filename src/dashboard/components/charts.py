"""Plotly figures, all sharing one visual language.

Formatting rules that apply everywhere: percentages for margins and returns,
compact currency for absolute amounts, and four decimals for multiples. Getting
this wrong is how a dashboard ends up showing "0.2397" where it means "24%".
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.dashboard.components import theme

# Ratios that read naturally as percentages.
PERCENT_RATIOS = {
    "gross_margin", "operating_margin", "net_margin", "return_on_equity",
    "return_on_assets", "return_on_invested_capital", "free_cash_flow_margin",
    "operating_cash_flow_margin", "debt_to_assets", "liabilities_to_assets",
    "capex_to_operating_cash_flow",
}
# Ratio values that are currency amounts, not multiples.
MONEY_RATIOS = {"working_capital", "free_cash_flow", "enterprise_value"}

# Statement fields that are not currency amounts. EPS is per-share (needs
# decimals, and "USD 7" hides the difference between 6.60 and 7.49), and share
# counts are units, not money.
PER_SHARE_FIELDS = {"eps", "eps_diluted"}
COUNT_FIELDS = {"weighted_average_shares", "weighted_average_shares_diluted"}


def format_statement_value(name: str, value: float | None, currency: str = "") -> str:
    """Format one statement line according to what the field actually measures."""
    if value is None:
        return "N/A"
    if name in PER_SHARE_FIELDS:
        return f"{value:,.2f}"
    if name in COUNT_FIELDS:
        return format_money(value)  # compact digits, no currency prefix
    return format_money(value, currency)


# Ratios where lower is better, so a decrease should read as good. Streamlit
# colours a negative delta red by default, which would mark falling leverage and
# faster collections as deterioration.
LOWER_IS_BETTER = {
    "debt_to_equity", "debt_to_assets", "liabilities_to_assets", "net_debt_to_ebitda",
    "days_sales_outstanding", "days_inventory_outstanding", "capex_to_operating_cash_flow",
}


def format_money(value: float | None, currency: str = "") -> str:
    if value is None:
        return "N/A"
    prefix = f"{currency} " if currency else ""
    magnitude = abs(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= divisor:
            return f"{prefix}{value / divisor:,.2f}{suffix}"
    return f"{prefix}{value:,.0f}"


def format_ratio(name: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    if name in MONEY_RATIOS:
        return format_money(value)
    if name in PERCENT_RATIOS:
        return f"{value * 100:.2f}%"
    return f"{value:,.2f}"


# The engine records a full sentence for how a ratio was computed. That belongs
# in a tooltip or the API, not in a narrow table column where it truncates to
# "average of ope...".
_BASIS_SHORT = {
    "average of opening and closing balance": "Average balance",
    "closing balance only (no prior period on file)": "Closing balance",
}


def short_basis(method: str | None) -> str:
    """Condense a ratio's computation basis for a table cell."""
    if not method:
        return "-"
    if method in _BASIS_SHORT:
        return _BASIS_SHORT[method]
    for key, short in _BASIS_SHORT.items():
        if method.startswith(key):
            return short
    return method if len(method) <= 28 else method[:27] + "…"


def humanise(name: str) -> str:
    acronyms = {"eps", "ebitda", "ebit", "roe", "roa", "roic", "fcf", "ocf", "dso", "dio", "ev"}
    return " ".join(
        w.upper() if w.lower() in acronyms else w.capitalize()
        for w in name.replace("_", " ").split()
    )


def trend_chart(
    points: dict[int, float | None], ratio_name: str, height: int = 240
) -> go.Figure | None:
    """A single ratio over time, with the area under it lightly filled."""
    usable = {int(k): v for k, v in points.items() if v is not None}
    if len(usable) < 2:
        return None

    years = sorted(usable)
    values = [usable[y] for y in years]
    as_percent = ratio_name in PERCENT_RATIOS
    display = [v * 100 for v in values] if as_percent else values

    figure = go.Figure(
        go.Scatter(
            x=[f"FY{y}" for y in years],
            y=display,
            mode="lines+markers",
            line={"color": theme.ACCENT, "width": 2.5, "shape": "spline", "smoothing": 0.5},
            marker={"size": 7, "color": theme.ACCENT, "line": {"width": 0}},
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.09)",
            hovertemplate="%{x}: %{y:.2f}" + ("%" if as_percent else "") + "<extra></extra>",
        )
    )
    layout = theme.plotly_layout(height=height)
    layout["title"] = {
        "text": humanise(ratio_name),
        "font": {"size": 12.5, "color": theme.TEXT_MUTED},
        "x": 0,
        "xanchor": "left",
    }
    if as_percent:
        layout["yaxis"]["ticksuffix"] = "%"
    figure.update_layout(**layout)
    return figure


def multi_trend_chart(
    series: dict[str, dict[int, float | None]], title: str, height: int = 340
) -> go.Figure | None:
    """Several ratios on one axis, for comparing shapes rather than levels."""
    figure = go.Figure()
    plotted = 0

    for index, (name, points) in enumerate(series.items()):
        usable = {int(k): v for k, v in points.items() if v is not None}
        if len(usable) < 2:
            continue
        years = sorted(usable)
        as_percent = name in PERCENT_RATIOS
        values = [usable[y] * 100 if as_percent else usable[y] for y in years]
        figure.add_trace(
            go.Scatter(
                x=[f"FY{y}" for y in years],
                y=values,
                name=humanise(name),
                mode="lines+markers",
                line={"color": theme.SERIES[index % len(theme.SERIES)], "width": 2.2},
                marker={"size": 6},
                hovertemplate="%{fullData.name}<br>%{x}: %{y:.2f}<extra></extra>",
            )
        )
        plotted += 1

    if plotted == 0:
        return None
    layout = theme.plotly_layout(height=height, showlegend=True)
    figure.update_layout(**layout)
    return figure


def altman_gauge(value: float | None, zone: str | None, height: int = 250) -> go.Figure:
    """Altman Z with its zone bands drawn in, so the number has context."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value if value is not None else 0,
            number={
                "font": {"size": 34, "color": theme.ZONE.get(zone or "", theme.TEXT)},
                "valueformat": ".2f",
            },
            gauge={
                "axis": {
                    "range": [0, 10],
                    "tickcolor": theme.TEXT_FAINT,
                    "tickfont": {"size": 10, "color": theme.TEXT_FAINT},
                },
                "bar": {"color": theme.ZONE.get(zone or "", theme.ACCENT), "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                # The published Altman bands: distress below 1.81, safe above 2.99.
                "steps": [
                    {"range": [0, 1.81], "color": "rgba(248, 113, 113, 0.20)"},
                    {"range": [1.81, 2.99], "color": "rgba(251, 191, 36, 0.18)"},
                    {"range": [2.99, 10], "color": "rgba(52, 211, 153, 0.16)"},
                ],
                "threshold": {
                    "line": {"color": theme.TEXT_MUTED, "width": 2},
                    "thickness": 0.75,
                    "value": value if value is not None else 0,
                },
            },
        )
    )
    layout = theme.plotly_layout(height=height)
    layout["margin"] = {"l": 22, "r": 22, "t": 14, "b": 4}
    figure.update_layout(**layout)
    return figure


def piotroski_bar(score: float | None, max_score: int = 9, height: int = 120) -> go.Figure:
    """Nine segments, filled to the score - reads faster than a number alone."""
    filled = int(score) if score is not None else 0
    colour = theme.POSITIVE if filled >= 7 else (theme.WARNING if filled >= 4 else theme.NEGATIVE)

    figure = go.Figure()
    for i in range(max_score):
        figure.add_trace(
            go.Bar(
                x=[1], y=["score"], orientation="h",
                marker={
                    "color": colour if i < filled else "rgba(255,255,255,0.07)",
                    "line": {"color": theme.INK, "width": 2},
                },
                hoverinfo="skip", showlegend=False,
            )
        )
    layout = theme.plotly_layout(height=height)
    layout.update(
        barmode="stack",
        xaxis={"visible": False, "range": [0, max_score]},
        yaxis={"visible": False},
        margin={"l": 0, "r": 0, "t": 6, "b": 6},
    )
    figure.update_layout(**layout)
    return figure


def category_bar(values: dict[str, float], title: str, height: int = 300) -> go.Figure | None:
    """Horizontal bars for a set of ratios in one category."""
    if not values:
        return None
    names = list(values)[::-1]
    numbers = [values[n] for n in names]

    figure = go.Figure(
        go.Bar(
            x=numbers,
            y=[humanise(n) for n in names],
            orientation="h",
            marker={
                "color": [theme.POSITIVE if v >= 0 else theme.NEGATIVE for v in numbers],
                "line": {"width": 0},
            },
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    layout = theme.plotly_layout(height=height)
    layout["title"] = {"text": title, "font": {"size": 13, "color": theme.TEXT_MUTED}, "x": 0}
    layout["margin"] = {"l": 8, "r": 8, "t": 34, "b": 8}
    figure.update_layout(**layout)
    return figure


def probability_bars(
    points: dict[int, float | None], height: int = 150, band_cuts=(0.05, 0.12, 0.30)
) -> go.Figure | None:
    """Distress probability by fiscal year, as small bars on a 0-1 axis.

    Colour steps at the model's risk-band cut-offs so a rising estimate is
    visible at a glance without reading the numbers.
    """
    usable = {int(k): v for k, v in points.items() if v is not None}
    if not usable:
        return None
    years = sorted(usable)
    values = [usable[y] for y in years]

    def colour(p: float) -> str:
        if p < band_cuts[0]:
            return theme.POSITIVE
        if p < band_cuts[1]:
            return theme.ACCENT
        if p < band_cuts[2]:
            return theme.WARNING
        return theme.NEGATIVE

    figure = go.Figure(
        go.Bar(
            x=[f"FY{y}" for y in years],
            y=values,
            marker={"color": [colour(v) for v in values], "line": {"width": 0}},
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        )
    )
    layout = theme.plotly_layout(height=height)
    layout["margin"] = {"l": 8, "r": 8, "t": 10, "b": 8}
    layout["yaxis"].update(tickformat=".0%", range=[0, max(0.35, max(values) * 1.25)])
    figure.update_layout(**layout)
    return figure


def revenue_profit_chart(
    years: list[int], revenue: list[float | None], net_income: list[float | None], height: int = 320
) -> go.Figure:
    """Revenue as bars with net income overlaid - the headline shape of a business."""
    labels = [f"FY{y}" for y in years]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels, y=revenue, name="Revenue",
            marker={"color": "rgba(45, 212, 191, 0.55)", "line": {"width": 0}},
            hovertemplate="Revenue %{x}: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels, y=net_income, name="Net income", mode="lines+markers",
            line={"color": theme.SERIES[1], "width": 2.5},
            marker={"size": 7},
            hovertemplate="Net income %{x}: %{y:,.0f}<extra></extra>",
        )
    )
    # No in-chart title: the panel heading already names it, and a title here
    # collides with the legend Plotly places along the top edge.
    figure.update_layout(**theme.plotly_layout(height=height, showlegend=True))
    return figure
