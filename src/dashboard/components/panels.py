"""Reusable rendering blocks: KPI rows, red-flag cards, statement tables, sources.

Each function does one thing and takes plain data from the API client, so the
page layout in `app.py` stays readable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components import charts, theme


def masthead(profile: dict) -> None:
    """Company identity line: ticker, name, and the classification chips."""
    chips = [c for c in (profile.get("sector"), profile.get("industry")) if c]
    if profile.get("exchange"):
        chips.append(profile["exchange"])
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(
        f'<div class="masthead">'
        f'<span class="ticker">{profile["ticker"]}</span>'
        f'<span class="name">{profile.get("name") or ""}</span>'
        f"{chip_html}</div>",
        unsafe_allow_html=True,
    )


def notice(message: str, kind: str = "info") -> None:
    st.markdown(
        f'<div class="notice{" warn" if kind == "warn" else ""}">{message}</div>',
        unsafe_allow_html=True,
    )


def _delta(current: float | None, previous: float | None, name: str) -> str | None:
    """Year-over-year change, in the same units the value is displayed in."""
    if current is None or previous is None:
        return None
    if name in charts.PERCENT_RATIOS:
        # The value renders as a percentage, so its change is in percentage
        # points, not a relative percentage. Spelled "pts" rather than the
        # terser "pp" - the abbreviation reads as a typo to most people.
        return f"{(current - previous) * 100:+.2f} pts"
    if name in charts.MONEY_RATIOS:
        return f"{charts.format_money(current - previous)}"
    if previous == 0:
        return None
    return f"{(current - previous) / abs(previous) * 100:+.1f}%"


def kpi_row(
    ratios_by_year: dict[int, dict[str, float | None]],
    year: int,
    names: list[str],
    columns: int = 4,
) -> None:
    """Metric cards with year-over-year deltas where a prior year exists."""
    current = ratios_by_year.get(year, {})
    previous = ratios_by_year.get(year - 1, {})

    for start in range(0, len(names), columns):
        row = st.columns(columns, gap="small")
        for slot, name in zip(row, names[start : start + columns]):
            value = current.get(name)
            with slot:
                delta = _delta(value, previous.get(name), name)
                st.metric(
                    label=charts.humanise(name),
                    value=charts.format_ratio(name, value),
                    delta=delta,
                    delta_color=(
                        "off"
                        if delta is None
                        else ("inverse" if name in charts.LOWER_IS_BETTER else "normal")
                    ),
                )


def _source_value(value) -> str:
    """Render a flag's trigger value at a sensible precision.

    A ratio needs decimals; a balance-sheet total does not. "147,357,000,000.0000"
    is technically right and completely unreadable.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if abs(value) >= 10_000:
        return charts.format_money(float(value))
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def red_flag_cards(flags: list[dict], limit: int | None = None) -> None:
    """One card per flag, colour-keyed by severity with its source values."""
    if not flags:
        notice(
            "No red flags were raised for this period. This is a computed result from the "
            "rule engine, not an absence of data."
        )
        return

    for flag in flags[:limit] if limit else flags:
        colour = theme.SEVERITY.get(flag["severity"], theme.TEXT_FAINT)
        sources = flag.get("source_values") or {}
        source_line = ""
        if sources:
            rendered = ", ".join(
                f"{charts.humanise(k)} = {_source_value(v)}"
                for k, v in list(sources.items())[:4]
                if not isinstance(v, dict)
            )
            if rendered:
                source_line = f'<div class="meta">{rendered}</div>'

        st.markdown(
            f'<div class="flag-card" style="border-left-color:{colour}">'
            f'<div class="title">{theme.severity_badge(flag["severity"])} '
            f'&nbsp;{flag["flag_name"]}</div>'
            f'<div class="body">{flag.get("explanation") or ""}</div>'
            f'<div class="meta">FY{flag["fiscal_year"]} &middot; '
            f'{flag.get("category") or "Uncategorised"}</div>'
            f"{source_line}</div>",
            unsafe_allow_html=True,
        )


def severity_summary(counts: dict[str, int]) -> None:
    """Counts by severity as a single compact row."""
    if not counts:
        return
    order = ["High", "Medium", "Low", "Info"]
    parts = [
        f'<span style="margin-right:1.1rem">{theme.severity_badge(s)} '
        f'<strong style="color:{theme.TEXT};font-size:1.05rem">{counts[s]}</strong></span>'
        for s in order
        if counts.get(s)
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)


def statement_table(lines: list[dict], currency: str = "") -> None:
    """Statements as a year-by-year table, with unreported fields marked.

    Fields the provider never sent show as "N/A - not reported" rather than a
    blank cell, so a gap in the data cannot be misread as a zero.
    """
    if not lines:
        notice("No statements on file for this period.")
        return

    ordered = sorted(lines, key=lambda line: line["fiscal_year"], reverse=True)
    field_names: list[str] = []
    for line in ordered:
        for name in line["values"]:
            if name not in field_names:
                field_names.append(name)

    table: dict[str, list[str]] = {}
    for line in ordered:
        column = []
        for name in field_names:
            value = line["values"].get(name)
            if value is None:
                reason = (line.get("missing_fields") or {}).get(name, "not reported")
                column.append(f"N/A - {reason.replace('_', ' ')}")
            else:
                column.append(charts.format_statement_value(name, value, currency))
        table[f"FY{line['fiscal_year']}"] = column

    frame = pd.DataFrame(table, index=[charts.humanise(n) for n in field_names])
    st.dataframe(frame, use_container_width=True, height=min(620, 42 + 35 * len(frame)))

    missing_total = sum(len(line.get("missing_fields") or {}) for line in ordered)
    if missing_total:
        st.markdown(
            f'<div class="caption">{missing_total} field(s) across these periods were not '
            f"reported by the data provider. They are shown as N/A and were never estimated.</div>",
            unsafe_allow_html=True,
        )


def ratio_table(ratios: list[dict]) -> None:
    """Every ratio in a category, including the ones that could not be computed."""
    calculable = [r for r in ratios if r["is_calculable"] and r["value"] is not None]
    blocked = [r for r in ratios if not r["is_calculable"]]

    if calculable:
        frame = pd.DataFrame(
            {
                "Ratio": [charts.humanise(r["name"]) for r in calculable],
                "Value": [charts.format_ratio(r["name"], r["value"]) for r in calculable],
                "Basis": [charts.short_basis(r.get("method")) for r in calculable],
            }
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)

    if blocked:
        with st.expander(f"{len(blocked)} ratio(s) could not be calculated - see why", expanded=False):
            for ratio in blocked:
                st.markdown(
                    f'<div class="caption" style="margin-bottom:.5rem">'
                    f'<strong style="color:{theme.TEXT_MUTED}">{charts.humanise(ratio["name"])}</strong>'
                    f'<br>{ratio.get("reason") or "No reason recorded."}</div>',
                    unsafe_allow_html=True,
                )


def sources_expander(sources: list[dict], label: str = "Sources") -> None:
    """The retrieved context behind a generated answer, so it can be audited."""
    if not sources:
        return
    with st.expander(f"{label} ({len(sources)} retrieved chunks)", expanded=False):
        for index, chunk in enumerate(sources, start=1):
            st.markdown(
                f'<div class="caption" style="margin-bottom:.15rem">'
                f'<strong style="color:{theme.ACCENT}">[{index}] {chunk["citation"]}</strong>'
                f' &middot; similarity {chunk["similarity"]:.3f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="caption" style="margin-bottom:.9rem">{chunk["excerpt"]}</div>',
                unsafe_allow_html=True,
            )


_BAND_COLOUR = {
    "Low": theme.POSITIVE,
    "Moderate": theme.ACCENT,
    "Elevated": theme.WARNING,
    "High": theme.NEGATIVE,
}


def _period_for_year(payload: dict, year: int | None) -> dict | None:
    periods = payload.get("periods") or []
    if not periods:
        return None
    if year is not None:
        match = next((p for p in periods if p["fiscal_year"] == year), None)
        if match:
            return match
    return max(periods, key=lambda p: p["fiscal_year"])


def model_probability_readout(payload: dict, year: int | None) -> None:
    """Compact distress-probability line for the Overview panel."""
    if not payload or not payload.get("available", True):
        return
    period = _period_for_year(payload, year)
    if period is None:
        return

    info = payload.get("model") or {}
    st.markdown(
        f'<div class="caption" style="margin-top:.9rem">Model estimate &middot; '
        f'{info.get("name", "distress model")}</div>',
        unsafe_allow_html=True,
    )
    if not period["is_scored"]:
        notice(period.get("reason") or "Not enough inputs to score this period.")
        return

    probability = period["probability"]
    band = period["risk_band"] or ""
    colour = _BAND_COLOUR.get(band, theme.TEXT_MUTED)
    roc = info.get("cv_roc_auc")
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:.6rem">'
        f'<span style="font-size:1.7rem;font-weight:680;color:{theme.TEXT}">{probability:.1%}</span>'
        f'{theme.badge(band, colour)}</div>'
        f'<div class="caption" style="margin-top:.3rem">Calibrated probability of financial '
        f'distress, from a classifier trained on {info.get("training_rows", 0):,} labelled '
        f'firm-years'
        + (f' (CV ROC-AUC {roc:.2f})' if roc else "")
        + ".</div>",
        unsafe_allow_html=True,
    )


def distress_model_section(payload: dict, year: int | None) -> None:
    """Full model-probability breakdown for the Risk tab."""
    if not payload or not payload.get("available", True):
        notice(
            payload.get("note")
            or "No distress model is trained. Run "
            "<code>python -m src.prediction.cli train</code>."
        )
        return

    info = payload.get("model") or {}
    st.markdown(
        '<div class="caption" style="margin-bottom:.8rem">A calibrated probability from a '
        "gradient-boosted classifier trained on ~78.7k firm-years of real American public "
        "companies (NYSE/NASDAQ, 1999-2018, ~6.6% bankrupt), using ratios scaled by total "
        "liabilities rather than total assets. It is a learned weighting of those inputs, "
        "reported alongside the Altman score, not instead of it.</div>",
        unsafe_allow_html=True,
    )

    period = _period_for_year(payload, year)
    points = {p["fiscal_year"]: p["probability"] for p in payload.get("periods", []) if p["is_scored"]}

    left, right = st.columns([1, 1.3], gap="medium")
    with left:
        if period and period["is_scored"]:
            band = period["risk_band"] or ""
            st.metric(
                f"FY{period['fiscal_year']} estimate",
                f"{period['probability']:.1%}",
                delta=band,
                delta_color="off",
            )
            factors = period.get("factors") or []
            if factors:
                rendered = "".join(
                    f'<div class="caption">&bull; {charts.humanise(f["feature"])} = '
                    f'{f["value"]:.3f} <span style="color:{theme.TEXT_FAINT}">'
                    f'(training median {f["training_median"]:.3f})</span></div>'
                    for f in factors
                )
                st.markdown(
                    f'<div class="caption" style="margin-top:.4rem;color:{theme.TEXT_MUTED}">'
                    f"Values on the risky side of the training distribution:</div>{rendered}",
                    unsafe_allow_html=True,
                )
        elif period:
            notice(period.get("reason") or "This period could not be scored.")
        else:
            notice("No scored periods for this company.")

    with right:
        figure = charts.probability_bars(points)
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    if payload.get("note"):
        st.markdown(f'<div class="caption">{payload["note"]}</div>', unsafe_allow_html=True)

    with st.expander("How this model was built", expanded=False):
        roc, pr = info.get("cv_roc_auc"), info.get("cv_pr_auc")
        base_pr = info.get("baseline_pr_auc")
        brier = info.get("cv_brier")
        rows = [
            ("Dataset", info.get("dataset", "American public companies, 1999-2018")),
            ("Training firm-years", f"{info.get('training_rows', 0):,} "
                                    f"({info.get('training_prevalence', 0):.1%} insolvent)"),
            ("Cross-validated ROC-AUC", f"{roc:.3f}" if roc else "-"),
            ("Cross-validated PR-AUC", f"{pr:.3f}" if pr else "-"),
            ("Baseline PR-AUC (naive linear rule)", f"{base_pr:.3f}" if base_pr else "-"),
            ("Brier score (lower is better)", f"{brier:.4f}" if brier else "-"),
            ("Model", f"{info.get('name', '')} v{info.get('version', '')}"),
        ]
        st.markdown(
            "".join(
                f'<div class="caption" style="display:flex;justify-content:space-between;gap:1rem">'
                f"<span>{k}</span><span style='color:{theme.TEXT_MUTED};text-align:right'>{v}</span></div>"
                for k, v in rows
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="caption" style="margin-top:.6rem">Trained on the same American '
            "public-company population this platform scores, but the training window ends "
            "in 2018 and skews toward small and mid caps, so a mega-cap's estimate is still "
            "somewhat outside the training distribution. Treat a band, not a decimal, as the "
            "signal.</div>",
            unsafe_allow_html=True,
        )


def verification_footer(answer: dict) -> None:
    """Groundedness and any unverified figures, shown with every answer."""
    st.markdown(
        theme.groundedness_pill(
            answer.get("groundedness", 0.0),
            answer.get("verified_claims", 0),
            answer.get("total_claims", 0),
        ),
        unsafe_allow_html=True,
    )
    unverified = answer.get("unverified") or []
    if unverified:
        notice(
            "These figures could not be traced back to the retrieved source data: "
            f"<strong>{', '.join(unverified)}</strong>. Treat them with caution.",
            kind="warn",
        )
    for note in answer.get("notes") or []:
        st.markdown(f'<div class="caption">{note}</div>', unsafe_allow_html=True)
