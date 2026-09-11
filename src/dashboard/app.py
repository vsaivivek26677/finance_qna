"""Streamlit dashboard - a thin client over the FastAPI backend.

It holds no database session, no model and no API key. Every figure on screen
came through an HTTP call, which is what makes the backend independently
deployable and testable.

Run with:  streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts the *script's* directory on sys.path, not the working
# directory, so `import src...` fails without this. Needed locally and on
# Streamlit Community Cloud alike.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.dashboard.api_client import ApiClient, ApiError  # noqa: E402
from src.dashboard.components import charts, panels, theme  # noqa: E402

OVERVIEW_KPIS = [
    "net_margin", "operating_margin", "return_on_equity", "current_ratio",
    "debt_to_equity", "free_cash_flow_margin", "asset_turnover", "interest_coverage",
]
TREND_RATIOS = ["net_margin", "operating_margin", "gross_margin"]

st.set_page_config(
    page_title="Financial Analytics Platform",
    page_icon="\N{CHART WITH UPWARDS TREND}",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Starting the analytics backend (first load only)…")
def _resolve_api_base_url() -> str:
    """On a single-process host (Streamlit Community Cloud) this starts the
    FastAPI backend in a background thread and returns its loopback URL. Locally,
    or with API_BASE_URL pointed at a remote backend, it just returns that URL.
    """
    from src.dashboard.embedded_backend import ensure_backend

    return ensure_backend()


API_BASE_URL = _resolve_api_base_url()


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(base_url=API_BASE_URL)


# Cached so switching tabs does not re-request everything. Short TTL keeps a
# freshly ingested company from being hidden behind a stale cache.
@st.cache_data(ttl=120, show_spinner=False)
def load_companies() -> list[dict]:
    return get_client().companies()


@st.cache_data(ttl=120, show_spinner=False)
def load_company_data(ticker: str) -> dict:
    client = get_client()
    return {
        "profile": client.profile(ticker),
        "ratios": client.ratios(ticker),
        "flags": client.red_flags(ticker),
        "statements": client.statements(ticker),
    }


@st.cache_data(ttl=120, show_spinner=False)
def load_distress(ticker: str, year: int) -> dict:
    return get_client().distress(ticker, fiscal_year=year)


@st.cache_data(ttl=120, show_spinner=False)
def load_distress_prediction(ticker: str) -> dict:
    return get_client().distress_prediction(ticker)


def ratios_by_year(payload: dict) -> dict[int, dict[str, float | None]]:
    """Flatten the API response into {year: {ratio_name: value}}."""
    return {
        period["fiscal_year"]: {
            r["name"]: r["value"] if r["is_calculable"] else None for r in period["ratios"]
        }
        for period in payload.get("periods", [])
    }


def render_sidebar(health: dict | None) -> tuple[str | None, int | None]:
    """Company and period selection, plus backend status."""
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:1.05rem;font-weight:700;color:{theme.TEXT};'
            f'letter-spacing:-.01em">Financial Analytics</div>'
            f'<div class="caption" style="margin-bottom:1rem">Grounded statement analysis</div>',
            unsafe_allow_html=True,
        )

        if health is None:
            st.error("Backend unreachable")
            st.markdown(
                '<div class="caption">Start it with:<br><code>uvicorn src.api.main:app --reload</code></div>',
                unsafe_allow_html=True,
            )
            return None, None

        try:
            companies = load_companies()
        except ApiError as exc:
            st.error(exc.message)
            return None, None

        if not companies:
            st.warning("No companies ingested yet.")
            st.markdown(
                '<div class="caption">Ingest one from the terminal:<br>'
                "<code>python -m src.ingestion.cli AAPL</code></div>",
                unsafe_allow_html=True,
            )
            return None, None

        options = [c["ticker"] for c in companies]
        labels = {c["ticker"]: f"{c['ticker']} - {c.get('name') or ''}".strip(" -") for c in companies}
        default = options.index("AAPL") if "AAPL" in options else 0

        ticker = st.selectbox(
            "Company", options, index=default, format_func=lambda t: labels.get(t, t)
        )
        years = next((c["fiscal_years"] for c in companies if c["ticker"] == ticker), [])
        year = st.selectbox("Fiscal year", years, index=0) if years else None

        st.divider()
        st.markdown("**Backend**", help="Live status of the API and its dependencies.")
        # Indexing is a separate step from ingestion, so "ready" alone hides the
        # case where most companies have ratios but no chunks - and the AI tabs
        # then fail for exactly those companies.
        analysed = health.get("companies_analysed", 0)
        indexed = health.get("companies_indexed", 0)
        index_state = (
            f"{indexed} of {analysed} companies"
            if analysed
            else ("ready" if health.get("rag_indexed") else "not built")
        )
        rows = [
            ("Companies", str(health.get("companies", 0))),
            ("Ratios", f"{health.get('ratios', 0):,}"),
            ("Retrieval index", index_state),
            ("Distress model", "ready" if health.get("distress_model_ready") else "not trained"),
            ("Language model", "configured" if health.get("llm_configured") else "not configured"),
        ]
        st.markdown(
            "".join(
                f'<div class="caption" style="display:flex;justify-content:space-between">'
                f"<span>{k}</span><span style='color:{theme.TEXT_MUTED}'>{v}</span></div>"
                for k, v in rows
            ),
            unsafe_allow_html=True,
        )
        if analysed and indexed < analysed:
            st.markdown(
                f'<div class="caption" style="margin-top:.5rem;color:{theme.WARNING}">'
                f"{analysed - indexed} company(s) are analysed but not indexed, so the AI "
                f"tabs cannot answer for them.<br>Fix with: "
                f"<code>python -m src.rag.cli index --all</code></div>",
                unsafe_allow_html=True,
            )

        st.divider()
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown(
            f'<div class="caption" style="margin-top:.6rem">API v{health.get("version", "?")} '
            f"&middot; {API_BASE_URL}</div>",
            unsafe_allow_html=True,
        )
        return ticker, year


def tab_overview(data: dict, ticker: str, year: int, client: ApiClient) -> None:
    by_year = ratios_by_year(data["ratios"])
    flags = data["flags"]
    year_flags = [f for f in flags["flags"] if f["fiscal_year"] == year]

    st.markdown("### Key metrics")
    panels.kpi_row(by_year, year, OVERVIEW_KPIS)

    st.write("")
    left, right = st.columns([1.35, 1], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("### Performance")
            statements = data["statements"]["income_statements"]
            ordered = sorted(statements, key=lambda s: s["fiscal_year"])
            if ordered:
                st.plotly_chart(
                    charts.revenue_profit_chart(
                        [s["fiscal_year"] for s in ordered],
                        [s["values"].get("revenue") for s in ordered],
                        [s["values"].get("net_income") for s in ordered],
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            series = {name: {y: v.get(name) for y, v in by_year.items()} for name in TREND_RATIOS}
            figure = charts.multi_trend_chart(series, "Margin trend", height=260)
            if figure:
                st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            st.markdown("### Bankruptcy risk")
            try:
                distress = load_distress(ticker, year)
            except ApiError as exc:
                panels.notice(exc.message, kind="warn")
                distress = {"scores": []}

            altman = next(
                (s for s in distress.get("scores", []) if s["name"].startswith("Altman Z-Score")),
                None,
            )
            fallback = next(
                (s for s in distress.get("scores", []) if "book value" in s["name"]), None
            )
            shown = altman if (altman and altman["is_calculable"]) else fallback

            if shown and shown["is_calculable"]:
                st.plotly_chart(
                    charts.altman_gauge(shown["value"], shown.get("zone")),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.markdown(
                    f'<div style="text-align:center;margin-top:-.6rem">'
                    f'{theme.zone_badge(shown.get("zone"))}</div>'
                    f'<div class="caption" style="text-align:center;margin-top:.5rem">'
                    f'{shown["name"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                panels.notice(
                    (shown or {}).get("reason")
                    or distress.get("note")
                    or "No distress score available for this period."
                )

            piotroski = next(
                (s for s in distress.get("scores", []) if "Piotroski" in s["name"]), None
            )
            if piotroski and piotroski["is_calculable"]:
                st.markdown(
                    f'<div class="caption" style="margin-top:.8rem">Piotroski F-Score &middot; '
                    f'<strong style="color:{theme.TEXT}">{piotroski["value"]:.0f} of 9</strong> '
                    f'&middot; {piotroski.get("interpretation") or ""}</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    charts.piotroski_bar(piotroski["value"]),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            try:
                panels.model_probability_readout(load_distress_prediction(ticker), year)
            except ApiError:
                pass  # the model is optional; the rest of the panel stands alone

        with st.container(border=True):
            st.markdown("### Red flags")
            panels.severity_summary(
                {
                    s: sum(1 for f in year_flags if f["severity"] == s)
                    for s in ("High", "Medium", "Low", "Info")
                }
            )
            st.write("")
            panels.red_flag_cards(year_flags, limit=3)
            if len(year_flags) > 3:
                st.markdown(
                    f'<div class="caption">{len(year_flags) - 3} more in the Risk tab.</div>',
                    unsafe_allow_html=True,
                )


def tab_statements(data: dict) -> None:
    currency = data["profile"].get("currency") or ""
    statements = data["statements"]
    labels = {
        "Income statement": statements["income_statements"],
        "Balance sheet": statements["balance_sheets"],
        "Cash flow": statements["cash_flow_statements"],
    }
    choice = st.radio("Statement", list(labels), horizontal=True, label_visibility="collapsed")
    st.write("")
    panels.statement_table(labels[choice], currency)


def tab_ratios(data: dict, ticker: str, year: int, client: ApiClient) -> None:
    payload = data["ratios"]
    by_year = ratios_by_year(payload)
    period = next((p for p in payload["periods"] if p["fiscal_year"] == year), None)
    if period is None:
        panels.notice("No ratios on file for this period.")
        return

    categories = [c for c in payload.get("categories", []) if c != "distress"]
    choice = st.selectbox(
        "Category", ["All"] + [charts.humanise(c) for c in categories], index=0
    )
    selected = None if choice == "All" else categories[
        [charts.humanise(c) for c in categories].index(choice)
    ]

    ratios = [
        r for r in period["ratios"]
        if r["category"] != "distress" and (selected is None or r["category"] == selected)
    ]

    left, right = st.columns([1, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown(f"### FY{year} values")
            panels.ratio_table(ratios)
    with right:
        with st.container(border=True):
            st.markdown("### Trend")
            names = [r["name"] for r in ratios if r["is_calculable"]]
            if not names:
                panels.notice("Nothing calculable to chart in this category.")
            else:
                chosen = st.selectbox(
                    "Ratio", names, format_func=charts.humanise, label_visibility="collapsed"
                )
                figure = charts.trend_chart(
                    {y: v.get(chosen) for y, v in by_year.items()}, chosen, height=300
                )
                if figure:
                    st.plotly_chart(
                        figure, use_container_width=True, config={"displayModeBar": False}
                    )
                else:
                    panels.notice("Needs at least two periods to plot a trend.")


def tab_risk(data: dict, ticker: str, year: int) -> None:
    flags = data["flags"]
    st.markdown("### All red flags")
    panels.severity_summary(flags.get("counts_by_severity", {}))
    st.markdown(
        '<div class="caption" style="margin:.6rem 0 1rem">Produced by a deterministic rule '
        "engine, not by a language model. Each flag lists the values that triggered it.</div>",
        unsafe_allow_html=True,
    )

    years = sorted({f["fiscal_year"] for f in flags["flags"]}, reverse=True)
    if years:
        chosen = st.selectbox(
            "Period", ["All periods"] + [f"FY{y}" for y in years], index=0
        )
        shown = (
            flags["flags"]
            if chosen == "All periods"
            else [f for f in flags["flags"] if f"FY{f['fiscal_year']}" == chosen]
        )
    else:
        shown = []
    panels.red_flag_cards(shown)

    st.write("")
    with st.container(border=True):
        st.markdown("### Model-estimated distress probability")
        try:
            panels.distress_model_section(load_distress_prediction(ticker), year)
        except ApiError as exc:
            panels.notice(exc.message, kind="warn")

    st.write("")
    with st.container(border=True):
        st.markdown("### Distress scores")
        try:
            distress = load_distress(ticker, year)
        except ApiError as exc:
            panels.notice(exc.message, kind="warn")
            return
        if distress.get("note"):
            panels.notice(distress["note"])
            st.write("")
        for score in distress.get("scores", []):
            if score["is_calculable"]:
                zone = f" &nbsp;{theme.zone_badge(score['zone'])}" if score.get("zone") else ""
                st.markdown(
                    f'<div style="margin-bottom:.55rem">'
                    f'<strong style="color:{theme.TEXT}">{score["name"]}</strong>: '
                    f'<span style="color:{theme.ACCENT};font-weight:650">{score["value"]:.4f}</span>'
                    f"{zone}"
                    f'<div class="caption">{score.get("interpretation") or ""}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="margin-bottom:.55rem">'
                    f'<strong style="color:{theme.TEXT_MUTED}">{score["name"]}</strong>: '
                    f'<span style="color:{theme.TEXT_FAINT}">Not available</span>'
                    f'<div class="caption">{score.get("reason") or ""}</div></div>',
                    unsafe_allow_html=True,
                )


def tab_summary(ticker: str, year: int, client: ApiClient, llm_ready: bool) -> None:
    st.markdown("### AI executive summary")
    st.markdown(
        '<div class="caption" style="margin-bottom:1rem">Generated only from verified figures. '
        "Every number is checked against the source data after generation; the groundedness "
        "score below reports how many were traceable.</div>",
        unsafe_allow_html=True,
    )
    if not llm_ready:
        panels.notice(
            "No language model is configured, so summaries are unavailable. Add "
            "<code>GROQ_API_KEY</code> to <code>.env</code> and restart the API. Every other "
            "tab works without it.",
            kind="warn",
        )
        return

    left, right = st.columns([3, 1])
    with right:
        regenerate = st.button("Regenerate", use_container_width=True)

    key = f"summary::{ticker}::{year}"
    if regenerate or key not in st.session_state:
        with st.spinner("Retrieving verified facts and generating..."):
            try:
                st.session_state[key] = client.summary(
                    ticker, fiscal_year=year, refresh=regenerate
                )
            except ApiError as exc:
                panels.notice(f"{exc.message}{' ' + exc.hint if exc.hint else ''}", kind="warn")
                return

    answer = st.session_state[key]
    with st.container(border=True):
        st.markdown(answer["answer"])
    st.write("")
    panels.verification_footer(answer)
    panels.sources_expander(answer.get("sources", []))


def tab_ask(ticker: str, year: int, client: ApiClient, llm_ready: bool) -> None:
    st.markdown("### Ask about this company")
    st.markdown(
        '<div class="caption" style="margin-bottom:1rem">Answers come only from this company\'s '
        "filed statements and computed ratios. If the data does not contain an answer, the "
        "assistant says so rather than estimating.</div>",
        unsafe_allow_html=True,
    )
    if not llm_ready:
        panels.notice(
            "No language model is configured. Add <code>GROQ_API_KEY</code> to "
            "<code>.env</code> and restart the API.",
            kind="warn",
        )
        return

    all_periods = st.toggle(
        f"Search all fiscal years (default: FY{year} only)",
        value=False,
        key=f"scope::{ticker}",
        help=(
            "Off, retrieval is limited to the fiscal year selected in the sidebar, so the "
            "answer matches the rest of the page. On, every period on file is searched - "
            "use it for trend questions."
        ),
    )
    scope_year = None if all_periods else year

    history_key = f"chat::{ticker}"
    history = st.session_state.setdefault(history_key, [])

    if not history:
        st.markdown('<div class="caption">Try one of these:</div>', unsafe_allow_html=True)
        suggestions = [
            f"What was the net margin in FY{year}?",
            "Is this company at risk of bankruptcy?",
            "What data is missing or not calculable?",
        ]
        for column, prompt in zip(st.columns(len(suggestions)), suggestions):
            if column.button(prompt, use_container_width=True, key=f"sugg::{prompt}"):
                st.session_state[f"pending::{ticker}"] = prompt
                st.rerun()
        st.write("")

    for entry in history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("answer"):
                panels.verification_footer(entry["answer"])
                panels.sources_expander(entry["answer"].get("sources", []))

    question = st.chat_input("Ask about revenue, margins, leverage, risk...")
    pending = st.session_state.pop(f"pending::{ticker}", None)
    question = question or pending

    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking the source data..."):
                try:
                    answer = client.ask(ticker, question, fiscal_year=scope_year)
                except ApiError as exc:
                    st.error(exc.message)
                    return
            st.markdown(answer["answer"])
            panels.verification_footer(answer)
            panels.sources_expander(answer.get("sources", []))
        history.append(
            {"role": "assistant", "content": answer["answer"], "answer": answer}
        )


def main() -> None:
    client = get_client()
    try:
        health = client.health()
    except ApiError:
        health = None

    ticker, year = render_sidebar(health)

    if health is None:
        st.markdown("# Financial Analytics Platform")
        panels.notice(
            "The dashboard is a thin client over the API and cannot show data without it. "
            f"Start the backend with <code>uvicorn src.api.main:app --reload</code>, then "
            f"reload this page. Expected at <code>{API_BASE_URL}</code>.",
            kind="warn",
        )
        return

    if not ticker:
        st.markdown("# Financial Analytics Platform")
        panels.notice(
            "No company is selected. Ingest one with "
            "<code>python -m src.ingestion.cli AAPL</code>, then use the sidebar."
        )
        return

    try:
        data = load_company_data(ticker)
    except ApiError as exc:
        st.error(exc.message)
        return

    panels.masthead(data["profile"])
    if year is None:
        panels.notice("No fiscal periods on file for this company.")
        return

    tabs = st.tabs(
        ["Overview", "Statements", "Ratio analytics", "Risk & red flags", "AI summary", "Ask AI"]
    )
    llm_ready = bool(health.get("llm_configured"))

    with tabs[0]:
        tab_overview(data, ticker, year, client)
    with tabs[1]:
        tab_statements(data)
    with tabs[2]:
        tab_ratios(data, ticker, year, client)
    with tabs[3]:
        tab_risk(data, ticker, year)
    with tabs[4]:
        tab_summary(ticker, year, client, llm_ready)
    with tabs[5]:
        tab_ask(ticker, year, client, llm_ready)


if __name__ == "__main__":
    main()
