"""Design system: one palette, one type scale, one Plotly template.

Every colour and spacing decision lives here rather than being sprinkled across
the pages, so the dashboard reads as one product instead of a stack of charts
that happen to share a page.

The palette is deliberately restrained: a deep navy ground, a single teal accent
for interactive and positive-neutral elements, and colour used sparingly enough
that when something turns amber or red it actually means something. Severity
colours are the one place saturation is high, because they are the signal a
reader is scanning for.
"""

from __future__ import annotations

# --- Palette ---------------------------------------------------------------

INK = "#0B1220"        # page ground
SURFACE = "#131C2E"    # cards and panels
SURFACE_ALT = "#1B2740"  # table stripes, hover
BORDER = "#243350"
BORDER_SOFT = "#1C2840"

TEXT = "#E8EEF7"
TEXT_MUTED = "#93A4BF"
TEXT_FAINT = "#647393"

ACCENT = "#2DD4BF"     # teal - links, focus, primary series
ACCENT_DEEP = "#0F766E"
ACCENT_SOFT = "rgba(45, 212, 191, 0.12)"

POSITIVE = "#34D399"
NEGATIVE = "#F87171"
WARNING = "#FBBF24"
INFO = "#60A5FA"

SEVERITY = {
    "High": NEGATIVE,
    "Medium": WARNING,
    "Low": INFO,
    "Info": TEXT_FAINT,
}

ZONE = {
    "Safe": POSITIVE,
    "Grey": WARNING,
    "Distress": NEGATIVE,
}

# Categorical series colours, ordered for contrast against the navy ground.
SERIES = [ACCENT, "#818CF8", "#F472B6", "#FBBF24", "#34D399", "#60A5FA", "#FB923C"]

FONT_STACK = (
    '"Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif'
)


def plotly_layout(height: int = 320, showlegend: bool = False) -> dict:
    """Shared Plotly layout. Charts inherit the page, they do not fight it."""
    return {
        "height": height,
        "showlegend": showlegend,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": TEXT_MUTED, "family": FONT_STACK, "size": 12},
        "margin": {"l": 8, "r": 8, "t": 28, "b": 8},
        "hoverlabel": {
            "bgcolor": SURFACE_ALT,
            "bordercolor": BORDER,
            "font": {"color": TEXT, "family": FONT_STACK, "size": 12},
        },
        "xaxis": {
            "gridcolor": BORDER_SOFT,
            "zerolinecolor": BORDER,
            "linecolor": BORDER,
            "tickfont": {"color": TEXT_FAINT},
        },
        "yaxis": {
            "gridcolor": BORDER_SOFT,
            "zerolinecolor": BORDER,
            "linecolor": BORDER,
            "tickfont": {"color": TEXT_FAINT},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": TEXT_MUTED, "size": 11},
        },
    }


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}

.stApp {{ background: {INK}; }}

/* Tighten Streamlit's generous default padding without cramping the content. */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

h1, h2, h4 {{ color: {TEXT}; font-weight: 600; letter-spacing: -0.015em; }}
h1 {{ font-size: 1.65rem; }}
h2 {{ font-size: 1.2rem; margin-top: 0.4rem; }}

/* Section labels. Streamlit's default h3 is nearly as large as h1, which makes
   every panel heading shout; these need to read as quiet signposts instead. */
[data-testid="stMarkdownContainer"] h3 {{
    font-size: 0.76rem !important; font-weight: 700; color: {TEXT_FAINT};
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 0 0 0.85rem 0; padding: 0;
}}

/* --- Masthead --- */
.masthead {{
    display: flex; align-items: baseline; gap: 0.9rem; flex-wrap: wrap;
    padding-bottom: 0.9rem; margin-bottom: 1.1rem;
    border-bottom: 1px solid {BORDER};
}}
.masthead .ticker {{ font-size: 1.8rem; font-weight: 700; color: {TEXT}; letter-spacing: -0.02em; }}
.masthead .name {{ font-size: 1.02rem; color: {TEXT_MUTED}; }}
.masthead .chip {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    color: {ACCENT}; background: {ACCENT_SOFT}; border: 1px solid {ACCENT_DEEP};
    padding: 0.2rem 0.55rem; border-radius: 999px;
}}

/* --- Metric cards --- */
div[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 0.95rem 1.1rem;
}}
div[data-testid="stMetric"] label p {{
    color: {TEXT_FAINT} !important; font-size: 0.73rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT}; font-size: 1.5rem !important; font-weight: 650; letter-spacing: -0.02em;
}}
div[data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; font-weight: 600; }}

/* --- Bordered containers used as panels --- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE}; border-color: {BORDER} !important; border-radius: 12px;
}}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.15rem; border-bottom: 1px solid {BORDER}; background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    height: 42px; padding: 0 1.05rem; background: transparent; border-radius: 8px 8px 0 0;
    color: {TEXT_FAINT}; font-size: 0.88rem; font-weight: 550;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important; background: {ACCENT_SOFT};
    border-bottom: 2px solid {ACCENT};
}}

/* --- Severity and zone badges --- */
.badge {{
    display: inline-block; padding: 0.14rem 0.55rem; border-radius: 999px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
}}
.flag-card {{
    border-left: 3px solid {BORDER}; background: {SURFACE};
    border-radius: 0 10px 10px 0; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}}
.flag-card .title {{ font-weight: 620; color: {TEXT}; font-size: 0.95rem; }}
.flag-card .body {{ color: {TEXT_MUTED}; font-size: 0.86rem; line-height: 1.55; margin-top: 0.3rem; }}
.flag-card .meta {{ color: {TEXT_FAINT}; font-size: 0.75rem; margin-top: 0.4rem; }}

/* --- Data availability notice --- */
.notice {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {INFO};
    border-radius: 0 10px 10px 0; padding: 0.75rem 1rem;
    color: {TEXT_MUTED}; font-size: 0.85rem; line-height: 1.55;
}}
.notice.warn {{ border-left-color: {WARNING}; }}

/* --- Groundedness pill --- */
.pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 999px;
    padding: 0.3rem 0.75rem; font-size: 0.78rem; color: {TEXT_MUTED}; font-weight: 550;
}}
.pill .dot {{ width: 7px; height: 7px; border-radius: 50%; }}

/* --- Tables --- */
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {{
    background: {SURFACE}; border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {{ font-size: 0.95rem; }}

/* --- Buttons --- */
.stButton button {{
    border-radius: 9px; border: 1px solid {BORDER}; background: {SURFACE_ALT};
    color: {TEXT}; font-weight: 550; font-size: 0.86rem; transition: all .15s ease;
}}
.stButton button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

/* --- Chat --- */
div[data-testid="stChatMessage"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 0.9rem 1.1rem;
}}

/* --- Misc --- */
hr {{ border-color: {BORDER}; }}
.caption {{ color: {TEXT_FAINT}; font-size: 0.78rem; line-height: 1.5; }}
a {{ color: {ACCENT}; }}
#MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def badge(text: str, colour: str) -> str:
    """An inline pill, tinted from a single colour so it never overpowers."""
    return (
        f'<span class="badge" style="color:{colour};'
        f"background:{colour}1F;border:1px solid {colour}55;\">{text}</span>"
    )


def severity_badge(severity: str) -> str:
    return badge(severity, SEVERITY.get(severity, TEXT_FAINT))


def zone_badge(zone: str | None) -> str:
    if not zone:
        return badge("N/A", TEXT_FAINT)
    return badge(f"{zone} zone", ZONE.get(zone, TEXT_FAINT))


def groundedness_pill(score: float, verified: int, total: int) -> str:
    """Show how much of an answer was traceable, not just that it was checked."""
    colour = POSITIVE if score >= 0.999 else (WARNING if score >= 0.8 else NEGATIVE)
    label = "fully grounded" if score >= 0.999 else f"{score:.0%} grounded"
    detail = f"{verified}/{total} figures verified" if total else "no figures stated"
    return (
        f'<div class="pill"><span class="dot" style="background:{colour}"></span>'
        f"<strong style='color:{colour}'>{label}</strong>"
        f'<span style="color:{TEXT_FAINT}">&middot; {detail}</span></div>'
    )
