"""Turn stored financial data into small, labelled, retrievable text chunks.

This module is where the anti-hallucination guarantee is actually built. Three
rules govern every chunk:

1. **Missing fields are stated, not omitted.** A chunk that simply leaves out
   `interest_expense` invites the model to fill the gap. A chunk that says
   "NOT AVAILABLE (not reported by the data source): interest_expense" gives it
   the words to refuse with.
2. **Every number carries its period and its label.** "Revenue: 391,035,000,000
   (391.04B)" inside a chunk headed "AAPL FY2024 - Income Statement" cannot be
   misattributed to another year.
3. **Every chunk records the machine-readable values behind its prose** in
   `source_values`. The guardrail checks the model's output against exactly
   these numbers, so verification never depends on re-parsing the text.

Chunks are derived from the database rather than from Layer 2's in-memory
objects, so what the LLM sees is precisely what was persisted and what the API
would serve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ratios.base import humanise  # re-exported: ratio-name display formatting
from src.db.models import (
    BalanceSheet,
    CashFlowStatement,
    Company,
    IncomeStatement,
    Ratio,
    RedFlag,
)

logger = logging.getLogger(__name__)

# Chunk types, used for retrieval filtering and for the evaluation set.
PROFILE = "profile"
INCOME_STATEMENT = "income_statement"
BALANCE_SHEET = "balance_sheet"
CASH_FLOW = "cash_flow"
RATIOS = "ratios"
DISTRESS = "distress"
RED_FLAG = "red_flag"
DATA_AVAILABILITY = "data_availability"


@dataclass
class Chunk:
    """One retrievable unit of verified financial fact."""

    chunk_id: str
    text: str
    ticker: str
    chunk_type: str
    period: str = "FY"
    fiscal_year: int | None = None
    category: str | None = None
    # Canonical name -> value for every number appearing in `text`. The guardrail
    # verifies the model's output against this, never against the prose.
    source_values: dict[str, float] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Flat metadata for the vector store (Chroma allows scalars only)."""
        return {
            "ticker": self.ticker,
            "chunk_type": self.chunk_type,
            "period": self.period,
            "fiscal_year": self.fiscal_year if self.fiscal_year is not None else -1,
            "category": self.category or "",
            "source_values_json": _compact_json(self.source_values),
        }

    @property
    def period_label(self) -> str:
        if self.fiscal_year is None:
            return "all periods"
        return f"FY{self.fiscal_year}" if self.period == "FY" else f"{self.period} {self.fiscal_year}"


def _compact_json(values: dict[str, float]) -> str:
    import json

    return json.dumps({k: v for k, v in values.items() if v is not None}, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

# Ratios most naturally read as percentages; rendered both ways so the model can
# quote either and the guardrail recognises both.
PERCENT_RATIOS = {
    "gross_margin",
    "operating_margin",
    "net_margin",
    "return_on_equity",
    "return_on_assets",
    "return_on_invested_capital",
    "free_cash_flow_margin",
    "operating_cash_flow_margin",
    "debt_to_assets",
    "liabilities_to_assets",
    "capex_to_operating_cash_flow",
}

# Ratio values that are currency amounts rather than multiples.
MONETARY_RATIOS = {"working_capital", "free_cash_flow", "enterprise_value"}


def format_money(value: float) -> str:
    """Exact figure plus a human-scale abbreviation, e.g. '391,035,000,000 (391.04B)'."""
    exact = f"{value:,.0f}"
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{exact} ({value / 1e9:,.2f}B)"
    if magnitude >= 1e6:
        return f"{exact} ({value / 1e6:,.2f}M)"
    return exact


def format_ratio(name: str, value: float) -> str:
    if name in MONETARY_RATIOS:
        return format_money(value)
    if name in PERCENT_RATIOS:
        return f"{value:.4f} ({value * 100:.2f}%)"
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Statement field selections
# ---------------------------------------------------------------------------

INCOME_FIELDS = [
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "research_and_development",
    "selling_general_admin",
    "operating_expenses",
    "operating_income",
    "ebitda",
    "depreciation_amortization",
    "interest_expense",
    "income_before_tax",
    "income_tax_expense",
    "net_income",
]
INCOME_NON_MONETARY = {"eps", "eps_diluted"}

BALANCE_FIELDS = [
    "cash_and_equivalents",
    "short_term_investments",
    "net_receivables",
    "inventory",
    "total_current_assets",
    "property_plant_equipment_net",
    "goodwill",
    "intangible_assets",
    "total_assets",
    "accounts_payable",
    "short_term_debt",
    "total_current_liabilities",
    "long_term_debt",
    "total_liabilities",
    "retained_earnings",
    "total_equity",
    "total_debt",
    "net_debt",
]

CASH_FLOW_FIELDS = [
    "net_income",
    "depreciation_amortization",
    "stock_based_compensation",
    "change_in_working_capital",
    "operating_cash_flow",
    "capital_expenditure",
    "investing_cash_flow",
    "dividends_paid",
    "common_stock_repurchased",
    "financing_cash_flow",
    "net_change_in_cash",
    "free_cash_flow",
]


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------


def _statement_chunk(
    row: Any,
    ticker: str,
    chunk_type: str,
    title: str,
    fields: list[str],
    non_monetary: set[str] | None = None,
) -> Chunk:
    """Render one statement, naming both what is reported and what is not."""
    non_monetary = non_monetary or set()
    period_label = f"FY{row.fiscal_year}" if row.period == "FY" else f"{row.period} {row.fiscal_year}"

    header = f"{ticker} {period_label} - {title}"
    context = []
    if row.period_end_date:
        context.append(f"period ended {row.period_end_date.isoformat()}")
    if row.reported_currency:
        context.append(f"reported in {row.reported_currency}")
    if context:
        header += f" ({', '.join(context)})"

    lines, values, missing = [header + "."], {}, []
    for name in fields:
        value = getattr(row, name, None)
        if value is None:
            missing.append(name)
            continue
        rendered = f"{value:,.2f}" if name in non_monetary else format_money(value)
        lines.append(f"{humanise(name)}: {rendered}.")
        values[name] = float(value)

    for name in sorted(non_monetary):
        value = getattr(row, name, None)
        if value is None:
            if name not in missing:
                missing.append(name)
            continue
        lines.append(f"{humanise(name)}: {value:,.2f}.")
        values[name] = float(value)

    if missing:
        reasons = row.is_missing_json or {}
        detail = ", ".join(
            f"{name} ({(reasons.get(name) or 'not reported').replace('_', ' ')})"
            for name in sorted(missing)
        )
        lines.append(
            f"NOT AVAILABLE - the following fields were not provided by the data source "
            f"and have not been estimated: {detail}."
        )
    else:
        lines.append("All fields listed above were reported by the data source.")

    return Chunk(
        chunk_id=f"{ticker}:{row.period}{row.fiscal_year}:{chunk_type}",
        text=" ".join(lines),
        ticker=ticker,
        chunk_type=chunk_type,
        period=row.period,
        fiscal_year=row.fiscal_year,
        source_values=values,
    )


def _profile_chunk(company: Company, years: list[int]) -> Chunk:
    parts = [f"{company.ticker} - Company Profile."]
    for label, value in (
        ("Name", company.name),
        ("Sector", company.sector),
        ("Industry", company.industry),
        ("Exchange", company.exchange),
        ("Country", company.country),
        ("Reporting currency", company.currency),
    ):
        parts.append(f"{label}: {value}." if value else f"{label}: NOT AVAILABLE.")

    if company.is_financial_sector:
        parts.append(
            "This company is classified in the financial sector. The Altman Z-Score and "
            "Z''-Score are NOT APPLICABLE to banks and insurers and are not reported for it, "
            "and several red-flag rules are deliberately not applied."
        )
    else:
        parts.append("Altman Z-Score models are applicable to this company (non-financial sector).")

    parts.append(
        f"Fiscal years on file: {', '.join(f'FY{y}' for y in sorted(years))}."
        if years
        else "No fiscal years on file."
    )
    return Chunk(
        chunk_id=f"{company.ticker}:profile",
        text=" ".join(parts),
        ticker=company.ticker,
        chunk_type=PROFILE,
        fiscal_year=None,
    )


def _ratio_chunks(ticker: str, period: str, year: int, rows: list[Ratio]) -> list[Chunk]:
    """One chunk per ratio category, plus a dedicated distress-score chunk."""
    chunks: list[Chunk] = []
    by_category: dict[str, list[Ratio]] = {}
    for row in rows:
        by_category.setdefault(row.category or "other", []).append(row)

    period_label = f"FY{year}" if period == "FY" else f"{period} {year}"

    for category, category_rows in sorted(by_category.items()):
        if category == "distress":
            chunks.append(_distress_chunk(ticker, period, year, category_rows))
            continue

        lines = [f"{ticker} {period_label} - {humanise(category)} Ratios."]
        values, uncalculable = {}, []
        for row in sorted(category_rows, key=lambda r: r.ratio_name):
            if row.is_calculable and row.ratio_value is not None:
                lines.append(
                    f"{humanise(row.ratio_name)}: {format_ratio(row.ratio_name, row.ratio_value)}."
                )
                values[row.ratio_name] = float(row.ratio_value)
                if row.method:
                    lines[-1] = lines[-1][:-1] + f" (basis: {row.method})."
            else:
                uncalculable.append((row.ratio_name, row.reason))

        if uncalculable:
            lines.append(
                "NOT AVAILABLE - the following could not be calculated and were not estimated: "
                + " ".join(f"{humanise(n)} - {r}." for n, r in sorted(uncalculable))
            )
        else:
            lines.append("Every ratio in this category was calculable for this period.")

        chunks.append(
            Chunk(
                chunk_id=f"{ticker}:{period}{year}:ratios:{category}",
                text=" ".join(lines),
                ticker=ticker,
                chunk_type=RATIOS,
                period=period,
                fiscal_year=year,
                category=category,
                source_values=values,
            )
        )
    return chunks


def _distress_chunk(ticker: str, period: str, year: int, rows: list[Ratio]) -> Chunk:
    """Distress scores with their zones, components and interpretations."""
    period_label = f"FY{year}" if period == "FY" else f"{period} {year}"
    lines = [f"{ticker} {period_label} - Distress and Earnings-Quality Scores."]
    values: dict[str, float] = {}
    by_name = {row.ratio_name: row for row in rows}

    for name, label in (
        ("altman_z_score", "Altman Z-Score"),
        ("altman_z_double_prime_score", "Altman Z''-Score (book-value variant)"),
        ("piotroski_f_score", "Piotroski F-Score"),
        ("beneish_m_score", "Beneish M-Score"),
    ):
        row = by_name.get(name)
        if row is None:
            continue
        if not row.is_calculable or row.ratio_value is None:
            lines.append(f"{label}: NOT AVAILABLE - {row.reason}")
            continue

        details = row.details_json or {}
        values[name] = float(row.ratio_value)

        if name.startswith("altman"):
            zone = details.get("zone", "unknown")
            thresholds = details.get("thresholds", {})
            lines.append(
                f"{label}: {row.ratio_value:.4f} - {zone} Zone "
                f"(Safe above {thresholds.get('safe_above')}, "
                f"Distress below {thresholds.get('distress_below')})."
            )
            components = ", ".join(f"{humanise(k)} {v:.4f}" for k, v in (row.inputs_json or {}).items())
            if components:
                lines.append(f"{label} components: {components}.")
        elif name == "piotroski_f_score":
            signals = details.get("signals", {})
            passed = [humanise(k) for k, v in signals.items() if v]
            failed = [humanise(k) for k, v in signals.items() if not v]
            lines.append(
                f"{label}: {row.ratio_value:.0f} out of {details.get('max_score', 9)} - "
                f"{details.get('interpretation', 'n/a')}."
            )
            lines.append(f"Signals passed: {', '.join(passed) or 'none'}.")
            lines.append(f"Signals failed: {', '.join(failed) or 'none'}.")
        else:
            lines.append(
                f"{label}: {row.ratio_value:.4f}. Threshold for elevated manipulation risk is "
                f"{details.get('threshold', -1.78)}; this score is "
                f"{'ABOVE' if details.get('flagged') else 'below'} it "
                f"({details.get('interpretation', 'n/a')}). The M-Score is a screening "
                f"indicator, not evidence of manipulation."
            )
            indices = ", ".join(f"{k.upper()} {v:.4f}" for k, v in (row.inputs_json or {}).items())
            if indices:
                lines.append(f"M-Score indices: {indices}.")

    return Chunk(
        chunk_id=f"{ticker}:{period}{year}:distress",
        text=" ".join(lines),
        ticker=ticker,
        chunk_type=DISTRESS,
        period=period,
        fiscal_year=year,
        category="distress",
        source_values=values,
    )


def _red_flag_chunks(ticker: str, period: str, year: int, rows: list[RedFlag]) -> list[Chunk]:
    """One chunk per flag, plus an explicit 'none raised' chunk when there are none.

    The empty case matters: without it, "were there any red flags in FY2021?"
    retrieves nothing and the model has no grounded basis for answering "no".
    """
    period_label = f"FY{year}" if period == "FY" else f"{period} {year}"

    risk_flags = [r for r in rows if r.severity != "Info"]
    if not risk_flags:
        text = (
            f"{ticker} {period_label} - Red Flags. No risk red flags were raised for this period "
            f"by the rule-based detector. This is a computed result, not an absence of data."
        )
        informational = [r for r in rows if r.severity == "Info"]
        if informational:
            text += " Informational notices for this period: " + "; ".join(
                f"{r.flag_name} - {r.explanation}" for r in informational
            )
        return [
            Chunk(
                chunk_id=f"{ticker}:{period}{year}:redflags:none",
                text=text,
                ticker=ticker,
                chunk_type=RED_FLAG,
                period=period,
                fiscal_year=year,
                category="none",
            )
        ]

    chunks = []
    for index, row in enumerate(rows):
        payload = row.source_json or {}
        values = {
            k: float(v)
            for k, v in (payload.get("source_values") or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        text = (
            f"{ticker} {period_label} - Red Flag [{row.severity} severity] {row.flag_name} "
            f"({payload.get('category', 'Uncategorised')}). {row.explanation}"
        )
        if values:
            text += " Source values: " + ", ".join(
                f"{humanise(k)} = {v:,.4f}" for k, v in sorted(values.items())
            ) + "."
        chunks.append(
            Chunk(
                chunk_id=f"{ticker}:{period}{year}:redflag:{index}:{row.flag_name[:40]}",
                text=text,
                ticker=ticker,
                chunk_type=RED_FLAG,
                period=period,
                fiscal_year=year,
                category=row.severity,
                source_values=values,
            )
        )
    return chunks


def _availability_chunk(
    ticker: str,
    period: str,
    year: int,
    statements: dict[str, Any],
    ratio_rows: list[Ratio],
) -> Chunk:
    """A dedicated chunk about what is *not* known for this period."""
    period_label = f"FY{year}" if period == "FY" else f"{period} {year}"
    lines = [f"{ticker} {period_label} - Data Availability and Limitations."]

    any_gap = False
    for label, row in statements.items():
        if row is None:
            lines.append(f"The {label.replace('_', ' ')} is NOT AVAILABLE for {period_label}.")
            any_gap = True
            continue
        missing = sorted((row.is_missing_json or {}).keys())
        if missing:
            any_gap = True
            lines.append(
                f"{label.replace('_', ' ').capitalize()} fields NOT AVAILABLE: {', '.join(missing)}."
            )

    uncalculable = [r for r in ratio_rows if not r.is_calculable]
    if uncalculable:
        any_gap = True
        lines.append(
            "Ratios that could not be calculated for this period: "
            + " ".join(f"{humanise(r.ratio_name)} - {r.reason}." for r in sorted(uncalculable, key=lambda r: r.ratio_name))
        )

    if not any_gap:
        lines.append(
            "All statement fields and all ratios were available for this period; no gaps."
        )
    lines.append(
        "Any figure not listed in the available data must be reported as NOT AVAILABLE "
        "rather than estimated."
    )

    return Chunk(
        chunk_id=f"{ticker}:{period}{year}:availability",
        text=" ".join(lines),
        ticker=ticker,
        chunk_type=DATA_AVAILABILITY,
        period=period,
        fiscal_year=year,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_chunks(session: Session, ticker: str, period: str = "FY") -> list[Chunk]:
    """Every chunk for one company, built from what the database actually holds."""
    ticker = ticker.strip().upper()
    company = session.scalar(select(Company).where(Company.ticker == ticker))
    if company is None:
        logger.warning("No company row for %s; nothing to chunk", ticker)
        return []

    def by_year(model) -> dict[int, Any]:
        return {
            row.fiscal_year: row
            for row in session.scalars(
                select(model).where(model.company_id == company.id, model.period == period)
            )
        }

    income, balance, cash = by_year(IncomeStatement), by_year(BalanceSheet), by_year(CashFlowStatement)

    ratios_by_year: dict[int, list[Ratio]] = {}
    for row in session.scalars(
        select(Ratio).where(Ratio.company_id == company.id, Ratio.period == period)
    ):
        ratios_by_year.setdefault(row.fiscal_year, []).append(row)

    flags_by_year: dict[int, list[RedFlag]] = {}
    for row in session.scalars(
        select(RedFlag).where(RedFlag.company_id == company.id, RedFlag.period == period)
    ):
        flags_by_year.setdefault(row.fiscal_year, []).append(row)

    years = sorted(set(income) | set(balance) | set(cash) | set(ratios_by_year))
    chunks: list[Chunk] = [_profile_chunk(company, years)]

    for year in years:
        if income.get(year) is not None:
            chunks.append(
                _statement_chunk(
                    income[year], ticker, INCOME_STATEMENT, "Income Statement",
                    INCOME_FIELDS, INCOME_NON_MONETARY,
                )
            )
        if balance.get(year) is not None:
            chunks.append(
                _statement_chunk(balance[year], ticker, BALANCE_SHEET, "Balance Sheet", BALANCE_FIELDS)
            )
        if cash.get(year) is not None:
            chunks.append(
                _statement_chunk(cash[year], ticker, CASH_FLOW, "Cash Flow Statement", CASH_FLOW_FIELDS)
            )

        ratio_rows = ratios_by_year.get(year, [])
        if ratio_rows:
            chunks.extend(_ratio_chunks(ticker, period, year, ratio_rows))

        if year in flags_by_year or ratio_rows:
            chunks.extend(_red_flag_chunks(ticker, period, year, flags_by_year.get(year, [])))

        chunks.append(
            _availability_chunk(
                ticker,
                period,
                year,
                {
                    "income_statement": income.get(year),
                    "balance_sheet": balance.get(year),
                    "cash_flow_statement": cash.get(year),
                },
                ratio_rows,
            )
        )

    logger.info("Built %d chunks for %s across %d periods", len(chunks), ticker, len(years))
    return chunks


def collect_source_values(chunks: Iterable[Chunk]) -> dict[str, float]:
    """Union of every verified number across chunks, keyed by period and name.

    This is the allow-list the guardrail checks generated numbers against.
    """
    values: dict[str, float] = {}
    for chunk in chunks:
        prefix = f"{chunk.period_label}:{chunk.category or chunk.chunk_type}"
        for name, value in chunk.source_values.items():
            values[f"{prefix}:{name}"] = value
    return values
