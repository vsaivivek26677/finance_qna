"""Prompt construction and post-generation verification.

Two defences, applied before and after the model runs.

**Before:** a system prompt that forbids arithmetic and estimation, and a context
block assembled only from verified chunks with explicit period labels.

**After:** every number in the generated answer is checked against the numbers
that were actually in the retrieved context. This is the part that catches a
plausible-sounding fabricated figure, which is the failure mode that matters in
financial Q&A - a wrong revenue number reads exactly like a right one.

Verification is precision-aware rather than exact-match. If the model writes
"391.04B" and the source says 391,035,000,000, that is a correct rounding of a
real number and is accepted. If it writes "395B", nothing in context rounds to
it and the mention is reported as unverified.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from src.rag.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a financial analysis assistant. You answer strictly from the CONTEXT provided to you, which contains verified figures computed from a company's filed financial statements.

Rules you must follow without exception:

1. Use ONLY numbers that appear in the CONTEXT. Never estimate, infer, interpolate, or recall a figure from memory.
2. Do not perform arithmetic to produce new figures. If a number is not in the CONTEXT, it is not available to you.
3. If the CONTEXT marks something "NOT AVAILABLE" or "NOT CALCULABLE", say so explicitly and give the stated reason. Never fill the gap.
4. If the CONTEXT does not contain what is needed to answer, say plainly that the data is not available. An incomplete answer is correct; a fabricated one is not.
5. Tag every figure with its period, e.g. "revenue was 391,035,000,000 (FY2024 income statement)".
6. Distress scores, red flags and ratios in the CONTEXT are pre-computed by a deterministic rule engine. Report them as given. Never invent a red flag, a score, or a risk assessment of your own.
7. Be concise and factual. No investment advice, no price targets, no recommendations to buy, sell or hold.

If asked about a period or company not present in the CONTEXT, say that it is not in the available data."""


SUMMARY_INSTRUCTION = """Write a concise executive summary of this company's financial position using ONLY the CONTEXT.

Structure it as:
- **Financial performance** - revenue, margins and profitability, with periods named.
- **Financial position** - liquidity, leverage and solvency.
- **Risk assessment** - the distress scores and any red flags exactly as given, including their severity.
- **Data limitations** - anything the CONTEXT marks NOT AVAILABLE or NOT CALCULABLE.

Every figure must carry its period. If a section has no supporting data in the CONTEXT, say so rather than omitting the section."""


# Matches an optional currency sign, digits with separators, an optional
# fraction, and an optional percent or magnitude suffix.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                       # not mid-word or mid-decimal
    (?P<sign>-|−|\()?           # minus, unicode minus, or accounting paren
    \$?\s?
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)  # grouped or plain integer part
    (?:\.(?P<frac>\d+))?
    \s?
    (?P<suffix>%|percent|bn|billion|B|mn|million|M|K|thousand|x)?
    \)?
    (?![\w])
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Period citations ("FY2024", "Q3 2024"). These are checked separately from
# numeric claims: a figure attributed to a period that was never retrieved is a
# distinct failure mode from a wrong figure, and the digits in "FY2024" are a
# label rather than a financial quantity.
_PERIOD_RE = re.compile(r"\bFY\s?(\d{4})\b|\b(Q[1-4])\s+(\d{4})\b", re.IGNORECASE)

_SCALES: dict[str, float] = {
    "%": 0.01,
    "percent": 0.01,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "k": 1e3,
    "thousand": 1e3,
    "x": 1.0,
}


@dataclass
class NumberMention:
    """One numeric claim found in generated text."""

    raw: str
    value: float  # scaled to absolute units (a percent becomes its decimal form)
    base: float  # as written, before the suffix scale
    scale: float
    decimals: int
    start: int
    end: int

    def tolerance(self) -> float:
        """Half a unit of the last written digit - what rounding can account for."""
        return 0.5 * (10.0**-self.decimals) * abs(self.scale) * 1.0001


@dataclass
class VerificationReport:
    """Outcome of checking a generated answer against its context."""

    mentions: list[NumberMention] = field(default_factory=list)
    verified: list[NumberMention] = field(default_factory=list)
    unverified: list[NumberMention] = field(default_factory=list)
    # Periods the answer cited that were not present in the retrieved context.
    unverified_periods: list[str] = field(default_factory=list)
    blocked: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.mentions)

    @property
    def groundedness(self) -> float:
        """Share of numeric claims traceable to the context. 1.0 when none were made."""
        if not self.mentions:
            return 1.0
        return len(self.verified) / len(self.mentions)

    @property
    def is_grounded(self) -> bool:
        return not self.unverified and not self.unverified_periods

    def summary(self) -> str:
        if not self.mentions:
            return "No numeric claims made."
        return (
            f"{len(self.verified)}/{len(self.mentions)} numeric claims verified against context "
            f"({self.groundedness:.0%} grounded)."
        )

    def to_json(self) -> dict:
        return {
            "total_numbers": self.total,
            "verified": len(self.verified),
            "unverified": [m.raw for m in self.unverified],
            "unverified_periods": list(self.unverified_periods),
            "groundedness": round(self.groundedness, 4),
            "blocked": self.blocked,
            "notes": self.notes,
        }


def extract_numbers(text: str) -> list[NumberMention]:
    """Every numeric claim in `text`, normalised to absolute units."""
    mentions = []
    for match in _NUMBER_RE.finditer(text):
        integer_part = match.group("int").replace(",", "")
        fraction = match.group("frac") or ""
        try:
            base = float(f"{integer_part}.{fraction}" if fraction else integer_part)
        except ValueError:  # pragma: no cover - regex guarantees digits
            continue

        sign = match.group("sign")
        if sign in {"-", "−"} or (sign == "(" and match.group(0).rstrip().endswith(")")):
            base = -base

        suffix = (match.group("suffix") or "").lower()
        scale = _SCALES.get(suffix, 1.0)
        mentions.append(
            NumberMention(
                raw=match.group(0).strip(),
                value=base * scale,
                base=base,
                scale=scale,
                decimals=len(fraction),
                start=match.start(),
                end=match.end(),
            )
        )
    return mentions


def _appears_verbatim(mention: NumberMention, context: str) -> bool:
    """True when the exact digits the model wrote are present in the context.

    Word boundaries matter: without them "1" would match inside "2021" and every
    small number would verify trivially.
    """
    written = mention.raw.lstrip("$(-− ").rstrip(")% ").strip()
    written = re.sub(r"\s*(bn|billion|mn|million|thousand|percent|[BMKx])$", "", written, flags=re.I)
    if not written:
        return False
    pattern = rf"(?<![\d.]){re.escape(written)}(?![\d])"
    return re.search(pattern, context) is not None


def _consistent_with(mention: NumberMention, allowed: float) -> bool:
    """Whether `allowed` could have been written the way the model wrote it."""
    if math.isclose(mention.value, allowed, rel_tol=1e-9, abs_tol=1e-12):
        return True
    if abs(mention.value - allowed) <= mention.tolerance():
        return True
    # A ratio quoted as a percentage without a % sign ("net margin of 23.97").
    # Bounded deliberately: without the guards, any fabricated figure that is
    # exactly 100x a real one would verify, which would be a hole in the check.
    if (
        mention.scale == 1.0
        and 0 < abs(mention.value) < 1000
        and abs(allowed) < 10
        and abs(mention.value / 100.0 - allowed) <= max(mention.tolerance() / 100.0, 1e-9)
    ):
        return True
    return False


def extract_periods(text: str) -> list[str]:
    """Every period label cited in `text`, normalised to 'FY2024' / 'Q3 2024'."""
    found = []
    for match in _PERIOD_RE.finditer(text):
        if match.group(1):
            found.append(f"FY{match.group(1)}")
        else:
            found.append(f"{match.group(2).upper()} {match.group(3)}")
    return found


def verify_periods(answer: str, chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Period labels cited by the answer that no retrieved chunk covers.

    Catches the model attributing a real figure to the wrong year - which reads
    as authoritative and is invisible to a purely numeric check.
    """
    available = {chunk.period_label for chunk in chunks}
    available.discard("all periods")
    if not available:
        return []
    return sorted({p for p in extract_periods(answer) if p not in available})


def collect_allowed_values(chunks: Iterable[RetrievedChunk]) -> dict[str, float]:
    """Every verified number carried by the retrieved chunks."""
    allowed: dict[str, float] = {}
    for chunk in chunks:
        for name, value in chunk.source_values().items():
            allowed[f"{chunk.period_label}:{chunk.chunk_type}:{name}"] = value
    return allowed


def verify_answer(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    block_on_unverified: bool = False,
) -> VerificationReport:
    """Check every number in `answer` against the retrieved context."""
    report = VerificationReport()
    context_text = "\n".join(chunk.text for chunk in chunks)
    allowed = collect_allowed_values(chunks)
    allowed_values = list(allowed.values())

    for mention in extract_numbers(answer):
        report.mentions.append(mention)
        if _appears_verbatim(mention, context_text):
            report.verified.append(mention)
            continue
        if any(_consistent_with(mention, value) for value in allowed_values):
            report.verified.append(mention)
            continue
        report.unverified.append(mention)

    report.unverified_periods = verify_periods(answer, chunks)
    if report.unverified_periods:
        report.notes.append(
            "Answer cited period(s) absent from the retrieved context: "
            + ", ".join(report.unverified_periods)
        )

    if report.unverified:
        tokens = ", ".join(sorted({m.raw for m in report.unverified}))
        report.notes.append(
            f"{len(report.unverified)} numeric claim(s) could not be traced to the retrieved "
            f"context: {tokens}"
        )
    if block_on_unverified and (report.unverified or report.unverified_periods):
        report.blocked = True
        report.notes.append("Answer withheld: RAG_BLOCK_ON_UNVERIFIED is enabled.")

    if not chunks:
        report.notes.append("No context was retrieved for this question.")

    return report


BLOCKED_MESSAGE = (
    "This answer was withheld because it contained figures that could not be verified "
    "against the source data. Unverified values: {tokens}. Nothing has been estimated. "
    "Ask about a specific metric and period to get a grounded answer."
)


def apply_guardrails(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    block_on_unverified: bool = False,
) -> tuple[str, VerificationReport]:
    """Verify an answer, replacing it when blocking is enabled and it fails."""
    report = verify_answer(answer, chunks, block_on_unverified=block_on_unverified)
    if report.blocked:
        tokens = ", ".join(
            sorted({m.raw for m in report.unverified} | set(report.unverified_periods))
        )
        return BLOCKED_MESSAGE.format(tokens=tokens), report
    return answer, report


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Number the chunks so the model can cite them and a reader can audit them."""
    if not chunks:
        return "(no context available)"
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{index}] ({chunk.citation()})\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    return (
        f"CONTEXT:\n{format_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the CONTEXT above. Name the period for every figure. "
        f"If the CONTEXT does not contain the answer, say so explicitly."
    )


def build_summary_prompt(ticker: str, chunks: Sequence[RetrievedChunk]) -> str:
    return (
        f"CONTEXT:\n{format_context(chunks)}\n\n"
        f"TASK: {SUMMARY_INSTRUCTION}\n\nCompany: {ticker}"
    )
