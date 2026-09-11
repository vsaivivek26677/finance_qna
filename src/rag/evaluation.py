"""Quantitative RAG evaluation against database ground truth.

The original plan called for `ragas`. Its published versions pin the pre-1.0
LangChain line, which conflicts with the LangChain already installed here, so it
was dropped rather than forcing an environment downgrade.

What replaced it is not a weaker substitute for this domain. `ragas` scores
faithfulness and answer relevancy by asking a second LLM to judge the first,
which is expensive, non-deterministic, and itself capable of hallucinating. Here
every question is generated *from the database*, so the correct answer, and the
chunk that should supply it, are both known exactly. That permits reference-based
metrics computed deterministically:

| Metric | Meaning | How it is computed |
|---|---|---|
| `faithfulness` | Numeric claims traceable to retrieved context | Guardrail verification (no LLM) |
| `answer_correctness` | The answer states the true value from the DB | Precision-aware numeric match |
| `context_recall` | The chunk holding the answer was retrieved | Known reference chunk id |
| `context_precision` | How highly that chunk ranked | Reciprocal rank |
| `refusal_correctness` | Says NOT AVAILABLE when the value genuinely is | Refusal-phrase detection |

`refusal_correctness` has no `ragas` equivalent and is the metric that matters
most here: the failure mode to prevent is inventing a figure that was never
reported.

Same run, same score. The only nondeterminism is the generating model itself.
"""

from __future__ import annotations

import logging
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Company, IncomeStatement, Ratio, RedFlag
from src.rag.chunker import humanise
from src.rag.guardrails import _consistent_with, extract_numbers
from src.rag.rag_pipeline import RagAnswer, RagPipeline
from src.rag.repository import log_rag_query

logger = logging.getLogger(__name__)

NUMERIC = "numeric"
REFUSAL = "refusal"
CATEGORICAL = "categorical"

_REFUSAL_PHRASES = (
    "not available",
    "not calculable",
    "not reported",
    "unavailable",
    "cannot be calculated",
    "could not be calculated",
    "does not contain",
    "is not in the",
    "no data",
    "not provided",
    "not applicable",
    "not present",
)


@dataclass
class EvalCase:
    """One question with an answer known from the database."""

    question: str
    ticker: str
    expected_kind: str
    reference_chunk_ids: set[str]
    fiscal_year: int | None = None
    expected_value: float | None = None
    expected_text: str | None = None
    metric_name: str = ""


@dataclass
class CaseResult:
    """Per-question scores plus the answer that produced them."""

    case: EvalCase
    answer: RagAnswer
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.scores.get("answer_correctness", 0.0) >= 1.0


@dataclass
class EvaluationReport:
    """Aggregate results over an evaluation set."""

    ticker: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    def mean(self, metric: str) -> float:
        values = [r.scores[metric] for r in self.results if metric in r.scores]
        return sum(values) / len(values) if values else 0.0

    def aggregate(self) -> dict[str, float]:
        metrics = sorted({m for r in self.results for m in r.scores})
        return {m: round(self.mean(m), 4) for m in metrics}

    def summary_line(self) -> str:
        scores = self.aggregate()
        passed = sum(1 for r in self.results if r.passed)
        return (
            f"{self.ticker}: {passed}/{self.total} answers correct | "
            f"faithfulness {scores.get('faithfulness', 0):.3f} | "
            f"context recall {scores.get('context_recall', 0):.3f} | "
            f"context precision {scores.get('context_precision', 0):.3f}"
        )


# ---------------------------------------------------------------------------
# Building the evaluation set from the database
# ---------------------------------------------------------------------------

# Ratios worth asking about: recognisable, and spread across categories.
_PROBE_RATIOS = [
    "net_margin",
    "gross_margin",
    "current_ratio",
    "return_on_equity",
    "debt_to_equity",
    "free_cash_flow_margin",
    "asset_turnover",
]


def build_eval_set(
    session: Session,
    ticker: str,
    period: str = "FY",
    max_years: int = 3,
    seed: int = 17,
) -> list[EvalCase]:
    """Generate questions whose correct answers come straight from the database."""
    ticker = ticker.strip().upper()
    company = session.scalar(select(Company).where(Company.ticker == ticker))
    if company is None:
        return []

    rng = random.Random(seed)
    ratios: dict[int, dict[str, Ratio]] = {}
    for row in session.scalars(
        select(Ratio).where(Ratio.company_id == company.id, Ratio.period == period)
    ):
        ratios.setdefault(row.fiscal_year, {})[row.ratio_name] = row

    income = {
        row.fiscal_year: row
        for row in session.scalars(
            select(IncomeStatement).where(
                IncomeStatement.company_id == company.id, IncomeStatement.period == period
            )
        )
    }
    flags: dict[int, list[RedFlag]] = {}
    for row in session.scalars(
        select(RedFlag).where(RedFlag.company_id == company.id, RedFlag.period == period)
    ):
        flags.setdefault(row.fiscal_year, []).append(row)

    years = sorted(ratios, reverse=True)[:max_years]
    cases: list[EvalCase] = []

    for year in years:
        year_ratios = ratios[year]

        # 1. Ratios that were computed - the answer is a specific number.
        for name in _PROBE_RATIOS:
            row = year_ratios.get(name)
            if row is None or not row.is_calculable or row.ratio_value is None:
                continue
            cases.append(
                EvalCase(
                    question=f"What was {ticker}'s {humanise(name)} in FY{year}?",
                    ticker=ticker,
                    fiscal_year=year,
                    expected_kind=NUMERIC,
                    expected_value=float(row.ratio_value),
                    reference_chunk_ids={f"{ticker}:{period}{year}:ratios:{row.category}"},
                    metric_name=name,
                )
            )

        # 2. Revenue and net income, straight off the income statement.
        statement = income.get(year)
        if statement is not None:
            for field_name in ("revenue", "net_income"):
                value = getattr(statement, field_name, None)
                if value is None:
                    continue
                cases.append(
                    EvalCase(
                        question=f"What was {ticker}'s {humanise(field_name)} in FY{year}?",
                        ticker=ticker,
                        fiscal_year=year,
                        expected_kind=NUMERIC,
                        expected_value=float(value),
                        reference_chunk_ids={f"{ticker}:{period}{year}:income_statement"},
                        metric_name=field_name,
                    )
                )

        # 3. The distress score and its zone - a number plus a label.
        z_row = year_ratios.get("altman_z_score") or year_ratios.get(
            "altman_z_double_prime_score"
        )
        if z_row is not None and z_row.is_calculable and z_row.ratio_value is not None:
            zone = (z_row.details_json or {}).get("zone")
            cases.append(
                EvalCase(
                    question=(
                        f"What was {ticker}'s Altman Z-Score in FY{year}, and which zone does "
                        f"it fall in?"
                    ),
                    ticker=ticker,
                    fiscal_year=year,
                    expected_kind=NUMERIC,
                    expected_value=float(z_row.ratio_value),
                    expected_text=zone,
                    reference_chunk_ids={f"{ticker}:{period}{year}:distress"},
                    metric_name="altman_z_score",
                )
            )

        # 4. Ratios that could NOT be computed - the correct answer is a refusal.
        #    These are the most valuable cases in the set.
        uncalculable = [r for r in year_ratios.values() if not r.is_calculable]
        if uncalculable:
            row = rng.choice(sorted(uncalculable, key=lambda r: r.ratio_name))
            cases.append(
                EvalCase(
                    question=f"What was {ticker}'s {humanise(row.ratio_name)} in FY{year}?",
                    ticker=ticker,
                    fiscal_year=year,
                    expected_kind=REFUSAL,
                    reference_chunk_ids={
                        f"{ticker}:{period}{year}:availability",
                        f"{ticker}:{period}{year}:ratios:{row.category}",
                        f"{ticker}:{period}{year}:distress",
                    },
                    metric_name=row.ratio_name,
                )
            )

        # 5. Red flags - present or explicitly absent.
        year_flags = [f for f in flags.get(year, []) if f.severity != "Info"]
        cases.append(
            EvalCase(
                question=f"Were any red flags raised for {ticker} in FY{year}?",
                ticker=ticker,
                fiscal_year=year,
                expected_kind=CATEGORICAL,
                expected_text=year_flags[0].flag_name if year_flags else "no red flags",
                reference_chunk_ids={
                    c
                    for c in (
                        f"{ticker}:{period}{year}:redflags:none",
                        *(
                            f"{ticker}:{period}{year}:redflag:{i}:{f.flag_name[:40]}"
                            for i, f in enumerate(flags.get(year, []))
                        ),
                    )
                },
                metric_name="red_flags",
            )
        )

    # 6. A question the data cannot answer, to check the model refuses rather
    #    than reaching for its pretraining.
    if years:
        cases.append(
            EvalCase(
                question=f"What was {ticker}'s revenue in FY1975?",
                ticker=ticker,
                fiscal_year=None,
                expected_kind=REFUSAL,
                reference_chunk_ids=set(),
                metric_name="out_of_range_period",
            )
        )

    logger.info("Built %d evaluation cases for %s", len(cases), ticker)
    return cases


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


# LLM output is full of typographic characters that are visually identical to
# ASCII but not equal to it: narrow no-break spaces inside numbers, curly quotes,
# en dashes. Comparing raw strings marks a correct answer wrong, so all text
# comparison goes through here first.
_TYPOGRAPHIC = str.maketrans(
    {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-",
    }
)


def normalise_text(text: str) -> str:
    """Fold typographic variants and whitespace, then lowercase."""
    # NFKC maps U+202F (narrow no-break space) and U+00A0 onto a plain space.
    folded = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHIC)
    return re.sub(r"\s+", " ", folded).strip().lower()


def looks_like_refusal(answer: str) -> bool:
    normalised = normalise_text(answer)
    return any(phrase in normalised for phrase in _REFUSAL_PHRASES)


def states_value(answer: str, expected: float) -> bool:
    """Whether the answer contains the expected number in any sane rendering."""
    return any(_consistent_with(mention, expected) for mention in extract_numbers(answer))


def _score_case(case: EvalCase, answer: RagAnswer) -> dict[str, float]:
    scores: dict[str, float] = {}

    # Retrieval: was the chunk holding the answer retrieved, and how highly?
    retrieved_ids = [chunk.chunk_id for chunk in answer.chunks]
    if case.reference_chunk_ids:
        hit_ranks = [
            index
            for index, chunk_id in enumerate(retrieved_ids, start=1)
            if chunk_id in case.reference_chunk_ids
        ]
        scores["context_recall"] = 1.0 if hit_ranks else 0.0
        scores["context_precision"] = 1.0 / min(hit_ranks) if hit_ranks else 0.0

    # Generation faithfulness, from the deterministic guardrail.
    scores["faithfulness"] = answer.verification.groundedness

    if case.expected_kind == NUMERIC:
        correct = states_value(answer.answer, case.expected_value or 0.0)
        if case.expected_text:
            correct = correct and normalise_text(case.expected_text) in normalise_text(answer.answer)
        scores["answer_correctness"] = 1.0 if correct else 0.0

    elif case.expected_kind == REFUSAL:
        refused = looks_like_refusal(answer.answer)
        # A refusal that still quotes an unverifiable number is not a refusal.
        invented = bool(answer.verification.unverified)
        scores["answer_correctness"] = 1.0 if (refused and not invented) else 0.0
        scores["refusal_correctness"] = scores["answer_correctness"]

    else:  # CATEGORICAL
        expected = normalise_text(case.expected_text or "")
        normalised = normalise_text(answer.answer)
        if expected == "no red flags":
            correct = looks_like_refusal(answer.answer) or any(
                phrase in normalised
                for phrase in ("no red flags", "none", "no risk red flags", "were not raised")
            )
        else:
            correct = expected in normalised
        scores["answer_correctness"] = 1.0 if correct else 0.0

    return scores


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_evaluation(
    session: Session,
    ticker: str,
    pipeline: RagPipeline | None = None,
    cases: Sequence[EvalCase] | None = None,
    max_years: int = 3,
    log: bool = True,
) -> EvaluationReport:
    """Run every case through the pipeline and score the answers."""
    ticker = ticker.strip().upper()
    pipeline = pipeline or RagPipeline()
    cases = list(cases) if cases is not None else build_eval_set(session, ticker, max_years=max_years)

    report = EvaluationReport(ticker=ticker)
    for case in cases:
        answer = pipeline.ask(
            case.question, ticker=ticker, fiscal_year=case.fiscal_year, session=None
        )
        scores = _score_case(case, answer)
        report.results.append(CaseResult(case=case, answer=answer, scores=scores))

        if log:
            try:
                log_rag_query(session, answer, scores=scores, query_type="eval")
            except Exception as exc:  # noqa: BLE001 - logging must not fail a run
                logger.warning("Could not log evaluation case: %s", exc)

    logger.info(report.summary_line())
    return report
