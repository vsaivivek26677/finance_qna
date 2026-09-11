"""Evaluation harness tests.

Two things must hold. The eval set must be built from database ground truth, so
the "correct" answers are facts rather than opinions. And the scorer must be
strict: a confident wrong answer, or a refusal that still smuggles in a number,
has to score zero.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from src.db.database import session_scope
from src.db.models import RagEvalLog
from src.rag.evaluation import (
    CATEGORICAL,
    NUMERIC,
    REFUSAL,
    build_eval_set,
    looks_like_refusal,
    run_evaluation,
    states_value,
)
from src.rag.embedder import HashingEmbedder
from src.rag.rag_pipeline import RagPipeline
from src.rag.vector_store import InMemoryVectorStore
from tests.db_fixtures import seed_and_analyze
from tests.test_rag_pipeline import FakeGroq


@pytest.fixture
def evaluable(temp_db):
    """A pipeline indexed over two years, with an LLM to be scripted per test."""
    pipeline = RagPipeline(
        embedder=HashingEmbedder(), store=InMemoryVectorStore(), llm=FakeGroq(), top_k=8
    )
    with session_scope() as session:
        seed_and_analyze(session, "AAPL", years=(2022, 2023))
        pipeline.index_company(session, "AAPL")
    return pipeline


class TestBuildEvalSet:
    def test_cases_are_generated_from_stored_data(self, evaluable):
        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
        assert cases
        assert all(c.ticker == "AAPL" for c in cases)

    def test_every_kind_is_represented(self, evaluable):
        with session_scope() as session:
            kinds = {c.expected_kind for c in build_eval_set(session, "AAPL")}
        assert kinds == {NUMERIC, REFUSAL, CATEGORICAL}

    def test_numeric_expectations_match_the_database(self, evaluable):
        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
        revenue = next(c for c in cases if c.metric_name == "revenue" and c.fiscal_year == 2023)
        assert revenue.expected_value == 383285

    def test_refusal_cases_target_genuinely_uncalculable_metrics(self, evaluable):
        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
        refusals = [c for c in cases if c.expected_kind == REFUSAL]
        # FY2022 has no prior year, so its year-over-year scores cannot compute.
        assert any(c.fiscal_year == 2022 for c in refusals)

    def test_an_out_of_range_period_is_probed(self, evaluable):
        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
        assert any(c.metric_name == "out_of_range_period" for c in cases)

    def test_reference_chunk_ids_match_real_chunks(self, evaluable):
        """A reference id that no chunk has would make recall unmeasurable."""
        indexed_ids = {c.chunk_id for c in evaluable.store.get({"ticker": "AAPL"}, limit=500)}
        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
        for case in cases:
            if case.reference_chunk_ids:
                assert case.reference_chunk_ids & indexed_ids, case.question

    def test_generation_is_deterministic(self, evaluable):
        with session_scope() as session:
            first = [c.question for c in build_eval_set(session, "AAPL")]
            second = [c.question for c in build_eval_set(session, "AAPL")]
        assert first == second

    def test_unknown_ticker_yields_no_cases(self, temp_db):
        with session_scope() as session:
            assert build_eval_set(session, "ZZZZ") == []


class TestScoringPrimitives:
    @pytest.mark.parametrize(
        "text",
        [
            "That figure is not available.",
            "It could not be calculated for FY2022.",
            "The context does not contain that value.",
            "Not applicable - financial sector.",
        ],
    )
    def test_refusals_are_recognised(self, text):
        assert looks_like_refusal(text)

    def test_typographic_characters_do_not_break_matching(self):
        """LLMs emit narrow no-break spaces and curly quotes inside figures.

        A real evaluation run marked two correct answers wrong because the model
        wrote "Below 1.0" rather than "Below 1.0".
        """
        from src.rag.evaluation import normalise_text

        answer = "Red flag: “Current Ratio Below 1.0” – medium severity"
        assert normalise_text("Current Ratio Below 1.0") in normalise_text(answer)

    def test_refusal_detection_is_normalised_too(self):
        assert looks_like_refusal("That value is not available for FY2022.")

    def test_a_confident_answer_is_not_a_refusal(self):
        assert not looks_like_refusal("Revenue was 383,285 in FY2023.")

    def test_states_value_accepts_faithful_renderings(self):
        assert states_value("Revenue was 383,285.", 383285)
        assert states_value("Revenue was 383.29K.", 383285)
        assert states_value("The margin was 25.31%.", 0.2531)

    def test_states_value_rejects_a_wrong_number(self):
        assert not states_value("Revenue was 400,000.", 383285)


class TestRunEvaluation:
    def test_report_covers_every_case_with_all_metrics(self, evaluable):
        with session_scope() as session:
            report = run_evaluation(session, "AAPL", evaluable, max_years=1)

        assert report.total > 0
        metrics = report.aggregate()
        for name in (
            "faithfulness",
            "answer_correctness",
            "context_recall",
            "context_precision",
        ):
            assert name in metrics, metrics

    def test_a_correct_answer_scores_one(self, evaluable):
        from src.rag.evaluation import EvalCase, _score_case

        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
            revenue = next(
                c for c in cases if c.metric_name == "revenue" and c.fiscal_year == 2023
            )
            evaluable._llm = FakeGroq("Revenue was 383,285 in FY2023 (income statement).")
            answer = evaluable.ask(revenue.question, "AAPL", fiscal_year=2023)

        scores = _score_case(revenue, answer)
        assert scores["answer_correctness"] == 1.0
        assert scores["faithfulness"] == 1.0

    def test_a_wrong_number_scores_zero(self, evaluable):
        from src.rag.evaluation import _score_case

        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
            revenue = next(
                c for c in cases if c.metric_name == "revenue" and c.fiscal_year == 2023
            )
            evaluable._llm = FakeGroq("Revenue was 500,000 in FY2023.")
            answer = evaluable.ask(revenue.question, "AAPL", fiscal_year=2023)

        scores = _score_case(revenue, answer)
        assert scores["answer_correctness"] == 0.0
        assert scores["faithfulness"] < 1.0

    def test_a_proper_refusal_scores_one(self, evaluable):
        from src.rag.evaluation import _score_case

        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
            refusal = next(c for c in cases if c.expected_kind == REFUSAL)
            evaluable._llm = FakeGroq("That metric is not available for that period.")
            answer = evaluable.ask(refusal.question, "AAPL", fiscal_year=refusal.fiscal_year)

        scores = _score_case(refusal, answer)
        assert scores["refusal_correctness"] == 1.0

    def test_a_refusal_that_invents_a_number_scores_zero(self, evaluable):
        """Hedged language does not excuse a fabricated figure."""
        from src.rag.evaluation import _score_case

        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
            refusal = next(c for c in cases if c.expected_kind == REFUSAL)
            evaluable._llm = FakeGroq(
                "That metric is not available, but it was roughly 7,777,777."
            )
            answer = evaluable.ask(refusal.question, "AAPL", fiscal_year=refusal.fiscal_year)

        scores = _score_case(refusal, answer)
        assert scores["refusal_correctness"] == 0.0

    def test_context_metrics_reflect_retrieval(self, evaluable):
        from src.rag.evaluation import _score_case

        with session_scope() as session:
            cases = build_eval_set(session, "AAPL")
            case = next(c for c in cases if c.metric_name == "net_margin")
            evaluable._llm = FakeGroq("Net margin was 0.2531 in FY2023.")
            answer = evaluable.ask(case.question, "AAPL", fiscal_year=case.fiscal_year)

        scores = _score_case(case, answer)
        assert scores["context_recall"] == 1.0
        assert 0.0 < scores["context_precision"] <= 1.0

    def test_missing_reference_chunk_scores_zero_recall(self, evaluable):
        from src.rag.evaluation import EvalCase, _score_case

        case = EvalCase(
            question="anything",
            ticker="AAPL",
            expected_kind=NUMERIC,
            expected_value=1.0,
            reference_chunk_ids={"AAPL:FY9999:does_not_exist"},
        )
        evaluable._llm = FakeGroq("Something.")
        answer = evaluable.ask("anything", "AAPL")
        scores = _score_case(case, answer)
        assert scores["context_recall"] == 0.0
        assert scores["context_precision"] == 0.0

    def test_every_case_is_logged_for_later_analysis(self, evaluable):
        with session_scope() as session:
            report = run_evaluation(session, "AAPL", evaluable, max_years=1)

        with session_scope() as session:
            rows = list(session.scalars(select(RagEvalLog).where(RagEvalLog.query_type == "eval")))
        assert len(rows) == report.total
        assert all(row.scores_json for row in rows)

    def test_summary_line_reports_the_headline_metrics(self, evaluable):
        with session_scope() as session:
            report = run_evaluation(session, "AAPL", evaluable, max_years=1)
        line = report.summary_line()
        assert "AAPL" in line and "faithfulness" in line and "context recall" in line
