"""Embedder, vector store and pipeline tests.

The LLM is always faked. What is under test is the plumbing around it: that the
right context is retrieved, that the prompt carries the rules, that whatever the
model says is verified, and that every query is logged.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from src.db.database import session_scope
from src.db.models import RagEvalLog
from src.rag import repository
from src.rag.chunker import DATA_AVAILABILITY, DISTRESS, PROFILE, RED_FLAG, build_chunks
from src.rag.embedder import HashingEmbedder, cosine_similarity, get_embedder
from src.rag.groq_client import Completion
from src.rag.rag_pipeline import RagPipeline
from src.rag.vector_store import InMemoryVectorStore
from tests.db_fixtures import seed_and_analyze


class FakeGroq:
    """Scripted stand-in for GroqClient."""

    def __init__(self, response: str = "Revenue was 383,285 in FY2023.", responses=None):
        self._responses = list(responses) if responses else None
        self._response = response
        self.calls: list[tuple[str, str]] = []
        self.finish_reason = "stop"

    def complete(self, system_prompt, user_prompt, temperature=None, max_tokens=None):
        self.calls.append((system_prompt, user_prompt))
        text = self._responses.pop(0) if self._responses else self._response
        return Completion(
            text=text,
            model="fake-llama",
            prompt_tokens=120,
            completion_tokens=30,
            latency_seconds=0.02,
            finish_reason=self.finish_reason,
        )

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1][1]


@pytest.fixture
def pipeline():
    return RagPipeline(
        embedder=HashingEmbedder(), store=InMemoryVectorStore(), llm=FakeGroq(), top_k=8
    )


@pytest.fixture
def indexed(temp_db, pipeline):
    with session_scope() as session:
        seed_and_analyze(session, "AAPL", years=(2022, 2023))
        pipeline.index_company(session, "AAPL")
    return pipeline


class TestEmbedder:
    def test_deterministic(self):
        embedder = HashingEmbedder()
        assert embedder.embed(["net margin"]) == embedder.embed(["net margin"])

    def test_dimension_and_normalisation(self):
        vector = HashingEmbedder(dimension=64).embed(["revenue and profit"])[0]
        assert len(vector) == 64
        assert sum(v * v for v in vector) == pytest.approx(1.0)

    def test_related_text_scores_higher_than_unrelated(self):
        embedder = HashingEmbedder()
        query, related, unrelated = embedder.embed(
            [
                "net margin profitability",
                "AAPL FY2023 Profitability Ratios Net Margin 0.2530",
                "AAPL FY2023 Balance Sheet Inventory Goodwill",
            ]
        )
        assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)

    def test_empty_text_is_a_zero_vector(self):
        assert all(v == 0.0 for v in HashingEmbedder().embed([""])[0])

    def test_factory_honours_the_configured_backend(self):
        # conftest pins the suite to the hashing backend.
        assert isinstance(get_embedder(), HashingEmbedder)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="unknown embedding backend"):
            get_embedder("nonsense")


class TestVectorStore:
    def test_upsert_and_count(self, temp_db, pipeline):
        with session_scope() as session:
            seed_and_analyze(session, "AAPL", years=(2023,))
            written = pipeline.index_company(session, "AAPL")
        assert written > 0
        assert pipeline.store.count("AAPL") == written

    def test_reindexing_replaces_rather_than_duplicates(self, temp_db, pipeline):
        with session_scope() as session:
            seed_and_analyze(session, "AAPL", years=(2023,))
            first = pipeline.index_company(session, "AAPL")
            pipeline.index_company(session, "AAPL")
        assert pipeline.store.count() == first

    def test_metadata_filtering(self, indexed):
        found = indexed.store.get({"chunk_type": DISTRESS})
        assert found and all(c.chunk_type == DISTRESS for c in found)

    def test_in_filter(self, indexed):
        found = indexed.store.get({"fiscal_year": {"$in": [2023]}})
        assert found and all(c.fiscal_year == 2023 for c in found)

    def test_delete_ticker(self, indexed):
        removed = indexed.store.delete_ticker("AAPL")
        assert removed > 0
        assert indexed.store.count("AAPL") == 0

    def test_indexing_nothing_is_harmless(self, temp_db, pipeline):
        with session_scope() as session:
            assert pipeline.index_company(session, "ZZZZ") == 0


class TestRetrieval:
    def test_finds_the_relevant_chunk(self, indexed):
        chunks = indexed.retrieve("What was the net margin?", ticker="AAPL", fiscal_year=2023)
        ids = [c.chunk_id for c in chunks]
        assert "AAPL:FY2023:ratios:profitability" in ids

    def test_scoped_to_the_requested_company(self, temp_db, pipeline):
        with session_scope() as session:
            seed_and_analyze(session, "AAPL", years=(2023,))
            seed_and_analyze(session, "MSFT", years=(2023,))
            pipeline.index_company(session, "AAPL")
            pipeline.index_company(session, "MSFT")

        chunks = pipeline.retrieve("revenue", ticker="MSFT")
        assert chunks and all(c.ticker == "MSFT" for c in chunks)

    def test_year_filter_keeps_the_profile_chunk(self, indexed):
        """The profile has no fiscal year but is always relevant."""
        chunks = indexed.retrieve("company sector", ticker="AAPL", fiscal_year=2023)
        years = {c.fiscal_year for c in chunks}
        assert years <= {2023, None}
        assert any(c.chunk_type == PROFILE for c in chunks)

    def test_year_filter_excludes_other_years(self, indexed):
        chunks = indexed.retrieve("revenue", ticker="AAPL", fiscal_year=2022)
        assert 2023 not in {c.fiscal_year for c in chunks}

    def test_top_k_is_respected(self, indexed):
        assert len(indexed.retrieve("revenue", ticker="AAPL", top_k=3)) == 3


class TestIntentCoverage:
    """Risk questions must always retrieve the deterministic risk chunks.

    Otherwise the model answers "is this company at risk?" from the balance
    sheet and reasons about solvency itself - the exact judgement the rule
    engine exists to make instead.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "Is the company at risk of bankruptcy?",
            "What is the Altman Z-Score?",
            "Any signs of earnings manipulation?",
            "Were there red flags?",
        ],
    )
    def test_risk_questions_retrieve_risk_chunks(self, indexed, question):
        types = {c.chunk_type for c in indexed.retrieve(question, ticker="AAPL", fiscal_year=2023)}
        assert DISTRESS in types
        assert RED_FLAG in types

    def test_data_questions_retrieve_the_availability_chunk(self, indexed):
        chunks = indexed.retrieve("What data is missing?", ticker="AAPL", fiscal_year=2023)
        assert DATA_AVAILABILITY in {c.chunk_type for c in chunks}

    def test_ordinary_questions_are_left_alone(self, indexed):
        """No intent match means pure similarity ranking, unmodified."""
        plain = indexed.retrieve("What was inventory?", ticker="AAPL", fiscal_year=2023, top_k=4)
        assert len(plain) == 4

    def test_budget_is_respected(self, indexed):
        chunks = indexed.retrieve(
            "Is the company at risk of bankruptcy?", ticker="AAPL", fiscal_year=2023, top_k=5
        )
        assert len(chunks) <= 5

    def test_added_chunks_carry_real_similarity_scores(self, indexed):
        """Merged-in chunks are re-queried, not stamped with a fake 1.0."""
        chunks = indexed.retrieve("Is there bankruptcy risk?", ticker="AAPL", fiscal_year=2023)
        assert all(c.score <= 1.0 for c in chunks)
        assert not all(c.score == 1.0 for c in chunks)

    def test_explicit_chunk_type_filter_disables_the_boost(self, indexed):
        chunks = indexed.retrieve(
            "bankruptcy risk", ticker="AAPL", fiscal_year=2023, chunk_types=[PROFILE]
        )
        assert {c.chunk_type for c in chunks} == {PROFILE}


class TestAsk:
    def test_returns_a_verified_answer(self, indexed):
        with session_scope() as session:
            result = indexed.ask("What was revenue in FY2023?", "AAPL", session=session)
        assert result.answer
        assert result.chunks
        assert result.model == "fake-llama"
        assert result.groundedness == 1.0

    def test_prompt_carries_the_rules_and_the_context(self, indexed):
        with session_scope() as session:
            indexed.ask("What was revenue in FY2023?", "AAPL", session=session)
        system, user = indexed.llm.calls[-1]
        assert "Use ONLY numbers that appear in the CONTEXT" in system
        assert "CONTEXT:" in user
        assert "383,285" in user

    def test_fabricated_number_is_caught(self, temp_db, pipeline):
        pipeline._llm = FakeGroq("Revenue was 999,999,999 in FY2023.")
        with session_scope() as session:
            seed_and_analyze(session, "AAPL", years=(2023,))
            pipeline.index_company(session, "AAPL")
            result = pipeline.ask("What was revenue?", "AAPL", session=session)

        assert result.groundedness < 1.0
        assert "999,999,999" in {m.raw for m in result.verification.unverified}

    def test_blocking_mode_withholds_the_answer(self, temp_db):
        pipeline = RagPipeline(
            embedder=HashingEmbedder(),
            store=InMemoryVectorStore(),
            llm=FakeGroq("Revenue was 999,999,999 in FY2023."),
            block_on_unverified=True,
        )
        with session_scope() as session:
            seed_and_analyze(session, "AAPL", years=(2023,))
            pipeline.index_company(session, "AAPL")
            result = pipeline.ask("What was revenue?", "AAPL", session=session)

        assert result.verification.blocked
        assert "withheld" in result.answer.lower()

    def test_no_context_means_no_llm_call(self, temp_db, pipeline):
        with session_scope() as session:
            result = pipeline.ask("What was revenue?", "NOPE", session=session)
        assert "not been ingested" in result.answer
        assert pipeline.llm.calls == []


class TestNoContextDiagnosis:
    """When there is nothing to answer from, name the step that is missing.

    Indexing is a separate command from ingestion, so the common failure is a
    company with statements and ratios but no chunks. Telling that user to "run
    ingestion" sends them to re-do work they have already done.
    """

    def test_company_never_ingested(self, temp_db, pipeline):
        with session_scope() as session:
            message = pipeline.diagnose(session, "NOPE")
        assert "not been ingested" in message
        assert "src.ingestion.cli NOPE" in message

    def test_ingested_but_ratios_not_computed(self, temp_db, pipeline):
        from tests.db_fixtures import seed_company

        with session_scope() as session:
            seed_company(session, "RAW", years=(2023,))
        with session_scope() as session:
            message = pipeline.diagnose(session, "RAW")
        assert "ratios have not been computed" in message
        assert "src.ratios.cli RAW" in message

    def test_analysed_but_not_indexed(self, temp_db, pipeline):
        """The case that actually bit: analysed, but never indexed."""
        with session_scope() as session:
            seed_and_analyze(session, "READY", years=(2023,))
        with session_scope() as session:
            message = pipeline.diagnose(session, "READY")
        assert "not indexed for retrieval" in message
        assert "src.rag.cli index READY" in message
        assert "ingestion" not in message.lower().split("index")[0]

    def test_indexed_but_wrong_period(self, temp_db, pipeline):
        with session_scope() as session:
            seed_and_analyze(session, "AAPL", years=(2022, 2023))
            pipeline.index_company(session, "AAPL")
        with session_scope() as session:
            message = pipeline.diagnose(session, "AAPL")
        assert "requested period" in message
        assert "FY2023" in message

    def test_summary_for_unindexed_company_says_how_to_fix_it(self, temp_db, pipeline):
        with session_scope() as session:
            seed_and_analyze(session, "READY", years=(2023,))
            result = pipeline.summarize("READY", session=session)
        assert "src.rag.cli index READY" in result.answer
        assert pipeline.llm.calls == []

    def test_diagnosis_without_a_session_still_helps(self, pipeline):
        assert "src.rag.cli index ZZZZ" in pipeline.diagnose(None, "ZZZZ")

    def test_truncated_output_is_flagged(self, indexed):
        indexed.llm.finish_reason = "length"
        with session_scope() as session:
            result = indexed.ask("Summarise everything", "AAPL", session=session)
        assert any("cut off" in note for note in result.verification.notes)

    def test_citations_are_available(self, indexed):
        with session_scope() as session:
            result = indexed.ask("What was revenue in FY2023?", "AAPL", session=session)
        assert result.citations
        assert "AAPL FY" in result.sources_block()


class TestSummary:
    def test_context_covers_risk_and_limitations(self, indexed):
        chunks = indexed.context_for_summary("AAPL")
        types = {c.chunk_type for c in chunks}
        assert {PROFILE, DISTRESS, RED_FLAG, DATA_AVAILABILITY} <= types

    def test_defaults_to_the_latest_year(self, indexed):
        years = {c.fiscal_year for c in indexed.context_for_summary("AAPL")} - {None}
        assert years == {2023}

    def test_specific_year_can_be_requested(self, indexed):
        years = {c.fiscal_year for c in indexed.context_for_summary("AAPL", 2022)} - {None}
        assert years == {2022}

    def test_summary_uses_the_summary_instruction(self, indexed):
        with session_scope() as session:
            indexed.summarize("AAPL", session=session)
        assert "executive summary" in indexed.llm.last_user_prompt.lower()
        assert "Data limitations" in indexed.llm.last_user_prompt

    def test_unknown_company_summary_is_honest(self, temp_db, pipeline):
        with session_scope() as session:
            result = pipeline.summarize("ZZZZ", session=session)
        assert "not been ingested" in result.answer
        assert not result.chunks


class TestLogging:
    def test_query_is_logged_with_scores_and_context(self, indexed):
        with session_scope() as session:
            indexed.ask("What was revenue in FY2023?", "AAPL", session=session)

        with session_scope() as session:
            row = session.scalar(select(RagEvalLog))
            assert row.ticker == "AAPL"
            assert row.query_type == "ask"
            assert row.groundedness == 1.0
            assert row.model == "fake-llama"
            assert row.retrieved_chunks_json["chunks"]
            assert row.latency_ms >= 0

    def test_unverified_claims_are_recorded(self, temp_db, pipeline):
        pipeline._llm = FakeGroq("Revenue was 12,345,678 in FY2023.")
        with session_scope() as session:
            seed_and_analyze(session, "AAPL", years=(2023,))
            pipeline.index_company(session, "AAPL")
            pipeline.ask("revenue?", "AAPL", session=session)

        with session_scope() as session:
            row = session.scalar(select(RagEvalLog))
            assert "12,345,678" in row.unverified_json["unverified"]

    def test_groundedness_summary_aggregates(self, indexed):
        with session_scope() as session:
            indexed.ask("What was revenue in FY2023?", "AAPL", session=session)
            indexed.ask("What was net income in FY2023?", "AAPL", session=session)

        with session_scope() as session:
            stats = repository.groundedness_summary(session, "AAPL")
        assert stats["queries_logged"] == 2
        assert stats["mean_groundedness"] == 1.0
        assert stats["blocked_answers"] == 0

    def test_logging_can_be_disabled(self, indexed, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "rag_log_queries", False, raising=False)
        with session_scope() as session:
            indexed.ask("What was revenue?", "AAPL", session=session)
        with session_scope() as session:
            assert session.scalar(select(RagEvalLog)) is None

    def test_answering_survives_a_logging_failure(self, indexed, monkeypatch):
        """A logging problem must never cost the user their answer."""

        def boom(*_args, **_kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr("src.rag.rag_pipeline.repository.log_rag_query", boom)
        with session_scope() as session:
            result = indexed.ask("What was revenue?", "AAPL", session=session)
        assert result.answer
        assert result.log_id is None
