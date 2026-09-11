"""FastAPI endpoint tests.

The contract worth protecting is not just "returns 200". It is that the API
reports *absence* faithfully: a ratio that could not be computed comes back with
a reason, a statement field the provider never sent is listed in
`missing_fields`, and a generated answer carries its own groundedness score. A
client rendering these correctly cannot show an estimate as a fact.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.api.main import app
from src.db.database import get_session_factory, session_scope
from src.rag.embedder import HashingEmbedder
from src.rag.groq_client import Completion, GroqError
from src.rag.rag_pipeline import RagPipeline
from src.rag.vector_store import InMemoryVectorStore
from tests.db_fixtures import seed_and_analyze


class FakeGroq:
    """Scripted LLM so endpoint tests never touch the network."""

    def __init__(self, response: str = "Revenue was 383,285 in FY2023.", error: bool = False):
        self._response = response
        self._error = error
        self.calls = 0

    def complete(self, system_prompt, user_prompt, temperature=None, max_tokens=None):
        self.calls += 1
        if self._error:
            raise GroqError("model unavailable")
        return Completion(
            text=self._response,
            model="fake-llama",
            prompt_tokens=100,
            completion_tokens=25,
            latency_seconds=0.01,
            finish_reason="stop",
        )


@pytest.fixture
def client(temp_db, monkeypatch):
    """TestClient bound to a seeded temporary database and a fake LLM."""
    with session_scope() as session:
        seed_and_analyze(session, "AAPL", years=(2022, 2023))
        seed_and_analyze(session, "MSFT", years=(2023,))

    monkeypatch.setattr("src.config.settings.groq_api_key", "test-key", raising=False)
    monkeypatch.setattr("src.config.settings.fmp_api_key", "test-key", raising=False)

    pipeline = RagPipeline(
        embedder=HashingEmbedder(), store=InMemoryVectorStore(), llm=FakeGroq(), top_k=8
    )
    with session_scope() as session:
        pipeline.index_company(session, "AAPL")
    dependencies.set_pipeline(pipeline)
    dependencies.clear_caches()

    # Bind request sessions to the temp database rather than the real one.
    # FastAPI needs the generator *function* here, not a call to it.
    app.dependency_overrides[dependencies.get_db] = _session

    with TestClient(app) as test_client:
        test_client.pipeline = pipeline
        yield test_client

    app.dependency_overrides.clear()
    dependencies.reset_pipeline()
    dependencies.clear_caches()


def _session():
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


class TestSystem:
    def test_health_reports_every_dependency(self, client):
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["database"] == "ok"
        assert payload["companies"] >= 2
        assert payload["ratios"] > 0
        assert payload["llm_configured"] is True

    def test_root_points_at_the_docs(self, client):
        assert client.get("/").json()["docs"] == "/docs"

    def test_openapi_schema_is_served(self, client):
        schema = client.get("/openapi.json").json()
        assert "/companies/{ticker}/ratios" in schema["paths"]


class TestCompanies:
    def test_list_includes_fiscal_years(self, client):
        rows = client.get("/companies").json()
        aapl = next(r for r in rows if r["ticker"] == "AAPL")
        assert aapl["fiscal_years"] == [2023, 2022]

    def test_profile(self, client):
        payload = client.get("/companies/AAPL").json()
        assert payload["name"] == "AAPL Inc."
        assert payload["sector"] == "Technology"

    def test_ticker_is_case_insensitive(self, client):
        assert client.get("/companies/aapl").status_code == 200

    def test_unknown_ticker_explains_what_to_do(self, client):
        response = client.get("/companies/ZZZZ")
        assert response.status_code == 404
        assert "POST /ingest/ZZZZ" in response.json()["detail"]

    def test_statements_carry_their_gaps(self, client, temp_db):
        """A field the provider never sent must be named, not silently absent."""
        with session_scope() as session:
            seed_and_analyze(
                session,
                "GAPS",
                years=(2023,),
                income_overrides={2023: {"interest_expense": None}},
                missing={"interest_expense": "not_reported"},
            )
        dependencies.clear_caches()

        payload = client.get("/companies/GAPS/statements").json()
        line = payload["income_statements"][0]
        assert line["values"]["interest_expense"] is None
        assert line["missing_fields"]["interest_expense"] == "not_reported"

    def test_statements_can_be_filtered_by_year(self, client):
        payload = client.get(
            "/companies/AAPL/statements", params={"fiscal_year": 2023}
        ).json()
        assert [s["fiscal_year"] for s in payload["income_statements"]] == [2023]


class TestRatios:
    def test_all_periods_returned_newest_first(self, client):
        payload = client.get("/companies/AAPL/ratios").json()
        assert [p["fiscal_year"] for p in payload["periods"]] == [2023, 2022]
        assert "profitability" in payload["categories"]

    def test_uncalculable_ratios_are_returned_with_a_reason(self, client):
        """Omitting them would let a client read absence as zero."""
        payload = client.get("/companies/AAPL/ratios", params={"fiscal_year": 2022}).json()
        ratios = {r["name"]: r for r in payload["periods"][0]["ratios"]}
        piotroski = ratios["piotroski_f_score"]
        assert piotroski["is_calculable"] is False
        assert piotroski["value"] is None
        assert "prior period" in piotroski["reason"]

    def test_calculable_only_filter(self, client):
        payload = client.get(
            "/companies/AAPL/ratios",
            params={"fiscal_year": 2022, "calculable_only": True},
        ).json()
        assert all(r["is_calculable"] for r in payload["periods"][0]["ratios"])

    def test_category_filter(self, client):
        payload = client.get(
            "/companies/AAPL/ratios", params={"category": "liquidity"}
        ).json()
        assert payload["categories"] == ["liquidity"]

    def test_ratio_series_shape(self, client):
        payload = client.get("/companies/AAPL/ratios/net_margin/series").json()
        assert payload["ratio_name"] == "net_margin"
        assert set(payload["points"]) == {"2022", "2023"}

    def test_method_is_exposed(self, client):
        """Average vs closing basis must reach the client, not just the database."""
        payload = client.get("/companies/AAPL/ratios", params={"fiscal_year": 2023}).json()
        roe = next(r for r in payload["periods"][0]["ratios"] if r["name"] == "return_on_equity")
        assert "average" in roe["method"]


class TestDistress:
    def test_scores_include_zone_and_components(self, client):
        payload = client.get("/companies/AAPL/distress-score").json()
        altman = next(s for s in payload["scores"] if s["name"] == "Altman Z-Score")
        assert altman["zone"] == "Safe"
        assert altman["thresholds"]["safe_above"] == 2.99
        assert "x3_ebit_to_assets" in altman["components"]

    def test_defaults_to_the_latest_year(self, client):
        assert client.get("/companies/AAPL/distress-score").json()["fiscal_year"] == 2023

    def test_financial_sector_is_explained_not_scored(self, client, temp_db):
        with session_scope() as session:
            seed_and_analyze(
                session, "BANK", years=(2023,), sector="Financial Services", is_financial=True
            )
        dependencies.clear_caches()

        payload = client.get("/companies/BANK/distress-score").json()
        assert payload["is_financial_sector"] is True
        assert "Not Applicable" in payload["note"] or "not defined" in payload["note"]
        altman = next(s for s in payload["scores"] if s["name"] == "Altman Z-Score")
        assert altman["is_calculable"] is False
        assert altman["value"] is None


class TestRedFlags:
    def test_flags_with_counts_and_sources(self, client):
        payload = client.get("/companies/AAPL/red-flags").json()
        assert payload["counts_by_severity"]
        flag = next(f for f in payload["flags"] if f["flag_name"] == "Current Ratio Below 1.0")
        assert flag["severity"] == "Medium"
        assert "current_ratio" in flag["source_values"]
        assert flag["category"] == "Liquidity Stress"

    def test_severity_filter(self, client):
        payload = client.get(
            "/companies/AAPL/red-flags", params={"min_severity": "High"}
        ).json()
        assert all(f["severity"] == "High" for f in payload["flags"])


class TestAi:
    def test_ask_returns_verification_with_the_answer(self, client):
        response = client.post(
            "/companies/AAPL/ask", json={"question": "What was revenue in FY2023?"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"]
        assert payload["groundedness"] == 1.0
        assert payload["sources"]
        assert payload["model"] == "fake-llama"

    def test_fabricated_figures_are_reported_to_the_client(self, client):
        client.pipeline._llm = FakeGroq("Revenue was 999,999,999 in FY2023.")
        payload = client.post(
            "/companies/AAPL/ask", json={"question": "What was revenue?"}
        ).json()
        assert payload["groundedness"] < 1.0
        assert "999,999,999" in payload["unverified"]

    def test_summary_is_generated_and_cached(self, client):
        first = client.get("/companies/AAPL/summary")
        assert first.status_code == 200
        calls = client.pipeline._llm.calls
        client.get("/companies/AAPL/summary")
        assert client.pipeline._llm.calls == calls  # served from cache

    def test_refresh_bypasses_the_cache(self, client):
        client.get("/companies/AAPL/summary")
        calls = client.pipeline._llm.calls
        client.get("/companies/AAPL/summary", params={"refresh": True})
        assert client.pipeline._llm.calls > calls

    def test_llm_failure_is_a_502_not_a_crash(self, client):
        client.pipeline._llm = FakeGroq(error=True)
        response = client.post("/companies/AAPL/ask", json={"question": "What was revenue?"})
        assert response.status_code == 502
        assert "language model is unavailable" in response.json()["detail"]

    def test_missing_key_is_a_clear_503(self, client, monkeypatch):
        monkeypatch.setattr("src.config.settings.groq_api_key", None, raising=False)
        response = client.post("/companies/AAPL/ask", json={"question": "What was revenue?"})
        assert response.status_code == 503
        assert "GROQ_API_KEY" in response.json()["detail"]

    def test_question_is_validated(self, client):
        assert client.post("/companies/AAPL/ask", json={"question": "x"}).status_code == 422


class TestIngestion:
    def test_missing_fmp_key_is_a_clear_503(self, client, monkeypatch):
        monkeypatch.setattr("src.config.settings.fmp_api_key", None, raising=False)
        response = client.post("/ingest/NEW")
        assert response.status_code == 503
        assert "FMP_API_KEY" in response.json()["detail"]


class TestCaching:
    def test_reads_are_cached_then_invalidated_by_a_write(self, client, temp_db):
        assert len(client.get("/companies").json()) == 2
        with session_scope() as session:
            seed_and_analyze(session, "NVDA", years=(2023,))
        assert len(client.get("/companies").json()) == 2  # still cached
        dependencies.clear_caches()
        assert len(client.get("/companies").json()) == 3
