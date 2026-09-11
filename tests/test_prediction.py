"""Layer 4 - financial-distress probability.

The training data (the American bankruptcy dataset) is never downloaded here: every test builds
a small synthetic frame with the same columns. What is checked is that the
feature definitions match between training and scoring, that an under-specified
company-year is refused rather than guessed at, and that the API surfaces the
model's provenance alongside the number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction import predict as predict_mod
from src.prediction.dataset import HORIZON_COLUMN, LABEL_COLUMN
from src.prediction.features import (
    FEATURE_NAMES,
    build_features_from_figures,
    usable_feature_count,
)
from src.prediction.model import DistressModel, train_model


# --- a small, separable stand-in for the real training set --------------------

def _synthetic_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = (rng.random(n) < 0.3).astype(int)
    rows = {}
    for name in FEATURE_NAMES:
        # healthy firms centre higher, distressed lower, with overlap
        base = rng.normal(0.5, 0.2, n)
        rows[name] = base - 0.6 * label + rng.normal(0, 0.05, n)
    frame = pd.DataFrame(rows)
    frame[LABEL_COLUMN] = label
    frame[HORIZON_COLUMN] = 1
    return frame


@pytest.fixture(scope="module")
def trained_model() -> DistressModel:
    return train_model(_synthetic_frame(), folds=3, seed=0)


# --- feature builder ---------------------------------------------------------

class TestFeatures:
    def test_all_eight_features_are_produced(self):
        features = build_features_from_figures(
            total_liabilities=600, net_income=80, operating_income=120,
            ebitda=150, retained_earnings=250, revenue=900,
            market_value=1000, gross_profit=350, operating_expenses=700,
        )
        assert set(features) == set(FEATURE_NAMES)
        assert features["net_income_to_liabilities"] == pytest.approx(80 / 600)
        assert features["ebit_to_liabilities"] == pytest.approx(120 / 600)
        assert features["ebitda_to_liabilities"] == pytest.approx(150 / 600)
        assert features["retained_earnings_to_liabilities"] == pytest.approx(250 / 600)
        assert features["revenue_to_liabilities"] == pytest.approx(900 / 600)
        assert features["market_value_to_liabilities"] == pytest.approx(1000 / 600)
        assert features["gross_margin"] == pytest.approx(350 / 900)
        assert features["operating_expense_ratio"] == pytest.approx(700 / 900)

    def test_missing_input_becomes_none_not_zero(self):
        features = build_features_from_figures(
            total_liabilities=600, net_income=80, operating_income=120,
            ebitda=150, retained_earnings=None, revenue=900,
            market_value=1000, gross_profit=None, operating_expenses=700,
        )
        assert features["retained_earnings_to_liabilities"] is None
        assert features["gross_margin"] is None
        assert usable_feature_count(features) == 6

    def test_zero_denominator_is_refused(self):
        features = build_features_from_figures(
            total_liabilities=0, net_income=0, operating_income=0,
            ebitda=0, retained_earnings=0, revenue=0,
            market_value=0, gross_profit=0, operating_expenses=0,
        )
        assert all(v is None for v in features.values())


# --- model training / scoring ---------------------------------------------------

class TestModel:
    def test_reports_lift_over_the_altman_baseline(self, trained_model):
        m = trained_model.metrics
        assert 0.5 < m["model"]["roc_auc"] <= 1.0
        assert m["baseline"]["roc_auc"] > 0.5
        assert m["model"]["pr_auc"] >= m["baseline"]["pr_auc"]

    def test_probability_is_a_probability(self, trained_model):
        healthy = {name: 0.6 for name in FEATURE_NAMES}
        distressed = {name: -0.2 for name in FEATURE_NAMES}
        assert 0.0 <= trained_model.probability(healthy) <= 1.0
        assert trained_model.probability(distressed) > trained_model.probability(healthy)

    def test_bands_are_ordered_by_probability(self, trained_model):
        assert trained_model.band(0.01) == "Low"
        assert trained_model.band(0.5) == "High"
        seen = [trained_model.band(p) for p in (0.01, 0.08, 0.2, 0.6)]
        assert seen == ["Low", "Moderate", "Elevated", "High"]

    def test_save_and_load_round_trip(self, trained_model, tmp_path):
        path = tmp_path / "m.joblib"
        trained_model.save(path)
        reloaded = DistressModel.load(path)
        probe = {name: 0.3 for name in FEATURE_NAMES}
        assert reloaded.probability(probe) == pytest.approx(trained_model.probability(probe))

    def test_explain_names_features_on_the_risky_side(self, trained_model):
        distressed = {name: -0.5 for name in FEATURE_NAMES}
        factors = trained_model.explain(distressed, top=3)
        assert 1 <= len(factors) <= 3
        assert all(f["direction"] == "raises estimate" for f in factors)


class TestScore:
    @pytest.fixture(autouse=True)
    def _use_synthetic_model(self, trained_model, tmp_path, monkeypatch):
        path = tmp_path / "distress_model.joblib"
        trained_model.save(path)
        monkeypatch.setattr(predict_mod.settings, "distress_model_path", str(path), raising=False)
        predict_mod.load_model.cache_clear()
        yield
        predict_mod.load_model.cache_clear()

    def test_enough_inputs_is_scored(self):
        estimate = predict_mod.score({name: 0.4 for name in FEATURE_NAMES})
        assert estimate.is_scored
        assert estimate.probability is not None
        assert estimate.risk_band in {"Low", "Moderate", "Elevated", "High"}

    def test_too_few_inputs_is_refused_with_a_reason(self):
        sparse = {name: None for name in FEATURE_NAMES}
        sparse[FEATURE_NAMES[0]] = 0.3
        sparse[FEATURE_NAMES[1]] = 0.3
        estimate = predict_mod.score(sparse)
        assert estimate.is_scored is False
        assert "need at least" in estimate.reason

    def test_low_band_carries_no_factor_noise(self):
        estimate = predict_mod.score({name: 5.0 for name in FEATURE_NAMES})
        assert estimate.risk_band == "Low"
        assert estimate.factors == []


# --- persistence against a seeded database -----------------------------------

class TestRepository:
    def test_feature_rows_come_from_stored_statements(self, db_session):
        from tests.db_fixtures import seed_company
        from src.prediction import repository

        seed_company(db_session, "AAPL", years=(2022, 2023))
        rows = repository.build_feature_rows(db_session, "AAPL")
        assert [r.fiscal_year for r in rows] == [2022, 2023]
        assert set(rows[0].features) == set(FEATURE_NAMES)

    def test_estimates_upsert_on_identity(self, db_session, trained_model, tmp_path, monkeypatch):
        from tests.db_fixtures import seed_company
        from src.prediction import repository

        path = tmp_path / "m.joblib"
        trained_model.save(path)
        monkeypatch.setattr(predict_mod.settings, "distress_model_path", str(path), raising=False)
        predict_mod.load_model.cache_clear()

        seed_company(db_session, "AAPL", years=(2022, 2023))
        rows = repository.build_feature_rows(db_session, "AAPL")
        estimates = [(r, predict_mod.score(r.features)) for r in rows]

        first = repository.save_estimates(db_session, "AAPL", estimates)
        second = repository.save_estimates(db_session, "AAPL", estimates)
        assert first == second == 2
        assert len(repository.get_estimates(db_session, "AAPL")) == 2
        predict_mod.load_model.cache_clear()


# --- API -------------------------------------------------------------------

class TestApi:
    @pytest.fixture
    def api_client(self, temp_db):
        """A TestClient over a seeded temp DB. The prediction endpoint needs no
        RAG pipeline, so this is lighter than test_api.py's fixture."""
        from fastapi.testclient import TestClient

        from src.api import dependencies
        from src.api.main import app
        from src.db.database import get_session_factory, session_scope
        from tests.db_fixtures import seed_company

        with session_scope() as session:
            seed_company(session, "AAPL", years=(2022, 2023))

        def _session():
            session = get_session_factory()()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[dependencies.get_db] = _session
        dependencies.clear_caches()
        with TestClient(app) as test_client:
            yield test_client
        app.dependency_overrides.clear()
        dependencies.clear_caches()

    @pytest.fixture
    def with_model(self, api_client, trained_model, tmp_path, monkeypatch):
        path = tmp_path / "distress_model.joblib"
        trained_model.save(path)
        monkeypatch.setattr("src.config.settings.distress_model_path", str(path), raising=False)
        predict_mod.load_model.cache_clear()
        from src.api import dependencies

        dependencies.clear_caches()
        yield api_client
        predict_mod.load_model.cache_clear()

    def test_prediction_endpoint_carries_model_provenance(self, with_model):
        payload = with_model.get("/companies/AAPL/distress-prediction").json()
        assert payload["available"] is True
        assert payload["model"]["training_rows"] > 0
        assert payload["model"]["cv_roc_auc"] is not None
        assert payload["periods"], "AAPL was seeded and should score"
        assert all("is_scored" in p for p in payload["periods"])

    def test_health_reports_the_model_is_ready(self, with_model):
        assert with_model.get("/health").json()["distress_model_ready"] is True

    def test_missing_model_is_a_clean_note_not_a_crash(self, api_client, monkeypatch):
        monkeypatch.setattr(
            "src.config.settings.distress_model_path", "/no/such/model.joblib", raising=False
        )
        predict_mod.load_model.cache_clear()
        from src.api import dependencies

        dependencies.clear_caches()
        payload = api_client.get("/companies/AAPL/distress-prediction").json()
        assert payload["available"] is False
        assert "train" in payload["note"].lower()
        predict_mod.load_model.cache_clear()
