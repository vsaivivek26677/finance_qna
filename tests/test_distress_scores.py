"""Tests for the composite distress and earnings-quality scores.

Expected scores are hand-computed from the published formulas, independently of
`src/`. These are the numbers the LLM will narrate, so a wrong coefficient here
would propagate as a confident, wrong statement about a company's solvency.
"""

from __future__ import annotations

import pytest

from src.ratios.distress_scores import (
    BENEISH_MANIPULATION_THRESHOLD,
    PIOTROSKI_MAX_SCORE,
    altman_z_double_prime_score,
    altman_z_score,
    beneish_m_score,
    piotroski_f_score,
)
from src.ratios.ratio_engine import compute_ratios
from tests.ratio_fixtures import REMOVE, make_bundle, make_pair

# --- Piotroski scenarios ---------------------------------------------------

# A prior year worse on every one of the six comparative signals.
PERFECT_PRIOR = {
    "income": {
        "revenue": 300000,
        "gross_profit": 78000,
        "net_income": 50000,
        "weighted_average_shares_diluted": 16000,
    },
    "balance": {"long_term_debt": 120000, "total_current_assets": 100000},
}

# A prior year better on every comparative signal, against a loss-making current.
WORST_CURRENT = {"income": {"net_income": -1000}, "cash_flow": {"operating_cash_flow": -2000}}
WORST_PRIOR = {
    "income": {
        "revenue": 500000,
        "gross_profit": 250000,
        "net_income": 50000,
        "weighted_average_shares_diluted": 15000,
    },
    "balance": {"long_term_debt": 50000, "total_current_assets": 200000},
}


class TestAltmanZScore:
    def test_z_score_matches_hand_calculation(self):
        result = compute_ratios(make_bundle())["altman_z_score"]
        assert result.is_calculable
        assert result.value == pytest.approx(7.934485817463457, rel=1e-12)
        assert result.details["zone"] == "Safe"

    def test_components_are_persisted_for_narration(self):
        """Layer 3 narrates the terms; it must never recompute or guess them."""
        result = compute_ratios(make_bundle())["altman_z_score"]
        assert result.inputs["x1_working_capital_to_assets"] == pytest.approx(-1742 / 352583)
        assert result.inputs["x2_retained_earnings_to_assets"] == pytest.approx(-214 / 352583)
        assert result.inputs["x3_ebit_to_assets"] == pytest.approx(114301 / 352583)
        assert result.inputs["x4_market_equity_to_liabilities"] == pytest.approx(2800000 / 290437)
        assert result.inputs["x5_revenue_to_assets"] == pytest.approx(383285 / 352583)

    @pytest.mark.parametrize(
        ("market_cap", "zone"),
        [
            (2800000, "Safe"),  # Z = 7.93
            (200000, "Grey"),  # Z = 2.56; a lower X4 alone lands in the grey band
        ],
    )
    def test_zone_from_market_value(self, market_cap, zone):
        result = compute_ratios(make_bundle(market={"market_cap": market_cap}))["altman_z_score"]
        assert result.details["zone"] == zone

    def test_distressed_profile_scores_in_the_distress_zone(self):
        """Negative working capital, an accumulated deficit and thin EBIT."""
        distressed = make_bundle(
            income={"operating_income": 1000, "revenue": 50000},
            balance={
                "total_current_assets": 20000,
                "total_current_liabilities": 80000,
                "retained_earnings": -100000,
                "total_liabilities": 340000,
                "total_equity": 12583,
            },
            market={"market_cap": 1000},
        )
        results = compute_ratios(distressed)
        assert results["altman_z_score"].value == pytest.approx(-0.4483420774849678, rel=1e-9)
        assert results["altman_z_score"].details["zone"] == "Distress"
        # The book-value variant agrees, so the fallback path flags it too.
        assert results["altman_z_double_prime_score"].details["zone"] == "Distress"

    def test_missing_retained_earnings_blocks_the_score(self):
        """A Z-Score with a term silently zeroed is worse than no Z-Score."""
        result = compute_ratios(make_bundle(balance={"retained_earnings": REMOVE}))[
            "altman_z_score"
        ]
        assert result.is_calculable is False
        assert "retained_earnings" in result.reason

    def test_no_market_data_blocks_the_market_variant(self):
        result = compute_ratios(make_bundle(with_market=False))["altman_z_score"]
        assert result.is_calculable is False
        assert "market" in result.reason


class TestFinancialSectorExclusion:
    @pytest.mark.parametrize(
        "name", ["altman_z_score", "altman_z_double_prime_score"]
    )
    def test_banks_get_not_applicable_not_a_number(self, name):
        result = compute_ratios(make_bundle(is_financial_sector=True, sector="Financial Services"))[
            name
        ]
        assert result.is_calculable is False
        assert result.value is None
        assert "Not Applicable - Financial Sector" in result.reason

    def test_non_financial_company_still_scores(self):
        result = compute_ratios(make_bundle(is_financial_sector=False))["altman_z_score"]
        assert result.is_calculable is True


class TestAltmanZDoublePrime:
    def test_z_double_prime_matches_hand_calculation(self):
        result = compute_ratios(make_bundle())["altman_z_double_prime_score"]
        assert result.value == pytest.approx(2.3687851572813625, rel=1e-12)
        assert result.details["zone"] == "Grey"

    def test_available_as_the_fallback_when_market_data_is_missing(self):
        """The documented degradation path: no price data, so book value is used."""
        results = compute_ratios(make_bundle(with_market=False))
        assert results["altman_z_score"].is_calculable is False
        assert results["altman_z_double_prime_score"].is_calculable is True
        assert "book-value" in results["altman_z_double_prime_score"].method

    def test_uses_book_equity_not_market_value(self):
        """Its score must not move when the share price does."""
        low = compute_ratios(make_bundle(market={"market_cap": 1000}))
        high = compute_ratios(make_bundle(market={"market_cap": 9_000_000}))
        assert low["altman_z_double_prime_score"].value == pytest.approx(
            high["altman_z_double_prime_score"].value
        )
        assert low["altman_z_score"].value != pytest.approx(high["altman_z_score"].value)

    def test_thresholds_differ_from_the_original_model(self):
        result = compute_ratios(make_bundle())["altman_z_double_prime_score"]
        assert result.details["thresholds"] == {"safe_above": 2.60, "distress_below": 1.10}


class TestPiotroskiFScore:
    def test_all_nine_signals_pass(self):
        bundle = make_pair(prior=PERFECT_PRIOR)
        result = compute_ratios(bundle)["piotroski_f_score"]
        assert result.value == 9.0
        assert result.details["interpretation"] == "Strong"
        assert all(result.details["signals"].values())
        assert len(result.details["signals"]) == PIOTROSKI_MAX_SCORE

    def test_all_nine_signals_fail(self):
        bundle = make_pair(current=WORST_CURRENT, prior=WORST_PRIOR)
        result = compute_ratios(bundle)["piotroski_f_score"]
        assert result.value == 0.0
        assert result.details["interpretation"] == "Weak"
        assert not any(result.details["signals"].values())

    def test_individual_signal_flips_the_score_by_one(self):
        baseline = compute_ratios(make_pair(prior=PERFECT_PRIOR))["piotroski_f_score"]
        # Issuing shares should cost exactly one point.
        issued = make_pair(
            current={"income": {"weighted_average_shares_diluted": 99999}},
            prior=PERFECT_PRIOR,
        )
        result = compute_ratios(issued)["piotroski_f_score"]
        assert result.value == baseline.value - 1
        assert result.details["signals"]["no_new_share_issuance"] is False

    def test_accruals_signal_compares_cash_with_earnings(self):
        bundle = make_pair(
            current={"cash_flow": {"operating_cash_flow": 1000}}, prior=PERFECT_PRIOR
        )
        signals = compute_ratios(bundle)["piotroski_f_score"].details["signals"]
        assert signals["cash_flow_exceeds_net_income"] is False

    def test_no_partial_score_without_a_prior_period(self):
        """Five of six possible signals must not be reported as 5 out of 9."""
        result = compute_ratios(make_bundle())["piotroski_f_score"]
        assert result.is_calculable is False
        assert result.value is None
        assert "prior period" in result.reason

    def test_missing_input_blocks_the_score(self):
        bundle = make_pair(current={"income": {"gross_profit": REMOVE}}, prior=PERFECT_PRIOR)
        result = compute_ratios(bundle)["piotroski_f_score"]
        assert result.is_calculable is False
        assert "gross_profit" in result.reason


class TestBeneishMScore:
    def test_flat_year_matches_hand_calculation(self):
        """With every index at 1.0 the score reduces to intercept + accruals."""
        result = compute_ratios(make_pair())["beneish_m_score"]
        assert result.value == pytest.approx(-2.6597905514446243, rel=1e-12)
        assert result.details["flagged"] is False
        for index in ("dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi"):
            assert result.inputs[index] == pytest.approx(1.0)
        assert result.inputs["tata"] == pytest.approx((96995 - 110543) / 352583)

    def test_manipulation_pattern_crosses_the_threshold(self):
        bundle = make_pair(
            current={
                "income": {"gross_profit": 100000, "depreciation_amortization": 5000},
                "balance": {"net_receivables": 121970},
                "cash_flow": {"operating_cash_flow": 10000},
            }
        )
        result = compute_ratios(bundle)["beneish_m_score"]
        assert result.value == pytest.approx(0.07824803403265435, rel=1e-9)
        assert result.value > BENEISH_MANIPULATION_THRESHOLD
        assert result.details["flagged"] is True
        assert result.inputs["dsri"] == pytest.approx(2.0)

    def test_no_score_without_a_prior_period(self):
        result = compute_ratios(make_bundle())["beneish_m_score"]
        assert result.is_calculable is False
        assert "prior period" in result.reason

    def test_depreciation_falls_back_to_the_cash_flow_statement(self):
        bundle = make_pair(current={"income": {"depreciation_amortization": REMOVE}})
        assert compute_ratios(bundle)["beneish_m_score"].is_calculable

    def test_missing_depreciation_everywhere_blocks_the_score(self):
        bundle = make_pair(
            current={
                "income": {"depreciation_amortization": REMOVE},
                "cash_flow": {"depreciation_amortization": REMOVE},
            }
        )
        result = compute_ratios(bundle)["beneish_m_score"]
        assert result.is_calculable is False
        assert "depreciation_amortization" in result.reason

    def test_zero_revenue_blocks_the_indices(self):
        bundle = make_pair(current={"income": {"revenue": 0}})
        result = compute_ratios(bundle)["beneish_m_score"]
        assert result.is_calculable is False


class TestScoreFunctionsAreDirectlyCallable:
    """The score functions are plain functions of a bundle, with no I/O."""

    def test_callable_without_the_engine(self):
        bundle = make_pair(prior=PERFECT_PRIOR)
        assert altman_z_score(bundle).value == pytest.approx(7.934485817463457)
        assert altman_z_double_prime_score(bundle).value > 0
        assert piotroski_f_score(bundle).value == 9.0
        assert beneish_m_score(bundle).value < 0
