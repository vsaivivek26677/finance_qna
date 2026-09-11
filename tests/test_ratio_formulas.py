"""Formula-level tests for every non-composite ratio.

Expected values were computed by hand from Apple's FY2023 filing, independently
of `src/`, and are asserted as literals. A refactor that quietly changes which
field feeds a formula fails here.
"""

from __future__ import annotations

import pytest

from src.ratios.base import NotCalculable, PeriodBundle
from src.ratios.ratio_engine import RATIO_REGISTRY, compute_ratios
from tests.ratio_fixtures import REMOVE, make_bundle, make_pair

# name -> value computed by hand from the FY2023 figures (no prior period, so
# every balance-sheet ratio uses closing balances).
EXPECTED = {
    "current_ratio": 0.9880116717592975,
    "quick_ratio": 0.9444421504665951,
    "cash_ratio": 0.4236174195501968,
    "working_capital": -1742.0,
    "gross_margin": 0.4413112957720756,
    "operating_margin": 0.2982141226502472,
    "net_margin": 0.2530623426432028,
    "return_on_equity": 1.5607601454639075,
    "return_on_assets": 0.27509834563776475,
    "return_on_invested_capital": 0.5626888293615087,
    "debt_to_equity": 1.7875325845589418,
    "debt_to_assets": 0.3150690759338936,
    "liabilities_to_assets": 0.8237407929480435,
    "interest_coverage": 29.062039155860667,
    "net_debt_to_ebitda": 0.6447544110634239,
    "asset_turnover": 1.087077369016657,
    "inventory_turnover": 33.82356657716001,
    "receivables_turnover": 6.284906124456834,
    "days_sales_outstanding": 58.07564866874519,
    "price_to_earnings": 28.86746739522656,
    "price_to_book": 45.05519261094841,
    "ev_to_ebitda": 22.898768081386105,
    "free_cash_flow": 99584.0,
    "free_cash_flow_margin": 0.25981710737440805,
    "cash_conversion_ratio": 1.1396773029537606,
}


@pytest.fixture
def results() -> dict:
    return compute_ratios(make_bundle())


@pytest.mark.parametrize(("name", "expected"), sorted(EXPECTED.items()))
def test_ratio_matches_hand_calculation(results, name, expected):
    result = results[name]
    assert result.is_calculable, f"{name} was not calculable: {result.reason}"
    assert result.value == pytest.approx(expected, rel=1e-12)


def test_every_registered_ratio_is_attempted():
    """compute_ratios returns an entry per registered ratio, never a silent gap."""
    results = compute_ratios(make_bundle())
    assert set(results) == {spec.name for spec in RATIO_REGISTRY}


class TestMissingInputsAreNamed:
    def test_missing_field_names_itself_in_the_reason(self):
        results = compute_ratios(make_bundle(balance={"total_current_assets": REMOVE}))
        result = results["current_ratio"]
        assert result.is_calculable is False
        assert result.value is None
        assert "total_current_assets" in result.reason
        assert "FY2023" in result.reason

    def test_ingestion_missing_reason_is_carried_through(self):
        """The 'why' recorded at ingestion survives into the ratio's reason."""
        bundle = make_bundle()
        bundle.balance.total_current_assets = None
        bundle.balance.is_missing_json = {"total_current_assets": "null_in_source"}
        result = compute_ratios(bundle)["current_ratio"]
        assert "null in source" in result.reason

    def test_absent_statement_is_reported(self):
        results = compute_ratios(make_bundle(no_income=True))
        assert results["net_margin"].is_calculable is False
        assert "income statement" in results["net_margin"].reason

    def test_zero_denominator_refuses_rather_than_dividing(self):
        results = compute_ratios(make_bundle(balance={"total_current_liabilities": 0}))
        result = results["current_ratio"]
        assert result.is_calculable is False
        assert "zero" in result.reason

    def test_not_calculable_ratios_still_appear_in_the_output(self):
        """A ratio that fails is reported as unavailable, never omitted."""
        results = compute_ratios(make_bundle(no_income=True))
        assert "net_margin" in results
        assert results["net_margin"].reason


class TestDerivations:
    def test_gross_profit_derived_from_cost_of_revenue(self):
        result = compute_ratios(make_bundle(income={"gross_profit": REMOVE}))["gross_margin"]
        assert result.is_calculable
        assert result.value == pytest.approx((383285 - 214137) / 383285)
        assert "derived" in result.method

    def test_total_debt_derived_from_components(self):
        result = compute_ratios(make_bundle(balance={"total_debt": REMOVE}))["debt_to_assets"]
        assert result.value == pytest.approx((15807 + 95281) / 352583)

    def test_free_cash_flow_derived_from_operating_less_capex(self):
        result = compute_ratios(make_bundle(cash_flow={"free_cash_flow": REMOVE}))[
            "free_cash_flow"
        ]
        assert result.value == pytest.approx(110543 - 10959)

    def test_capex_sign_convention_does_not_flip_fcf(self):
        """Sources disagree on the sign of capex; magnitude is what matters."""
        positive = compute_ratios(
            make_bundle(cash_flow={"free_cash_flow": REMOVE, "capital_expenditure": 10959})
        )["free_cash_flow"]
        assert positive.value == pytest.approx(110543 - 10959)

    def test_cash_ratio_falls_back_to_components(self):
        result = compute_ratios(
            make_bundle(balance={"cash_and_short_term_investments": REMOVE})
        )["cash_ratio"]
        assert result.value == pytest.approx((29965 + 31590) / 145308)
        assert "short-term investments" in result.method

    def test_diluted_eps_falls_back_to_basic(self):
        result = compute_ratios(make_bundle(income={"eps_diluted": REMOVE}))[
            "earnings_per_share_diluted"
        ]
        assert result.value == pytest.approx(6.16)
        assert "basic" in result.method


class TestAveragingBasis:
    def test_closing_balance_used_without_a_prior_period(self):
        result = compute_ratios(make_bundle())["return_on_equity"]
        assert result.value == pytest.approx(96995 / 62146)
        assert "closing balance only" in result.method

    def test_average_balance_used_when_prior_is_available(self):
        bundle = make_pair(prior={"balance": {"total_equity": 50672}})
        result = compute_ratios(bundle)["return_on_equity"]
        assert result.value == pytest.approx(96995 / ((62146 + 50672) / 2))
        assert "average" in result.method

    def test_basis_is_always_reported(self):
        """Two periods on different bases must never be compared silently."""
        for name in ("return_on_equity", "return_on_assets", "asset_turnover"):
            assert compute_ratios(make_bundle())[name].method


class TestMeaninglessRatiosAreRefused:
    def test_negative_equity_blocks_return_on_equity(self):
        result = compute_ratios(make_bundle(balance={"total_equity": -5000}))["return_on_equity"]
        assert result.is_calculable is False
        assert "negative" in result.reason

    def test_negative_equity_blocks_debt_to_equity(self):
        result = compute_ratios(make_bundle(balance={"total_equity": -5000}))["debt_to_equity"]
        assert result.is_calculable is False

    def test_negative_earnings_block_price_to_earnings(self):
        result = compute_ratios(make_bundle(income={"net_income": -1000}))["price_to_earnings"]
        assert result.is_calculable is False
        assert "not meaningful" in result.reason

    def test_negative_ebitda_blocks_ev_multiple(self):
        result = compute_ratios(make_bundle(income={"ebitda": -500}))["ev_to_ebitda"]
        assert result.is_calculable is False

    def test_zero_inventory_blocks_turnover(self):
        result = compute_ratios(make_bundle(balance={"inventory": 0}))["inventory_turnover"]
        assert result.is_calculable is False
        assert "undefined" in result.reason

    def test_zero_interest_expense_blocks_coverage(self):
        """No debt service is not distress — the ratio is simply undefined."""
        result = compute_ratios(make_bundle(income={"interest_expense": 0}))["interest_coverage"]
        assert result.is_calculable is False
        assert "not itself a sign of distress" in result.reason

    def test_interest_expense_sign_does_not_flip_coverage(self):
        negative = compute_ratios(make_bundle(income={"interest_expense": -3933}))
        assert negative["interest_coverage"].value == pytest.approx(114301 / 3933)


class TestValuationNeedsMarketData:
    @pytest.mark.parametrize(
        "name", ["price_to_earnings", "price_to_book", "price_to_sales", "ev_to_ebitda"]
    )
    def test_no_market_data_means_no_valuation_ratio(self, name):
        result = compute_ratios(make_bundle(with_market=False))[name]
        assert result.is_calculable is False
        assert "market" in result.reason.lower()

    def test_market_cap_derived_from_price_and_shares(self):
        bundle = make_bundle(
            market={"market_cap": REMOVE, "close_price": 180.0, "shares_outstanding": 15600.0}
        )
        assert compute_ratios(bundle)["price_to_book"].value == pytest.approx(
            180.0 * 15600.0 / 62146
        )


class TestROICTaxHandling:
    def test_effective_tax_rate_from_reported_expense(self):
        result = compute_ratios(make_bundle())["return_on_invested_capital"]
        assert result.inputs["effective_tax_rate"] == pytest.approx(16741 / 113736)

    def test_pretax_loss_sets_tax_rate_to_zero_and_says_so(self):
        result = compute_ratios(
            make_bundle(income={"income_before_tax": -5000, "income_tax_expense": 0})
        )["return_on_invested_capital"]
        assert result.inputs["effective_tax_rate"] == 0.0
        assert "undefined" in result.method

    def test_tax_rate_is_clamped_to_a_sane_range(self):
        result = compute_ratios(
            make_bundle(income={"income_before_tax": 1000, "income_tax_expense": 5000})
        )["return_on_invested_capital"]
        assert result.inputs["effective_tax_rate"] == 1.0


class TestBundleHelpers:
    def test_period_label_formats(self):
        assert make_bundle(2023).period_label == "FY2023"
        assert make_bundle(2023, period="Q3").period_label == "Q3 2023"

    def test_require_prior_without_prior_raises(self):
        with pytest.raises(NotCalculable, match="no prior period"):
            make_bundle().require_prior("income", "revenue")

    def test_get_returns_none_instead_of_raising(self):
        assert make_bundle(income={"ebitda": REMOVE}).get("income", "ebitda") is None

    def test_non_finite_result_is_rejected(self):
        """A computation producing inf/nan is reported, never persisted as a value."""
        bundle: PeriodBundle = make_bundle(
            income={"revenue": 1e308, "gross_profit": 1e308},
            balance={"total_assets": 1e-308},
        )
        result = compute_ratios(bundle)["asset_turnover"]
        assert result.is_calculable is False
        assert "non-finite" in result.reason
