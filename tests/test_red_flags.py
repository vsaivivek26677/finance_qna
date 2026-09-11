"""Red flag rule tests.

Each rule is checked twice: it fires on the condition it describes, and it stays
silent otherwise. False positives matter as much as misses here — these flags are
handed to the LLM as ground truth, so a spurious one becomes a confident,
incorrect statement about a real company.
"""

from __future__ import annotations

import pytest

from src.ratios.base import PeriodAnalysis, PeriodBundle
from src.ratios.ratio_engine import compute_ratios
from src.ratios.red_flags import (
    RULE_REGISTRY,
    THRESHOLDS,
    RedFlagResult,
    RuleSpec,
    Severity,
    detect_all,
    detect_for_period,
)
from tests.ratio_fixtures import REMOVE, make_bundle


def analysis_of(bundle: PeriodBundle) -> PeriodAnalysis:
    return PeriodAnalysis(bundle=bundle, ratios=compute_ratios(bundle))


def flags_for(*bundles: PeriodBundle, **kwargs) -> dict[str, RedFlagResult]:
    """Flags for the last bundle, keyed by name. Bundles are passed oldest first."""
    history = [analysis_of(b) for b in bundles]
    return {f.flag_name: f for f in detect_for_period(history, **kwargs)}


def chain(*bundles: PeriodBundle) -> list[PeriodBundle]:
    """Link bundles oldest-to-newest via `.prior`, as the engine does."""
    for previous, current in zip(bundles, bundles[1:]):
        current.prior = previous
    return list(bundles)


class TestBaseline:
    def test_healthy_company_raises_only_what_is_true(self):
        """Apple FY2023: solvent and profitable, but its current ratio is below 1."""
        flags = flags_for(make_bundle())
        assert set(flags) == {"Current Ratio Below 1.0"}

    def test_every_rule_is_registered(self):
        assert len(RULE_REGISTRY) == 15


class TestFinancialSectorScoping:
    """Rules whose premise fails for banks are skipped, not reported as risks."""

    BANK = dict(is_financial_sector=True, sector="Financial Services")

    def test_bank_specific_false_positives_are_suppressed(self):
        # Apple's own figures produce a sub-1.0 current ratio; classified as a
        # bank, that rule must not fire at all.
        flags = flags_for(make_bundle(**self.BANK))
        assert "Current Ratio Below 1.0" not in flags

    def test_interest_coverage_not_flagged_for_a_bank(self):
        """Interest expense is a bank's cost of funding, not a debt burden."""
        flags = flags_for(make_bundle(income={"operating_income": 3000}, **self.BANK))
        assert "Interest Coverage Danger" not in flags
        assert "Interest Coverage Danger" in flags_for(
            make_bundle(income={"operating_income": 3000})
        )

    def test_negative_operating_cash_flow_not_flagged_for_a_bank(self):
        periods = chain(
            make_bundle(2022, cash_flow={"operating_cash_flow": -5000}, **self.BANK),
            make_bundle(2023, cash_flow={"operating_cash_flow": -8000}, **self.BANK),
        )
        assert "Negative Operating Cash Flow (Sustained)" not in flags_for(*periods)

    def test_the_skip_is_disclosed_not_silent(self):
        flag = flags_for(make_bundle(**self.BANK))["Financial-Sector Rules Not Applied"]
        assert flag.severity is Severity.INFO
        assert "weak_interest_coverage" in flag.source_values["skipped_rules"]
        assert len(flag.source_values["skipped_rules"]) == 8

    def test_universally_valid_rules_still_run_for_banks(self):
        """Losses and manipulation risk are not sector-specific."""
        flags = flags_for(make_bundle(income={"net_income": -2000}, **self.BANK))
        assert "Net Loss" in flags

    def test_non_financial_company_gets_no_skip_notice(self):
        assert "Financial-Sector Rules Not Applied" not in flags_for(make_bundle())


class TestSolvencyFlags:
    def test_altman_distress_zone_flagged_high(self):
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
        flag = flags_for(distressed)["Altman Z-Score in Distress Zone"]
        assert flag.severity is Severity.HIGH
        assert flag.source_values["zone"] == "Distress"
        assert "-0.45" in flag.explanation

    def test_grey_zone_is_low_severity(self):
        flags = flags_for(make_bundle(market={"market_cap": 200000}))
        assert flags["Altman Z-Score in Grey Zone"].severity is Severity.LOW

    def test_falls_back_to_book_value_variant_without_market_data(self):
        """No price history must not mean no distress assessment."""
        distressed = make_bundle(
            income={"operating_income": 1000, "revenue": 50000},
            balance={
                "total_current_assets": 20000,
                "total_current_liabilities": 80000,
                "retained_earnings": -100000,
                "total_liabilities": 340000,
                "total_equity": 12583,
            },
            with_market=False,
        )
        flag = flags_for(distressed)["Altman Z-Score in Distress Zone"]
        assert "book value" in flag.source_values["variant"]

    def test_financial_sector_raises_no_altman_flag(self):
        """A bank must not be flagged by a model that does not apply to it."""
        flags = flags_for(make_bundle(is_financial_sector=True))
        assert not any("Altman" in name for name in flags)

    def test_negative_equity_flagged_high(self):
        flag = flags_for(make_bundle(balance={"total_equity": -5000}))[
            "Negative Shareholders' Equity"
        ]
        assert flag.severity is Severity.HIGH
        assert flag.source_values["total_equity"] == -5000

    def test_positive_equity_not_flagged(self):
        assert "Negative Shareholders' Equity" not in flags_for(make_bundle())

    @pytest.mark.parametrize(
        ("operating_income", "expected_phrase"),
        [(3000, "does not cover"), (5500, "little headroom")],
    )
    def test_interest_coverage_danger(self, operating_income, expected_phrase):
        # Interest expense is 3,933, so coverage is 0.76x and 1.40x respectively.
        flag = flags_for(make_bundle(income={"operating_income": operating_income}))[
            "Interest Coverage Danger"
        ]
        assert flag.severity is Severity.HIGH
        assert expected_phrase in flag.explanation

    def test_healthy_coverage_not_flagged(self):
        assert "Interest Coverage Danger" not in flags_for(make_bundle())

    def test_no_interest_expense_does_not_flag(self):
        """A debt-free company must not be flagged for undefined coverage."""
        flags = flags_for(make_bundle(income={"interest_expense": 0}))
        assert "Interest Coverage Danger" not in flags


class TestLiquidityFlags:
    def test_current_ratio_below_one(self):
        flag = flags_for(make_bundle())["Current Ratio Below 1.0"]
        assert flag.severity is Severity.MEDIUM
        assert flag.source_values["current_ratio"] == pytest.approx(143566 / 145308)

    def test_healthy_current_ratio_not_flagged(self):
        flags = flags_for(make_bundle(balance={"total_current_liabilities": 100000}))
        assert "Current Ratio Below 1.0" not in flags

    def test_quick_ratio_below_threshold(self):
        flags = flags_for(make_bundle(balance={"inventory": 90000}))
        assert "Quick Ratio Below 0.5" in flags

    def test_sustained_negative_operating_cash_flow(self):
        periods = chain(
            make_bundle(2022, cash_flow={"operating_cash_flow": -5000}),
            make_bundle(2023, cash_flow={"operating_cash_flow": -8000}),
        )
        flag = flags_for(*periods)["Negative Operating Cash Flow (Sustained)"]
        assert flag.severity is Severity.HIGH
        assert "2 consecutive" in flag.explanation

    def test_single_negative_period_is_not_sustained(self):
        periods = chain(
            make_bundle(2022, cash_flow={"operating_cash_flow": 50000}),
            make_bundle(2023, cash_flow={"operating_cash_flow": -8000}),
        )
        assert "Negative Operating Cash Flow (Sustained)" not in flags_for(*periods)

    def test_no_flag_when_history_is_too_short(self):
        one_period = make_bundle(2023, cash_flow={"operating_cash_flow": -8000})
        assert "Negative Operating Cash Flow (Sustained)" not in flags_for(one_period)


class TestProfitabilityFlags:
    def test_three_consecutive_margin_declines(self):
        periods = chain(
            *[
                make_bundle(year, income={"operating_income": oi})
                for year, oi in [(2020, 120000), (2021, 100000), (2022, 80000), (2023, 60000)]
            ]
        )
        flag = flags_for(*periods)["Margin Compression"]
        assert flag.severity is Severity.MEDIUM
        assert "3 consecutive periods" in flag.explanation

    def test_a_single_recovery_year_breaks_the_streak(self):
        periods = chain(
            *[
                make_bundle(year, income={"operating_income": oi})
                for year, oi in [(2020, 120000), (2021, 100000), (2022, 110000), (2023, 60000)]
            ]
        )
        assert "Margin Compression" not in flags_for(*periods)

    def test_net_loss_single_period(self):
        flag = flags_for(make_bundle(income={"net_income": -2000}))["Net Loss"]
        assert flag.severity is Severity.MEDIUM

    def test_consecutive_losses_escalate_to_high(self):
        periods = chain(
            make_bundle(2022, income={"net_income": -1000}),
            make_bundle(2023, income={"net_income": -2000}),
        )
        flags = flags_for(*periods)
        assert flags["Sustained Net Losses"].severity is Severity.HIGH
        assert "Net Loss" not in flags  # escalated, not duplicated

    def test_negative_fcf_despite_profit(self):
        flag = flags_for(
            make_bundle(cash_flow={"free_cash_flow": -5000, "operating_cash_flow": 2000})
        )["Negative Free Cash Flow Despite Positive Net Income"]
        assert flag.source_values["free_cash_flow"] == -5000

    def test_negative_fcf_with_a_loss_is_not_this_flag(self):
        """The signal is the *divergence*, not negative cash flow on its own."""
        flags = flags_for(
            make_bundle(
                income={"net_income": -1000}, cash_flow={"free_cash_flow": -5000}
            )
        )
        assert "Negative Free Cash Flow Despite Positive Net Income" not in flags


class TestLeverageFlags:
    def test_debt_to_equity_rising_more_than_twenty_percent(self):
        periods = chain(
            make_bundle(2022, balance={"total_debt": 50000}),
            make_bundle(2023, balance={"total_debt": 111088}),
        )
        flag = flags_for(*periods)["Debt-to-Equity Rising Sharply"]
        assert flag.source_values["change_pct"] > THRESHOLDS["debt_to_equity_yoy_increase"]

    def test_stable_leverage_not_flagged(self):
        periods = chain(make_bundle(2022), make_bundle(2023))
        assert "Debt-to-Equity Rising Sharply" not in flags_for(*periods)

    def test_sector_comparison_skipped_without_a_median(self):
        """No benchmark is supplied, so no benchmark claim is made."""
        assert "Leverage Well Above Sector Median" not in flags_for(make_bundle())

    def test_sector_comparison_fires_when_a_median_is_supplied(self):
        flags = flags_for(make_bundle(), sector_medians={"debt_to_equity": 0.5})
        flag = flags["Leverage Well Above Sector Median"]
        assert flag.source_values["sector_median"] == 0.5
        assert flag.source_values["multiple_of_median"] == pytest.approx(1.7875325845589418 / 0.5)


class TestEarningsQualityFlags:
    def test_beneish_above_threshold(self):
        periods = chain(
            make_bundle(2022),
            make_bundle(
                2023,
                income={"gross_profit": 100000, "depreciation_amortization": 5000},
                balance={"net_receivables": 121970},
                cash_flow={"operating_cash_flow": 10000},
            ),
        )
        flag = flags_for(*periods)["Beneish M-Score Above Manipulation Threshold"]
        assert flag.severity is Severity.HIGH
        assert "screening signal, not evidence" in flag.explanation
        assert flag.source_values["dsri"] == pytest.approx(2.0)

    def test_clean_year_not_flagged(self):
        periods = chain(make_bundle(2022), make_bundle(2023))
        assert "Beneish M-Score Above Manipulation Threshold" not in flags_for(*periods)

    def test_weak_piotroski_flagged_with_failed_signals(self):
        periods = chain(
            make_bundle(
                2022,
                income={
                    "revenue": 500000,
                    "gross_profit": 250000,
                    "net_income": 50000,
                    "weighted_average_shares_diluted": 15000,
                },
                balance={"long_term_debt": 50000, "total_current_assets": 200000},
            ),
            make_bundle(
                2023, income={"net_income": -1000}, cash_flow={"operating_cash_flow": -2000}
            ),
        )
        flag = flags_for(*periods)["Weak Piotroski F-Score"]
        assert flag.source_values["f_score"] == 0.0
        assert len(flag.source_values["failed_signals"]) == 9

    def test_receivables_outpacing_revenue(self):
        periods = chain(
            make_bundle(2022, balance={"net_receivables": 40000}),
            make_bundle(2023, balance={"net_receivables": 60985}),
        )
        flag = flags_for(*periods)["Receivables Growing Faster Than Revenue"]
        assert flag.source_values["gap"] > THRESHOLDS["receivables_growth_gap"]

    def test_proportional_growth_not_flagged(self):
        periods = chain(
            make_bundle(2022, income={"revenue": 300000}, balance={"net_receivables": 47000}),
            make_bundle(2023),
        )
        assert "Receivables Growing Faster Than Revenue" not in flags_for(*periods)


class TestDataIntegrityFlag:
    def test_missing_critical_field_is_surfaced(self):
        flag = flags_for(make_bundle(balance={"total_assets": REMOVE}))[
            "Missing Critical Financial Data"
        ]
        assert flag.severity is Severity.INFO
        assert "total_assets" in flag.source_values["missing_fields"]["balance_sheet"]

    def test_absent_statement_is_surfaced(self):
        flag = flags_for(make_bundle(no_income=True))["Missing Critical Financial Data"]
        assert flag.source_values["missing_fields"]["income_statement"] == [
            "entire statement not on file"
        ]

    def test_complete_data_raises_no_integrity_flag(self):
        assert "Missing Critical Financial Data" not in flags_for(make_bundle())


class TestRunner:
    def test_flags_sorted_by_severity(self):
        periods = chain(
            make_bundle(2022, income={"net_income": -1000}),
            make_bundle(
                2023,
                income={"net_income": -2000, "operating_income": 1000},
                balance={"total_equity": -5000, "total_assets": REMOVE},
            ),
        )
        history = [analysis_of(b) for b in periods]
        flags = detect_for_period(history)
        severities = [f.severity for f in flags]
        assert severities == sorted(severities, key=lambda s: ["High", "Medium", "Low", "Info"].index(s.value))

    def test_detect_all_covers_every_period(self):
        periods = chain(make_bundle(2021), make_bundle(2022), make_bundle(2023))
        by_year = detect_all([analysis_of(b) for b in periods])
        assert set(by_year) == {2021, 2022, 2023}

    def test_each_period_sees_only_its_own_history(self):
        """A flag on FY2022 must not depend on FY2023 data that did not exist yet."""
        periods = chain(
            make_bundle(2022, cash_flow={"operating_cash_flow": 50000}),
            make_bundle(2023, cash_flow={"operating_cash_flow": -8000}),
        )
        by_year = detect_all([analysis_of(b) for b in periods])
        names_2022 = {f.flag_name for f in by_year[2022]}
        assert "Negative Operating Cash Flow (Sustained)" not in names_2022

    def test_flags_carry_their_period(self):
        by_year = detect_all([analysis_of(make_bundle(2023))])
        assert all(f.fiscal_year == 2023 and f.period == "FY" for f in by_year[2023])

    def test_a_failing_rule_does_not_suppress_the_others(self, monkeypatch):
        def exploding_rule(window):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "src.ratios.red_flags.RULE_REGISTRY",
            [RuleSpec(fn=exploding_rule), *RULE_REGISTRY],
        )
        flags = flags_for(make_bundle())
        assert "Current Ratio Below 1.0" in flags

    def test_empty_history_returns_no_flags(self):
        assert detect_for_period([]) == []
