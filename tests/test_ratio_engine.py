"""Engine tests: loading from the database, market pairing, and persistence."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import func, select

from src.db.database import session_scope
from src.db.models import (
    BalanceSheet,
    CashFlowStatement,
    Company,
    IncomeStatement,
    MarketData,
    Ratio,
    RedFlag,
)
from src.ratios import repository
from src.ratios.ratio_engine import (
    MARKET_DATA_WINDOW_DAYS,
    analyze_ticker,
    load_history,
    run_analysis,
)
from tests.db_fixtures import seed_company
from tests.ratio_fixtures import INCOME_FY2023


class TestLoadHistory:
    def test_bundles_are_oldest_first_and_linked(self, temp_db):
        with session_scope() as session:
            seed_company(session, years=(2021, 2022, 2023))
            history = load_history(session, "AAPL")

        assert [b.fiscal_year for b in history] == [2021, 2022, 2023]
        assert history[0].prior is None
        assert history[1].prior is history[0]
        assert history[2].prior is history[1]

    def test_company_metadata_reaches_the_bundle(self, temp_db):
        with session_scope() as session:
            seed_company(session, sector="Financial Services", is_financial=True)
            history = load_history(session, "AAPL")
        assert history[-1].is_financial_sector is True
        assert history[-1].sector == "Financial Services"

    def test_unknown_ticker_returns_empty(self, temp_db):
        with session_scope() as session:
            assert load_history(session, "ZZZZ") == []

    def test_lookup_is_case_insensitive(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            assert len(load_history(session, "aapl")) == 2

    def test_year_with_only_one_statement_still_loads(self, temp_db):
        """Statements are joined per year, not assumed to arrive together."""
        with session_scope() as session:
            company = seed_company(session, years=(2023,))
            session.add(
                IncomeStatement(
                    company_id=company.id,
                    period="FY",
                    fiscal_year=2022,
                    period_end_date=date(2022, 9, 30),
                    is_missing_json={},
                    **INCOME_FY2023,
                )
            )
            session.flush()
            history = load_history(session, "AAPL")

        partial = next(b for b in history if b.fiscal_year == 2022)
        assert partial.income is not None
        assert partial.balance is None


class TestMarketDataPairing:
    def test_nearest_observation_within_the_window_is_used(self, temp_db):
        with session_scope() as session:
            company = seed_company(session, years=(2023,))
            session.add(
                MarketData(
                    company_id=company.id, date=date(2023, 9, 15), close_price=1.0, market_cap=1
                )
            )
            session.flush()
            history = load_history(session, "AAPL")

        # 29 Sep is one day from the 30 Sep period end; 15 Sep is fifteen.
        assert history[0].market.date == date(2023, 9, 29)

    def test_observation_outside_the_window_is_not_paired(self, temp_db):
        """A stale price must never be applied to a period it does not belong to."""
        with session_scope() as session:
            seed_company(session, years=(2023,), market_dates={2023: date(2023, 1, 15)})
            history = load_history(session, "AAPL")

        assert history[0].market is None

    def test_window_boundary(self, temp_db):
        inside = date(2023, 9, 30)
        with session_scope() as session:
            seed_company(session, years=(2023,), market_dates={2023: inside})
            history = load_history(session, "AAPL")
        assert history[0].market is not None
        assert MARKET_DATA_WINDOW_DAYS == 45

    def test_valuation_ratios_unavailable_without_a_paired_price(self, temp_db):
        with session_scope() as session:
            seed_company(session, years=(2023,), market_dates={2023: date(2023, 1, 15)})
            analyses, _ = analyze_ticker(session, "AAPL")

        result = analyses[0].result("price_to_earnings")
        assert result.is_calculable is False
        assert analyses[0].result("altman_z_double_prime_score").is_calculable is True


class TestPersistence:
    def test_ratios_and_flags_are_written(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            report = run_analysis(session, "AAPL")

        assert report.ratios_computed > 0
        assert report.fiscal_years == [2022, 2023]
        with session_scope() as session:
            assert session.scalar(select(func.count()).select_from(Ratio)) > 0
            assert session.scalar(select(func.count()).select_from(RedFlag)) > 0

    def test_not_calculable_ratios_are_stored_with_their_reason(self, temp_db):
        """The absence of a ratio is itself a fact the database records."""
        with session_scope() as session:
            seed_company(session, years=(2023,))  # no prior year -> no F-Score
            run_analysis(session, "AAPL")

        with session_scope() as session:
            row = session.scalar(
                select(Ratio).where(Ratio.ratio_name == "piotroski_f_score")
            )
            assert row.is_calculable is False
            assert row.ratio_value is None
            assert "prior period" in row.reason

    def test_distress_components_are_persisted_for_the_rag_layer(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            run_analysis(session, "AAPL")

        with session_scope() as session:
            row = session.scalar(
                select(Ratio).where(
                    Ratio.ratio_name == "altman_z_score", Ratio.fiscal_year == 2023
                )
            )
            assert row.details_json["zone"] == "Safe"
            assert "x3_ebit_to_assets" in row.inputs_json
            assert row.method

    def test_method_is_persisted(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            run_analysis(session, "AAPL")
        with session_scope() as session:
            row = session.scalar(
                select(Ratio).where(
                    Ratio.ratio_name == "return_on_equity", Ratio.fiscal_year == 2023
                )
            )
            assert "average" in row.method

    def test_rerun_updates_rather_than_duplicating(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            run_analysis(session, "AAPL")
            first = session.scalar(select(func.count()).select_from(Ratio))
            run_analysis(session, "AAPL")
            second = session.scalar(select(func.count()).select_from(Ratio))
        assert first == second

    def test_red_flags_are_replaced_not_accumulated(self, temp_db):
        """Rules change between runs; stale flags must not survive."""
        with session_scope() as session:
            seed_company(session)
            run_analysis(session, "AAPL")
            first = session.scalar(select(func.count()).select_from(RedFlag))
            run_analysis(session, "AAPL")
            second = session.scalar(select(func.count()).select_from(RedFlag))
        assert first == second

    def test_no_store_leaves_the_database_untouched(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            run_analysis(session, "AAPL", persist=False)
        with session_scope() as session:
            assert session.scalar(select(func.count()).select_from(Ratio)) == 0

    def test_red_flag_source_values_are_persisted(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            run_analysis(session, "AAPL")
        with session_scope() as session:
            row = session.scalar(
                select(RedFlag).where(RedFlag.flag_name == "Current Ratio Below 1.0")
            )
            assert row.severity == "Medium"
            assert row.source_json["source_values"]["current_ratio"] == pytest.approx(
                143566 / 145308
            )
            assert row.source_json["category"] == "Liquidity Stress"


class TestReadHelpers:
    def test_get_ratios_filters_by_year_and_calculability(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            run_analysis(session, "AAPL")

        with session_scope() as session:
            # With complete data and a prior year, FY2023 computes all 35.
            assert len(repository.get_ratios(session, "AAPL", fiscal_year=2023)) == 35
            assert (
                len(repository.get_ratios(session, "AAPL", fiscal_year=2023, calculable_only=True))
                == 35
            )
            # FY2022 is the earliest year, so the year-over-year scores cannot run.
            all_2022 = repository.get_ratios(session, "AAPL", fiscal_year=2022)
            calculable_2022 = repository.get_ratios(
                session, "AAPL", fiscal_year=2022, calculable_only=True
            )
            missing = {r.ratio_name for r in all_2022} - {r.ratio_name for r in calculable_2022}

        assert len(all_2022) == 35
        assert missing == {"piotroski_f_score", "beneish_m_score"}

    def test_get_ratio_series_across_years(self, temp_db):
        with session_scope() as session:
            seed_company(session, years=(2021, 2022, 2023))
            run_analysis(session, "AAPL")
        with session_scope() as session:
            series = repository.get_ratio_series(session, "AAPL", "net_margin")
        assert set(series) == {2021, 2022, 2023}
        assert all(v == pytest.approx(96995 / 383285) for v in series.values())

    def test_get_red_flags_filters_by_severity(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            run_analysis(session, "AAPL")
        with session_scope() as session:
            everything = repository.get_red_flags(session, "AAPL")
            high_only = repository.get_red_flags(session, "AAPL", min_severity="High")
        assert len(everything) >= len(high_only)

    def test_read_helpers_tolerate_unknown_tickers(self, temp_db):
        with session_scope() as session:
            assert repository.get_ratios(session, "ZZZZ") == []
            assert repository.get_red_flags(session, "ZZZZ") == []


class TestReport:
    def test_report_counts_and_coverage(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            report = run_analysis(session, "AAPL")
        assert report.ratios_computed + report.ratios_not_calculable == 35 * 2
        assert 0 < report.coverage <= 1
        assert "High" in report.flags_by_severity or "Medium" in report.flags_by_severity

    def test_missing_data_produces_a_warning(self, temp_db):
        with session_scope() as session:
            report = run_analysis(session, "NOPE")
        assert report.warnings
        assert "ingestion first" in report.warnings[0]

    def test_financial_sector_warning_explains_the_skip(self, temp_db):
        with session_scope() as session:
            seed_company(session, sector="Financial Services", is_financial=True)
            report = run_analysis(session, "AAPL")
        assert any("financial-sector" in w for w in report.warnings)

    def test_no_market_data_warning_names_the_fallback(self, temp_db):
        with session_scope() as session:
            seed_company(session, market_dates={2022: date(2022, 1, 1), 2023: date(2023, 1, 1)})
            report = run_analysis(session, "AAPL")
        assert any("Z''-Score" in w for w in report.warnings)

    def test_summary_line_is_human_readable(self, temp_db):
        with session_scope() as session:
            seed_company(session)
            report = run_analysis(session, "AAPL")
        line = report.summary_line()
        assert "AAPL" in line and "2022-2023" in line and "coverage" in line
