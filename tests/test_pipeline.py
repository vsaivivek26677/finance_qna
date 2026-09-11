"""Pipeline and persistence tests, driven by a stubbed FMP client."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import func, select

from src.db.database import session_scope
from src.db.models import BalanceSheet, Company, IncomeStatement, IngestionRun, MarketData
from src.ingestion import repository
from src.ingestion.fmp_client import FMPError
from src.ingestion.ingest_pipeline import IngestionPipeline
from src.ingestion.schemas import MarketDataPoint, PeriodType
from tests import fixtures


class StubFMPClient:
    """Stands in for FMPClient without touching the network."""

    def __init__(self, shape: str = "v3", fail_profile: bool = False, statements=None):
        self.shape = shape
        self.fail_profile = fail_profile
        self._statements = statements
        self.request_count = 0
        self.variant = shape

    def get_profile(self, symbol):
        if self.fail_profile:
            raise FMPError("profile endpoint unavailable")
        return {**fixtures.PROFILE_V3, "symbol": symbol}

    def get_all_statements(self, symbol, period="annual", limit=5):
        if self._statements is not None:
            return self._statements
        return fixtures.statements_payload(self.shape)

    def close(self):
        pass


class StubYFClient:
    """Returns two months of prices, or raises if configured to fail."""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def get_fiscal_year_end_prices(self, symbol, years=5):
        if self.fail:
            from src.ingestion.yfinance_client import YFinanceUnavailableError

            raise YFinanceUnavailableError("yfinance blew up")
        return [
            MarketDataPoint(
                ticker=symbol,
                date=date(2023, 8, 31),
                close_price=187.87,
                market_cap=2_900_000_000_000,
                shares_outstanding=15_500_000_000,
            ),
            MarketDataPoint(
                ticker=symbol,
                date=date(2023, 9, 29),
                close_price=171.21,
                market_cap=2_650_000_000_000,
                shares_outstanding=15_500_000_000,
            ),
        ]


def make_pipeline(**kwargs) -> IngestionPipeline:
    return IngestionPipeline(
        fmp_client=StubFMPClient(**{k: v for k, v in kwargs.items() if k in {"shape", "fail_profile", "statements"}}),
        yf_client=StubYFClient(fail=kwargs.get("yf_fail", False)),
        include_market_data=kwargs.get("include_market_data", True),
    )


class TestFetch:
    def test_fetch_returns_normalized_bundle(self):
        data, errors = make_pipeline().fetch("aapl")
        assert errors == []
        assert data.profile.ticker == "AAPL"
        assert data.profile.name == "Apple Inc."
        assert len(data.income_statements) == 1
        assert data.income_statements[0].revenue == 383285000000
        assert len(data.market_data) == 2
        assert data.fiscal_years == [2023]

    def test_profile_failure_is_recorded_not_fatal(self):
        data, errors = make_pipeline(fail_profile=True).fetch("AAPL")
        assert any("profile" in e for e in errors)
        assert data.profile.ticker == "AAPL"  # placeholder keeps the FK intact
        assert len(data.income_statements) == 1  # statements still ingested

    def test_market_data_failure_is_recorded_not_fatal(self):
        data, errors = make_pipeline(yf_fail=True).fetch("AAPL")
        assert any("market_data" in e for e in errors)
        assert data.market_data == []
        assert len(data.balance_sheets) == 1

    def test_market_data_can_be_skipped(self):
        data, errors = make_pipeline(include_market_data=False).fetch("AAPL")
        assert data.market_data == []
        assert errors == []

    def test_missing_data_map_is_keyed_by_period(self):
        data, _ = make_pipeline(shape="stable").fetch("AAPL")
        missing = data.missing_data_map()
        assert "FY2023" in missing
        # goodwill is absent from the stable balance-sheet fixture.
        assert "goodwill" in missing["FY2023"]["balance_sheet"]


class TestRun:
    def test_full_run_persists_everything(self, temp_db):
        report = make_pipeline().run("AAPL")
        assert report.status == "success"
        assert report.records_written == {
            "income_statements": 1,
            "balance_sheets": 1,
            "cash_flow_statements": 1,
            "market_data": 2,
        }

        with session_scope() as session:
            company = repository.get_company(session, "AAPL")
            assert company is not None
            assert company.name == "Apple Inc."
            assert company.is_financial_sector is False
            assert session.scalar(select(func.count()).select_from(IncomeStatement)) == 1
            assert session.scalar(select(func.count()).select_from(MarketData)) == 2

    def test_statement_values_round_trip(self, temp_db):
        make_pipeline().run("AAPL")
        with session_scope() as session:
            rows = repository.get_statements(session, "AAPL", "income_statement")
            assert len(rows) == 1
            assert rows[0].revenue == 383285000000
            assert rows[0].fiscal_year == 2023
            assert rows[0].period == "FY"
            assert rows[0].period_end_date == date(2023, 9, 30)
            assert rows[0].data_source == "fmp"

    def test_missing_fields_persisted_as_json(self, temp_db):
        make_pipeline(shape="stable").run("AAPL")
        with session_scope() as session:
            sheet = session.scalar(select(BalanceSheet))
            assert sheet.is_missing_json["goodwill"] == "not_reported"

    def test_rerun_updates_rather_than_duplicates(self, temp_db):
        make_pipeline().run("AAPL")
        make_pipeline().run("AAPL")
        with session_scope() as session:
            assert session.scalar(select(func.count()).select_from(Company)) == 1
            assert session.scalar(select(func.count()).select_from(IncomeStatement)) == 1
            assert session.scalar(select(func.count()).select_from(MarketData)) == 2

    def test_restatement_overwrites_stale_figures(self, temp_db):
        make_pipeline().run("AAPL")
        restated = fixtures.statements_payload("v3")
        restated["income_statement"] = [{**fixtures.INCOME_V3, "revenue": 400000000000}]
        make_pipeline(statements=restated).run("AAPL")
        with session_scope() as session:
            row = session.scalar(select(IncomeStatement))
            assert row.revenue == 400000000000

    def test_run_is_partial_when_market_data_missing(self, temp_db):
        report = make_pipeline(yf_fail=True).run("AAPL")
        assert report.status == "partial"
        assert any("market_data" in e for e in report.errors)

    def test_run_fails_when_no_statements(self, temp_db):
        empty = {"income_statement": [], "balance_sheet": [], "cash_flow": []}
        report = make_pipeline(statements=empty).run("AAPL")
        assert report.status == "failed"

    def test_critical_missing_fields_reported(self, temp_db):
        stripped = fixtures.statements_payload("v3")
        stripped["income_statement"] = [
            {k: v for k, v in fixtures.INCOME_V3.items() if k != "revenue"}
        ]
        report = make_pipeline(statements=stripped).run("AAPL")
        assert "revenue" in report.critical_missing["FY2023"]
        assert report.status == "partial"

    def test_ingestion_run_is_audited(self, temp_db):
        make_pipeline().run("AAPL")
        with session_scope() as session:
            run = session.scalar(select(IngestionRun))
            assert run.ticker == "AAPL"
            assert run.status == "success"
            assert run.period == "annual"
            assert run.finished_at is not None

    def test_failed_run_leaves_no_orphaned_company(self, temp_db):
        class FlakyStub(StubFMPClient):
            def get_all_statements(self, symbol, period="annual", limit=5):
                raise FMPError("upstream exploded (403 Legacy Endpoint)")

        pipeline = IngestionPipeline(
            fmp_client=FlakyStub(), yf_client=StubYFClient(), include_market_data=True
        )
        report = pipeline.run("BAD")

        assert report.status == "failed"
        with session_scope() as session:
            # The profile fetch upserts Company before statements are known to have
            # failed; a failed run must not leave that row behind as a phantom.
            assert repository.get_company(session, "BAD") is None
            assert session.scalar(select(func.count()).select_from(Company)) == 0
            run = session.scalar(select(IngestionRun).where(IngestionRun.ticker == "BAD"))
            assert run is not None
            assert run.status == "failed"

    def test_failed_rerun_does_not_delete_existing_company(self, temp_db):
        make_pipeline().run("AAPL")  # successful first run, real data on file

        class FlakyStub(StubFMPClient):
            def get_all_statements(self, symbol, period="annual", limit=5):
                raise FMPError("upstream exploded (403 Legacy Endpoint)")

        pipeline = IngestionPipeline(
            fmp_client=FlakyStub(), yf_client=StubYFClient(), include_market_data=True
        )
        report = pipeline.run("AAPL")

        assert report.status == "failed"
        with session_scope() as session:
            company = repository.get_company(session, "AAPL")
            assert company is not None  # pre-existing company must survive
            assert company.name == "Apple Inc."
            assert session.scalar(select(func.count()).select_from(IncomeStatement)) == 1

    def test_run_many_continues_past_a_failure(self, temp_db):
        class FlakyStub(StubFMPClient):
            def get_all_statements(self, symbol, period="annual", limit=5):
                if symbol == "BAD":
                    raise FMPError("upstream exploded")
                return fixtures.statements_payload("v3")

        pipeline = IngestionPipeline(
            fmp_client=FlakyStub(), yf_client=StubYFClient(), include_market_data=True
        )
        reports = pipeline.run_many(["BAD", "AAPL"])

        assert [r.status for r in reports] == ["failed", "success"]
        with session_scope() as session:
            # The good ticker still landed; the bad one is audited, not silent.
            assert repository.get_company(session, "AAPL") is not None
            statuses = {r.ticker: r.status for r in session.scalars(select(IngestionRun))}
            assert statuses["BAD"] == "failed"


class TestRepositoryHelpers:
    def test_get_company_is_case_insensitive(self, temp_db):
        make_pipeline().run("AAPL")
        with session_scope() as session:
            assert repository.get_company(session, "aapl") is not None

    def test_latest_market_data_is_most_recent(self, temp_db):
        make_pipeline().run("AAPL")
        with session_scope() as session:
            latest = repository.get_latest_market_data(session, "AAPL")
            assert latest.date == date(2023, 9, 29)

    def test_unknown_ticker_returns_empty(self, temp_db):
        with session_scope() as session:
            assert repository.get_company(session, "ZZZZ") is None
            assert repository.get_statements(session, "ZZZZ", "income_statement") == []

    def test_partial_profile_does_not_erase_known_values(self, temp_db):
        make_pipeline().run("AAPL")
        from src.ingestion.schemas import CompanyProfile

        with session_scope() as session:
            repository.upsert_company(session, CompanyProfile(ticker="AAPL"))
        with session_scope() as session:
            company = repository.get_company(session, "AAPL")
            assert company.name == "Apple Inc."  # not clobbered by an empty profile


class TestFinancialSectorFlag:
    def test_bank_is_flagged_for_z_score_exclusion(self, temp_db):
        class BankStub(StubFMPClient):
            def get_profile(self, symbol):
                return fixtures.PROFILE_BANK

        pipeline = IngestionPipeline(
            fmp_client=BankStub(), yf_client=StubYFClient(), include_market_data=False
        )
        pipeline.run("JPM")
        with session_scope() as session:
            assert repository.get_company(session, "JPM").is_financial_sector is True
