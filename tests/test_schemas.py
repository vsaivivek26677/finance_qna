"""Schema-level guarantees: period handling, missing maps, sector classification."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.ingestion.schemas import (
    BalanceSheetRecord,
    CashFlowRecord,
    CompanyFinancials,
    CompanyProfile,
    IncomeStatementRecord,
    IngestionReport,
    MissingReason,
    PeriodType,
)


class TestPeriodValidation:
    @pytest.mark.parametrize(
        ("raw", "expected"), [("FY", "FY"), ("fy", "FY"), ("annual", "FY"), ("", "FY"), ("q2", "Q2")]
    )
    def test_period_normalization(self, raw, expected):
        rec = IncomeStatementRecord(ticker="AAPL", period=raw, fiscal_year=2023)
        assert rec.period == expected

    @pytest.mark.parametrize("raw", ["Q5", "H1", "FY2023", "junk"])
    def test_invalid_period_rejected(self, raw):
        with pytest.raises(ValidationError):
            IncomeStatementRecord(ticker="AAPL", period=raw, fiscal_year=2023)

    @pytest.mark.parametrize("year", [1800, 3000, 0])
    def test_implausible_fiscal_year_rejected(self, year):
        with pytest.raises(ValidationError):
            IncomeStatementRecord(ticker="AAPL", fiscal_year=year)

    def test_period_label(self):
        annual = IncomeStatementRecord(ticker="AAPL", period="FY", fiscal_year=2023)
        quarterly = IncomeStatementRecord(ticker="AAPL", period="Q3", fiscal_year=2024)
        assert annual.period_label == "FY2023"
        assert quarterly.period_label == "Q3 2024"

    def test_ticker_is_normalized(self):
        assert IncomeStatementRecord(ticker=" aapl ", fiscal_year=2023).ticker == "AAPL"


class TestMissingSemantics:
    def test_all_numeric_fields_default_to_none_not_zero(self):
        rec = BalanceSheetRecord(ticker="X", fiscal_year=2023)
        assert rec.total_assets is None
        assert rec.inventory is None

    def test_missing_json_serializes_reasons(self):
        rec = IncomeStatementRecord(
            ticker="X",
            fiscal_year=2023,
            missing_fields={"ebitda": MissingReason.NOT_REPORTED},
        )
        assert rec.missing_json() == {"ebitda": "not_reported"}

    def test_missing_critical_fields_per_statement_kind(self):
        income = IncomeStatementRecord(ticker="X", fiscal_year=2023, revenue=100.0)
        # net_income and operating_income are critical and absent; revenue is present.
        assert income.missing_critical_fields() == {"net_income", "operating_income"}

        balance = BalanceSheetRecord(ticker="X", fiscal_year=2023)
        assert "total_assets" in balance.missing_critical_fields()

        cash = CashFlowRecord(ticker="X", fiscal_year=2023, operating_cash_flow=1.0)
        assert cash.missing_critical_fields() == set()

    def test_zero_is_not_treated_as_missing(self):
        income = IncomeStatementRecord(
            ticker="X", fiscal_year=2023, revenue=0.0, net_income=0.0, operating_income=0.0
        )
        assert income.missing_critical_fields() == set()


class TestCompanyProfile:
    @pytest.mark.parametrize(
        "sector", ["Financial Services", "financials", "Banks", "Insurance", "Capital Markets"]
    )
    def test_financial_sectors_flagged(self, sector):
        assert CompanyProfile(ticker="X", sector=sector).is_financial_sector is True

    @pytest.mark.parametrize("sector", ["Technology", "Healthcare", "Energy", None])
    def test_non_financial_sectors_not_flagged(self, sector):
        assert CompanyProfile(ticker="X", sector=sector).is_financial_sector is False


class TestCompanyFinancials:
    def _bundle(self) -> CompanyFinancials:
        return CompanyFinancials(
            profile=CompanyProfile(ticker="X", sector="Technology"),
            income_statements=[
                IncomeStatementRecord(
                    ticker="X",
                    fiscal_year=2023,
                    missing_fields={"ebitda": MissingReason.NOT_REPORTED},
                ),
                IncomeStatementRecord(ticker="X", fiscal_year=2022),
            ],
            balance_sheets=[
                BalanceSheetRecord(
                    ticker="X",
                    fiscal_year=2023,
                    missing_fields={"goodwill": MissingReason.NULL_IN_SOURCE},
                )
            ],
        )

    def test_fiscal_years_descending(self):
        assert self._bundle().fiscal_years == [2023, 2022]

    def test_missing_data_map_shape(self):
        missing = self._bundle().missing_data_map()
        assert missing["FY2023"]["income_statement"] == ["ebitda"]
        assert missing["FY2023"]["balance_sheet"] == ["goodwill"]
        assert "FY2022" not in missing  # nothing missing there


class TestIngestionReport:
    def test_summary_line(self):
        report = IngestionReport(
            ticker="AAPL",
            period=PeriodType.ANNUAL,
            status="partial",
            records_written={"income_statements": 5, "balance_sheets": 5},
        )
        line = report.summary_line()
        assert "PARTIAL" in line and "AAPL" in line and "income_statements=5" in line

    def test_empty_report_summary(self):
        report = IngestionReport(ticker="X", period=PeriodType.QUARTER)
        assert "no records" in report.summary_line()
