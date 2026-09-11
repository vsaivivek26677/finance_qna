"""Normalizer tests — the layer that must never silently invent a number."""

from __future__ import annotations

from datetime import date

import pytest

from src.ingestion.normalizer import (
    build_balance_sheet,
    build_cash_flow,
    build_income_statement,
    build_profile,
    build_statements,
    coerce_date,
    coerce_number,
    extract_numeric_fields,
)
from src.ingestion.schemas import MissingReason
from tests import fixtures


class TestCoerceNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (42, 42.0),
            (42.5, 42.5),
            ("42", 42.0),
            ("1,234.5", 1234.5),
            ("$1,234", 1234.0),
            ("(500)", -500.0),  # accounting notation for negative
            (-7, -7.0),
            (0, 0.0),
        ],
    )
    def test_parses_numbers(self, raw, expected):
        assert coerce_number(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "-", "N/A", "None", "null", "abc", [], {}])
    def test_rejects_non_numbers(self, raw):
        assert coerce_number(raw) is None

    def test_booleans_are_not_numbers(self):
        # True -> 1.0 would fabricate a financial figure out of a flag.
        assert coerce_number(True) is None
        assert coerce_number(False) is None

    def test_zero_is_preserved_not_treated_as_missing(self):
        assert coerce_number(0) == 0.0
        assert coerce_number("0") == 0.0


class TestCoerceDate:
    def test_iso_date(self):
        assert coerce_date("2023-09-30") == date(2023, 9, 30)

    def test_datetime_string(self):
        assert coerce_date("2023-11-02 18:08:27") == date(2023, 11, 2)

    def test_passthrough_date(self):
        assert coerce_date(date(2020, 1, 1)) == date(2020, 1, 1)

    @pytest.mark.parametrize("raw", [None, "", "not-a-date"])
    def test_unusable_returns_none(self, raw):
        assert coerce_date(raw) is None


class TestExtractNumericFields:
    ALIASES = {"revenue": ["revenue"], "profit": ["grossProfit", "profit"]}

    def test_missing_key_is_flagged_not_reported(self):
        values, missing = extract_numeric_fields({"revenue": 100}, self.ALIASES)
        assert values["profit"] is None
        assert missing["profit"] is MissingReason.NOT_REPORTED
        assert "revenue" not in missing

    def test_explicit_null_is_distinguished_from_absent(self):
        values, missing = extract_numeric_fields({"revenue": 100, "profit": None}, self.ALIASES)
        assert missing["profit"] is MissingReason.NULL_IN_SOURCE

    def test_unparseable_value_is_flagged(self):
        values, missing = extract_numeric_fields({"revenue": "lots"}, self.ALIASES)
        assert values["revenue"] is None
        assert missing["revenue"] is MissingReason.UNPARSEABLE

    def test_second_alias_used_when_first_absent(self):
        values, missing = extract_numeric_fields({"profit": 50}, self.ALIASES)
        assert values["profit"] == 50.0
        assert "profit" not in missing

    def test_zero_is_a_real_value_not_a_miss(self):
        values, missing = extract_numeric_fields({"revenue": 0, "profit": 0}, self.ALIASES)
        assert values["revenue"] == 0.0
        assert missing == {}


class TestBuildStatements:
    def test_income_statement_from_v3(self):
        rec = build_income_statement(fixtures.INCOME_V3, "AAPL")
        assert rec.ticker == "AAPL"
        assert rec.period == "FY"
        assert rec.fiscal_year == 2023
        assert rec.period_end_date == date(2023, 9, 30)
        assert rec.filing_date == date(2023, 11, 3)
        assert rec.reported_currency == "USD"
        assert rec.revenue == 383285000000
        assert rec.eps_diluted == 6.13
        assert rec.missing_fields == {}

    def test_v3_and_stable_shapes_normalize_identically(self):
        """The two FMP API generations must produce the same canonical record."""
        for builder, v3, stable in (
            (build_income_statement, fixtures.INCOME_V3, fixtures.INCOME_STABLE),
            (build_cash_flow, fixtures.CASH_FLOW_V3, fixtures.CASH_FLOW_STABLE),
        ):
            a = builder(v3, "AAPL").model_dump(exclude={"fetched_at"})
            b = builder(stable, "AAPL").model_dump(exclude={"fetched_at"})
            assert a == b, f"{builder.__name__} differs between API shapes"

    def test_balance_sheet_alias_fallback(self):
        v3 = build_balance_sheet(fixtures.BALANCE_V3, "AAPL")
        stable = build_balance_sheet(fixtures.BALANCE_STABLE, "AAPL")
        # netReceivables (v3) and accountsReceivables (stable) are the same field.
        assert v3.net_receivables == stable.net_receivables == 60985000000

    def test_field_absent_only_from_stable_is_flagged(self):
        stable = build_balance_sheet(fixtures.BALANCE_STABLE, "AAPL")
        assert stable.goodwill is None
        assert stable.missing_fields["goodwill"] is MissingReason.NOT_REPORTED

    def test_fiscal_year_falls_back_to_statement_date(self):
        raw = {k: v for k, v in fixtures.INCOME_V3.items() if k != "calendarYear"}
        rec = build_income_statement(raw, "AAPL")
        assert rec.fiscal_year == 2023

    def test_row_without_year_or_date_is_rejected(self):
        raw = {k: v for k, v in fixtures.INCOME_V3.items() if k not in {"calendarYear", "date"}}
        with pytest.raises(ValueError, match="fiscal year"):
            build_income_statement(raw, "AAPL")

    def test_quarterly_period_preserved(self):
        raw = {**fixtures.INCOME_V3, "period": "Q3"}
        assert build_income_statement(raw, "AAPL", period_hint="Q1").period == "Q3"

    def test_bad_row_is_skipped_not_fatal(self):
        payloads = {
            "income_statement": [fixtures.INCOME_V3, {"revenue": 1}],  # 2nd has no year/date
            "balance_sheet": [fixtures.BALANCE_V3],
            "cash_flow": [fixtures.CASH_FLOW_V3],
        }
        income, balance, cash, errors = build_statements(payloads, "AAPL")
        assert len(income) == 1
        assert len(balance) == 1 and len(cash) == 1
        assert len(errors) == 1 and "income_statement[1]" in errors[0]


class TestBuildProfile:
    def test_profile_fields(self):
        profile = build_profile(fixtures.PROFILE_V3, "AAPL")
        assert profile.name == "Apple Inc."
        assert profile.sector == "Technology"
        assert profile.exchange == "NASDAQ"  # exchangeShortName wins over exchange
        assert profile.market_cap == 2794144143000
        assert profile.is_financial_sector is False

    def test_financial_sector_detected(self):
        profile = build_profile(fixtures.PROFILE_BANK, "JPM")
        assert profile.is_financial_sector is True

    def test_missing_profile_fields_flagged(self):
        profile = build_profile({"companyName": "Tiny Co"}, "TINY")
        assert profile.sector is None
        assert profile.missing_fields["sector"] is MissingReason.NOT_REPORTED
        assert profile.is_financial_sector is False
