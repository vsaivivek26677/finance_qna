"""Chunker tests.

The chunks are the model's entire view of the world, so what matters most is not
what they contain but what they refuse to leave out: a field the source never
reported must appear as "NOT AVAILABLE", not as silence.
"""

from __future__ import annotations

import json

import pytest

from src.db.database import session_scope
from src.rag.chunker import (
    DATA_AVAILABILITY,
    DISTRESS,
    INCOME_STATEMENT,
    PROFILE,
    RATIOS,
    RED_FLAG,
    build_chunks,
    format_money,
    format_ratio,
    humanise,
)
from tests.db_fixtures import seed_and_analyze


@pytest.fixture
def chunks(temp_db):
    with session_scope() as session:
        seed_and_analyze(session, "AAPL", years=(2022, 2023))
        return build_chunks(session, "AAPL")


def by_type(chunks, chunk_type, year=None):
    return [
        c for c in chunks if c.chunk_type == chunk_type and (year is None or c.fiscal_year == year)
    ]


def one(chunks, chunk_type, year=None):
    found = by_type(chunks, chunk_type, year)
    assert found, f"no {chunk_type} chunk for {year}"
    return found[0]


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (391035000000, "391,035,000,000 (391.04B)"),
            (5_500_000, "5,500,000 (5.50M)"),
            (1234, "1,234"),
            (-23405, "-23,405"),
        ],
    )
    def test_money_shows_exact_and_human_scale(self, value, expected):
        """Both renderings appear so the model can quote either one faithfully."""
        assert format_money(value) == expected

    def test_percent_ratios_get_both_forms(self):
        assert format_ratio("net_margin", 0.2397) == "0.2397 (23.97%)"

    def test_multiples_stay_plain(self):
        assert format_ratio("current_ratio", 0.8673) == "0.8673"

    def test_monetary_ratios_are_formatted_as_money(self):
        assert "(-23.41M)" in format_ratio("working_capital", -23405000)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("net_margin", "Net Margin"), ("return_on_equity", "Return On Equity"), ("ebitda", "EBITDA")],
    )
    def test_humanise(self, raw, expected):
        assert humanise(raw) == expected


class TestCoverage:
    def test_every_chunk_type_is_produced(self, chunks):
        produced = {c.chunk_type for c in chunks}
        assert {PROFILE, INCOME_STATEMENT, RATIOS, DISTRESS, RED_FLAG, DATA_AVAILABILITY} <= produced

    def test_chunk_ids_are_deterministic_and_unique(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert one(chunks, INCOME_STATEMENT, 2023).chunk_id == "AAPL:FY2023:income_statement"

    def test_rebuilding_produces_identical_ids(self, temp_db):
        """Stable ids are what make re-indexing an update instead of a duplicate."""
        with session_scope() as session:
            seed_and_analyze(session, "AAPL", years=(2023,))
            first = {c.chunk_id for c in build_chunks(session, "AAPL")}
            second = {c.chunk_id for c in build_chunks(session, "AAPL")}
        assert first == second

    def test_unknown_ticker_yields_nothing(self, temp_db):
        with session_scope() as session:
            assert build_chunks(session, "ZZZZ") == []

    def test_every_chunk_is_labelled_with_company_and_period(self, chunks):
        for chunk in chunks:
            assert chunk.ticker == "AAPL"
            assert chunk.text.startswith("AAPL")
            if chunk.fiscal_year is not None:
                assert f"FY{chunk.fiscal_year}" in chunk.text


class TestStatementChunks:
    def test_figures_appear_in_both_renderings(self, chunks):
        text = one(chunks, INCOME_STATEMENT, 2023).text
        assert "383,285" in text
        assert "Revenue" in text

    def test_source_values_are_machine_readable(self, chunks):
        chunk = one(chunks, INCOME_STATEMENT, 2023)
        assert chunk.source_values["revenue"] == 383285
        assert chunk.source_values["net_income"] == 96995

    def test_period_context_is_stated(self, chunks):
        text = one(chunks, INCOME_STATEMENT, 2023).text
        assert "period ended 2023-09-30" in text
        assert "reported in USD" in text

    def test_complete_statement_says_so(self, chunks):
        assert "All fields listed above were reported" in one(chunks, INCOME_STATEMENT, 2023).text


class TestMissingDataIsStatedNotOmitted:
    def test_absent_field_is_named_as_not_available(self, temp_db):
        with session_scope() as session:
            seed_and_analyze(
                session,
                "AAPL",
                years=(2023,),
                income_overrides={2023: {"interest_expense": None}},
                missing={"interest_expense": "not_reported"},
            )
            chunks = build_chunks(session, "AAPL")

        text = one(chunks, INCOME_STATEMENT, 2023).text
        assert "NOT AVAILABLE" in text
        assert "interest_expense (not reported)" in text
        assert "have not been estimated" in text

    def test_uncalculable_ratios_are_named_with_their_reason(self, chunks):
        # FY2022 is the earliest year, so the year-over-year scores cannot run.
        text = one(chunks, DISTRESS, 2022).text
        assert "Piotroski F-Score: NOT AVAILABLE" in text
        assert "prior period" in text

    def test_availability_chunk_summarises_the_gaps(self, chunks):
        text = one(chunks, DATA_AVAILABILITY, 2022).text
        assert "Piotroski F Score" in text or "piotroski" in text.lower()
        assert "must be reported as NOT AVAILABLE rather than estimated" in text

    def test_availability_chunk_confirms_completeness_when_there_are_no_gaps(self, chunks):
        text = one(chunks, DATA_AVAILABILITY, 2023).text
        assert "no gaps" in text or "NOT AVAILABLE" in text


class TestDistressChunk:
    def test_score_zone_and_components_are_all_present(self, chunks):
        text = one(chunks, DISTRESS, 2023).text
        assert "Altman Z-Score:" in text
        assert "Safe Zone" in text
        assert "Safe above 2.99" in text
        assert "components:" in text

    def test_piotroski_signals_are_enumerated(self, chunks):
        text = one(chunks, DISTRESS, 2023).text
        assert "Signals passed:" in text
        assert "Signals failed:" in text

    def test_beneish_is_framed_as_a_screen_not_a_verdict(self, chunks):
        text = one(chunks, DISTRESS, 2023).text
        assert "screening indicator, not evidence of manipulation" in text

    def test_financial_sector_exclusion_is_explained(self, temp_db):
        with session_scope() as session:
            seed_and_analyze(
                session, "JPM", years=(2023,), sector="Financial Services", is_financial=True
            )
            chunks = build_chunks(session, "JPM")

        profile = one(chunks, PROFILE).text
        assert "NOT APPLICABLE to banks and insurers" in profile
        assert "Not Applicable - Financial Sector" in one(chunks, DISTRESS, 2023).text


class TestRedFlagChunks:
    def test_a_raised_flag_becomes_its_own_chunk(self, chunks):
        texts = [c.text for c in by_type(chunks, RED_FLAG, 2023)]
        assert any("Current Ratio Below 1.0" in t for t in texts)
        assert any("Medium severity" in t for t in texts)

    def test_flag_source_values_are_retained(self, chunks):
        flag = next(c for c in by_type(chunks, RED_FLAG, 2023) if "Current Ratio" in c.text)
        assert flag.source_values["current_ratio"] == pytest.approx(143566 / 145308)

    def test_absence_of_flags_is_itself_a_chunk(self, temp_db):
        """Without this, 'were there red flags?' would retrieve nothing at all."""
        with session_scope() as session:
            seed_and_analyze(
                session,
                "SAFE",
                years=(2023,),
                balance_overrides={2023: {"total_current_liabilities": 50000}},
            )
            chunks = build_chunks(session, "SAFE")

        flag_chunks = by_type(chunks, RED_FLAG, 2023)
        assert len(flag_chunks) == 1
        assert "No risk red flags were raised" in flag_chunks[0].text
        assert "computed result, not an absence of data" in flag_chunks[0].text


class TestMetadata:
    def test_metadata_is_flat_and_scalar_for_the_vector_store(self, chunks):
        for chunk in chunks:
            for key, value in chunk.metadata().items():
                assert isinstance(value, (str, int, float, bool)), f"{key} is {type(value)}"

    def test_profile_chunk_uses_the_no_year_sentinel(self, chunks):
        assert one(chunks, PROFILE).metadata()["fiscal_year"] == -1

    def test_source_values_round_trip_through_metadata(self, chunks):
        chunk = one(chunks, INCOME_STATEMENT, 2023)
        restored = json.loads(chunk.metadata()["source_values_json"])
        assert restored["revenue"] == 383285
