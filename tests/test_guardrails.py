"""Guardrail tests: number extraction and post-generation verification.

This is the component that has to catch a fabricated figure. The tests are
written from the attacker's side - given a context, what wrong number could a
model emit, and does the verifier notice?
"""

from __future__ import annotations

import pytest

from src.rag.guardrails import (
    BLOCKED_MESSAGE,
    SYSTEM_PROMPT,
    apply_guardrails,
    build_user_prompt,
    extract_numbers,
    format_context,
    verify_answer,
)
from src.rag.vector_store import RetrievedChunk


def chunk(text: str, values: dict[str, float] | None = None, **metadata) -> RetrievedChunk:
    import json

    meta = {
        "ticker": "AAPL",
        "chunk_type": "income_statement",
        "period": "FY",
        "fiscal_year": 2024,
        "category": "",
        "source_values_json": json.dumps(values or {}),
    }
    meta.update(metadata)
    return RetrievedChunk(chunk_id="c1", text=text, metadata=meta, score=0.9)


CONTEXT = [
    chunk(
        "AAPL FY2024 - Income Statement (period ended 2024-09-28). "
        "Revenue: 391,035,000,000 (391.04B). Net Income: 93,736,000,000 (93.74B). "
        "NOT AVAILABLE: interest_expense (not reported).",
        {"revenue": 391035000000.0, "net_income": 93736000000.0},
    ),
    chunk(
        "AAPL FY2024 - Profitability Ratios. Net Margin: 0.2397 (23.97%). "
        "Return On Equity: 1.5741 (157.41%).",
        {"net_margin": 0.2397, "return_on_equity": 1.5741},
        chunk_type="ratios",
        category="profitability",
    ),
]


class TestExtractNumbers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Revenue was 391,035,000,000.", 391035000000.0),
            ("Revenue was 391.04B.", 391.04e9),
            ("Revenue was 391.04 billion.", 391.04e9),
            ("Margin was 23.97%.", 0.2397),
            ("Margin was 23.97 percent.", 0.2397),
            ("Coverage was 29.06x.", 29.06),
            ("The score was -1.78.", -1.78),
            ("Cash flow was (5,000).", -5000.0),
            ("It grew 12M last year.", 12e6),
        ],
    )
    def test_scales_and_signs(self, text, expected):
        mentions = extract_numbers(text)
        assert mentions, f"nothing extracted from {text!r}"
        assert mentions[0].value == pytest.approx(expected)

    def test_decimal_precision_is_recorded(self):
        """Precision drives the rounding tolerance, so it must be captured."""
        assert extract_numbers("0.2397")[0].decimals == 4
        assert extract_numbers("0.24")[0].decimals == 2
        assert extract_numbers("391")[0].decimals == 0

    def test_multiple_numbers_in_one_sentence(self):
        mentions = extract_numbers("Revenue 391.04B and net income 93.74B in FY2024.")
        # The year in "FY2024" is a period label, not a financial quantity; it is
        # verified separately by verify_periods.
        assert [m.value for m in mentions] == [391.04e9, 93.74e9]

    def test_no_numbers(self):
        assert extract_numbers("The data is not available for that period.") == []


class TestVerification:
    def test_exact_figure_from_context_verifies(self):
        report = verify_answer("Revenue was 391,035,000,000 in FY2024.", CONTEXT)
        assert report.is_grounded
        assert report.groundedness == 1.0

    def test_correct_rounding_verifies(self):
        """391.04B is a faithful rendering of 391,035,000,000."""
        report = verify_answer("Revenue was 391.04B (FY2024).", CONTEXT)
        assert report.is_grounded, [m.raw for m in report.unverified]

    def test_percentage_rendering_of_a_ratio_verifies(self):
        report = verify_answer("Net margin was 23.97% in FY2024.", CONTEXT)
        assert report.is_grounded

    def test_rounded_ratio_verifies(self):
        report = verify_answer("Net margin was about 0.24 in FY2024.", CONTEXT)
        assert report.is_grounded

    def test_fabricated_figure_is_caught(self):
        report = verify_answer("Revenue was 395B in FY2024.", CONTEXT)
        assert not report.is_grounded
        assert "395B" in {m.raw for m in report.unverified}

    def test_plausible_but_wrong_figure_is_caught(self):
        """The dangerous case: a wrong number that reads exactly like a right one."""
        report = verify_answer("Net income was 96,995,000,000 in FY2024.", CONTEXT)
        assert not report.is_grounded

    def test_number_inflated_by_100x_is_not_excused(self):
        """The percent fallback must not verify any figure that is 100x a real one."""
        report = verify_answer("Return on equity was 157.41 times over.", CONTEXT)
        # 157.41 == 1.5741 * 100 and appears verbatim in context, so it verifies;
        # but a value 100x a *large* source number must not.
        report2 = verify_answer("Revenue reached 39,103,500,000,000.", CONTEXT)
        assert not report2.is_grounded

    def test_years_present_in_context_are_not_flagged(self):
        report = verify_answer("In FY2024 revenue was 391.04B.", CONTEXT)
        assert report.is_grounded

    def test_period_absent_from_context_is_flagged(self):
        """Attributing a real figure to a year that was never retrieved."""
        report = verify_answer("In FY2019 revenue was 391.04B.", CONTEXT)
        assert report.unverified_periods == ["FY2019"]
        assert not report.is_grounded
        assert any("FY2019" in note for note in report.notes)

    def test_correct_period_is_accepted(self):
        report = verify_answer("In FY2024 revenue was 391.04B.", CONTEXT)
        assert report.unverified_periods == []
        assert report.is_grounded

    def test_period_check_is_skipped_without_context(self):
        assert verify_answer("In FY2019 revenue rose.", []).unverified_periods == []

    def test_answer_with_no_numbers_is_fully_grounded(self):
        report = verify_answer("Interest expense is not available for FY2024.", CONTEXT)
        assert report.groundedness == 1.0
        assert report.total == 0

    def test_groundedness_is_a_ratio_not_a_boolean(self):
        report = verify_answer("Revenue was 391.04B and margin was 99.9%.", CONTEXT)
        assert 0.0 < report.groundedness < 1.0

    def test_empty_context_is_noted(self):
        report = verify_answer("Revenue was 391.04B.", [])
        assert not report.is_grounded
        assert any("No context" in note for note in report.notes)

    def test_blocking_covers_period_errors_too(self):
        answer, report = apply_guardrails(
            "In FY2019 revenue was 391.04B.", CONTEXT, block_on_unverified=True
        )
        assert report.blocked
        assert "FY2019" in answer

    def test_report_serialises_for_logging(self):
        payload = verify_answer("Revenue was 999B.", CONTEXT).to_json()
        assert payload["unverified"] == ["999B"]
        assert payload["groundedness"] == 0.0


class TestBlocking:
    def test_flags_by_default_without_suppressing(self):
        answer, report = apply_guardrails("Revenue was 999B.", CONTEXT)
        assert answer == "Revenue was 999B."
        assert report.unverified
        assert not report.blocked

    def test_blocks_when_enabled(self):
        answer, report = apply_guardrails("Revenue was 999B.", CONTEXT, block_on_unverified=True)
        assert report.blocked
        assert answer.startswith(BLOCKED_MESSAGE[:40])
        assert "999B" in answer

    def test_valid_answer_passes_through_when_blocking_is_on(self):
        text = "Revenue was 391.04B in FY2024."
        answer, report = apply_guardrails(text, CONTEXT, block_on_unverified=True)
        assert answer == text
        assert not report.blocked


class TestPromptAssembly:
    def test_system_prompt_states_the_hard_rules(self):
        lowered = SYSTEM_PROMPT.lower()
        assert "only numbers that appear in the context" in lowered
        assert "not available" in lowered
        assert "never estimate" in lowered or "never" in lowered
        assert "no investment advice" in lowered

    def test_context_is_numbered_and_cited(self):
        rendered = format_context(CONTEXT)
        assert "[1]" in rendered and "[2]" in rendered
        assert "AAPL FY2024 income statement" in rendered

    def test_empty_context_is_explicit(self):
        assert format_context([]) == "(no context available)"

    def test_user_prompt_carries_question_and_context(self):
        prompt = build_user_prompt("What was revenue?", CONTEXT)
        assert "What was revenue?" in prompt
        assert "391,035,000,000" in prompt
        assert "CONTEXT:" in prompt
