"""Dashboard tests.

Streamlit's rendering is not unit-testable without its runtime, so what is tested
here is everything underneath it: the API client's error handling, and the pure
formatting functions that decide whether a reader sees "24%" or "0.2397".

Those formatters are where a finance dashboard actually goes wrong. A margin
rendered as a raw decimal, EPS shown as currency-rounded, or a falling leverage
ratio painted red are all silent errors that look fine until someone acts on them.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.dashboard.api_client import DEFAULT_TIMEOUT, LONG_TIMEOUT, ApiClient, ApiError
from src.dashboard.components import charts, panels

BASE = "http://testserver"


@pytest.fixture
def client() -> ApiClient:
    return ApiClient(base_url=BASE, timeout=5.0)


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (416_161_000_000, "416.16B"),
            (5_500_000, "5.50M"),
            (-23_405_000_000, "-23.41B"),
            (1_234, "1.23K"),
            (250, "250"),
            (None, "N/A"),
        ],
    )
    def test_money_is_compact(self, value, expected):
        assert charts.format_money(value) == expected

    def test_currency_prefix(self):
        assert charts.format_money(416_161_000_000, "USD") == "USD 416.16B"

    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("net_margin", 0.2692, "26.92%"),
            ("return_on_equity", 1.7142, "171.42%"),
            ("current_ratio", 0.8933, "0.89"),
            ("debt_to_equity", 1.5241, "1.52"),
            ("working_capital", -17_674_000_000, "-17.67B"),
            ("net_margin", None, "N/A"),
        ],
    )
    def test_ratios_render_in_their_natural_units(self, name, value, expected):
        """A margin shown as 0.2692 is the classic finance-dashboard error."""
        assert charts.format_ratio(name, value) == expected

    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("eps", 7.24, "7.24"),
            ("eps_diluted", 6.60, "6.60"),
            ("weighted_average_shares", 14_950_000_000, "14.95B"),
            ("revenue", 416_161_000_000, "USD 416.16B"),
        ],
    )
    def test_statement_values_respect_what_the_field_measures(self, name, value, expected):
        """EPS is per-share and share counts are units - neither is currency."""
        assert charts.format_statement_value(name, value, "USD") == expected

    def test_humanise_preserves_finance_acronyms(self):
        assert charts.humanise("return_on_equity") == "Return On Equity"
        assert charts.humanise("ev_to_ebitda") == "EV To EBITDA"

    def test_basis_is_condensed_for_tables(self):
        assert charts.short_basis("average of opening and closing balance") == "Average balance"
        assert charts.short_basis(None) == "-"
        assert len(charts.short_basis("x" * 80)) <= 28

    def test_lower_is_better_set_covers_leverage(self):
        """Falling leverage must be able to render green, not red."""
        assert "debt_to_equity" in charts.LOWER_IS_BETTER
        assert "net_margin" not in charts.LOWER_IS_BETTER


class TestDeltas:
    """A percentage ratio's YoY change is in points, not a relative percent,
    and it must not render as the cryptic "pp"."""

    def test_percentage_ratio_delta_is_in_points(self):
        # net margin 24.00% -> 26.15% is a +2.15 point move.
        assert panels._delta(0.2615, 0.2400, "net_margin") == "+2.15 pts"

    def test_percentage_ratio_delta_never_uses_pp(self):
        assert "pp" not in panels._delta(0.10, 0.05, "return_on_equity")

    def test_multiple_ratio_delta_is_relative_percent(self):
        # current ratio 1.20 -> 1.05 is a -12.5% relative move.
        assert panels._delta(1.05, 1.20, "current_ratio") == "-12.5%"

    def test_money_ratio_delta_is_compact_currency(self):
        assert panels._delta(3.0e9, 1.0e9, "free_cash_flow") == "2.00B"

    def test_delta_is_none_without_a_prior_value(self):
        assert panels._delta(0.25, None, "net_margin") is None
        assert panels._delta(1.0, 0.0, "current_ratio") is None


class TestCharts:
    def test_trend_needs_two_points(self):
        assert charts.trend_chart({2023: 0.25}, "net_margin") is None
        assert charts.trend_chart({2022: 0.2, 2023: 0.25}, "net_margin") is not None

    def test_trend_ignores_missing_years(self):
        figure = charts.trend_chart({2021: 0.2, 2022: None, 2023: 0.25}, "net_margin")
        assert list(figure.data[0].x) == ["FY2021", "FY2023"]

    def test_percentage_ratios_are_scaled_for_display(self):
        figure = charts.trend_chart({2022: 0.20, 2023: 0.25}, "net_margin")
        assert list(figure.data[0].y) == [20.0, 25.0]

    def test_multiples_are_not_scaled(self):
        figure = charts.trend_chart({2022: 1.2, 2023: 1.5}, "current_ratio")
        assert list(figure.data[0].y) == [1.2, 1.5]

    def test_multi_trend_returns_none_when_nothing_plottable(self):
        assert charts.multi_trend_chart({"net_margin": {2023: 0.2}}, "x") is None

    def test_gauge_handles_a_missing_score(self):
        assert charts.altman_gauge(None, None) is not None

    def test_piotroski_bar_has_nine_segments(self):
        assert len(charts.piotroski_bar(6).data) == 9


class TestApiClient:
    @respx.mock
    def test_health(self, client):
        respx.get(f"{BASE}/health").mock(return_value=httpx.Response(200, json={"status": "ok"}))
        assert client.health()["status"] == "ok"

    @respx.mock
    def test_query_params_omit_none(self, client):
        route = respx.get(f"{BASE}/companies/AAPL/ratios").mock(
            return_value=httpx.Response(200, json={})
        )
        client.ratios("AAPL", fiscal_year=None, category="liquidity")
        params = route.calls.last.request.url.params
        assert "fiscal_year" not in params
        assert params["category"] == "liquidity"

    @respx.mock
    def test_ask_posts_the_question(self, client):
        route = respx.post(f"{BASE}/companies/AAPL/ask").mock(
            return_value=httpx.Response(200, json={"answer": "x"})
        )
        client.ask("AAPL", "What was revenue?", fiscal_year=2024)
        import json as _json

        body = _json.loads(route.calls.last.request.content)
        assert body == {"question": "What was revenue?", "fiscal_year": 2024}

    @respx.mock
    def test_backend_error_detail_is_surfaced(self, client):
        """The user should see the backend's own message, not a status code."""
        respx.get(f"{BASE}/companies/ZZZZ").mock(
            return_value=httpx.Response(404, json={"detail": "ZZZZ is not in the database."})
        )
        with pytest.raises(ApiError) as caught:
            client.profile("ZZZZ")
        assert "not in the database" in caught.value.message
        assert caught.value.status_code == 404

    @respx.mock
    def test_validation_errors_are_readable(self, client):
        respx.post(f"{BASE}/companies/AAPL/ask").mock(
            return_value=httpx.Response(
                422,
                json={"detail": [{"msg": "String too short", "loc": ["body", "question"]}]},
            )
        )
        with pytest.raises(ApiError, match="String too short"):
            client.ask("AAPL", "x")

    @respx.mock
    def test_connection_failure_explains_how_to_fix_it(self, client):
        """A blank page helps nobody; the hint names the command to run."""
        respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ApiError) as caught:
            client.health()
        assert "Cannot reach the API" in caught.value.message
        assert "uvicorn" in caught.value.hint

    @respx.mock
    def test_timeout_is_a_typed_error(self, client):
        respx.get(f"{BASE}/companies").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ApiError, match="did not respond in time"):
            client.companies()

    @respx.mock
    def test_non_json_error_still_produces_a_message(self, client):
        respx.get(f"{BASE}/companies").mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(ApiError, match="500"):
            client.companies()

    @respx.mock
    def test_generation_uses_a_longer_timeout_than_reads(self, client, monkeypatch):
        """A 30s read timeout would abort a summary that is progressing normally."""
        assert LONG_TIMEOUT > DEFAULT_TIMEOUT

        seen: list[float | None] = []
        real_request = httpx.request

        def capture(method, url, **kwargs):
            seen.append(kwargs.get("timeout"))
            return real_request(method, url, **kwargs)

        monkeypatch.setattr("src.dashboard.api_client.httpx.request", capture)
        respx.get(f"{BASE}/companies/AAPL/summary").mock(
            return_value=httpx.Response(200, json={"answer": "x"})
        )
        respx.get(f"{BASE}/companies").mock(return_value=httpx.Response(200, json=[]))

        client.summary("AAPL")
        client.companies()
        # Generation overrides with LONG_TIMEOUT; reads use the client's own.
        assert seen == [LONG_TIMEOUT, client.timeout]
