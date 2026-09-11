"""FMP client tests — quota safety, error mapping, API-generation fallback."""

from __future__ import annotations

import httpx
import pytest
import respx

from src.ingestion.fmp_client import (
    FMPAuthError,
    FMPClient,
    FMPEndpointUnavailableError,
    FMPError,
    FMPNotFoundError,
    FMPRateLimitError,
)
from tests import fixtures

BASE = "https://financialmodelingprep.com"
STABLE_INCOME = f"{BASE}/stable/income-statement"
V3_INCOME = f"{BASE}/api/v3/income-statement/AAPL"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch):
    """Retry backoff must not make the suite slow."""
    monkeypatch.setattr("src.ingestion.fmp_client.time.sleep", lambda *_: None)


class TestRequests:
    @respx.mock
    def test_successful_fetch(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(200, json=[fixtures.INCOME_V3])
        )
        rows = fmp_client.get_income_statement("AAPL", limit=1)
        assert len(rows) == 1
        assert rows[0]["revenue"] == 383285000000
        assert fmp_client.request_count == 1

    @respx.mock
    def test_api_key_is_sent(self, fmp_client: FMPClient):
        route = respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(200, json=[fixtures.INCOME_V3])
        )
        fmp_client.get_income_statement("AAPL")
        assert route.calls.last.request.url.params["apikey"] == "test-key"
        assert route.calls.last.request.url.params["symbol"] == "AAPL"

    def test_missing_key_raises_before_any_request(self):
        client = FMPClient(api_key=None, cache_raw=False)
        with pytest.raises(FMPAuthError, match="FMP_API_KEY"):
            client.get_income_statement("AAPL")
        assert client.request_count == 0

    @respx.mock
    def test_empty_list_is_not_found(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(return_value=httpx.Response(200, json=[]))
        with pytest.raises(FMPNotFoundError):
            fmp_client.get_income_statement("NOPE")


class TestErrorHandling:
    @respx.mock
    def test_rate_limit_is_retried_then_raises(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(return_value=httpx.Response(429, text="Limit Reach"))
        with pytest.raises(FMPError):
            fmp_client.get_income_statement("AAPL")
        assert fmp_client.request_count == fmp_client.max_retries

    @respx.mock
    def test_transient_500_then_success(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(
            side_effect=[
                httpx.Response(500, text="boom"),
                httpx.Response(200, json=[fixtures.INCOME_V3]),
            ]
        )
        assert len(fmp_client.get_income_statement("AAPL")) == 1
        assert fmp_client.request_count == 2

    @respx.mock
    def test_error_object_in_200_body_is_raised(self, fmp_client: FMPClient):
        # FMP frequently returns HTTP 200 with an error object in the payload.
        respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(200, json={"Error Message": "Invalid API KEY."})
        )
        with pytest.raises(FMPError, match="Invalid API KEY"):
            fmp_client.get_income_statement("AAPL")

    @respx.mock
    def test_quota_message_maps_to_rate_limit_error(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(200, json={"Error Message": "Limit Reach for today"})
        )
        with pytest.raises(FMPRateLimitError):
            fmp_client.get_income_statement("AAPL")

    @respx.mock
    def test_bad_key_raises_auth_error(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(return_value=httpx.Response(401, text="Invalid API key"))
        with pytest.raises(FMPAuthError):
            fmp_client.get_income_statement("AAPL")


class TestBudget:
    @respx.mock
    def test_daily_budget_is_enforced(self):
        client = FMPClient(api_key="k", daily_budget=2, cache_raw=False, max_retries=1)
        respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(200, json=[fixtures.INCOME_V3])
        )
        client.get_income_statement("AAPL")
        client.get_income_statement("AAPL")
        with pytest.raises(FMPRateLimitError, match="budget"):
            client.get_income_statement("AAPL")
        assert client.request_count == 2


class TestApiVariantFallback:
    @respx.mock
    def test_legacy_endpoint_error_falls_back_to_v3(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(
                403, text="Exclusive Endpoint: This endpoint is not available under your plan."
            )
        )
        respx.get(V3_INCOME).mock(return_value=httpx.Response(200, json=[fixtures.INCOME_V3]))

        rows = fmp_client.get_income_statement("AAPL")
        assert len(rows) == 1
        # The working variant sticks for the rest of the session.
        assert fmp_client.variant == "v3"

    @respx.mock
    def test_fallback_also_triggered_by_body_message(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(
                200, json={"Error Message": "Legacy Endpoint: please migrate."}
            )
        )
        respx.get(V3_INCOME).mock(return_value=httpx.Response(200, json=[fixtures.INCOME_V3]))
        assert len(fmp_client.get_income_statement("AAPL")) == 1

    @respx.mock
    def test_v3_url_puts_symbol_in_path(self):
        client = FMPClient(api_key="k", variant="v3", cache_raw=False)
        route = respx.get(V3_INCOME).mock(
            return_value=httpx.Response(200, json=[fixtures.INCOME_V3])
        )
        client.get_income_statement("AAPL")
        assert "symbol" not in route.calls.last.request.url.params

    @respx.mock
    def test_symbol_gated_by_plan_reports_the_reason(self, fmp_client: FMPClient):
        """FMP's free tier gates some symbols entirely, answering 402.

        Observed for real on 11 of 38 large-cap tickers, so this path matters:
        it must degrade with a clear reason, not look like a bad API key.
        """
        gated = httpx.Response(
            402,
            text=(
                "Premium Query Parameter: 'Special Endpoint : This value set for 'symbol' "
                "is not available under your current subscription"
            ),
        )
        respx.get(STABLE_INCOME).mock(return_value=gated)
        respx.get(V3_INCOME).mock(return_value=gated)
        with pytest.raises(FMPEndpointUnavailableError, match="plan limit"):
            fmp_client.get_income_statement("AAPL")

    @respx.mock
    def test_plan_limit_does_not_look_like_a_bad_key(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(402, text="This endpoint requires a paid plan")
        )
        respx.get(V3_INCOME).mock(
            return_value=httpx.Response(402, text="This endpoint requires a paid plan")
        )
        with pytest.raises(FMPEndpointUnavailableError, match="paid FMP plan"):
            fmp_client.get_income_statement("AAPL")

    @respx.mock
    def test_unavailable_on_both_variants_propagates(self, fmp_client: FMPClient):
        unavailable = httpx.Response(403, text="Exclusive Endpoint")
        respx.get(STABLE_INCOME).mock(return_value=unavailable)
        respx.get(V3_INCOME).mock(return_value=unavailable)
        with pytest.raises(FMPEndpointUnavailableError):
            fmp_client.get_income_statement("AAPL")


class TestGetAllStatements:
    @respx.mock
    def test_one_missing_statement_does_not_lose_the_others(self, fmp_client: FMPClient):
        respx.get(STABLE_INCOME).mock(
            return_value=httpx.Response(200, json=[fixtures.INCOME_V3])
        )
        respx.get(f"{BASE}/stable/balance-sheet-statement").mock(
            return_value=httpx.Response(200, json=[fixtures.BALANCE_V3])
        )
        respx.get(f"{BASE}/stable/cash-flow-statement").mock(
            return_value=httpx.Response(200, json=[])  # no data for this one
        )
        payloads = fmp_client.get_all_statements("AAPL")
        assert len(payloads["income_statement"]) == 1
        assert len(payloads["balance_sheet"]) == 1
        assert payloads["cash_flow"] == []
