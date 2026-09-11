"""Financial Modeling Prep HTTP client.

Handles the two things that actually break free-tier ingestion in practice:

1. **Two API generations.** FMP is migrating from `/api/v3/...` to `/stable/...`
   and which one a key can use depends on when the key was issued. The client
   tries the configured variant, and on an endpoint-availability error retries
   once against the other, remembering which worked for the rest of the session.
2. **A 250 request/day quota.** A runaway loop can burn a whole day's budget in
   seconds, so the client counts requests and refuses to exceed the configured
   budget instead of hammering the API into a ban.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

ApiVariant = Literal["stable", "v3"]

# Substrings FMP uses when a key's plan cannot reach an endpoint. Seeing one of
# these means "try the other API generation", not "this ticker has no data".
_ENDPOINT_UNAVAILABLE_MARKERS = (
    "legacy endpoint",
    "exclusive endpoint",
    "special endpoint",
    "not available under your current subscription",
    "premium query parameter",
    "upgrade your plan",
)

# FMP signals plan limits with 402 as well as 401/403. It uses the same wording
# for two different restrictions: an endpoint the plan cannot reach (worth
# retrying on the other API generation) and a *symbol* outside the free tier's
# allowed list (retrying will not help). Both surface the same way, so the
# client tries the fallback once and then reports it clearly.
_PLAN_LIMIT_STATUSES = (401, 402, 403)


class FMPError(RuntimeError):
    """Base class for all FMP client failures."""


class FMPAuthError(FMPError):
    """Missing or rejected API key."""


class FMPRateLimitError(FMPError):
    """HTTP 429, or the local daily budget being exhausted."""


class FMPEndpointUnavailableError(FMPError):
    """The key's plan cannot reach this endpoint on this API generation."""


class FMPNotFoundError(FMPError):
    """The ticker exists as a request but the API returned no data for it."""


class FMPClient:
    """Thin, well-behaved wrapper over the FMP REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        variant: ApiVariant = "stable",
        timeout: float | None = None,
        max_retries: int | None = None,
        daily_budget: int | None = None,
        cache_raw: bool | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or settings.fmp_api_key
        self.base_url = (base_url or settings.fmp_base_url).rstrip("/")
        self.variant: ApiVariant = variant
        self.timeout = timeout if timeout is not None else settings.fmp_timeout_seconds
        self.max_retries = max_retries if max_retries is not None else settings.fmp_max_retries
        self.daily_budget = (
            daily_budget if daily_budget is not None else settings.fmp_daily_request_budget
        )
        self.cache_raw = settings.ingest_cache_raw_json if cache_raw is None else cache_raw
        self.request_count = 0
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=self.timeout)

    # --- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "FMPClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- request plumbing --------------------------------------------------

    def _build_request(
        self, endpoint: str, symbol: str, params: dict[str, Any], variant: ApiVariant
    ) -> tuple[str, dict[str, Any]]:
        """URL + query params differ between the two API generations."""
        query = {k: v for k, v in params.items() if v is not None}
        query["apikey"] = self.api_key
        if variant == "stable":
            url = f"{self.base_url}/stable/{endpoint}"
            query["symbol"] = symbol
        else:
            url = f"{self.base_url}/api/v3/{endpoint}/{symbol}"
        return url, query

    def _check_budget(self) -> None:
        if self.request_count >= self.daily_budget:
            raise FMPRateLimitError(
                f"Local request budget of {self.daily_budget} exhausted "
                f"({self.request_count} made). Raise FMP_DAILY_REQUEST_BUDGET or wait for reset."
            )

    @staticmethod
    def _detect_payload_error(payload: Any) -> str | None:
        """FMP often returns HTTP 200 with an error object in the body."""
        if isinstance(payload, dict):
            for key in ("Error Message", "error", "message"):
                if key in payload and payload[key]:
                    return str(payload[key])
        return None

    def _request(
        self, endpoint: str, symbol: str, params: dict[str, Any], variant: ApiVariant
    ) -> Any:
        if not self.api_key:
            raise FMPAuthError(
                "FMP_API_KEY is not set. Add it to .env (free key: "
                "https://site.financialmodelingprep.com/developer/docs)."
            )

        url, query = self._build_request(endpoint, symbol, params, variant)
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._check_budget()
            self.request_count += 1
            try:
                response = self._client.get(url, params=query)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("FMP network error (%s, attempt %d): %s", endpoint, attempt, exc)
                time.sleep(min(2**attempt, 8))
                continue

            if response.status_code == 429:
                # Respect the API's own backoff hint when it sends one.
                wait = float(response.headers.get("Retry-After", min(2**attempt, 8)))
                logger.warning("FMP rate limited on %s; sleeping %.1fs", endpoint, wait)
                last_exc = FMPRateLimitError(f"429 from FMP on {endpoint}")
                time.sleep(wait)
                continue

            if response.status_code in _PLAN_LIMIT_STATUSES:
                body = response.text.lower()
                if any(marker in body for marker in _ENDPOINT_UNAVAILABLE_MARKERS):
                    raise FMPEndpointUnavailableError(
                        f"{endpoint} unavailable for {symbol} on the '{variant}' API "
                        f"({response.status_code}). This is a plan limit: either the endpoint "
                        f"or the symbol itself is outside the free tier's allowed list. "
                        f"Detail: {response.text[:160]}"
                    )
                if response.status_code == 402:
                    raise FMPEndpointUnavailableError(
                        f"{endpoint} requires a paid FMP plan for {symbol} ({response.status_code}): "
                        f"{response.text[:160]}"
                    )
                raise FMPAuthError(
                    f"FMP rejected the API key ({response.status_code}): {response.text[:200]}"
                )

            if response.status_code >= 500:
                last_exc = FMPError(f"FMP server error {response.status_code} on {endpoint}")
                logger.warning("FMP %d on %s (attempt %d)", response.status_code, endpoint, attempt)
                time.sleep(min(2**attempt, 8))
                continue

            if response.status_code >= 400:
                raise FMPError(f"FMP {response.status_code} on {endpoint}: {response.text[:200]}")

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise FMPError(f"Non-JSON response from {endpoint}: {response.text[:200]}") from exc

            message = self._detect_payload_error(payload)
            if message:
                if any(marker in message.lower() for marker in _ENDPOINT_UNAVAILABLE_MARKERS):
                    raise FMPEndpointUnavailableError(
                        f"{endpoint} unavailable on '{variant}' API: {message}"
                    )
                if "limit" in message.lower():
                    raise FMPRateLimitError(message)
                raise FMPError(f"FMP error on {endpoint}: {message}")

            self._cache_raw(endpoint, symbol, params, payload)
            return payload

        raise FMPError(
            f"FMP request to {endpoint} failed after {self.max_retries} attempts"
        ) from last_exc

    def _get(self, endpoint: str, symbol: str, **params: Any) -> Any:
        """Request an endpoint, falling back to the other API generation once."""
        try:
            return self._request(endpoint, symbol, params, self.variant)
        except FMPEndpointUnavailableError as exc:
            fallback: ApiVariant = "v3" if self.variant == "stable" else "stable"
            logger.info("Switching FMP API variant %s -> %s (%s)", self.variant, fallback, exc)
            payload = self._request(endpoint, symbol, params, fallback)
            self.variant = fallback  # stick with what works for the rest of the run
            return payload

    def _cache_raw(self, endpoint: str, symbol: str, params: dict[str, Any], payload: Any) -> None:
        """Persist raw JSON so re-runs and debugging do not spend more quota."""
        if not self.cache_raw:
            return
        try:
            out_dir: Path = settings.raw_cache_dir / symbol.upper()
            out_dir.mkdir(parents=True, exist_ok=True)
            period = params.get("period", "na")
            (out_dir / f"{endpoint}_{period}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # caching is a convenience, never a hard failure
            logger.debug("Could not cache raw payload for %s/%s: %s", symbol, endpoint, exc)

    @staticmethod
    def _as_list(payload: Any, symbol: str, endpoint: str) -> list[dict[str, Any]]:
        if payload is None:
            raise FMPNotFoundError(f"No {endpoint} data returned for {symbol}")
        if isinstance(payload, dict):
            # A few endpoints wrap the array; unwrap the first list-valued key.
            for value in payload.values():
                if isinstance(value, list):
                    payload = value
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list) or not payload:
            raise FMPNotFoundError(f"No {endpoint} data returned for {symbol}")
        return [row for row in payload if isinstance(row, dict)]

    # --- public endpoints --------------------------------------------------

    def get_profile(self, symbol: str) -> dict[str, Any]:
        payload = self._get("profile", symbol)
        rows = self._as_list(payload, symbol, "profile")
        return rows[0]

    def get_income_statement(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[dict[str, Any]]:
        payload = self._get("income-statement", symbol, period=period, limit=limit)
        return self._as_list(payload, symbol, "income-statement")

    def get_balance_sheet(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[dict[str, Any]]:
        payload = self._get("balance-sheet-statement", symbol, period=period, limit=limit)
        return self._as_list(payload, symbol, "balance-sheet-statement")

    def get_cash_flow(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[dict[str, Any]]:
        payload = self._get("cash-flow-statement", symbol, period=period, limit=limit)
        return self._as_list(payload, symbol, "cash-flow-statement")

    def get_all_statements(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch all three statements. A failure on one does not lose the others."""
        out: dict[str, list[dict[str, Any]]] = {}
        for kind, fetch in (
            ("income_statement", self.get_income_statement),
            ("balance_sheet", self.get_balance_sheet),
            ("cash_flow", self.get_cash_flow),
        ):
            try:
                out[kind] = fetch(symbol, period=period, limit=limit)
            except FMPNotFoundError as exc:
                logger.warning("%s: %s", symbol, exc)
                out[kind] = []
        return out
