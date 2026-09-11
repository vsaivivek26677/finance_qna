"""HTTP client for the FastAPI backend.

The dashboard holds no database session, no model and no API keys. Everything it
shows comes through here, which is what keeps the presentation layer swappable
and the business logic in one place.

Every call returns either data or a typed `ApiError`. Streamlit renders a blank
page on an uncaught exception, so failures are turned into values the UI can
present as an explanation rather than a stack trace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
# Generation and ingestion are slow by nature; a short timeout would abort work
# that is progressing normally.
LONG_TIMEOUT = 180.0


class ApiError(Exception):
    """A backend call failed, with a message fit to show a user."""

    def __init__(self, message: str, status_code: int | None = None, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.hint = hint


@dataclass
class ApiClient:
    """Thin wrapper over the backend's REST endpoints."""

    base_url: str
    timeout: float = DEFAULT_TIMEOUT

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        try:
            response = httpx.request(
                method,
                self._url(path),
                params={k: v for k, v in (params or {}).items() if v is not None},
                json=json,
                timeout=timeout or self.timeout,
            )
        except httpx.ConnectError as exc:
            raise ApiError(
                "Cannot reach the API.",
                hint=f"Start it with `uvicorn src.api.main:app --reload` (expected at {self.base_url}).",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError(
                "The API did not respond in time.",
                hint="Generation and ingestion can take a while on the free tiers.",
            ) from exc

        if response.status_code >= 400:
            detail = _extract_detail(response)
            raise ApiError(detail, status_code=response.status_code)
        return response.json()

    # --- reads -------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=10.0)

    def companies(self) -> list[dict]:
        return self._request("GET", "/companies")

    def profile(self, ticker: str) -> dict:
        return self._request("GET", f"/companies/{ticker}")

    def statements(self, ticker: str, period: str = "FY", fiscal_year: int | None = None) -> dict:
        return self._request(
            "GET",
            f"/companies/{ticker}/statements",
            params={"period": period, "fiscal_year": fiscal_year},
        )

    def ratios(
        self,
        ticker: str,
        fiscal_year: int | None = None,
        category: str | None = None,
        calculable_only: bool = False,
    ) -> dict:
        return self._request(
            "GET",
            f"/companies/{ticker}/ratios",
            params={
                "fiscal_year": fiscal_year,
                "category": category,
                "calculable_only": calculable_only,
            },
        )

    def ratio_series(self, ticker: str, ratio_name: str) -> dict:
        return self._request("GET", f"/companies/{ticker}/ratios/{ratio_name}/series")

    def distress(self, ticker: str, fiscal_year: int | None = None) -> dict:
        return self._request(
            "GET", f"/companies/{ticker}/distress-score", params={"fiscal_year": fiscal_year}
        )

    def distress_prediction(self, ticker: str) -> dict:
        return self._request("GET", f"/companies/{ticker}/distress-prediction")

    def red_flags(self, ticker: str, fiscal_year: int | None = None) -> dict:
        return self._request(
            "GET", f"/companies/{ticker}/red-flags", params={"fiscal_year": fiscal_year}
        )

    # --- generation and ingestion -----------------------------------------

    def summary(self, ticker: str, fiscal_year: int | None = None, refresh: bool = False) -> dict:
        return self._request(
            "GET",
            f"/companies/{ticker}/summary",
            params={"fiscal_year": fiscal_year, "refresh": refresh},
            timeout=LONG_TIMEOUT,
        )

    def ask(self, ticker: str, question: str, fiscal_year: int | None = None) -> dict:
        return self._request(
            "POST",
            f"/companies/{ticker}/ask",
            json={"question": question, "fiscal_year": fiscal_year},
            timeout=LONG_TIMEOUT,
        )

    def ingest(self, ticker: str, years: int = 5) -> dict:
        return self._request(
            "POST",
            f"/ingest/{ticker}",
            json={"years": years, "period": "annual"},
            timeout=LONG_TIMEOUT,
        )


def _extract_detail(response: httpx.Response) -> str:
    """Pull the backend's own message out, falling back to the status line."""
    try:
        payload = response.json()
    except ValueError:
        return f"{response.status_code}: {response.text[:200]}"

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list) and detail:  # FastAPI validation errors
        first = detail[0]
        location = " -> ".join(str(p) for p in first.get("loc", []))
        return f"{first.get('msg', 'Invalid request')} ({location})"
    return str(detail or f"{response.status_code}: {response.text[:200]}")
