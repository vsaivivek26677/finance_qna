"""Groq chat completions client.

Groq's free tier is fast and genuinely free, with two operational quirks worth
handling explicitly:

* **Models get decommissioned.** Model ids are retired on a rolling basis, and a
  hard-coded id that worked last month returns a 404 today. The client falls back
  to a second model on a model-not-found error and reports which one answered, so
  a retired id degrades the run instead of breaking it.
* **Rate limits are per-minute, not per-day.** A 429 is almost always worth
  waiting out rather than failing, so the client honours `Retry-After`.

Temperature defaults to 0.1: this is factual extraction from a supplied context,
not prose generation, and near-greedy decoding keeps quoted figures stable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


class GroqError(RuntimeError):
    """Base class for Groq client failures."""


class GroqAuthError(GroqError):
    """Missing or rejected API key."""


class GroqRateLimitError(GroqError):
    """Rate limited and out of retries."""


class GroqModelUnavailableError(GroqError):
    """Every configured model was rejected by the API."""


@dataclass
class Completion:
    """One model response plus the metadata worth logging."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def was_truncated(self) -> bool:
        """A cut-off answer can end mid-number, so callers need to know."""
        return self.finish_reason == "length"


class GroqClient:
    """Thin wrapper over the Groq SDK with retries and model fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model = model or settings.groq_model
        self.fallback_model = fallback_model or settings.groq_fallback_model
        self.temperature = temperature if temperature is not None else settings.groq_temperature
        self.max_tokens = max_tokens or settings.groq_max_tokens
        self.timeout = timeout or settings.groq_timeout_seconds
        self.max_retries = max_retries or settings.groq_max_retries
        self.request_count = 0
        self._client = client
        self._active_model: str | None = None

    # --- plumbing ----------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise GroqAuthError(
                "GROQ_API_KEY is not set. Add it to .env (free key: https://console.groq.com)."
            )
        try:
            from groq import Groq  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise GroqError(
                "The groq package is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        self._client = Groq(api_key=self.api_key, timeout=self.timeout)
        return self._client

    @property
    def is_configured(self) -> bool:
        """Whether a key is present, without constructing a client."""
        return bool(self.api_key) or self._client is not None

    def _candidate_models(self) -> list[str]:
        models = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models.append(self.fallback_model)
        return models

    @staticmethod
    def _is_model_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in ("model_not_found", "does not exist", "decommission", "model `", "404")
        )

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "invalid api key" in text or "authentication" in text or "401" in text

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        text = str(exc).lower()
        return "rate limit" in text or "429" in text or "too many requests" in text

    @staticmethod
    def _retry_after(exc: Exception, attempt: int) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        try:
            return float(headers.get("retry-after") or headers.get("Retry-After") or 0) or min(
                2**attempt, 20
            )
        except (TypeError, ValueError):
            return min(2**attempt, 20)

    # --- public API --------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Send one chat completion, retrying and falling back as needed."""
        client = self._get_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None

        for model in self._candidate_models():
            for attempt in range(1, self.max_retries + 1):
                self.request_count += 1
                started = time.monotonic()
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature if temperature is None else temperature,
                        max_tokens=max_tokens or self.max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001 - SDK raises a wide range
                    last_error = exc
                    if self._is_auth_error(exc):
                        raise GroqAuthError(f"Groq rejected the API key: {exc}") from exc
                    if self._is_model_error(exc):
                        logger.warning("Groq model %s unavailable (%s); trying next", model, exc)
                        break  # move to the fallback model
                    if self._is_rate_limit(exc):
                        wait = self._retry_after(exc, attempt)
                        logger.warning("Groq rate limited; sleeping %.1fs", wait)
                        time.sleep(wait)
                        continue
                    logger.warning("Groq request failed (attempt %d): %s", attempt, exc)
                    time.sleep(min(2**attempt, 8))
                    continue

                latency = time.monotonic() - started
                choice = response.choices[0]
                usage = getattr(response, "usage", None)
                self._active_model = model
                return Completion(
                    text=(choice.message.content or "").strip(),
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_seconds=latency,
                    finish_reason=getattr(choice, "finish_reason", None),
                )

        if last_error is not None and self._is_rate_limit(last_error):
            raise GroqRateLimitError(f"Groq rate limit not cleared: {last_error}") from last_error
        raise GroqModelUnavailableError(
            f"No Groq model succeeded (tried {', '.join(self._candidate_models())}): {last_error}"
        ) from last_error

    def list_models(self) -> list[str]:
        """Model ids the key can currently use - handy when one is retired."""
        client = self._get_client()
        try:
            response = client.models.list()
        except Exception as exc:  # noqa: BLE001
            raise GroqError(f"Could not list Groq models: {exc}") from exc
        return sorted(getattr(item, "id", "") for item in getattr(response, "data", []))
