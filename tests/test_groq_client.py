"""Groq client tests: error mapping, retries and model fallback.

Groq retires model ids on a rolling schedule, so the fallback path is not a
theoretical edge case - it is what keeps the pipeline working the week a model
is decommissioned.
"""

from __future__ import annotations

import pytest

from src.rag.groq_client import (
    Completion,
    GroqAuthError,
    GroqClient,
    GroqError,
    GroqModelUnavailableError,
    GroqRateLimitError,
)


class FakeResponse:
    def __init__(self, text="hello", finish_reason="stop", prompt_tokens=10, completion_tokens=5):
        message = type("Message", (), {"content": text})()
        choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
        self.choices = [choice]
        self.usage = type(
            "Usage", (), {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )()


class FakeSDK:
    """Stands in for the Groq SDK, scripted per model id."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[str] = []
        outer = self

        class Completions:
            def create(self, model, messages, temperature, max_tokens):
                outer.calls.append(model)
                action = outer.behaviour.get(model, FakeResponse())
                if isinstance(action, list):
                    action = action.pop(0)
                if isinstance(action, Exception):
                    raise action
                return action

        self.chat = type("Chat", (), {"completions": Completions()})()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("src.rag.groq_client.time.sleep", lambda *_: None)


# Fixed ids so the tests do not break every time Groq retires a model and the
# configured defaults change.
PRIMARY = "primary-model"
FALLBACK = "fallback-model"


def make_client(behaviour, **kwargs):
    kwargs.setdefault("model", PRIMARY)
    kwargs.setdefault("fallback_model", FALLBACK)
    return GroqClient(api_key="k", client=FakeSDK(behaviour), max_retries=2, **kwargs)


class TestCompletion:
    def test_successful_call(self):
        client = make_client({PRIMARY: FakeResponse("The answer.")})
        result = client.complete("system", "user")
        assert result.text == "The answer."
        assert result.model == PRIMARY
        assert result.total_tokens == 15
        assert not result.was_truncated

    def test_truncation_is_surfaced(self):
        client = make_client(
            {PRIMARY: FakeResponse("cut off mid-", finish_reason="length")}
        )
        assert client.complete("s", "u").was_truncated

    def test_request_count_tracked(self):
        client = make_client({PRIMARY: FakeResponse()})
        client.complete("s", "u")
        client.complete("s", "u")
        assert client.request_count == 2


class TestErrorHandling:
    def test_missing_key_raises_before_any_call(self):
        client = GroqClient(api_key=None)
        with pytest.raises(GroqAuthError, match="GROQ_API_KEY"):
            client.complete("s", "u")

    def test_is_configured_reflects_the_key(self):
        assert not GroqClient(api_key=None).is_configured
        assert GroqClient(api_key="k").is_configured

    def test_auth_error_is_not_retried(self):
        sdk = FakeSDK({PRIMARY: RuntimeError("Invalid API Key provided")})
        client = GroqClient(api_key="k", client=sdk, model=PRIMARY, fallback_model=FALLBACK, max_retries=3)
        with pytest.raises(GroqAuthError):
            client.complete("s", "u")
        assert len(sdk.calls) == 1  # no point retrying a bad key

    def test_transient_failure_then_success(self):
        client = make_client(
            {PRIMARY: [RuntimeError("connection reset"), FakeResponse("ok")]}
        )
        assert client.complete("s", "u").text == "ok"

    def test_rate_limit_is_retried_then_raised(self):
        client = make_client(
            {
                PRIMARY: RuntimeError("rate limit exceeded"),
                FALLBACK: RuntimeError("rate limit exceeded"),
            }
        )
        with pytest.raises(GroqRateLimitError):
            client.complete("s", "u")

    def test_all_models_failing_raises_clearly(self):
        client = make_client(
            {
                PRIMARY: RuntimeError("boom"),
                FALLBACK: RuntimeError("boom"),
            }
        )
        with pytest.raises(GroqModelUnavailableError, match="No Groq model succeeded"):
            client.complete("s", "u")


class TestModelFallback:
    def test_decommissioned_model_falls_back(self):
        """The scenario that actually happens: a model id is retired."""
        sdk = FakeSDK(
            {
                PRIMARY: RuntimeError(
                    "model_not_found: model has been decommissioned"
                ),
                FALLBACK: FakeResponse("fallback answer"),
            }
        )
        client = GroqClient(api_key="k", client=sdk, model=PRIMARY, fallback_model=FALLBACK, max_retries=2)
        result = client.complete("s", "u")
        assert result.text == "fallback answer"
        assert result.model == FALLBACK

    def test_model_error_is_not_retried_on_the_same_model(self):
        sdk = FakeSDK(
            {
                PRIMARY: RuntimeError("model_not_found"),
                FALLBACK: FakeResponse("ok"),
            }
        )
        GroqClient(
            api_key="k", client=sdk, model=PRIMARY, fallback_model=FALLBACK, max_retries=3
        ).complete("s", "u")
        assert sdk.calls == [PRIMARY, FALLBACK]

    def test_identical_fallback_is_not_tried_twice(self):
        sdk = FakeSDK({"m": RuntimeError("model_not_found")})
        client = GroqClient(api_key="k", client=sdk, model="m", fallback_model="m", max_retries=1)
        with pytest.raises(GroqError):
            client.complete("s", "u")
        assert sdk.calls == ["m"]
