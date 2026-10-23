from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from ragbench.config import EnvSettings
from ragbench.llm.client import LLMClient, LLMError, MissingApiKeyError, scrub
from ragbench.llm.fake import FakeLLMClient
from ragbench.llm.roles import make_client

KEY = "gsk_" + "A1b2C3d4" * 6
MSGS = [{"role": "user", "content": "hello"}]
REQ = httpx.Request("POST", "https://api.example.test/v1/chat/completions")


def completion(text="hi", prompt=11, completion_tokens=3):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion_tokens),
    )


class StubSDK:
    """Stands in for openai.OpenAI: replays a script of results/exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def rate_limit(retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    return openai.RateLimitError(
        "rate limited", response=httpx.Response(429, request=REQ, headers=headers), body=None
    )


def make(script, **kw):
    sleeps: list[float] = []
    client = LLMClient(
        "https://api.example.test/v1",
        KEY,
        "test-model",
        sdk=StubSDK(script),
        sleep=sleeps.append,
        **kw,
    )
    return client, sleeps


def test_returns_text_tokens_latency_and_model():
    client, _ = make([completion("answer", 20, 5)])
    r = client.chat(MSGS, temperature=0.5, max_tokens=100)
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.model) == (
        "answer",
        20,
        5,
        "test-model",
    )
    assert r.latency_s >= 0 and r.n_retries == 0 and not r.cached


def test_sends_expected_request_fields():
    client, _ = make([completion()])
    client.chat(MSGS, temperature=0.3, max_tokens=50, json_mode=True, seed=7)
    sent = client._sdk.calls[0]
    assert sent["model"] == "test-model" and sent["temperature"] == 0.3 and sent["max_tokens"] == 50
    assert sent["response_format"] == {"type": "json_object"} and sent["seed"] == 7


def test_retries_rate_limits_with_backoff_and_counts_them():
    client, sleeps = make([rate_limit(), rate_limit("3"), completion("ok")])
    r = client.chat(MSGS, temperature=0.5, max_tokens=10)
    assert r.text == "ok" and r.n_retries == 2
    assert len(sleeps) == 2 and sleeps[1] == 3.0  # honours Retry-After


def test_gives_up_after_max_attempts_with_scrubbed_error():
    client, _ = make([rate_limit()] * 3, max_attempts=3)
    with pytest.raises(LLMError, match="after 2 retries"):
        client.chat(MSGS, temperature=0.5, max_tokens=10)


def test_long_retry_after_fails_fast_instead_of_burning_retries():
    client, sleeps = make([rate_limit("320"), completion("never reached")])
    with pytest.raises(LLMError, match="daily quota"):
        client.chat(MSGS, temperature=0.5, max_tokens=10)
    assert not sleeps and len(client._sdk.calls) == 1


def test_retry_after_is_parsed_from_the_provider_message_when_no_header():
    err = openai.RateLimitError(
        "Rate limit reached ... Please try again in 5m17.952s. Need more tokens?",
        response=httpx.Response(429, request=REQ),
        body=None,
    )
    assert LLMClient._retry_after(err) == pytest.approx(317.952)
    client, sleeps = make([err])
    with pytest.raises(LLMError):
        client.chat(MSGS, temperature=0.5, max_tokens=10)
    assert not sleeps


def test_short_provider_message_wait_is_honoured():
    err = openai.RateLimitError(
        "Please try again in 2.5s", response=httpx.Response(429, request=REQ), body=None
    )
    client, sleeps = make([err, completion("ok")])
    assert client.chat(MSGS, temperature=0.5, max_tokens=10).text == "ok"
    assert sleeps == [2.5]


def test_authentication_error_is_not_retried_and_hides_key():
    err = openai.AuthenticationError(
        f"bad key {KEY}", response=httpx.Response(401, request=REQ), body=None
    )
    client, sleeps = make([err])
    with pytest.raises(LLMError) as exc:
        client.chat(MSGS, temperature=0.5, max_tokens=10)
    assert KEY not in str(exc.value) and not sleeps


def test_other_api_errors_are_scrubbed():
    err = openai.BadRequestError(
        f"invalid request for key {KEY}",
        response=httpx.Response(400, request=REQ),
        body=None,
    )
    client, _ = make([err])
    with pytest.raises(LLMError) as exc:
        client.chat(MSGS, temperature=0.5, max_tokens=10)
    assert KEY not in str(exc.value)


def test_key_never_appears_in_repr_or_str():
    client, _ = make([])
    assert KEY not in repr(client) and KEY not in str(client)


def test_cache_hit_skips_the_network_and_keeps_original_latency():
    cache: dict = {}
    client, _ = make([completion("first")], cache=cache)
    a = client.chat(MSGS, temperature=0.5, max_tokens=10)
    b = client.chat(MSGS, temperature=0.5, max_tokens=10)
    assert len(client._sdk.calls) == 1
    assert (b.text, b.cached, b.latency_s) == ("first", True, a.latency_s) and not a.cached


def test_cache_salt_and_settings_make_distinct_entries():
    cache: dict = {}
    client, _ = make([completion("t1"), completion("t2"), completion("t3")], cache=cache)
    assert client.chat(MSGS, temperature=0.5, max_tokens=10, cache_salt="trial1").text == "t1"
    assert client.chat(MSGS, temperature=0.5, max_tokens=10, cache_salt="trial2").text == "t2"
    assert client.chat(MSGS, temperature=0.9, max_tokens=10, cache_salt="trial1").text == "t3"


def test_cache_key_does_not_include_the_api_key():
    client_a, _ = make([completion("x")], cache={})
    key_a = client_a._cache_key({"m": 1})
    other = LLMClient(
        "https://api.example.test/v1", "gsk_other" + "z" * 30, "test-model", sdk=StubSDK([])
    )
    assert other._cache_key({"m": 1}) == key_a


def test_missing_key_raises_without_leaking_anything():
    with pytest.raises(MissingApiKeyError):
        LLMClient("https://x", "", "m")


def test_scrub_masks_key_shapes():
    assert KEY not in scrub(f"error with {KEY} inside")
    assert "sk-proj-" + "a" * 20 not in scrub("sk-proj-" + "a" * 20)
    assert "secretvalue" not in scrub("token secretvalue", ["secretvalue"])


def test_make_client_requires_a_key():
    env = EnvSettings(_env_file=None, generator_api_key=None)
    with pytest.raises(MissingApiKeyError, match="GENERATOR_API_KEY"):
        make_client("generator", env=env, use_cache=False)


def test_byok_key_overrides_env_and_disables_the_shared_cache():
    env = EnvSettings(_env_file=None, generator_api_key="gsk_envkey" + "e" * 30)
    client = make_client("generator", api_key=KEY, env=env)
    assert client._cache is None and client._secret == KEY


def test_fake_client_records_calls_and_tokens():
    fake = FakeLLMClient()
    r = fake.chat(MSGS, temperature=0.5, max_tokens=10, seed=3, cache_salt="s")
    assert fake.calls[0]["seed"] == 3 and fake.calls[0]["cache_salt"] == "s"
    assert r.completion_tokens >= 1 and r.model == "fake-llm"


def test_bad_request_error_keeps_provider_code_and_failed_generation_scrubbed():
    body = {
        "error": {
            "message": "Failed to validate JSON.",
            "code": "json_validate_failed",
            "failed_generation": f'{{"claims": ["x"]}} leaked {KEY}',
        }
    }
    err = openai.BadRequestError("400", response=httpx.Response(400, request=REQ), body=body)
    client, _ = make([err])
    with pytest.raises(LLMError) as exc:
        client.chat(MSGS, temperature=0.0, max_tokens=10, json_mode=True)
    assert exc.value.code == "json_validate_failed"
    assert '{"claims": ["x"]}' in (exc.value.failed_generation or "")
    assert KEY not in (exc.value.failed_generation or "") and KEY not in str(exc.value)


def test_rate_limit_waits_are_logged_without_any_key_material(caplog):
    import logging

    caplog.set_level(logging.INFO)
    client, _ = make([rate_limit("3"), completion("ok")])
    client.chat(MSGS, temperature=0.5, max_tokens=10)
    waits = [r.getMessage() for r in caplog.records if "rate-limited; waiting" in r.getMessage()]
    assert waits and "3.0s" in waits[0] and "attempt 1 of 5" in waits[0]
    assert KEY not in caplog.text
