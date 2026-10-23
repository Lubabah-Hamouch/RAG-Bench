"""OpenAI-compatible chat client with timing, token usage, retries, disk cache and key scrubbing.

One class serves every role and vendor: Groq and OpenAI both speak the same protocol, so swapping a
model is only a matter of base URL, key and model id (see `ragbench.llm.roles`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import openai

Message = dict[str, str]

log = logging.getLogger(__name__)

_KEY_SHAPES = re.compile(r"\b(?:gsk_|sk-or-v1-|sk-proj-|sk-)[A-Za-z0-9_\-]{8,}")
_MAX_CONCURRENCY = threading.BoundedSemaphore(int(os.environ.get("RAGBENCH_MAX_CONCURRENCY", "4")))
_TRY_AGAIN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)
MAX_RETRY_WAIT_S = 90.0  # a longer ask (e.g. a daily token quota) is not worth waiting for
_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


class LLMError(RuntimeError):
    """A failed model call. Messages are scrubbed of anything key-shaped.

    `code` is the provider's error code when it gave one (e.g. "json_validate_failed"), and
    `failed_generation` the text the model produced, when the provider returns it.
    """

    def __init__(
        self, message: str, *, code: str | None = None, failed_generation: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.failed_generation = failed_generation


class MissingApiKeyError(LLMError):
    pass


def scrub(text: str, secrets: Sequence[str] = ()) -> str:
    """Remove key-shaped strings (and any explicitly known secrets) from text."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return _KEY_SHAPES.sub("[REDACTED]", text)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    latency_s: float  # the successful attempt only; retry back-off is excluded
    prompt_tokens: int
    completion_tokens: int
    n_retries: int = 0
    reasoning_tokens: int = 0  # hidden reasoning, already included in completion_tokens
    finish_reason: str | None = None  # "length" means the output budget ran out
    cached: bool = False  # a cache hit keeps the original latency so benchmark timings stay valid


class ChatClient(Protocol):
    model: str

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        seed: int | None = None,
        cache_salt: str = "",
    ) -> LLMResponse: ...


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        role: str = "generator",
        cache: Any = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
        max_attempts: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        sdk: Any = None,
    ) -> None:
        if not api_key:
            raise MissingApiKeyError(f"no API key configured for the {role} role")
        self.model = model
        self.role = role
        self._base_url = base_url
        self._secret = api_key  # kept only to scrub it from error text
        self._cache = cache
        self._extra_body = dict(extra_body or {})
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._sdk = sdk or openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0
        )

    def __repr__(self) -> str:
        return f"LLMClient(role={self.role!r}, model={self.model!r})"

    def list_models(self) -> list[str]:
        """Model ids this key can use. Used to validate a bring-your-own key cheaply."""
        try:
            return sorted(m.id for m in self._sdk.models.list().data)
        except openai.AuthenticationError:
            raise LLMError(
                f"authentication failed for the {self.role} role (HTTP 401): check the API key",
                code="auth_failed",
            ) from None
        except openai.RateLimitError as exc:
            raise LLMError(
                scrub(f"{self.role} rate-limited: {exc}", [self._secret]), code="rate_limited"
            ) from None
        except openai.OpenAIError as exc:
            raise LLMError(scrub(f"{self.role} call failed: {exc}", [self._secret])) from None

    def _cache_key(self, payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        """Seconds the provider asks us to wait: the Retry-After header, else its message text."""
        response = getattr(exc, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        try:
            if header is not None:
                return float(header)
        except ValueError:
            pass
        m = _TRY_AGAIN.search(str(exc))
        if m:
            return float(m.group(1) or 0) * 60 + float(m.group(2))
        return None

    def _backoff(self, attempt: int, exc: Exception) -> float:
        asked = self._retry_after(exc)
        if asked is not None:
            return asked
        return min(2.0**attempt, 30.0) + random.uniform(0, 0.5)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
        seed: int | None = None,
        cache_salt: str = "",
    ) -> LLMResponse:
        key = self._cache_key(
            {
                "base_url": self._base_url,
                "model": self.model,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "seed": seed,
                "salt": cache_salt,
                "extra_body": self._extra_body,
            }
        )
        if self._cache is not None and key in self._cache:
            hit: LLMResponse = self._cache[key]
            return dataclasses.replace(hit, cached=True)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if seed is not None:
            kwargs["seed"] = seed
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        retries = 0
        for attempt in range(self._max_attempts):
            try:
                with _MAX_CONCURRENCY:
                    started = time.perf_counter()
                    completion = self._sdk.chat.completions.create(**kwargs)
                    latency = time.perf_counter() - started
            except _RETRYABLE as exc:
                asked = self._retry_after(exc)
                if asked is not None and asked > MAX_RETRY_WAIT_S:
                    raise LLMError(
                        scrub(
                            f"{self.role} call rate-limited; provider asks to wait {asked:.0f}s "
                            f"(likely a daily quota): {exc}",
                            [self._secret],
                        ),
                        code="rate_limit_long",
                    ) from None
                if attempt == self._max_attempts - 1:
                    raise LLMError(
                        scrub(
                            f"{self.role} call failed after {retries} retries: {exc}",
                            [self._secret],
                        ),
                        code="rate_limited" if isinstance(exc, openai.RateLimitError) else None,
                    ) from None
                wait = self._backoff(attempt, exc)
                log.info(
                    "%s call rate-limited; waiting %.1fs (attempt %d of %d)",
                    self.role,
                    wait,
                    attempt + 1,
                    self._max_attempts,
                )
                self._sleep(wait)
                retries += 1
                continue
            except openai.AuthenticationError:
                raise LLMError(
                    f"authentication failed for the {self.role} role (HTTP 401): check the API key",
                    code="auth_failed",
                ) from None
            except openai.NotFoundError as exc:
                raise LLMError(
                    scrub(f"{self.role} model not found: {exc}", [self._secret]),
                    code="model_not_found",
                ) from None
            except openai.BadRequestError as exc:
                body = exc.body if isinstance(exc.body, dict) else {}
                err = body.get("error", body) if isinstance(body, dict) else {}
                failed = err.get("failed_generation") if isinstance(err, dict) else None
                raise LLMError(
                    scrub(f"{self.role} call failed: {exc}", [self._secret]),
                    code=err.get("code") if isinstance(err, dict) else None,
                    failed_generation=scrub(str(failed), [self._secret]) if failed else None,
                ) from None
            except openai.OpenAIError as exc:
                raise LLMError(scrub(f"{self.role} call failed: {exc}", [self._secret])) from None

            usage = completion.usage
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", 0) or 0
            response = LLMResponse(
                text=completion.choices[0].message.content or "",
                model=self.model,
                latency_s=latency,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                n_retries=retries,
                reasoning_tokens=int(reasoning),
                finish_reason=getattr(completion.choices[0], "finish_reason", None),
            )
            if self._cache is not None:
                self._cache[key] = response
            return response
        raise LLMError("unreachable")  # pragma: no cover
