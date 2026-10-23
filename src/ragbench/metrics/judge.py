"""The judge: a second model asked to return validated JSON.

Reasoning models (gpt-oss) spend part of `max_tokens` on hidden reasoning, so a reply can be cut
off before any JSON is written. The judge detects that (finish_reason == "length" or empty text)
and retries with double the budget, up to a cap.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ragbench.llm.client import ChatClient, LLMError
from ragbench.strategies.base import CallRecord

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class JudgeError(LLMError):
    """The judge could not produce valid JSON within its retry budget."""


def extract_json(text: str) -> object:
    """Parse a JSON object from a reply, tolerating code fences and surrounding prose."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


class Judge:
    def __init__(
        self,
        client: ChatClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        max_tokens_cap: int = 6000,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_cap = max_tokens_cap
        self.max_retries = max_retries

    @staticmethod
    def _salvage(text: str | None, schema: type[T]) -> T | None:
        if not text:
            return None
        try:
            return schema.model_validate(extract_json(text))
        except (json.JSONDecodeError, ValidationError):
            return None

    def ask(self, name: str, prompt: str, schema: type[T]) -> tuple[T, list[CallRecord]]:
        """Send `prompt` and parse the reply as `schema`; also return every call made."""
        calls: list[CallRecord] = []
        budget = self.max_tokens
        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=budget,
                    json_mode=True,
                    cache_salt=f"judge:{name}:{attempt}",
                )
            except LLMError as exc:
                if exc.code != "json_validate_failed":
                    raise
                # The provider rejected malformed JSON-mode output. Salvage the text if it is
                # parseable after all; otherwise retry with a larger budget (truncation is the
                # usual cause when a reasoning model runs out of tokens mid-answer).
                salvaged = self._salvage(exc.failed_generation, schema)
                if salvaged is not None:
                    return salvaged, calls
                last_error = "provider rejected the JSON output"
                budget = min(budget * 2, self.max_tokens_cap)
                continue
            calls.append(
                CallRecord(
                    stage=f"judge:{name}",
                    model=resp.model,
                    latency_s=resp.latency_s,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    n_retries=resp.n_retries,
                    cached=resp.cached,
                    reasoning_tokens=resp.reasoning_tokens,
                )
            )
            if resp.finish_reason == "length" or not resp.text.strip():
                last_error = f"output budget exhausted at {budget} tokens"
                budget = min(budget * 2, self.max_tokens_cap)
                continue
            try:
                return schema.model_validate(extract_json(resp.text)), calls
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = f"invalid JSON: {str(exc)[:160]}"
        raise JudgeError(f"judge call {name!r} failed after {len(calls)} attempts: {last_error}")
