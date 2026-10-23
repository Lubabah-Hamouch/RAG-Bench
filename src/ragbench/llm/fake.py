"""A deterministic stand-in for the model. Tests and the default CLI mode use it (no network)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ragbench.llm.client import LLMResponse, Message

# A responder returns the reply text, or (text, finish_reason) to simulate a truncated reply.
Responder = Callable[[Sequence[Message]], str | tuple[str, str]]

_CRITIQUE = """Passage relevance:
[1] Relevant
[2] Relevant
[3] Irrelevant
[4] Irrelevant

Answerable from context: Yes
Factual accuracy: Fully supported
Unsupported or incorrect claims: None
Completeness: Complete
Missing information from the context: None
Utility (1-5): 4
Revision instructions: No changes needed"""


def default_responder(messages: Sequence[Message]) -> str:
    """Recognises each strategy stage by its prompt and returns a canned, stage-shaped reply."""
    text = "\n".join(m["content"] for m in messages)
    if "No reference documents are provided" in text:
        return "Fake no-context answer."
    if "reading retrieved passages" in text:
        return "- Fake relevant fact [1]\n- Fake relevant fact [2]"
    if "synthesizing an answer" in text:
        return "Fake synthesized draft."
    if "editing a draft answer" in text:
        return "Fake refined answer."
    if "strict reviewer" in text:
        return _CRITIQUE
    if "Revise the initial answer" in text:
        return "Fake revised answer."
    if "Initial answer:" in text:
        return "Fake initial answer."
    if "## Summary" in text:
        return (
            "## Summary\nFake summary.\n\n## Key Points\n- one\n- two\n- three\n\n"
            "## Elaboration\nFake."
        )
    return "Fake baseline answer."


class FakeLLMClient:
    """Implements `ChatClient`; records every call so tests can inspect prompts and settings."""

    def __init__(self, responder: Responder | None = None, model: str = "fake-llm") -> None:
        self.model = model
        self._responder = responder or default_responder
        self.calls: list[dict[str, Any]] = []
        self.available_models: list[str] = [model]

    def list_models(self) -> list[str]:
        return list(self.available_models)

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
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "seed": seed,
                "cache_salt": cache_salt,
            }
        )
        out = self._responder(messages)
        reply, finish = (out, "stop") if isinstance(out, str) else out
        prompt_chars = sum(len(m["content"]) for m in messages)
        return LLMResponse(
            text=reply,
            model=self.model,
            latency_s=0.0,
            prompt_tokens=prompt_chars // 4,
            completion_tokens=max(1, len(reply) // 4),
            finish_reason=finish,
        )
