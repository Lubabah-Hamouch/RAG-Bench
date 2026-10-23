"""Strategy interface and the result type every strategy returns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragbench.llm.client import ChatClient, Message
from ragbench.prompts import is_refusal
from ragbench.types import RetrievedChunk


@dataclass(frozen=True)
class Stage:
    name: str
    text: str


@dataclass(frozen=True)
class CallRecord:
    stage: str
    model: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    n_retries: int = 0
    cached: bool = False
    reasoning_tokens: int = 0  # hidden reasoning; already counted in completion_tokens


@dataclass
class StrategyResult:
    strategy: str
    answer: str  # the final answer; metrics are computed on this text only
    trace: list[Stage] = field(default_factory=list)  # every stage's output, in order
    calls: list[CallRecord] = field(default_factory=list)
    critique: str | None = None  # Self-RAG only: shown to the user, never scored as the answer

    @property
    def abstained(self) -> bool:
        """The answer states that the context lacks the information."""
        return is_refusal(self.answer)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    def display(self) -> str:
        """What a user sees: the answer, plus the self-critique when the strategy produced one."""
        if self.critique:
            return f"**Answer**\n{self.answer}\n\n**Self-critique**\n{self.critique}"
        return self.answer


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Numbered passages. The numbers are for reference only; no strategy must cite them."""
    return "\n\n".join(f"[{i}] {rc.chunk.text}" for i, rc in enumerate(chunks, start=1))


class Recorder:
    """Runs model calls for one strategy run and records stages and per-call metrics."""

    def __init__(
        self,
        client: ChatClient,
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        cache_salt: str,
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._salt = cache_salt
        self.trace: list[Stage] = []
        self.calls: list[CallRecord] = []

    def call(self, stage: str, messages: Sequence[Message]) -> str:
        resp = self._client.chat(
            messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            seed=self._seed,
            cache_salt=f"{self._salt}:{stage}",
        )
        text = resp.text.strip()
        self.trace.append(Stage(stage, text))
        self.calls.append(
            CallRecord(
                stage=stage,
                model=resp.model,
                latency_s=resp.latency_s,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                n_retries=resp.n_retries,
                cached=resp.cached,
                reasoning_tokens=resp.reasoning_tokens,
            )
        )
        return text


def system_user(system: str, user: str) -> list[Message]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def user_only(user: str) -> list[Message]:
    return [{"role": "user", "content": user}]


class Strategy(ABC):
    name: str
    label: str
    uses_retrieval: bool = True

    @abstractmethod
    def run(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        client: ChatClient,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        trial: int = 0,
    ) -> StrategyResult:
        """Answer `question`. All strategies receive the same `chunks` (identical retrieval)."""

    def _recorder(
        self,
        client: ChatClient,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        trial: int,
    ) -> Recorder:
        return Recorder(
            client,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            cache_salt=f"{self.name}:trial{trial}",
        )
