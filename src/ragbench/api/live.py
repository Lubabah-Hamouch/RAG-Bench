"""Live runs: one question through all five conditions with a visitor-supplied (BYOK) key.

The key exists only as an argument to `ClientFactory` for the duration of one run. Clients built
with a key bypass the disk cache (see `ragbench.llm.roles.make_client`), and every error message
is scrubbed before it leaves this module.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ragbench.api.assemble import answer_from_record, chunk_out, question_out
from ragbench.api.schemas import QuestionResultOut, RetrievalOut
from ragbench.benchmark import build_record, retrieval_record
from ragbench.config import ExperimentSettings, load_experiment_settings, resolve_chroma_dir
from ragbench.llm.client import ChatClient, LLMError, scrub
from ragbench.llm.roles import make_client
from ragbench.metrics.cost import Pricing
from ragbench.metrics.judge import Judge, JudgeError
from ragbench.metrics.scoring import evaluate_answer
from ragbench.pipeline import Retriever
from ragbench.questions import Question
from ragbench.strategies.registry import ALL_NAMES, REGISTRY, get_strategy

log = logging.getLogger(__name__)

# (role, api_key) -> client. The default builds a cache-free client from the visitor's key.
ClientFactory = Callable[[str, str], ChatClient]
Event = tuple[str, dict[str, Any]]


class RunCancelled(Exception):
    """The client went away; stop before starting another model call."""


class _CancelAware:
    """Wraps a client so a cancelled run refuses to start its next model call.

    A call already in flight cannot be interrupted, but this bounds the wasted work (and the
    visitor's quota spent) to that one call, instead of the rest of the run.
    """

    def __init__(self, inner: ChatClient, cancelled: Callable[[], bool]) -> None:
        self._inner = inner
        self._cancelled = cancelled
        self.model = inner.model

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        if self._cancelled():
            raise RunCancelled
        return self._inner.chat(messages, **kwargs)


def default_client_factory(role: str, api_key: str) -> ChatClient:
    return make_client("generator" if role == "generator" else "judge", api_key=api_key)


class LazyRetriever:
    """Builds the embedder and Chroma store on first use, so replay-only servers start fast."""

    # Chroma's client construction is not safe to run concurrently (two builds on one index race
    # and one fails), so every instance in the process builds under one shared lock.
    _BUILD_LOCK = threading.Lock()

    def __init__(self, factory: Callable[[], Retriever] | None = None) -> None:
        self._factory = factory or _default_retriever
        self._retriever: Retriever | None = None
        self.error: str | None = None  # why the last build failed, if it did

    @property
    def ready(self) -> bool:
        return self._retriever is not None

    def _build(self) -> Retriever:
        with self._BUILD_LOCK:
            if self._retriever is None:
                try:
                    self._retriever = self._factory()
                    self.error = None
                except Exception as exc:
                    self.error = type(exc).__name__
                    raise
            return self._retriever

    def _warm_quietly(self) -> None:
        try:
            self._build()
        except Exception as exc:  # noqa: BLE001 - surfaced through `error` and /api/health
            log.warning("retriever warm-up failed: %s", type(exc).__name__)

    def warm(self) -> None:
        """Start loading in a background thread (idempotent). Queries wait for it if needed."""
        if not self.ready:
            threading.Thread(
                target=self._warm_quietly, name="retriever-warmup", daemon=True
            ).start()

    def query(self, text: str, top_k: int = 4) -> list[Any]:
        return self._build().query(text, top_k)


def _default_retriever() -> Retriever:
    from ragbench.retrieval.embedder import Embedder
    from ragbench.retrieval.store import VectorStore

    s = load_experiment_settings()
    embedder = Embedder(s.retrieval.embedding_model)
    # Load the model now, not on the first query: `ready` (and /api/health's live_ready) must only
    # turn true once a live run will not wait for the embedding model.
    embedder.warm()
    return VectorStore(resolve_chroma_dir(s), embedder)


def describe_error(exc: Exception, secrets: tuple[str, ...] = ()) -> tuple[int, str, str]:
    """Map an exception to (http_status, stable_code, scrubbed_message)."""
    if isinstance(exc, JudgeError):
        return 502, "judge_failed", scrub(str(exc), secrets)
    if isinstance(exc, LLMError):
        if exc.code == "auth_failed":
            return 401, "invalid_key", "The provider rejected the API key. Check it and try again."
        if exc.code in ("rate_limit_long", "rate_limited"):
            return (
                429,
                "provider_quota",
                "Your Groq rate limit or daily quota was reached. Wait and retry, or skip scoring.",
            )
        if exc.code == "model_not_found":
            return 502, "model_unavailable", scrub(str(exc), secrets)
        return 502, "upstream_error", scrub(str(exc), secrets)
    return 500, "internal_error", "Unexpected server error."  # no details: nothing to leak


def _error_event(exc: Exception, secrets: tuple[str, ...]) -> Event:
    status, code, message = describe_error(exc, secrets)
    return "error", {"error": code, "message": message, "status": status}


def live_events(
    *,
    question: Question,
    free_text: bool,
    score: bool,
    api_key: str,
    retriever: Retriever,
    factory: ClientFactory,
    settings: ExperimentSettings,
    pricing: Pricing,
    titles: Mapping[str, str],
    models: Mapping[str, str],
    budget_s: float,
    cancelled: Callable[[], bool] = lambda: False,
) -> Iterator[Event]:
    """Yield (event, payload): retrieval, strategy_started, strategy_done, metrics, done | error."""
    secrets = (api_key,)
    started = time.monotonic()

    try:
        gen_client = _CancelAware(factory("generator", api_key), cancelled)
        chunks = retriever.query(question.question, settings.retrieval.top_k)
        # the time budget starts after retrieval, so a cold embedder load is not charged to it
        deadline = time.monotonic() + budget_s

        def timed_out() -> bool:
            return time.monotonic() > deadline

        ret_rec = retrieval_record(question, chunks)
        retrieval = RetrievalOut(chunks=[chunk_out(c, titles) for c in ret_rec["chunks"]])
        q_out = question_out(question, free_text=free_text)
        yield (
            "retrieval",
            {"question": q_out.model_dump(), "chunks": retrieval.model_dump()["chunks"]},
        )

        judge: Judge | None = None
        if score:
            judge = Judge(
                _CancelAware(factory("judge", api_key), cancelled),
                temperature=settings.judge.temperature,
                max_tokens=settings.judge.max_tokens,
                max_tokens_cap=settings.judge.max_tokens_cap,
                max_retries=settings.judge.max_retries,
            )

        answers = {}
        for name in ALL_NAMES:
            if timed_out():
                yield (
                    "error",
                    {"error": "timeout", "message": "The run took too long.", "status": 504},
                )
                return
            yield "strategy_started", {"strategy": name, "label": REGISTRY[name].label}
            result = get_strategy(name).run(
                question.question,
                chunks,
                client=gen_client,
                temperature=settings.generation.temperature,
                max_tokens=settings.generation.max_tokens,
                trial=0,
            )
            answer = answer_from_record(
                build_record(question, 0, result, None, None), question, pricing
            )
            answers[name] = answer
            yield "strategy_done", answer.model_dump()
            log.info("live %s generated at +%.0fs", name, time.monotonic() - started)

            if judge is not None:
                if timed_out():
                    yield (
                        "error",
                        {"error": "timeout", "message": "The run took too long.", "status": 504},
                    )
                    return
                try:
                    metrics = evaluate_answer(
                        question.question,
                        result,
                        chunks,
                        judge=judge,
                        gold_facts=question.gold_fact_texts,
                        pricing=pricing,
                        mattr_window=settings.metrics.mattr_window,
                    )
                except JudgeError as exc:
                    _, code, message = describe_error(exc, secrets)
                    yield "metrics", {"strategy": name, "error": code, "message": message}
                    continue
                scored = answer_from_record(
                    build_record(question, 0, result, metrics, None), question, pricing
                )
                answers[name] = scored
                yield "metrics", {"strategy": name, "metrics": scored.metrics.model_dump()}
                log.info("live %s scored at +%.0fs", name, time.monotonic() - started)

        final = QuestionResultOut(
            mode="live",
            n_trials=1,  # a single trial: no consistency score and no confidence intervals
            judged_trials=1 if judge is not None else 0,
            models=dict(models),
            question=q_out,
            retrieval=retrieval,
            answers=answers,
        )
        yield "done", final.model_dump()
    except RunCancelled:
        log.info("live run cancelled by the client")
        return
    except LLMError as exc:
        log.warning("live run failed: %s (%s)", type(exc).__name__, exc.code)
        yield _error_event(exc, secrets)
    except Exception as exc:  # noqa: BLE001 - never surface internals (or a key) to the client
        log.warning(
            "live run failed: %s", type(exc).__name__
        )  # type only: no message, no traceback
        yield _error_event(exc, secrets)
