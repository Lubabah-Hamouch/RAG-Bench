"""HTTP routes. Replay is open; live endpoints require a visitor-supplied key (BYOK)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from ragbench.api.live import (
    ClientFactory,
    LazyRetriever,
    describe_error,
    live_events,
)
from ragbench.api.replay import ReplayStore, ReplayUnavailable, strategy_infos
from ragbench.api.schemas import (
    ErrorOut,
    HealthOut,
    QuestionOut,
    QuestionResultOut,
    StrategyInfo,
    ValidateOut,
)
from ragbench.api.security import (
    KEY_HEADER,
    InflightGuard,
    RateLimiter,
    client_ip,
    is_secure,
    valid_key_shape,
)
from ragbench.api.settings import ApiSettings
from ragbench.config import ExperimentSettings
from ragbench.llm.client import LLMError
from ragbench.metrics.cost import Pricing
from ragbench.questions import Question, QuestionSet

NO_STORE = {"Cache-Control": "no-store"}


@dataclass
class AppContext:
    settings: ApiSettings
    experiment: ExperimentSettings
    questions: QuestionSet
    replay: ReplayStore
    retriever: LazyRetriever
    client_factory: ClientFactory
    pricing: Pricing
    titles: dict[str, str]
    models: dict[str, str]
    run_limiter: RateLimiter
    validate_limiter: RateLimiter
    guard: InflightGuard


class RunRequest(BaseModel):
    question: str | None = None  # free text
    question_id: str | None = None  # a benchmark question (enables coverage and FCS)
    score: bool = True  # also judge the answers (about 3 extra judge calls per strategy)


def error_response(
    status: int, code: str, message: str, retry_after: float | None = None
) -> JSONResponse:
    headers = dict(NO_STORE)
    if retry_after is not None:
        headers["Retry-After"] = str(max(1, round(retry_after)))
    body = ErrorOut(error=code, message=message, retry_after_s=retry_after)
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status, headers=headers)


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter(prefix="/api")

    # ---- replay (no key, no model calls) ----------------------------------------------------

    @router.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut(
            status="ok",
            replay_available=ctx.replay.available,
            questions=len(ctx.questions.questions),
            models=ctx.models,
            require_https=ctx.settings.require_https,
            live_ready=ctx.retriever.ready,
            live_error=ctx.retriever.error,
        )

    @router.get("/strategies", response_model=list[StrategyInfo])
    def strategies() -> list[StrategyInfo]:
        return strategy_infos()

    @router.get("/questions", response_model=list[QuestionOut])
    def questions() -> list[QuestionOut]:
        return ctx.replay.questions_out()

    @router.get("/results/summary", response_model=None)
    def summary() -> Any:
        try:
            return ctx.replay.summary()
        except ReplayUnavailable as exc:
            return error_response(503, "replay_unavailable", str(exc))

    @router.get("/results/{qid}", response_model=QuestionResultOut)
    def result(qid: str) -> Any:
        try:
            return ctx.replay.result(qid)
        except KeyError:
            return error_response(404, "unknown_question", f"No benchmark question {qid!r}.")
        except ReplayUnavailable as exc:
            return error_response(503, "replay_unavailable", str(exc))

    # ---- shared checks for the key-bearing endpoints ----------------------------------------

    def check_key(request: Request) -> tuple[str | None, JSONResponse | None]:
        if ctx.settings.require_https and not is_secure(request, ctx.settings):
            return None, error_response(
                400, "https_required", "API keys are only accepted over HTTPS on this server."
            )
        key = request.headers.get(KEY_HEADER)
        if not key:
            return None, error_response(
                401, "key_required", f"Send your Groq API key in the {KEY_HEADER} header."
            )
        if not valid_key_shape(key):
            # never echo the value back
            return None, error_response(400, "invalid_key_format", "The API key looks malformed.")
        return key, None

    # ---- live: validate a key ---------------------------------------------------------------

    @router.post("/live/validate", response_model=ValidateOut)
    async def validate(request: Request) -> Any:
        key, err = check_key(request)
        if err is not None or key is None:
            return err
        ok, wait = ctx.validate_limiter.check(client_ip(request, ctx.settings))
        if not ok:
            return error_response(429, "rate_limited", "Too many validation attempts.", wait)

        def probe() -> ValidateOut:
            gen = ctx.client_factory("generator", key)
            judge = ctx.client_factory("judge", key)
            gen_ok = gen.model in gen.list_models()  # type: ignore[attr-defined]
            judge_ok = judge.model in judge.list_models()  # type: ignore[attr-defined]
            if gen_ok and judge_ok:
                msg = "Key accepted; both models are available."
            else:
                missing = [
                    m for m, ok_ in ((gen.model, gen_ok), (judge.model, judge_ok)) if not ok_
                ]
                msg = "Key accepted, but model(s) not available to it: " + ", ".join(missing)
            return ValidateOut(
                valid=gen_ok and judge_ok,
                generator_model=gen.model,
                generator_available=gen_ok,
                judge_model=judge.model,
                judge_available=judge_ok,
                message=msg,
            )

        try:
            out = await run_in_threadpool(probe)
        except LLMError as exc:
            status, code, message = describe_error(exc, (key,))
            return error_response(status, code, message)
        except Exception as exc:  # noqa: BLE001
            status, code, message = describe_error(exc, (key,))
            return error_response(status, code, message)
        return JSONResponse(out.model_dump(), headers=NO_STORE)

    # ---- live: run all five conditions ------------------------------------------------------

    @router.post("/run", response_model=None)
    async def run(
        request: Request,
        body: RunRequest,
        stream: bool = Query(True, description="Server-sent events (default) or one JSON reply."),
    ) -> Any:
        key, err = check_key(request)
        if err is not None or key is None:
            return err

        if body.question_id:
            found = next((q for q in ctx.questions.questions if q.id == body.question_id), None)
            if found is None:
                return error_response(404, "unknown_question", "No such benchmark question.")
            question, free_text = found, False
        else:
            text = " ".join((body.question or "").split())
            if not text:
                return error_response(
                    422, "invalid_request", "Provide a question or a question_id."
                )
            if len(text) > ctx.settings.max_question_chars:
                return error_response(
                    422,
                    "question_too_long",
                    f"Questions are limited to {ctx.settings.max_question_chars} characters.",
                )
            question = Question(id="live", type="lookup", question=text)
            free_text = True

        ip = client_ip(request, ctx.settings)
        ok, wait = ctx.run_limiter.check(ip)
        if not ok:
            return error_response(429, "rate_limited", "Too many live runs; slow down.", wait)
        refused = ctx.guard.acquire(ip)
        if refused:
            msg = (
                "A run is already in progress for this client."
                if refused == "run_in_progress"
                else "The server is at capacity; try again shortly."
            )
            return error_response(429 if refused == "run_in_progress" else 503, refused, msg, 5.0)

        cancel = threading.Event()
        events = live_events(
            question=question,
            free_text=free_text,
            score=body.score,
            api_key=key,
            retriever=ctx.retriever,
            factory=ctx.client_factory,
            settings=ctx.experiment,
            pricing=ctx.pricing,
            titles=ctx.titles,
            models=ctx.models,
            budget_s=ctx.settings.run_timeout_s,
            cancelled=cancel.is_set,
        )

        if not stream:
            try:
                collected = await run_in_threadpool(list, events)
            finally:
                ctx.guard.release(ip)
            for name, payload in collected:
                if name == "error":
                    return error_response(
                        payload["status"],
                        payload["error"],
                        payload["message"],
                        payload.get("retry_after_s"),
                    )
            final = next(payload for name, payload in collected if name == "done")
            return JSONResponse(final, headers=NO_STORE)

        async def source() -> Any:
            try:
                async for name, payload in iterate_in_threadpool(events):
                    yield {"event": name, "data": json.dumps(payload)}
            finally:
                cancel.set()
                ctx.guard.release(ip)

        async def on_client_close(_scope: Any) -> None:
            # the visitor cancelled or left: stop spending their quota and free their slot now,
            # rather than after the step that is currently running finishes
            cancel.set()
            ctx.guard.release(ip)

        return EventSourceResponse(
            source(), headers=NO_STORE, client_close_handler_callable=on_client_close
        )

    return router
