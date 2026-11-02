"""FastAPI application factory.

    uvicorn ragbench.api.main:app --reload        (or: ragbench serve)

Replay endpoints need no key and no model calls. Live endpoints need the visitor's own Groq key in
the X-Groq-Api-Key header; the server holds no LLM key of its own.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from ragbench.api.live import ClientFactory, LazyRetriever, default_client_factory
from ragbench.api.replay import ReplayStore
from ragbench.api.routes import AppContext, build_router
from ragbench.api.security import KEY_HEADER, InflightGuard, RateLimiter, install_log_redaction
from ragbench.api.settings import ApiSettings
from ragbench.config import (
    ExperimentSettings,
    get_env_settings,
    load_experiment_settings,
    repo_root,
)
from ragbench.ingest.manifest import load_manifest
from ragbench.metrics.cost import Pricing, load_pricing
from ragbench.pipeline import Retriever
from ragbench.questions import QuestionSet, load_questions

log = logging.getLogger(__name__)

API_CSP = "default-src 'none'; frame-ancestors 'none'"
_INLINE_SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)


def ui_csp(index_html: str) -> str:
    """CSP for the built frontend: only its own origin, plus the hashes of its inline scripts."""
    hashes = " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(s.encode("utf-8")).digest()).decode() + "'"
        for s in _INLINE_SCRIPT.findall(index_html)
    )
    return (
        f"default-src 'self'; script-src 'self' {hashes}; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    )


class UiFiles(StaticFiles):
    """The built frontend (ui/dist), served from the API's origin with its own CSP."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory=directory, html=True)
        self.csp = ui_csp((directory / "index.html").read_text(encoding="utf-8"))

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Content-Security-Policy"] = self.csp
        return response


def create_app(
    *,
    settings: ApiSettings | None = None,
    experiment: ExperimentSettings | None = None,
    results_dir: Path | None = None,
    questions: QuestionSet | None = None,
    retriever: Retriever | None = None,
    client_factory: ClientFactory | None = None,
    pricing: Pricing | None = None,
    titles: dict[str, str] | None = None,
    models: dict[str, str] | None = None,
    ui_dir: Path | None = None,
) -> FastAPI:
    """Build the app. Every dependency can be injected, which is how the tests avoid the network.

    If the frontend has been built (`ui/dist`, or `ui_dir`), it is served at "/" from the same
    origin as the API. Without a build, the server is API-only, as in local development.
    """
    install_log_redaction()
    settings = settings or ApiSettings()
    experiment = experiment or load_experiment_settings()
    questions = questions or load_questions()
    pricing = pricing if pricing is not None else load_pricing()
    if titles is None:
        titles = {d.id: d.title for d in load_manifest().documents}
    if models is None:
        env = get_env_settings()
        models = {"generator": env.generator_model, "judge": env.judge_model}

    ctx = AppContext(
        settings=settings,
        experiment=experiment,
        questions=questions,
        replay=ReplayStore(
            results_dir or repo_root() / "evaluation" / "results", questions, pricing, titles
        ),
        retriever=LazyRetriever(lambda: retriever) if retriever is not None else LazyRetriever(),
        client_factory=client_factory or default_client_factory,
        pricing=pricing,
        titles=titles,
        models=models,
        run_limiter=RateLimiter(settings.rate_limit_runs, settings.rate_limit_runs_window_s),
        validate_limiter=RateLimiter(
            settings.rate_limit_validate, settings.rate_limit_validate_window_s
        ),
        guard=InflightGuard(settings.max_concurrent_runs),
    )

    app = FastAPI(
        title="RAGBench API",
        version="0.1.0",
        description="Replay of the recorded benchmark, plus live runs with your own Groq key.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[KEY_HEADER, "Content-Type"],
        expose_headers=["Retry-After"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        # UiFiles sets its own policy; everything else (the API) gets the strict one
        response.headers.setdefault("Content-Security-Policy", API_CSP)
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            {"error": "http_error", "message": str(exc.detail)}, status_code=exc.status_code
        )

    if settings.warm_retriever:
        ctx.retriever.warm()

    app.include_router(build_router(ctx))
    # mounted last, so /api, /docs and /openapi.json take precedence over the static files
    ui_dir = ui_dir or repo_root() / "ui" / "dist"
    if (ui_dir / "index.html").is_file():
        app.mount("/", UiFiles(ui_dir), name="ui")
        log.info("serving the frontend from %s", ui_dir)
    app.state.ctx = ctx
    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """`ragbench.api.main:app` for uvicorn, built on first access rather than at import.

    Importing this module must stay side-effect free: building the app starts loading the
    embedding model in the background, which tests and tools that only need `create_app` should
    not pay for (and two apps loading the same index at once would race).
    """
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
