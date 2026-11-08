"""FastAPI application factory.

    uvicorn ragbench.api.main:app --reload        (or: ragbench serve)

Replay endpoints need no key and no model calls. Live endpoints need the visitor's own Groq key in
the X-Groq-Api-Key header; the server holds no LLM key of its own.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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

API_CSP = "default-src 'none'; frame-ancestors 'none'"

# Only for FastAPI's two documentation pages, which load their code from a CDN. Swagger UI needs an
# inline bootstrap script; ReDoc needs web fonts, generated styles, a blob worker and its logo.
_SWAGGER_CDN = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/"
_REDOC_CDN = "https://cdn.jsdelivr.net/npm/redoc@2/"
DOCS_CSP = (
    "default-src 'none'; "
    f"script-src 'unsafe-inline' {_SWAGGER_CDN} {_REDOC_CDN}; "
    f"style-src 'unsafe-inline' {_SWAGGER_CDN} https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src data: https://fastapi.tiangolo.com https://cdn.redoc.ly; "
    "worker-src blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


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
) -> FastAPI:
    """Build the app. Every dependency can be injected, which is how the tests avoid the network."""
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

    docs_paths = {app.docs_url, app.redoc_url}

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        csp = DOCS_CSP if request.url.path in docs_paths else API_CSP
        response.headers["Content-Security-Policy"] = csp
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            {"error": "http_error", "message": str(exc.detail)}, status_code=exc.status_code
        )

    if settings.warm_retriever:
        ctx.retriever.warm()

    app.include_router(build_router(ctx))
    app.state.ctx = ctx
    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """`ragbench.api.main:app` for uvicorn, built on first access rather than at import.

    Importing this module must stay side-effect free: building the app reads its config and, with
    RAGBENCH_WARM_RETRIEVER set, starts loading the embedding model in the background, which tests
    and tools that only need `create_app` should not pay for.
    """
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
