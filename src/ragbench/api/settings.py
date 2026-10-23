"""API configuration, read from RAGBENCH_* environment variables (all optional)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LOCAL_DEV_ORIGINS = [
    "http://localhost:5173",  # Vite
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGBENCH_", env_file=".env", extra="ignore")

    # CORS: local development origins by default; set the real domain(s) when deploying.
    cors_origins: list[str] = Field(default_factory=lambda: list(LOCAL_DEV_ORIGINS))

    # Hosted mode: refuse to accept an API key over plain HTTP.
    require_https: bool = False
    # Only honour X-Forwarded-For / X-Forwarded-Proto when behind a proxy you control.
    trust_proxy_headers: bool = False

    # Load the embedding model in the background at startup (it takes ~1 minute cold), so the
    # first live request does not pay for it. Replay endpoints never wait for it.
    warm_retriever: bool = True

    # Abuse protection for the live endpoint (per client IP, in-memory).
    max_question_chars: int = 500
    run_timeout_s: float = 240.0  # whole live run, checked between steps
    rate_limit_runs: int = 6  # live runs allowed per window
    rate_limit_runs_window_s: float = 600.0
    rate_limit_validate: int = 10  # key validations per window
    rate_limit_validate_window_s: float = 60.0
    max_concurrent_runs: int = 4  # across all clients; each client may have 1 in flight
