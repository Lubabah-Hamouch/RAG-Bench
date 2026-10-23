"""Build generator / judge clients from environment configuration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import diskcache  # type: ignore[import-untyped]

from ragbench.config import EnvSettings, get_env_settings, repo_root
from ragbench.llm.client import LLMClient, MissingApiKeyError

Role = Literal["generator", "judge"]
log = logging.getLogger(__name__)


def resolve_cache_dir() -> Path:
    """Disk cache for model calls: `.cache/llm_cache` in the repo (gitignored).

    Override with RAGBENCH_CACHE_DIR.
    """
    override = os.environ.get("RAGBENCH_CACHE_DIR")
    if override:
        return Path(override)
    return repo_root() / ".cache" / "llm_cache"


def make_client(
    role: Role,
    api_key: str | None = None,
    *,
    use_cache: bool = True,
    env: EnvSettings | None = None,
) -> LLMClient:
    """Create the client for a role.

    `api_key` is the bring-your-own-key override: it replaces the environment key for this client
    only, and disables the shared disk cache so one visitor's results can never reach another.
    """
    env = env or get_env_settings()
    if role == "generator":
        base_url, model, env_key = (
            env.generator_base_url,
            env.generator_model,
            env.generator_api_key,
        )
        effort = env.generator_reasoning_effort
    else:
        base_url, model, env_key = env.judge_base_url, env.judge_model, env.judge_api_key
        effort = env.judge_reasoning_effort

    key = api_key or (env_key.get_secret_value() if env_key else "")
    if not key:
        raise MissingApiKeyError(
            f"no API key for the {role} role: set {role.upper()}_API_KEY in .env or pass a key"
        )
    if env.judge_model == env.generator_model:
        log.warning("judge and generator use the same model (%s); scores will be biased", model)

    cache = diskcache.Cache(str(resolve_cache_dir())) if use_cache and api_key is None else None
    extra_body = {"reasoning_effort": effort} if effort else None
    return LLMClient(base_url, key, model, role=role, cache=cache, extra_body=extra_body)
