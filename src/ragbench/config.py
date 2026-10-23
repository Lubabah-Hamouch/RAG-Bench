"""Typed configuration: experiment settings from YAML, models and keys from the environment."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    """Repository root (the directory holding pyproject.toml)."""
    return Path(__file__).resolve().parents[2]


class RetrievalSettings(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_chars: int = 600
    chunk_overlap: int = 50
    top_k: int = 4
    distance: str = "cosine"


class GenerationSettings(BaseModel):
    temperature: float = 0.5
    max_tokens: int = 700


class JudgeSettings(BaseModel):
    temperature: float = 0.0
    # Starting output budget per judge call. gpt-oss spends part of it on hidden reasoning, so a
    # truncated reply is retried with double the budget, up to max_tokens_cap.
    max_tokens: int = 1500
    max_tokens_cap: int = 6000
    max_retries: int = 2


class BenchmarkSettings(BaseModel):
    n_trials: int = 2
    judge_trials: int = 1
    bootstrap_resamples: int = 10_000
    seed: int = 20251209


class MetricsSettings(BaseModel):
    mattr_window: int = 25


class PathSettings(BaseModel):
    corpus_dir: Path = Path("data/corpus")
    corpus_manifest: Path = Path("data/corpus/manifest.yaml")
    processed_dir: Path = Path("data/processed")
    chroma_dir: Path = Path("data/chroma")
    questions: Path = Path("data/questions.yaml")


class ExperimentSettings(BaseModel):
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()
    judge: JudgeSettings = JudgeSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()
    metrics: MetricsSettings = MetricsSettings()
    paths: PathSettings = PathSettings()


class EnvSettings(BaseSettings):
    """Models and keys. Keys are SecretStr so they never appear in repr() or logs."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    generator_base_url: str = "https://api.groq.com/openai/v1"
    generator_api_key: SecretStr | None = None
    generator_model: str = "qwen/qwen3.8-27b"
    generator_reasoning_effort: str = "none"

    judge_base_url: str = "https://api.groq.com/openai/v1"
    judge_api_key: SecretStr | None = None
    judge_model: str = "openai/gpt-oss-20b"
    judge_reasoning_effort: str = "low"


def resolve_chroma_dir(settings: ExperimentSettings | None = None) -> Path:
    """Where the Chroma index lives: the configured in-repo path (`data/chroma`, gitignored).

    `RAGBENCH_CHROMA_DIR` overrides it. The index is rebuildable with `ragbench ingest`.
    """
    override = os.environ.get("RAGBENCH_CHROMA_DIR")
    if override:
        return Path(override)
    settings = settings or load_experiment_settings()
    return repo_root() / settings.paths.chroma_dir


def load_experiment_settings(path: Path | None = None) -> ExperimentSettings:
    path = path or repo_root() / "configs" / "settings.yaml"
    with path.open(encoding="utf-8") as fh:
        return ExperimentSettings.model_validate(yaml.safe_load(fh) or {})


@lru_cache
def get_env_settings() -> EnvSettings:
    return EnvSettings()
