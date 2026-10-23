"""Corpus manifest: which documents are indexed, in what role, and how references are cut."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from ragbench.config import repo_root


class ReferenceSpan(BaseModel):
    start: str
    end: str | None = None


class DocumentSpec(BaseModel):
    id: str
    file: str
    title: str
    role: Literal["core", "distractor"]
    source_url: str
    retrieved: date
    license: str
    local: bool
    references: ReferenceSpan | None = None
    notes: str | None = None


class Manifest(BaseModel):
    documents: list[DocumentSpec]


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or repo_root() / "data" / "corpus" / "manifest.yaml"
    with path.open(encoding="utf-8") as fh:
        return Manifest.model_validate(yaml.safe_load(fh))
