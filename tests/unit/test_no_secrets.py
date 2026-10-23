"""Guard: no key-shaped strings in any non-ignored text file, and no key files present."""

from __future__ import annotations

import re
import subprocess

from ragbench.config import repo_root

ROOT = repo_root()
KEY_SHAPES = re.compile(r"\bgsk_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{20,}")
SKIP_SUFFIXES = {".pdf", ".png", ".docx", ".ipynb", ".json"}  # binary, or generated/outputs


def _candidate_files():
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")
        return [ROOT / p for p in out if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        skip = {
            ".venv",
            ".git",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }
        return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.parts)]


def test_no_key_shaped_strings():
    hits = []
    for path in _candidate_files():
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if KEY_SHAPES.search(text):
            hits.append(str(path.relative_to(ROOT)))
    assert not hits, f"key-shaped strings in: {hits}"


def test_no_apikey_file_in_repo():
    names = [p.name.lower() for p in _candidate_files()]
    assert not [n for n in names if n.startswith("apikey") or n == ".env"]
