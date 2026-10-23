"""Prompt templates, stored as plain .txt files next to this module and rendered by name."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# The one sentence every retrieval strategy is told to use when the context cannot answer.
REFUSAL = "The provided context does not contain this information."

_PLACEHOLDER = re.compile(r"\{([a-z_0-9]+)\}")


@lru_cache
def load_template(name: str) -> str:
    return (Path(__file__).parent / f"{name}.txt").read_text(encoding="utf-8")


def placeholders(name: str) -> set[str]:
    return set(_PLACEHOLDER.findall(load_template(name)))


def render(name: str, **values: str) -> str:
    """Fill `{placeholder}` fields. Strict: missing and unexpected values both raise.

    Substitution is single-pass, so text inserted for one field is never re-scanned (a retrieved
    chunk that happens to contain "{question}" cannot inject anything).
    """
    expected = placeholders(name)
    if set(values) != expected:
        raise KeyError(f"template {name!r} needs {sorted(expected)}, got {sorted(values)}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], load_template(name)).strip()


def is_refusal(text: str) -> bool:
    """True if the answer states that the context lacks the information."""
    return REFUSAL.lower().rstrip(".") in text.lower()
