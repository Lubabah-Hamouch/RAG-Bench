"""Judge validation: agreement between the LLM judge and a human on a labelled claim sample."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

LABELS = ("supported", "unsupported", "contradicted")


def load_labelled(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # coverage rows (judge_label covered / not covered) are a different judgment: see below
    return [
        r
        for r in rows
        if r.get("human_label", "").strip() and r.get("judge_label", "").strip().lower() in LABELS
    ]


def agreement(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Accuracy and Cohen's kappa of judge vs human, plus the supported-vs-not binary version."""
    if not rows:
        return {"n": 0}
    judge = [r["judge_label"].strip().lower() for r in rows]
    human = [r["human_label"].strip().lower() for r in rows]
    unknown = sorted({h for h in human if h not in LABELS})
    if unknown:
        raise ValueError(f"human_label must be one of {LABELS}; found {unknown}")
    n = len(rows)
    binary_j = [x == "supported" for x in judge]
    binary_h = [x == "supported" for x in human]
    out: dict[str, Any] = {
        "n": n,
        "accuracy_3way": sum(a == b for a, b in zip(judge, human, strict=True)) / n,
        "accuracy_supported_vs_not": sum(a == b for a, b in zip(binary_j, binary_h, strict=True))
        / n,
        "disagreements": [
            {"sample_id": r.get("sample_id"), "judge": j, "human": h}
            for r, j, h in zip(rows, judge, human, strict=True)
            if j != h
        ],
    }
    if len(set(judge) | set(human)) > 1:
        out["kappa_3way"] = float(cohen_kappa_score(judge, human))
    if len(set(binary_j) | set(binary_h)) > 1:
        out["kappa_supported_vs_not"] = float(cohen_kappa_score(binary_j, binary_h))
    return out


COVERAGE_LABELS = ("covered", "not covered")


def load_coverage_labelled(path: Path) -> list[dict[str, str]]:
    """Gold-fact coverage rows that a human has labelled (judge_label is covered / not covered)."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [
        r
        for r in rows
        if r.get("human_label", "").strip()
        and r.get("judge_label", "").strip().lower() in COVERAGE_LABELS
    ]


def coverage_agreement(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Judge vs human on whole-fact coverage. Too few rows for kappa, so list each case."""
    if not rows:
        return {"n": 0}
    cases = []
    for r in rows:
        human = r["human_label"].strip().lower()
        if human not in COVERAGE_LABELS:
            raise ValueError(
                f"coverage human_label must be one of {COVERAGE_LABELS}; got {human!r}"
            )
        cases.append(
            {
                "sample_id": r.get("sample_id"),
                "judge": r["judge_label"].strip().lower(),
                "human": human,
            }
        )
    return {
        "n": len(cases),
        "agree": sum(c["judge"] == c["human"] for c in cases),
        "cases": cases,
    }
