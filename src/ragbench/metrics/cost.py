"""Token and dollar cost from recorded calls. Unknown prices give None, never a guess."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from ragbench.config import repo_root
from ragbench.strategies.base import CallRecord


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


Pricing = Mapping[str, ModelPrice]


def load_pricing(path: Path | None = None) -> dict[str, ModelPrice]:
    path = path or repo_root() / "configs" / "pricing.yaml"
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return {
        model: ModelPrice(float(spec["input_per_mtok"]), float(spec["output_per_mtok"]))
        for model, spec in (raw.get("models") or {}).items()
    }


def call_cost(record: CallRecord, pricing: Pricing) -> float | None:
    """Dollar cost of one call; hidden reasoning tokens are inside completion_tokens (billed)."""
    price = pricing.get(record.model)
    if price is None:
        return None
    return (
        record.prompt_tokens * price.input_per_mtok
        + record.completion_tokens * price.output_per_mtok
    ) / 1_000_000


def total_cost(records: Iterable[CallRecord], pricing: Pricing) -> float | None:
    """Sum of call costs; None if any call's model has no known price."""
    total = 0.0
    for record in records:
        cost = call_cost(record, pricing)
        if cost is None:
            return None
        total += cost
    return total


def cost_per_supported_claim(cost_usd: float | None, n_supported: int) -> float | None:
    if cost_usd is None or n_supported <= 0:
        return None
    return cost_usd / n_supported
