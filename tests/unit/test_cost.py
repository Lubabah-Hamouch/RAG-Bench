import pytest

from ragbench.metrics.cost import (
    ModelPrice,
    call_cost,
    cost_per_supported_claim,
    load_pricing,
    total_cost,
)
from ragbench.strategies.base import CallRecord

PRICING = {"m": ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0)}


def rec(model="m", p=1000, c=500):
    return CallRecord("s", model, 0.1, p, c)


def test_cost_arithmetic():
    assert call_cost(rec(), PRICING) == pytest.approx((1000 * 1.0 + 500 * 4.0) / 1e6)


def test_unknown_model_has_no_cost_rather_than_a_guess():
    assert call_cost(rec(model="mystery"), PRICING) is None
    assert total_cost([rec(), rec(model="mystery")], PRICING) is None


def test_total_cost_sums_calls():
    assert total_cost([rec(), rec(p=2000, c=0)], PRICING) == pytest.approx(0.003 + 0.002)
    assert total_cost([], PRICING) == 0.0


def test_cost_per_supported_claim():
    assert cost_per_supported_claim(0.006, 3) == pytest.approx(0.002)
    assert cost_per_supported_claim(0.006, 0) is None
    assert cost_per_supported_claim(None, 3) is None


def test_shipped_pricing_file_covers_the_two_configured_models():
    pricing = load_pricing()
    assert set(pricing) >= {"qwen/qwen3.8-27b", "openai/gpt-oss-20b"}
    assert pricing["openai/gpt-oss-20b"] == ModelPrice(0.075, 0.30)
    assert all(p.input_per_mtok > 0 and p.output_per_mtok > 0 for p in pricing.values())
