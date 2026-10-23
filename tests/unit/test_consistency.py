import numpy as np
import pytest

from ragbench.metrics.consistency import semantic_consistency, structural_consistency


class MapEmbedder:
    name = "map"

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts):
        v = np.array([self.vectors[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


E = MapEmbedder({"a": [1, 0, 0], "a2": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1], "ab": [1, 1, 0]})


def test_identical_answers_are_perfectly_consistent():
    assert semantic_consistency(["a", "a2", "a"], E) == pytest.approx(1.0)


def test_orthogonal_answers_have_zero_consistency():
    assert semantic_consistency(["a", "b", "c"], E) == pytest.approx(0.0)


def test_three_trials_average_three_pairs():
    # pairs: (a,a2)=1, (a,b)=0, (a2,b)=0  ->  1/3
    assert semantic_consistency(["a", "a2", "b"], E) == pytest.approx(1 / 3)


def test_partial_similarity_uses_cosine():
    assert semantic_consistency(["a", "ab"], E) == pytest.approx(np.sqrt(0.5))


def test_fewer_than_two_answers_is_undefined():
    assert semantic_consistency(["a"], E) is None
    assert semantic_consistency([], E) is None


def test_structural_consistency_is_one_for_equal_lengths():
    assert structural_consistency(["a b c", "d e f", "g h i"]) == pytest.approx(1.0)


def test_structural_consistency_penalises_length_variation():
    texts = [" ".join(["w"] * n) for n in (10, 20, 0)]
    # counts 10, 20, 0: mean 10, population sd sqrt(200/3) -> cv 0.8165
    assert structural_consistency(texts) == pytest.approx(1 - np.sqrt(200 / 3) / 10)


def test_structural_consistency_is_clipped_to_zero_and_handles_empty():
    assert structural_consistency(["", "", " ".join(["w"] * 30)]) == 0.0
    assert structural_consistency(["", ""]) is None
    assert structural_consistency(["one"]) is None
