import pytest

from ragbench.metrics.lexical import lexical_metrics, mattr, sentences, words


def test_words_lowercases_and_drops_punctuation():
    assert words("Hello, World! hello") == ["hello", "world", "hello"]


def test_sentence_splitting_handles_abbreviations():
    assert len(sentences("Dr. Smith wrote this. It was good.")) == 2


@pytest.mark.parametrize(
    ("tokens", "window", "expected"),
    [
        (list("abab"), 2, 1.0),  # windows ab, ba, ab: all fully unique
        (list("aaa"), 2, 0.5),  # windows aa, aa: 1 unique of 2
        (list("abac"), 3, (2 / 3 + 3 / 3) / 2),  # aba -> 2/3, bac -> 3/3
        (list("abcd"), 4, 1.0),
        (list("aabb"), 4, 0.5),
    ],
)
def test_mattr_matches_hand_computed_values(tokens, window, expected):
    assert mattr(tokens, window) == pytest.approx(expected)


def test_mattr_is_undefined_for_text_shorter_than_one_window():
    assert mattr(list("abc"), 25) is None


def test_mattr_matches_a_naive_implementation_on_longer_text():
    tokens = ("the cat sat on the mat while the dog ran past the old red barn " * 6).split()
    window = 10
    windows = [tokens[i : i + window] for i in range(len(tokens) - window + 1)]
    naive = sum(len(set(w)) / window for w in windows) / len(windows)
    assert mattr(tokens, window) == pytest.approx(naive)


def test_lexical_metrics_on_a_small_text():
    m = lexical_metrics("One two. Three four five.")
    assert (m.chars, m.words, m.sentences, m.unique_vocab) == (25, 5, 2, 5)
    assert m.avg_word_length == pytest.approx(3.8)
    assert m.words_per_sentence == pytest.approx(2.5)
    assert m.ttr == 1.0 and m.mattr is None  # too short for a 25-word window


def test_short_answers_get_ttr_one_but_no_mattr_unlike_the_original_report():
    m = lexical_metrics("Halts execution for Assert only.")
    assert m.ttr == 1.0 and m.mattr is None


def test_repetition_lowers_mattr_for_long_text():
    varied = " ".join(f"w{i}" for i in range(60))
    repetitive = " ".join(["same words again"] * 20)
    assert lexical_metrics(varied).mattr == pytest.approx(1.0)
    assert lexical_metrics(repetitive).mattr < 0.2


def test_empty_text_is_safe():
    m = lexical_metrics("")
    assert m.words == 0 and m.avg_word_length is None and m.ttr is None and m.mattr is None
