from ragbench.config import EnvSettings, load_experiment_settings, repo_root, resolve_chroma_dir
from ragbench.llm.roles import resolve_cache_dir


def test_chroma_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGBENCH_CHROMA_DIR", str(tmp_path / "x"))
    assert resolve_chroma_dir() == tmp_path / "x"


def test_chroma_dir_and_llm_cache_default_to_the_project_folder(monkeypatch):
    monkeypatch.delenv("RAGBENCH_CHROMA_DIR", raising=False)
    monkeypatch.delenv("RAGBENCH_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", "C:/Users/someone/AppData/Local")  # must not matter
    assert resolve_chroma_dir() == repo_root() / "data" / "chroma"
    assert resolve_cache_dir() == repo_root() / ".cache" / "llm_cache"


def test_experiment_settings_match_plan():
    s = load_experiment_settings()
    assert (s.retrieval.chunk_chars, s.retrieval.top_k, s.retrieval.distance) == (600, 4, "cosine")
    assert s.retrieval.embedding_model == "all-MiniLM-L6-v2"
    assert (s.benchmark.n_trials, s.benchmark.judge_trials) == (2, 1)


def test_judge_model_differs_from_generator_by_default():
    env = EnvSettings(_env_file=None)
    assert env.judge_model != env.generator_model


def test_env_override_swaps_judge_without_code_change(monkeypatch):
    monkeypatch.setenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-mini")
    env = EnvSettings(_env_file=None)
    assert (env.judge_base_url, env.judge_model) == ("https://api.openai.com/v1", "gpt-4o-mini")


def test_api_keys_never_appear_in_repr(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "gsk_" + "x" * 40)
    env = EnvSettings(_env_file=None)
    assert "xxxx" not in repr(env)
    assert "xxxx" not in str(env)
