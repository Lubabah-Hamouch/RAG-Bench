"""Run the benchmark: questions x strategies x trials, with metrics, resumably.

Results are appended to `runs.jsonl` (one JSON object per question/strategy/trial) as each answer
completes, so an interrupted run (a daily quota, a crash) resumes where it stopped and nothing
already paid for is repeated. The loop is trial-major: all of trial 0 (the judged trial) finishes
before trial 1 starts, so a quota stop never leaves the judged data incomplete.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragbench.config import ExperimentSettings
from ragbench.llm.client import ChatClient, LLMError
from ragbench.metrics.cost import Pricing
from ragbench.metrics.judge import Judge, JudgeError
from ragbench.metrics.scoring import AnswerMetrics, evaluate_answer
from ragbench.pipeline import Retriever
from ragbench.questions import Question
from ragbench.strategies.base import StrategyResult
from ragbench.strategies.registry import ALL_NAMES, get_strategy
from ragbench.types import RetrievedChunk

RUNS_FILE = "runs.jsonl"
RETRIEVAL_FILE = "retrieval.jsonl"
LOG_FILE = "run_log.jsonl"

Key = tuple[str, str, int]  # (question id, strategy, trial)


@dataclass
class Outcome:
    completed: int  # answers newly produced in this invocation
    skipped: int  # answers already in runs.jsonl
    judge_errors: int
    quota_stopped: bool
    seconds: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _append(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def load_runs(out_dir: Path) -> dict[Key, dict[str, Any]]:
    """Latest record per (question, strategy, trial)."""
    return {(r["qid"], r["strategy"], r["trial"]): r for r in _read_jsonl(out_dir / RUNS_FILE)}


def load_retrieval(out_dir: Path) -> dict[str, dict[str, Any]]:
    return {r["qid"]: r for r in _read_jsonl(out_dir / RETRIEVAL_FILE)}


def _is_complete(rec: dict[str, Any] | None, needs_judge: bool) -> bool:
    return rec is not None and (not needs_judge or bool(rec.get("judged")))


def _metrics_dict(m: AnswerMetrics) -> dict[str, Any]:
    g = m.groundedness
    return {
        "lexical": asdict(m.lexical),
        "groundedness": {
            "claims": g.claims,
            "labels": [v.label for v in g.verdicts],
            "reasons": [v.reason for v in g.verdicts],
            "evidence_passages": [v.evidence_passage for v in g.verdicts],
            "n_supported": g.n_supported,
            "n_unsupported": g.n_unsupported,
            "n_contradicted": g.n_contradicted,
            "n_missing_verdicts": g.n_missing_verdicts,
            "G": g.groundedness,
            "H": g.hallucination_rate,
            "X": g.contradiction_rate,
            "abstained": g.abstained,
        },
        "coverage": {"covered": m.coverage.covered, "C": m.coverage.coverage},
        "fcs": m.fcs,
        "gen_cost_usd": m.gen_cost_usd,
        "judge_cost_usd": m.judge_cost_usd,
        "judge_prompt_tokens": m.judge_prompt_tokens,
        "judge_completion_tokens": m.judge_completion_tokens,
        "cost_per_supported_claim": m.cost_per_supported_claim,
        "judge_calls": [asdict(c) for c in g.judge_calls + m.coverage.judge_calls],
    }


def build_record(
    q: Question,
    trial: int,
    result: StrategyResult,
    metrics: AnswerMetrics | None,
    judge_error: str | None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "qid": q.id,
        "qtype": q.type,
        "strategy": result.strategy,
        "trial": trial,
        "answer": result.answer,
        "critique": result.critique,
        "trace": [{"stage": s.name, "text": s.text} for s in result.trace],
        "abstained": result.abstained,
        "calls": [asdict(c) for c in result.calls],
        "gen_latency_s": result.latency_s,
        "gen_prompt_tokens": result.prompt_tokens,
        "gen_completion_tokens": result.completion_tokens,
        "judged": metrics is not None,
        "judge_error": judge_error,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if metrics is not None:
        rec.update(_metrics_dict(metrics))
    return rec


def retrieval_record(q: Question, chunks: Sequence[RetrievedChunk]) -> dict[str, Any]:
    return {
        "qid": q.id,
        "chunks": [
            {
                "rank": h.rank,
                "similarity": h.similarity,
                "chunk_id": h.chunk.chunk_id,
                "doc_id": h.chunk.doc_id,
                "role": h.chunk.role,
                "page": h.chunk.page,
                "text": h.chunk.text,
            }
            for h in chunks
        ],
    }


def run_benchmark(
    questions: Sequence[Question],
    retriever: Retriever,
    client: ChatClient,
    judge: Judge,
    pricing: Pricing,
    settings: ExperimentSettings,
    out_dir: Path,
    *,
    strategies: Sequence[str] | None = None,
    trials: int | None = None,
    judge_trials: int | None = None,
    log: Callable[[str], None] = print,
) -> Outcome:
    trials = settings.benchmark.n_trials if trials is None else trials
    judge_trials = settings.benchmark.judge_trials if judge_trials is None else judge_trials
    names = list(strategies or ALL_NAMES)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_runs(out_dir)
    have_retrieval = load_retrieval(out_dir)
    started = time.perf_counter()
    completed = skipped = judge_errors = 0
    quota = False

    chunks_by_q: dict[str, list[RetrievedChunk]] = {}
    try:
        for trial in range(trials):
            needs_judge = trial < judge_trials
            for q in questions:
                if q.id not in chunks_by_q:
                    chunks_by_q[q.id] = retriever.query(q.question, settings.retrieval.top_k)
                    if q.id not in have_retrieval:
                        _append(out_dir / RETRIEVAL_FILE, retrieval_record(q, chunks_by_q[q.id]))
                        have_retrieval[q.id] = {}
                chunks = chunks_by_q[q.id]
                for name in names:
                    if _is_complete(done.get((q.id, name, trial)), needs_judge):
                        skipped += 1
                        continue
                    result = get_strategy(name).run(
                        q.question,
                        chunks,
                        client=client,
                        temperature=settings.generation.temperature,
                        max_tokens=settings.generation.max_tokens,
                        trial=trial,
                    )
                    metrics: AnswerMetrics | None = None
                    judge_error: str | None = None
                    if needs_judge:
                        try:
                            metrics = evaluate_answer(
                                q.question,
                                result,
                                chunks,
                                judge=judge,
                                gold_facts=q.gold_fact_texts,
                                pricing=pricing,
                                mattr_window=settings.metrics.mattr_window,
                            )
                        except JudgeError as exc:
                            judge_error = str(exc)
                            judge_errors += 1
                    rec = build_record(q, trial, result, metrics, judge_error)
                    _append(out_dir / RUNS_FILE, rec)
                    done[(q.id, name, trial)] = rec
                    completed += 1
                    summary = ""
                    if metrics is not None:
                        g, c, f = (
                            metrics.groundedness.groundedness,
                            metrics.coverage.coverage,
                            metrics.fcs,
                        )
                        summary = f" G={_fmt(g)} C={_fmt(c)} FCS={_fmt(f)}"
                    elif judge_error:
                        summary = " JUDGE-ERROR"
                    log(f"  t{trial} {q.id} {name:13s} {result.latency_s:5.1f}s{summary}")
    except LLMError as exc:
        if exc.code != "rate_limit_long":
            raise
        quota = True
        log(f"STOPPED: provider quota reached ({exc}). Progress is saved; rerun to resume.")

    outcome = Outcome(completed, skipped, judge_errors, quota, time.perf_counter() - started)
    _append(
        out_dir / LOG_FILE,
        {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            **asdict(outcome),
            "trials": trials,
            "judge_trials": judge_trials,
            "questions": [q.id for q in questions],
            "strategies": names,
        },
    )
    return outcome


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


CONTEXT_SUPPORT_FILE = "context_support.jsonl"


def run_context_support(
    questions: Sequence[Question],
    out_dir: Path,
    judge: Judge,
    log: Callable[[str], None] = print,
) -> int:
    """Judge, per question, whether the retrieved context contains each gold fact. Resumable."""
    from ragbench.metrics.coverage import score_context_support

    retrieval = load_retrieval(out_dir)
    done = {r["qid"] for r in _read_jsonl(out_dir / CONTEXT_SUPPORT_FILE)}
    written = 0
    for q in questions:
        if q.id in done or not q.gold_facts or q.id not in retrieval:
            continue
        context = "\n\n".join(f"[{c['rank']}] {c['text']}" for c in retrieval[q.id]["chunks"])
        result = score_context_support(q.question, context, q.gold_fact_texts, judge)
        _append(
            out_dir / CONTEXT_SUPPORT_FILE,
            {
                "qid": q.id,
                "supported": result.supported,
                "recall": result.recall,
                "judge_tokens": sum(
                    c.prompt_tokens + c.completion_tokens for c in result.judge_calls
                ),
            },
        )
        log(f"  {q.id} context support {result.recall:.2f}")
        written += 1
    return written
