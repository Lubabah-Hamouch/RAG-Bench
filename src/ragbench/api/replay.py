"""Replay: serve the recorded benchmark (evaluation/results/) with no model calls."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragbench.api.assemble import answer_from_record, chunk_out, question_out
from ragbench.api.schemas import (
    ConsistencyOut,
    QuestionOut,
    QuestionResultOut,
    RetrievalOut,
    StrategyInfo,
    TrialAnswerOut,
)
from ragbench.benchmark import CONTEXT_SUPPORT_FILE, RETRIEVAL_FILE, RUNS_FILE, _read_jsonl
from ragbench.metrics.consistency import structural_consistency
from ragbench.metrics.cost import Pricing
from ragbench.questions import QuestionSet
from ragbench.strategies.registry import ALL_NAMES, REGISTRY

SUMMARY_FILE = "summary.json"
_FILES = (RUNS_FILE, RETRIEVAL_FILE, CONTEXT_SUPPORT_FILE, SUMMARY_FILE)


class ReplayUnavailable(LookupError):
    """No recorded results for the request (run `ragbench bench` and `ragbench analyze`)."""


def strategy_infos() -> list[StrategyInfo]:
    calls = {"baseline": 1, "structured": 1, "rrr": 3, "self_rag": 3, "no_retrieval": 1}
    return [
        StrategyInfo(
            name=n,
            label=REGISTRY[n].label,
            uses_retrieval=REGISTRY[n].uses_retrieval,
            calls=calls[n],
        )
        for n in ALL_NAMES
    ]


class ReplayStore:
    def __init__(
        self,
        results_dir: Path,
        questions: QuestionSet,
        pricing: Pricing,
        titles: Mapping[str, str],
    ) -> None:
        self.results_dir = results_dir
        self.questions = questions
        self.pricing = pricing
        self.titles = titles
        self._lock = threading.Lock()
        self._stamp: tuple[float, ...] | None = None
        self._runs: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._retrieval: dict[str, dict[str, Any]] = {}
        self._support: dict[str, dict[str, Any]] = {}
        self._summary: dict[str, Any] | None = None

    # ---- loading (re-read when a results file changes) --------------------------------------

    def _current_stamp(self) -> tuple[float, ...]:
        return tuple(
            (self.results_dir / f).stat().st_mtime if (self.results_dir / f).exists() else 0.0
            for f in _FILES
        )

    def _refresh(self) -> None:
        with self._lock:
            stamp = self._current_stamp()
            if stamp == self._stamp:
                return
            runs = _read_jsonl(self.results_dir / RUNS_FILE)
            self._runs = {(r["qid"], r["strategy"], r["trial"]): r for r in runs}
            self._retrieval = {r["qid"]: r for r in _read_jsonl(self.results_dir / RETRIEVAL_FILE)}
            self._support = {
                r["qid"]: r for r in _read_jsonl(self.results_dir / CONTEXT_SUPPORT_FILE)
            }
            path = self.results_dir / SUMMARY_FILE
            self._summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            self._stamp = stamp

    @property
    def available(self) -> bool:
        self._refresh()
        return bool(self._runs)

    # ---- public API -------------------------------------------------------------------------

    def questions_out(self) -> list[QuestionOut]:
        return [question_out(q) for q in self.questions.questions]

    def summary(self) -> dict[str, Any]:
        self._refresh()
        if self._summary is None:
            raise ReplayUnavailable("summary.json not found; run `ragbench analyze`")
        return self._summary

    def models(self) -> dict[str, str]:
        self._refresh()
        meta = (self._summary or {}).get("meta", {})
        return {k: meta[k] for k in ("generator", "judge") if k in meta}

    def result(self, qid: str) -> QuestionResultOut:
        self._refresh()
        q = next((x for x in self.questions.questions if x.id == qid), None)
        if q is None:
            raise KeyError(qid)
        by_strategy: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for (rq, strat, trial), rec in self._runs.items():
            if rq == qid:
                by_strategy[strat][trial] = rec
        if not by_strategy or qid not in self._retrieval:
            raise ReplayUnavailable(f"no recorded results for {qid}")

        per_q = ((self._summary or {}).get("per_question") or {}).get(qid, {}).get("strategies", {})
        answers = {}
        for name in ALL_NAMES:
            trials = by_strategy.get(name)
            if not trials:
                continue
            first = trials[min(trials)]
            texts = [trials[t]["answer"] for t in sorted(trials)]
            consistency = None
            if len(texts) >= 2 and not q.expected_abstain:
                semantic = per_q.get(name, {}).get("consistency_semantic")
                consistency = ConsistencyOut(
                    semantic=semantic,
                    structural=structural_consistency(texts),
                    n_samples=len(texts),
                )
            others = [
                TrialAnswerOut(trial=t, answer=trials[t]["answer"])
                for t in sorted(trials)
                if t != first["trial"]
            ]
            answers[name] = answer_from_record(
                first, q, self.pricing, consistency=consistency, other_trials=others
            )

        retrieval = RetrievalOut(
            chunks=[chunk_out(c, self.titles) for c in self._retrieval[qid]["chunks"]],
            context_recall_judge=(self._support.get(qid) or {}).get("recall"),
        )
        n_trials = max(len(t) for t in by_strategy.values())
        judged = max(sum(bool(r.get("judged")) for r in t.values()) for t in by_strategy.values())
        return QuestionResultOut(
            mode="replay",
            n_trials=n_trials,
            judged_trials=judged,
            models=self.models(),
            question=question_out(q),
            retrieval=retrieval,
            answers=answers,
        )

    # ---- static export (for the keyless hosted demo) ----------------------------------------

    def export_static(self, out_dir: Path) -> list[Path]:
        """Write the same JSON the API serves, as static files for a backend-less deployment."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results").mkdir(exist_ok=True)
        written: list[Path] = []

        def write(path: Path, payload: Any) -> None:
            text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            path.write_text(text, encoding="utf-8")
            written.append(path)

        write(
            out_dir / "questions.json", json.dumps([q.model_dump() for q in self.questions_out()])
        )
        write(out_dir / "strategies.json", json.dumps([s.model_dump() for s in strategy_infos()]))
        write(out_dir / "summary.json", self.summary())
        for q in self.questions.questions:
            write(out_dir / "results" / f"{q.id}.json", self.result(q.id).model_dump_json())
        return written
