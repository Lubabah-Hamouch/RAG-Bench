"""Turn benchmark records into summary statistics.

Unit of analysis is the question. Quality metrics (G, C, FCS) use the judged trial(s) on the
answerable questions; consistency and efficiency use every trial. The out-of-corpus question is
reported separately as abstention behaviour, never averaged into quality.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ragbench.benchmark import CONTEXT_SUPPORT_FILE, _read_jsonl, load_retrieval, load_runs
from ragbench.metrics.consistency import semantic_consistency, structural_consistency
from ragbench.metrics.cost import Pricing, total_cost
from ragbench.metrics.lexical import words
from ragbench.metrics.stats import Estimate, aggregate, paired_difference, question_means
from ragbench.questions import Question
from ragbench.retrieval.embedder import EmbedderLike
from ragbench.strategies.base import CallRecord
from ragbench.strategies.registry import ALL_NAMES

CONTROL = "no_retrieval"
BASELINE = "baseline"
QUALITY = ("G", "C", "FCS")
FACT_OVERLAP = 0.8  # share of a gold quote's words that must appear in one retrieved chunk


def _est(e: Estimate | None) -> dict[str, float | int] | None:
    return None if e is None else {"mean": e.mean, "lo": e.lo, "hi": e.hi, "n": e.n}


def fact_in_chunks(quote: str, chunk_texts: Sequence[str]) -> bool:
    """Is the gold quote (approximately) present in a single retrieved chunk?"""
    tokens = set(words(quote))
    if not tokens:
        return False
    return any(len(tokens & set(words(t))) / len(tokens) >= FACT_OVERLAP for t in chunk_texts)


def context_recall(q: Question, chunk_texts: Sequence[str]) -> float | None:
    """Share of the question's gold facts whose supporting text reached the generator's context."""
    if not q.gold_facts:
        return None
    return sum(fact_in_chunks(g.quote, chunk_texts) for g in q.gold_facts) / len(q.gold_facts)


def _calls(rec: Mapping[str, Any]) -> list[CallRecord]:
    return [CallRecord(**c) for c in rec["calls"]]


def _quality(rec: Mapping[str, Any], answerable: bool) -> dict[str, float | None]:
    """G, C, FCS for one judged answer. A refusal on an answerable question is a miss (FCS 0)."""
    if not rec.get("judged"):
        return {"G": None, "C": None, "FCS": None, "X": None}
    g = rec["groundedness"]
    refused = bool(rec["abstained"] or g["abstained"])
    if answerable and refused:
        return {"G": None, "C": rec["coverage"]["C"], "FCS": 0.0, "X": None}
    return {"G": g["G"], "C": rec["coverage"]["C"], "FCS": rec["fcs"], "X": g["X"]}


def summarize(
    questions: Sequence[Question],
    out_dir: Path,
    embedder: EmbedderLike,
    pricing: Pricing,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    strategies: Sequence[str] = tuple(ALL_NAMES),
) -> dict[str, Any]:
    runs = load_runs(out_dir)
    retrieval = load_retrieval(out_dir)
    qmap = {q.id: q for q in questions}
    answerable = [q for q in questions if not q.expected_abstain]
    types = sorted({q.type for q in answerable})

    # ---- per (strategy, question) series --------------------------------------------------
    quality: dict[str, dict[str, dict[str, list[float | None]]]] = {
        s: {m: defaultdict(list) for m in (*QUALITY, "X")} for s in strategies
    }
    lexical: dict[str, dict[str, dict[str, list[float | None]]]] = {
        s: {m: defaultdict(list) for m in ("words", "sentences", "mattr", "ttr", "avg_word_length")}
        for s in strategies
    }
    efficiency: dict[str, dict[str, dict[str, list[float | None]]]] = {
        s: {m: defaultdict(list) for m in ("latency_s", "tokens", "cost_usd", "calls")}
        for s in strategies
    }
    consistency: dict[str, dict[str, dict[str, float | None]]] = {
        s: {"semantic": {}, "structural": {}} for s in strategies
    }
    answers: dict[tuple[str, str], list[str]] = defaultdict(list)
    abstain_flags: dict[str, dict[str, list[bool]]] = {s: defaultdict(list) for s in strategies}

    for (qid, strat, _trial), rec in sorted(runs.items()):
        if qid not in qmap or strat not in quality:
            continue
        q = qmap[qid]
        is_answerable = not q.expected_abstain
        answers[(strat, qid)].append(rec["answer"])
        abstain_flags[strat][qid].append(bool(rec["abstained"]))
        if is_answerable:
            for m, v in _quality(rec, True).items():
                if rec.get("judged"):
                    quality[strat][m][qid].append(v)
            lex = rec["lexical"] if rec.get("lexical") else None
            if lex:
                for m in lexical[strat]:
                    lexical[strat][m][qid].append(lex.get(m))
            else:
                nwords = len(words(rec["answer"]))
                lexical[strat]["words"][qid].append(float(nwords))
            efficiency[strat]["latency_s"][qid].append(rec["gen_latency_s"])
            efficiency[strat]["tokens"][qid].append(
                rec["gen_prompt_tokens"] + rec["gen_completion_tokens"]
            )
            efficiency[strat]["cost_usd"][qid].append(total_cost(_calls(rec), pricing))
            efficiency[strat]["calls"][qid].append(len(rec["calls"]))

    for (strat, qid), texts in answers.items():
        if qmap[qid].expected_abstain:
            continue
        consistency[strat]["semantic"][qid] = semantic_consistency(texts, embedder)
        consistency[strat]["structural"][qid] = structural_consistency(texts)

    def agg(
        series: Mapping[str, Sequence[float | None]], subset: set[str] | None = None
    ) -> Estimate | None:
        chosen = {k: v for k, v in series.items() if subset is None or k in subset}
        return aggregate(chosen, n_resamples, seed)

    # ---- headline tables --------------------------------------------------------------------
    result: dict[str, Any] = {"strategies": {}, "by_type": {}, "paired": {}, "per_question": {}}
    for s in strategies:
        row: dict[str, Any] = {}
        for m in (*QUALITY, "X"):
            row[m] = _est(agg(quality[s][m]))
        for m in lexical[s]:
            row[m] = _est(agg(lexical[s][m]))
        for m in efficiency[s]:
            row[m] = _est(agg(efficiency[s][m]))
        for m in ("semantic", "structural"):
            row[f"consistency_{m}"] = _est(
                aggregate({k: [v] for k, v in consistency[s][m].items()}, n_resamples, seed)
            )
        result["strategies"][s] = row

    for t in types:
        ids = {q.id for q in answerable if q.type == t}
        result["by_type"][t] = {
            s: {m: _est(agg(quality[s][m], ids)) for m in QUALITY} for s in strategies
        }

    # ---- paired differences on per-question means --------------------------------------------
    def qm(s: str, m: str) -> dict[str, float]:
        return question_means(quality[s][m])

    for ref_name, ref in (("vs_baseline", BASELINE), ("vs_control", CONTROL)):
        result["paired"][ref_name] = {}
        for s in strategies:
            if s == ref:
                continue
            result["paired"][ref_name][s] = {}
            for m in QUALITY:
                cmp = paired_difference(qm(s, m), qm(ref, m), n_resamples, seed)
                result["paired"][ref_name][s][m] = (
                    None if cmp is None else {"diff": _est(cmp.diff), "wilcoxon_p": cmp.wilcoxon_p}
                )

    # ---- abstention -------------------------------------------------------------------------
    abstention: dict[str, Any] = {}
    for s in strategies:
        out_ids = [q.id for q in questions if q.expected_abstain]
        out_flags = [f for qid in out_ids for f in abstain_flags[s].get(qid, [])]
        ans_flags = [f for q in answerable for f in abstain_flags[s].get(q.id, [])]
        abstention[s] = {
            "out_of_corpus_abstain_rate": (sum(out_flags) / len(out_flags)) if out_flags else None,
            "out_of_corpus_n": len(out_flags),
            "false_abstain_rate_answerable": (sum(ans_flags) / len(ans_flags))
            if ans_flags
            else None,
            "answerable_n": len(ans_flags),
        }
    result["abstention"] = abstention

    # out-of-corpus answers from the control, judged against the retrieved context
    control_out: list[dict[str, Any]] = []
    for q in questions:
        if q.expected_abstain:
            for (qid, strat, trial), rec in sorted(runs.items()):
                if qid == q.id and strat == CONTROL and rec.get("judged"):
                    control_out.append(
                        {
                            "qid": qid,
                            "trial": trial,
                            "G": rec["groundedness"]["G"],
                            "n_claims": len(rec["groundedness"]["claims"]),
                        }
                    )
    result["control_out_of_corpus"] = control_out

    # ---- retrieval diagnostics -------------------------------------------------------------------
    sims_by_rank: dict[int, list[float]] = defaultdict(list)
    recalls: dict[str, float | None] = {}
    distractor_hits = 0
    for qid, rec in retrieval.items():
        if qid not in qmap or not rec.get("chunks"):
            continue
        for ch in rec["chunks"]:
            sims_by_rank[ch["rank"]].append(ch["similarity"])
            distractor_hits += ch["role"] == "distractor"
        recalls[qid] = context_recall(qmap[qid], [c["text"] for c in rec["chunks"]])
    defined = [v for v in recalls.values() if v is not None]
    support = {r["qid"]: r for r in _read_jsonl(out_dir / CONTEXT_SUPPORT_FILE)}
    judge_recalls = {qid: r["recall"] for qid, r in support.items() if qid in qmap}
    result["retrieval"] = {
        "judge_context_recall_by_question": judge_recalls,
        "mean_judge_context_recall": (sum(judge_recalls.values()) / len(judge_recalls))
        if judge_recalls
        else None,
        "mean_similarity_by_rank": {
            str(r): sum(v) / len(v) for r, v in sorted(sims_by_rank.items())
        },
        "distractor_chunks_in_top4": distractor_hits,
        "questions": len(recalls),
        "context_recall_by_question": recalls,
        "mean_context_recall": (sum(defined) / len(defined)) if defined else None,
    }

    # ---- per-question detail (for heatmaps and tables) ----------------------------
    for q in questions:
        result["per_question"][q.id] = {
            "type": q.type,
            "question": q.question,
            "strategies": {
                s: {
                    **{m: qm(s, m).get(q.id) for m in QUALITY},
                    "consistency_semantic": consistency[s]["semantic"].get(q.id),
                    "context_recall_judge": judge_recalls.get(q.id),
                    "abstain_rate": (
                        sum(abstain_flags[s][q.id]) / len(abstain_flags[s][q.id])
                        if abstain_flags[s].get(q.id)
                        else None
                    ),
                }
                for s in strategies
            },
        }

    # ---- totals ---------------------------------------------------------------------------------
    gen_tokens = judge_tokens = n_calls = cached = retries = 0
    gen_cost = judge_cost = 0.0
    cost_known = True
    for rec in runs.values():
        for c in rec["calls"]:
            gen_tokens += c["prompt_tokens"] + c["completion_tokens"]
            n_calls += 1
            cached += bool(c["cached"])
            retries += c["n_retries"]
        cost = total_cost(_calls(rec), pricing)
        cost_known &= cost is not None
        gen_cost += cost or 0.0
        if rec.get("judged"):
            judge_tokens += rec["judge_prompt_tokens"] + rec["judge_completion_tokens"]
            judge_cost += rec["judge_cost_usd"] or 0.0
            for c in rec["judge_calls"]:
                n_calls += 1
                cached += bool(c["cached"])
                retries += c["n_retries"]
    result["totals"] = {
        "answers": len(runs),
        "judged_answers": sum(bool(r.get("judged")) for r in runs.values()),
        "model_calls": n_calls,
        "cached_calls": cached,
        "rate_limit_retries": retries,
        "generator_tokens": gen_tokens,
        "judge_tokens": judge_tokens,
        "generator_list_cost_usd": gen_cost if cost_known else None,
        "judge_list_cost_usd": judge_cost,
    }
    return result


def sample_claims_for_labeling(out_dir: Path, path: Path, n: int = 30, seed: int = 0) -> int:
    """Write a CSV of judged claims for a human to label (the judge-validation set).

    The sample is stratified so both verdict types are represented. The `human_label` column is
    left blank on purpose: agreement can only be computed once a person fills it in.
    """
    runs = load_runs(out_dir)
    retrieval = load_retrieval(out_dir)
    pool: list[dict[str, str]] = []
    for (qid, strat, _), rec in sorted(runs.items()):
        if not rec.get("judged") or qid not in retrieval:
            continue
        g = rec["groundedness"]
        passages = "\n".join(f"[{c['rank']}] {c['text']}" for c in retrieval[qid]["chunks"])
        for claim, label, reason in zip(g["claims"], g["labels"], g["reasons"], strict=True):
            pool.append(
                {
                    "qid": qid,
                    "strategy": strat,
                    "claim": claim,
                    "judge_label": label,
                    "judge_reason": reason,
                    "passages": passages,
                }
            )
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in pool:
        by_label[item["judge_label"]].append(item)
    for items in by_label.values():
        rng.shuffle(items)
    chosen: list[dict[str, str]] = []
    # take non-supported claims first (they are rarer and matter most), then fill with supported
    for label in ("contradicted", "unsupported", "supported"):
        quota = n - len(chosen) if label == "supported" else min(len(by_label[label]), n // 3)
        chosen.extend(by_label[label][:quota])
    chosen = chosen[:n]
    rng.shuffle(chosen)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sample_id", *pool[0].keys(), "human_label", "notes"]
            if pool
            else ["sample_id"],
        )
        writer.writeheader()
        for i, item in enumerate(chosen, start=1):
            writer.writerow({"sample_id": i, **item, "human_label": "", "notes": ""})
    return len(chosen)


def run_totals_from_log(out_dir: Path) -> dict[str, Any]:
    from ragbench.benchmark import LOG_FILE, _read_jsonl

    entries = _read_jsonl(out_dir / LOG_FILE)
    return {
        "invocations": len(entries),
        "active_minutes": sum(e["seconds"] for e in entries) / 60,
        "quota_stops": sum(bool(e["quota_stopped"]) for e in entries),
        "judge_errors": sum(e["judge_errors"] for e in entries),
        "first_start": entries[0]["at"] if entries else None,
        "last_end": entries[-1]["at"] if entries else None,
        "entries": [asdict_safe(e) for e in entries],
    }


def asdict_safe(e: Mapping[str, Any]) -> dict[str, Any]:
    return dict(e)
