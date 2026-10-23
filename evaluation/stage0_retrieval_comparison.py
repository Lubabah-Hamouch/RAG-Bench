"""Stage 0: compare retrieval variants by judge-based context recall (an experiment, not pipeline code).

Variants, on Q03, Q08, Q09, Q11, Q12 (plus a Q13 abstention check):
  current k=4 | flat k=8 | LLM query decomposition (merged, <=6 chunks) | document-balanced k=6

Adoption bar for decomposition (fixed in advance):
  1. multi-hop context recall >= flat k=8's + 0.15
  2. no more chunks than k=8
  3. no lookup regression versus the current k=4
  4. Q13 still abstains under the decomposed retrieval

Run: python evaluation/stage0_retrieval_comparison.py
"""

from __future__ import annotations

import json

from ragbench.config import load_experiment_settings, repo_root, resolve_chroma_dir
from ragbench.llm.client import LLMError
from ragbench.llm.roles import make_client
from ragbench.metrics.coverage import score_context_support
from ragbench.metrics.judge import Judge
from ragbench.questions import load_questions
from ragbench.retrieval.embedder import Embedder
from ragbench.retrieval.store import VectorStore
from ragbench.strategies.registry import get_strategy
from ragbench.types import RetrievedChunk

TEST_IDS = ["Q03", "Q08", "Q09", "Q11", "Q12"]
MULTI_HOP = {"Q09", "Q11", "Q12"}
LOOKUP = {"Q03", "Q08"}
OUT = repo_root() / "evaluation" / "results" / "stage0"
MAX_MERGED = 6
PER_QUERY = 3

DECOMPOSE_PROMPT = """You prepare search queries for a document retrieval system.
Decide whether answering the question needs information about several distinct topics, papers or systems that a single search would not find together.
- If the question is about ONE topic, return {"sub_queries": []}.
- If it compares or relates two or more distinct things, return one self-contained search query per thing (at most 3). Each sub-query must be about a single topic and make sense without the others.
Do not answer the question.

Question: {question}

Return JSON only: {"sub_queries": ["..."]}"""


def decompose(client, question: str) -> list[str]:
    prompt = DECOMPOSE_PROMPT.replace("{question}", question)
    resp = client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
        json_mode=True,
        cache_salt="stage0:decompose",
    )
    try:
        subs = [s.strip() for s in json.loads(resp.text).get("sub_queries", []) if str(s).strip()]
    except (json.JSONDecodeError, AttributeError):
        return []
    return subs[:3] if len(subs) >= 2 else []  # one sub-query is not a decomposition


def merge_round_robin(lists: list[list[RetrievedChunk]], cap: int) -> list[RetrievedChunk]:
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for rank in range(max(len(x) for x in lists)):
        for lst in lists:
            if rank < len(lst) and lst[rank].chunk.chunk_id not in seen:
                seen.add(lst[rank].chunk.chunk_id)
                merged.append(lst[rank])
    return merged[:cap]


def doc_balanced(
    store: VectorStore, question: str, k: int = 6, n_docs: int = 3
) -> list[RetrievedChunk]:
    hits = store.query(question, 24)
    chosen: list[RetrievedChunk] = []
    docs: set[str] = set()
    for h in hits:
        if h.chunk.doc_id not in docs and len(docs) < n_docs:
            docs.add(h.chunk.doc_id)
            chosen.append(h)
    for h in hits:
        if len(chosen) >= k:
            break
        if h not in chosen:
            chosen.append(h)
    return chosen[:k]


def context_text(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(chunks, start=1))


def main() -> None:
    s = load_experiment_settings()
    qs = {q.id: q for q in load_questions().questions}
    store = VectorStore(resolve_chroma_dir(s), Embedder(s.retrieval.embedding_model))
    gen = make_client("generator")
    judge = Judge(
        make_client("judge"),
        temperature=s.judge.temperature,
        max_tokens=s.judge.max_tokens,
        max_tokens_cap=s.judge.max_tokens_cap,
        max_retries=s.judge.max_retries,
    )
    print(f"generator={gen.model} judge={judge.client.model}")

    # ---- 1. decomposition (also on Q13 to see whether its retrieval would change) -------------
    subqueries: dict[str, list[str]] = {}
    for qid in [*TEST_IDS, "Q13"]:
        subqueries[qid] = decompose(gen, qs[qid].question)
        print(f"  decompose {qid}: {subqueries[qid] or 'NOT SPLIT'}")

    def decomposed(qid: str) -> list[RetrievedChunk]:
        q = qs[qid]
        subs = subqueries[qid]
        if not subs:
            return store.query(q.question, 4)  # not split: exactly the current retrieval
        lists = [store.query(q.question, PER_QUERY)] + [store.query(sq, PER_QUERY) for sq in subs]
        return merge_round_robin(lists, MAX_MERGED)

    variants = {
        "current k=4": lambda qid: store.query(qs[qid].question, 4),
        "flat k=8": lambda qid: store.query(qs[qid].question, 8),
        "LLM decomposition": decomposed,
        "doc-balanced k=6": lambda qid: doc_balanced(store, qs[qid].question, 6),
    }

    # ---- 2. judge-based context recall -------------------------------------------------------
    results: dict[str, dict[str, dict]] = {v: {} for v in variants}
    for vname, fn in variants.items():
        for qid in TEST_IDS:
            chunks = fn(qid)
            res = score_context_support(
                qs[qid].question, context_text(chunks), qs[qid].gold_fact_texts, judge
            )
            results[vname][qid] = {
                "recall": res.recall,
                "supported": res.supported,
                "n_chunks": len(chunks),
                "docs": sorted({h.chunk.doc_id.split("_")[0] for h in chunks}),
                "chunk_ids": [h.chunk.chunk_id for h in chunks],
                "judge_tokens": sum(c.prompt_tokens + c.completion_tokens for c in res.judge_calls),
            }
            print(f"  {vname:18s} {qid} recall={res.recall:.2f} chunks={len(chunks)}")

    # ---- 3. Q13 abstention under decomposed retrieval ----------------------------------------
    q13 = qs["Q13"]
    cur13 = [h.chunk.chunk_id for h in store.query(q13.question, 4)]
    dec13_chunks = decomposed("Q13")
    dec13 = [h.chunk.chunk_id for h in dec13_chunks]
    q13_changed = cur13 != dec13
    abstain: dict[str, bool] = {}
    for name in ("baseline", "structured", "rrr", "self_rag"):
        r = get_strategy(name).run(
            q13.question,
            dec13_chunks,
            client=gen,
            temperature=s.generation.temperature,
            max_tokens=s.generation.max_tokens,
            trial=0,
        )
        abstain[name] = r.abstained
    print(f"  Q13 retrieval changed by decomposition: {q13_changed}; abstained: {abstain}")

    # ---- 4. adoption bar ----------------------------------------------------------------------
    def mean(vname: str, ids: set[str]) -> float:
        vals = [results[vname][q]["recall"] for q in ids]
        return sum(vals) / len(vals)

    mh = {v: mean(v, MULTI_HOP) for v in variants}
    lk = {v: mean(v, LOOKUP) for v in variants}
    dec, k8, cur = "LLM decomposition", "flat k=8", "current k=4"
    max_dec_chunks = max(results[dec][q]["n_chunks"] for q in TEST_IDS)
    max_k8_chunks = max(results[k8][q]["n_chunks"] for q in TEST_IDS)
    criteria = {
        "1_multihop_recall_gain_over_k8_ge_0.15": {
            "value": mh[dec] - mh[k8],
            "passed": mh[dec] - mh[k8] >= 0.15,
        },
        "2_no_more_chunks_than_k8": {
            "value": [max_dec_chunks, max_k8_chunks],
            "passed": max_dec_chunks <= max_k8_chunks,
        },
        "3_no_lookup_regression_vs_current": {
            "value": {q: [results[dec][q]["recall"], results[cur][q]["recall"]] for q in LOOKUP},
            "passed": all(results[dec][q]["recall"] >= results[cur][q]["recall"] for q in LOOKUP),
        },
        "4_q13_still_abstains": {
            "value": {"retrieval_changed": q13_changed, "abstained": abstain},
            "passed": all(abstain.values()),
        },
    }
    adopted = all(c["passed"] for c in criteria.values())

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "variants": results,
        "multi_hop_recall": mh,
        "lookup_recall": lk,
        "subqueries": subqueries,
        "criteria": criteria,
        "decomposition_adopted": adopted,
        "judge_tokens_total": sum(r["judge_tokens"] for v in results.values() for r in v.values()),
    }
    (OUT / "stage0.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("\n== context recall (judge-based) ==")
    print(
        f"{'variant':18s} "
        + " ".join(f"{q:>5s}" for q in TEST_IDS)
        + "  | multi-hop  lookup  | chunks (max)"
    )
    for v in variants:
        row = " ".join(f"{results[v][q]['recall']:5.2f}" for q in TEST_IDS)
        mx = max(results[v][q]["n_chunks"] for q in TEST_IDS)
        print(f"{v:18s} {row}  |   {mh[v]:.2f}      {lk[v]:.2f}   | {mx}")
    print("\n== adoption criteria ==")
    for name, c in criteria.items():
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {name}: {c['value']}")
    print(f"\nDECOMPOSITION ADOPTED: {adopted}")


if __name__ == "__main__":
    try:
        main()
    except LLMError as exc:
        raise SystemExit(
            f"STOPPED by a provider error (no partial result is reported): {exc}"
        ) from None
