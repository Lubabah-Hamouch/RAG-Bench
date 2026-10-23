"""Stage 0b: candidate pool + local cross-encoder reranker (an experiment, not pipeline code).

Retrieve a pool of 20 chunks by dense similarity, rerank them with a local cross-encoder, keep the
top 4 or top 6. Compared with the Stage 0 variants (read from stage0.json, not re-judged) using the
same metric: judge-based context recall on Q03, Q08 (lookups) and Q09, Q11, Q12 (multi-hop).

Criteria fixed BEFORE running (the reranker should capture k=8's within-document gains cheaply):
  1. mean recall on Q03, Q08, Q09 >= 0.765  (75% of k=8's gain over current k=4: 0.557 -> 0.833)
  2. at most 6 chunks (<= 75% of k=8's context)
  3. no lookup regression: Q03 and Q08 each >= current k=4
  Expectation (not pass/fail): Q11 stays unsolved, because it is a cross-document problem.
  Outstanding before any adoption: Q13 must still abstain (needs generator tokens; deferred).

Run: python evaluation/stage0b_rerank.py   (uses the judge model only; reranking is local)
"""

from __future__ import annotations

import json

from sentence_transformers import CrossEncoder

from ragbench.config import load_experiment_settings, repo_root, resolve_chroma_dir
from ragbench.llm.client import LLMError
from ragbench.llm.roles import make_client
from ragbench.metrics.coverage import score_context_support
from ragbench.metrics.judge import Judge
from ragbench.questions import load_questions
from ragbench.retrieval.embedder import Embedder
from ragbench.retrieval.store import VectorStore
from ragbench.types import RetrievedChunk

TEST_IDS = ["Q03", "Q08", "Q09", "Q11", "Q12"]
MULTI_HOP = ["Q09", "Q11", "Q12"]
LOOKUP = ["Q03", "Q08"]
WITHIN_DOC = ["Q03", "Q08", "Q09"]
POOL = 20
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RES = repo_root() / "evaluation" / "results"


def context_text(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(chunks, start=1))


def main() -> None:
    s = load_experiment_settings()
    qs = {q.id: q for q in load_questions().questions}
    store = VectorStore(resolve_chroma_dir(s), Embedder(s.retrieval.embedding_model))
    judge = Judge(
        make_client("judge"),
        temperature=s.judge.temperature,
        max_tokens=s.judge.max_tokens,
        max_tokens_cap=s.judge.max_tokens_cap,
        max_retries=s.judge.max_retries,
    )
    print(f"judge={judge.client.model} reranker={RERANKER} pool={POOL}")
    ce = CrossEncoder(RERANKER, device="cpu")

    prior = json.loads((RES / "stage0" / "stage0.json").read_text(encoding="utf-8"))["variants"]
    lengths = {
        r["chunk_id"]: len(r["text"])
        for r in map(
            json.loads, (repo_root() / "data/processed/chunks.jsonl").open(encoding="utf-8")
        )
    }

    def rerank(qid: str, keep: int) -> list[RetrievedChunk]:
        pool = store.query(qs[qid].question, POOL)
        scores = ce.predict([(qs[qid].question, h.chunk.text) for h in pool])
        order = sorted(range(len(pool)), key=lambda i: float(scores[i]), reverse=True)
        return [pool[i] for i in order[:keep]]

    new: dict[str, dict[str, dict]] = {}
    for keep in (4, 6):
        name = f"rerank pool{POOL}->{keep}"
        new[name] = {}
        for qid in TEST_IDS:
            chunks = rerank(qid, keep)
            res = score_context_support(
                qs[qid].question, context_text(chunks), qs[qid].gold_fact_texts, judge
            )
            dense_rank = {h.chunk.chunk_id: h.rank for h in store.query(qs[qid].question, POOL)}
            new[name][qid] = {
                "recall": res.recall,
                "supported": res.supported,
                "n_chunks": len(chunks),
                "docs": sorted({h.chunk.doc_id.split("_")[0] for h in chunks}),
                "chunk_ids": [h.chunk.chunk_id for h in chunks],
                "dense_ranks_of_kept": [dense_rank[h.chunk.chunk_id] for h in chunks],
                "judge_tokens": sum(c.prompt_tokens + c.completion_tokens for c in res.judge_calls),
            }
            print(
                f"  {name:20s} {qid} recall={res.recall:.2f} dense ranks kept={new[name][qid]['dense_ranks_of_kept']}"
            )

    all_variants = {**prior, **new}

    def mean(v: str, ids: list[str]) -> float:
        return sum(all_variants[v][q]["recall"] for q in ids) / len(ids)

    def avg_chars(v: str) -> float:
        return sum(
            sum(lengths[c] for c in all_variants[v][q]["chunk_ids"]) for q in TEST_IDS
        ) / len(TEST_IDS)

    cur, k8 = "current k=4", "flat k=8"
    gain_target = mean(cur, WITHIN_DOC) + 0.75 * (mean(k8, WITHIN_DOC) - mean(cur, WITHIN_DOC))
    criteria: dict[str, dict] = {}
    for name in new:
        mx = max(new[name][q]["n_chunks"] for q in TEST_IDS)
        criteria[name] = {
            "1_within_doc_mean": {
                "value": mean(name, WITHIN_DOC),
                "target": gain_target,
                "passed": mean(name, WITHIN_DOC) >= gain_target,
            },
            "2_max_chunks_le_6": {"value": mx, "passed": mx <= 6},
            "3_no_lookup_regression": {
                "value": {q: [new[name][q]["recall"], prior[cur][q]["recall"]] for q in LOOKUP},
                "passed": all(new[name][q]["recall"] >= prior[cur][q]["recall"] for q in LOOKUP),
            },
            "expectation_q11_unsolved": {
                "value": new[name]["Q11"]["recall"],
                "as_expected": new[name]["Q11"]["recall"] <= prior[cur]["Q11"]["recall"],
            },
        }

    (RES / "stage0").mkdir(parents=True, exist_ok=True)
    (RES / "stage0" / "stage0b.json").write_text(
        json.dumps(
            {
                "reranker": RERANKER,
                "pool": POOL,
                "variants": new,
                "criteria": criteria,
                "judge_tokens_total": sum(
                    r["judge_tokens"] for v in new.values() for r in v.values()
                ),
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print("\n== context recall (judge-based), Stage 0 variants reused ==")
    print(
        f"{'variant':22s} "
        + " ".join(f"{q:>5s}" for q in TEST_IDS)
        + "  | multi-hop  lookup  within-doc | chunks  ctx chars"
    )
    for v in all_variants:
        row = " ".join(f"{all_variants[v][q]['recall']:5.2f}" for q in TEST_IDS)
        mx = max(all_variants[v][q]["n_chunks"] for q in TEST_IDS)
        print(
            f"{v:22s} {row}  |   {mean(v, MULTI_HOP):.2f}      {mean(v, LOOKUP):.2f}     {mean(v, WITHIN_DOC):.2f}   | {mx:>4d}  {avg_chars(v):7.0f}"
        )
    print(f"\nwithin-document target (>= 75% of k=8's gain): {gain_target:.3f}")
    for name, c in criteria.items():
        print(f"\n{name}")
        for crit, d in c.items():
            flag = d.get("passed", d.get("as_expected"))
            print(f"  {'PASS' if flag else 'FAIL'}  {crit}: {d['value']}")


if __name__ == "__main__":
    try:
        main()
    except LLMError as exc:
        raise SystemExit(
            f"STOPPED by a provider error (no partial result is reported): {exc}"
        ) from None
