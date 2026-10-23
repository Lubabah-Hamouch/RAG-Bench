"""Command-line entry point. Subcommands are added phase by phase."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from collections import Counter
from typing import TYPE_CHECKING

from ragbench.config import load_experiment_settings, repo_root, resolve_chroma_dir

if TYPE_CHECKING:
    from ragbench.llm.client import ChatClient
    from ragbench.retrieval.store import VectorStore


def _cmd_audit() -> None:
    from ragbench.ingest.audit import run_audit

    sys.stdout.write(run_audit())


def _cmd_ingest(chunk_chars: int | None, overlap: int | None) -> None:
    from ragbench.ingest.chunking import chunk_corpus, length_stats, write_chunks_jsonl
    from ragbench.ingest.corpus import load_cleaned_documents
    from ragbench.retrieval.embedder import Embedder
    from ragbench.retrieval.store import VectorStore

    s = load_experiment_settings()
    chunk_chars = chunk_chars or s.retrieval.chunk_chars
    overlap = overlap if overlap is not None else s.retrieval.chunk_overlap

    t0 = time.perf_counter()
    docs = load_cleaned_documents()
    chunks = chunk_corpus(docs, chunk_chars, overlap)
    write_chunks_jsonl(chunks, repo_root() / s.paths.processed_dir / "chunks.jsonl")

    chroma_dir = resolve_chroma_dir(s)
    store = VectorStore(chroma_dir, Embedder(s.retrieval.embedding_model))
    n = store.rebuild(chunks)

    per_doc = Counter(c.doc_id for c in chunks)
    print(f"Indexed {n} chunks from {len(docs)} documents in {time.perf_counter() - t0:.1f}s")
    for doc_id, count in per_doc.items():
        print(f"  {doc_id:32s} {count:4d}")
    print("Chunk length (chars):", json.dumps(length_stats(chunks)))
    print(f"Embedding model: {s.retrieval.embedding_model} | distance: cosine")
    print(f"Chroma dir: {chroma_dir}")


def _build_store() -> VectorStore:
    from ragbench.retrieval.embedder import Embedder
    from ragbench.retrieval.store import VectorStore

    s = load_experiment_settings()
    return VectorStore(resolve_chroma_dir(s), Embedder(s.retrieval.embedding_model))


def _cmd_ask(
    question: str, live: bool, names: list[str] | None, trial: int, show_trace: bool
) -> None:
    from ragbench.llm.fake import FakeLLMClient
    from ragbench.pipeline import run_question
    from ragbench.strategies.registry import REGISTRY

    s = load_experiment_settings()
    client: ChatClient
    if live:
        from ragbench.llm.roles import make_client

        client = make_client("generator")
        print(f"[live] generator model: {client.model}", file=sys.stderr)
    else:
        client = FakeLLMClient()
        print("[fake LLM: canned replies. Use --live to call the real model]", file=sys.stderr)

    result = run_question(
        question,
        _build_store(),
        client,
        strategies=names,
        top_k=s.retrieval.top_k,
        temperature=s.generation.temperature,
        max_tokens=s.generation.max_tokens,
        trial=trial,
    )
    print("Retrieved context (shared by every strategy):")
    for h in result.chunks:
        c = h.chunk
        print(f"  [{h.rank}] sim={h.similarity:.3f}  {c.doc_id}  p.{c.page}")
    for name, r in result.results.items():
        label = REGISTRY[name].label
        flag = "  [ABSTAINED]" if r.abstained else ""
        print(
            f"\n=== {label} | {r.n_calls} call(s) | {r.latency_s:.2f}s | "
            f"{r.prompt_tokens} in / {r.completion_tokens} out tokens{flag}"
        )
        print(textwrap.indent(r.display(), "  "))
        if show_trace and len(r.trace) > 1:
            for stage in r.trace:
                print(f"  --- stage: {stage.name}")
                print(textwrap.indent(stage.text, "      "))


def _cmd_budget(trials: int | None, margin: float, questions: int | None) -> None:
    from ragbench.budget import load_budget_config, project, render
    from ragbench.questions import load_questions

    questions = questions or len(load_questions().questions)
    cfg = load_budget_config()
    t = trials or load_experiment_settings().benchmark.n_trials
    scenarios = {
        f"judge all {t} trials": project(cfg, n_questions=questions, trials=t, margin=margin),
        **{
            f"judge {j} of {t} trials": project(
                cfg, n_questions=questions, trials=t, judge_trials=j, margin=margin
            )
            for j in range(t - 1, 0, -1)
        },
    }
    print(
        f"Projected load: {questions} questions x {t} trials "
        f"({cfg.generator_model} generates, {cfg.judge_model} judges)"
    )
    print(f"Safety margin: {margin:.0%}. Free-tier limits come from configs/budget.yaml.\n")
    print(render(scenarios))
    for name, p in scenarios.items():
        for label, load in (("generator", p.generator), ("judge", p.judge)):
            verdict = "fits in one day" if load.fits_in_one_day else f"needs {load.days} days"
            print(
                f"  {name:22s} {label:9s} {verdict} "
                f"(tokens {load.tpd_use:.0%}, requests {load.rpd_use:.0%} of daily)"
            )


def _cmd_bench(
    trials: int | None,
    judge_trials: int | None,
    qids: list[str] | None,
    names: list[str] | None,
    out: str | None,
) -> None:
    from pathlib import Path

    from ragbench.benchmark import run_benchmark
    from ragbench.llm.roles import make_client
    from ragbench.metrics.cost import load_pricing
    from ragbench.metrics.judge import Judge
    from ragbench.questions import load_questions

    s = load_experiment_settings()
    questions = load_questions().questions
    if qids:
        questions = [q for q in questions if q.id in qids]
    gen = make_client("generator")
    judge_client = make_client("judge")
    print(f"generator={gen.model} judge={judge_client.model} questions={len(questions)}")
    judge = Judge(
        judge_client,
        temperature=s.judge.temperature,
        max_tokens=s.judge.max_tokens,
        max_tokens_cap=s.judge.max_tokens_cap,
        max_retries=s.judge.max_retries,
    )
    out_dir = Path(out) if out else repo_root() / "evaluation" / "results"
    outcome = run_benchmark(
        questions,
        _build_store(),
        gen,
        judge,
        load_pricing(),
        s,
        out_dir,
        strategies=names,
        trials=trials,
        judge_trials=judge_trials,
    )
    print(
        f"done: {outcome.completed} new, {outcome.skipped} already done, "
        f"{outcome.judge_errors} judge errors, quota_stopped={outcome.quota_stopped}, "
        f"{outcome.seconds / 60:.1f} min"
    )


def _cmd_analyze(out: str | None) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from ragbench.analysis import run_totals_from_log, sample_claims_for_labeling, summarize
    from ragbench.config import get_env_settings
    from ragbench.metrics.cost import load_pricing
    from ragbench.questions import load_questions
    from ragbench.report import make_figures, write_outputs
    from ragbench.retrieval.embedder import Embedder

    s = load_experiment_settings()
    qs = load_questions().questions
    out_dir = Path(out) if out else repo_root() / "evaluation" / "results"
    summary = summarize(
        qs,
        out_dir,
        Embedder(s.retrieval.embedding_model),
        load_pricing(),
        n_resamples=s.benchmark.bootstrap_resamples,
        seed=s.benchmark.seed,
    )
    env = get_env_settings()
    comp: dict[str, int] = {}
    for q in qs:
        comp[q.type] = comp.get(q.type, 0) + 1
    meta = {
        "n_questions": len(qs),
        "n_answerable": sum(not q.expected_abstain for q in qs),
        "composition": ", ".join(f"{n} {t.replace('_', '-')}" for t, n in comp.items()),
        "trials": s.benchmark.n_trials,
        "judge_trials": s.benchmark.judge_trials,
        "generator": env.generator_model,
        "judge": env.judge_model,
        "log": run_totals_from_log(out_dir),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    excluded = ["Q11"]  # retrieval-limited: its context held 1 of 3 gold facts
    kept = [q for q in qs if q.id not in excluded]
    sens_summary = summarize(
        kept,
        out_dir,
        Embedder(s.retrieval.embedding_model),
        load_pricing(),
        n_resamples=s.benchmark.bootstrap_resamples,
        seed=s.benchmark.seed,
    )
    note_path = out_dir / "sensitivity_note.md"
    sensitivity = {
        "excluded": excluded,
        "summary": sens_summary,
        "note": note_path.read_text(encoding="utf-8") if note_path.exists() else None,
    }
    write_outputs(summary, meta, out_dir, sensitivity)
    figs = make_figures(summary, meta, out_dir / "figures")
    n = sample_claims_for_labeling(out_dir, out_dir / "claims_to_label.csv")
    print(f"wrote summary.json, summary.md, {len(figs)} figures, {n} claims to label in {out_dir}")


def _cmd_context_support(out: str | None) -> None:
    from pathlib import Path

    from ragbench.benchmark import run_context_support
    from ragbench.llm.roles import make_client
    from ragbench.metrics.judge import Judge
    from ragbench.questions import load_questions

    s = load_experiment_settings()
    judge = Judge(
        make_client("judge"),
        temperature=s.judge.temperature,
        max_tokens=s.judge.max_tokens,
        max_tokens_cap=s.judge.max_tokens_cap,
        max_retries=s.judge.max_retries,
    )
    out_dir = Path(out) if out else repo_root() / "evaluation" / "results"
    n = run_context_support(load_questions().questions, out_dir, judge)
    print(f"judged context support for {n} question(s)")


def _cmd_serve(host: str, port: int, reload: bool) -> None:
    import logging

    import uvicorn

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
    app_log = logging.getLogger("ragbench")
    app_log.setLevel(logging.INFO)
    app_log.addHandler(handler)

    print(f"RAGBench API on http://{host}:{port}  (docs at /docs). Replay needs no key; live runs")
    print("need the visitor's Groq key in the X-Groq-Api-Key header.")
    uvicorn.run("ragbench.api.main:app", host=host, port=port, reload=reload)


def _cmd_export_replay(out: str | None) -> None:
    from pathlib import Path

    from ragbench.api.main import create_app

    store = create_app().state.ctx.replay
    target = Path(out) if out else repo_root() / "ui" / "public" / "replay"
    written = store.export_static(target)
    print(f"wrote {len(written)} static replay files to {target}")


def _cmd_retrieve(question: str, top_k: int | None, as_json: bool) -> None:
    from ragbench.retrieval.embedder import Embedder
    from ragbench.retrieval.store import VectorStore

    s = load_experiment_settings()
    k = top_k or s.retrieval.top_k
    store = VectorStore(resolve_chroma_dir(s), Embedder(s.retrieval.embedding_model))
    hits = store.query(question, k)
    if as_json:
        print(
            json.dumps(
                [{"rank": h.rank, "similarity": h.similarity, **h.chunk.to_dict()} for h in hits],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for h in hits:
        c = h.chunk
        print(f"[{h.rank}] sim={h.similarity:.3f}  {c.doc_id}  p.{c.page}  ({c.role})")
        print(textwrap.indent(textwrap.fill(h.chunk.text, 100), "    "))
        print()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page
    parser = argparse.ArgumentParser(prog="ragbench")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit", help="extract + clean the corpus and write docs/corpus_audit.md")

    p_ingest = sub.add_parser("ingest", help="chunk the corpus and (re)build the Chroma index")
    p_ingest.add_argument("--chunk-chars", type=int)
    p_ingest.add_argument("--overlap", type=int)

    p_ret = sub.add_parser("retrieve", help="show the top-k chunks for a question")
    p_ret.add_argument("question")
    p_ret.add_argument("-k", "--top-k", type=int)
    p_ret.add_argument("--json", action="store_true")

    p_bench = sub.add_parser("bench", help="run the benchmark (live; resumable)")
    p_bench.add_argument("--trials", type=int)
    p_bench.add_argument("--judge-trials", type=int)
    p_bench.add_argument("--questions", help="comma-separated ids, e.g. Q02,Q09")
    p_bench.add_argument("--strategies", help="comma-separated subset")
    p_bench.add_argument("--out", help="results directory (default evaluation/results)")

    p_srv = sub.add_parser("serve", help="run the API (replay + bring-your-own-key live runs)")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--reload", action="store_true")

    p_exp = sub.add_parser("export-replay", help="write the replay data as static JSON files")
    p_exp.add_argument("--out", help="default: ui/public/replay")

    p_cs = sub.add_parser(
        "context-support", help="judge whether retrieved context holds the gold facts"
    )
    p_cs.add_argument("--out")

    p_an = sub.add_parser("analyze", help="summarise benchmark results: tables and figures")
    p_an.add_argument("--out", help="results directory (default evaluation/results)")

    p_bud = sub.add_parser("budget", help="project call/token volume against free-tier limits")
    p_bud.add_argument("--trials", type=int)
    p_bud.add_argument("--margin", type=float, default=0.25)
    p_bud.add_argument("--questions", type=int, help="default: size of data/questions.yaml")

    p_ask = sub.add_parser(
        "ask", help="run a question through the strategies (fake LLM by default)"
    )
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--live", action="store_true", help="call the real generator (uses your key)"
    )
    p_ask.add_argument("--strategies", help="comma-separated subset, e.g. baseline,self_rag")
    p_ask.add_argument("--trial", type=int, default=0)
    p_ask.add_argument("--trace", action="store_true", help="show intermediate stages")

    args = parser.parse_args(argv)
    if args.command == "serve":
        _cmd_serve(args.host, args.port, args.reload)
    elif args.command == "export-replay":
        _cmd_export_replay(args.out)
    elif args.command == "context-support":
        _cmd_context_support(args.out)
    elif args.command == "analyze":
        _cmd_analyze(args.out)
    elif args.command == "bench":
        _cmd_bench(
            args.trials,
            args.judge_trials,
            args.questions.split(",") if args.questions else None,
            args.strategies.split(",") if args.strategies else None,
            args.out,
        )
    elif args.command == "budget":
        _cmd_budget(args.trials, args.margin, args.questions)
    elif args.command == "ask":
        names = args.strategies.split(",") if args.strategies else None
        _cmd_ask(args.question, args.live, names, args.trial, args.trace)
    elif args.command == "audit":
        _cmd_audit()
    elif args.command == "ingest":
        _cmd_ingest(args.chunk_chars, args.overlap)
    elif args.command == "retrieve":
        _cmd_retrieve(args.question, args.top_k, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
