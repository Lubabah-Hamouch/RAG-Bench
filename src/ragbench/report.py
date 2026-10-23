"""Render the summary as markdown tables and matplotlib figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ragbench.strategies.registry import ALL_NAMES, REGISTRY  # noqa: E402

COLORS = {
    "baseline": "#4C78A8",
    "structured": "#F58518",
    "rrr": "#54A24B",
    "self_rag": "#B279A2",
    "no_retrieval": "#8C8C8C",
}


def label(name: str) -> str:
    return REGISTRY[name].label


def fmt(e: dict[str, Any] | None, digits: int = 2) -> str:
    if not e:
        return "n/a"
    return f"{e['mean']:.{digits}f} [{e['lo']:.{digits}f}, {e['hi']:.{digits}f}]"


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _diff(row: dict[str, Any], metric: str) -> str:
    return fmt(row[metric]["diff"]) if row.get(metric) else "n/a"


def render_sensitivity(
    full: dict[str, Any],
    excl: dict[str, Any],
    excluded: list[str],
    n_full: int,
    note: str | None,
) -> list[str]:
    """Side-by-side tables: all answerable questions vs the same analysis without `excluded`."""
    names = [n for n in ALL_NAMES if n in full["strategies"]]
    ex = ", ".join(excluded)
    n_excl = n_full - len(excluded)
    lines = [
        f"## Sensitivity: excluding {ex}",
        "",
        "Recomputed from the same recorded answers and judgements (no new model calls) on the "
        f"{n_excl} remaining answerable questions, shown beside the original {n_full}. "
        f"{ex} is excluded because its retrieved context held too little of the needed evidence "
        "(judge-based context recall 0.33), so its scores mostly measure retrieval failure.",
        "",
        f"**Quality (mean [95% CI]; original n={n_full} | excluding {ex}, n={n_excl})**",
        "",
        "| strategy | G (all) | G (excl.) | C (all) | C (excl.) | FCS (all) | FCS (excl.) |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in names:
        a, b = full["strategies"][n], excl["strategies"][n]
        cells = " | ".join(f"{fmt(a[m])} | {fmt(b[m])}" for m in ("G", "C", "FCS"))
        lines.append(f"| {label(n)} | {cells} |")
    lines += [
        "",
        "**Retrieval vs no retrieval: paired difference (strategy minus control)**",
        "",
        "| strategy | dG (all) | dG (excl.) | dC (all) | dC (excl.) | dFCS (all) | dFCS (excl.) |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in full["paired"]["vs_control"]:
        a, b = full["paired"]["vs_control"][n], excl["paired"]["vs_control"][n]
        cells = " | ".join(f"{_diff(a, m)} | {_diff(b, m)}" for m in ("G", "C", "FCS"))
        lines.append(f"| {label(n)} | {cells} |")
    lines += [
        "",
        "**Strategy vs Baseline: paired difference in FCS**",
        "",
        "| strategy | dFCS (all) | p (all) | dFCS (excl.) | p (excl.) |",
        "|---|---|---|---|---|",
    ]
    for n in full["paired"]["vs_baseline"]:
        a, b = full["paired"]["vs_baseline"][n], excl["paired"]["vs_baseline"][n]
        pa = a["FCS"]["wilcoxon_p"] if a.get("FCS") else None
        pb = b["FCS"]["wilcoxon_p"] if b.get("FCS") else None
        da, db = _diff(a, "FCS"), _diff(b, "FCS")
        lines.append(f"| {label(n)} | {da} | {_num(pa, 3)} | {db} | {_num(pb, 3)} |")
    lines += [
        "",
        "**Refusals on answerable questions** (all -> excluding)",
        "",
        "| strategy | refused an answerable question |",
        "|---|---|",
    ]
    for n in names:
        a = full["abstention"][n]["false_abstain_rate_answerable"]
        b = excl["abstention"][n]["false_abstain_rate_answerable"]
        lines.append(f"| {label(n)} | {_num(a)} -> {_num(b)} |")
    if note:
        lines += ["", "**Reading of the sensitivity analysis**", "", note.strip()]
    lines.append("")
    return lines


JUDGE_VALIDATION = (
    "- **Judge validation (human-labelled sample).** On 30 claims the judge agreed with a "
    "human label 28 times (93%; 95% CI for accuracy [0.79, 0.98]; Cohen's kappa 0.879, "
    "3-way). All three disagreements found (two claim-level, one coverage-level) ran "
    "against the retrieval-vs-control gap: two were lenient toward the no-retrieval control "
    'and one wrongly marked an RRR claim contradicted, so the "retrieval clearly helps" '
    "finding is likely conservative, not overstated. Limits: n=30; not randomly drawn "
    "(weighted toward Control and Structured claims, plus one deliberately targeted suspect "
    "case); a single labeller who also wrote the gold facts; and coverage and FCS are "
    "validated far less than groundedness (one coverage case checked, and it went against "
    "the judge). See `claims_to_label.csv`."
)


def question_numbering(per_question: dict[str, Any]) -> list[str]:
    """Display numbers (Q1..Q10, as in the web app) against the research IDs used in the data.

    Three of the 13 candidate questions were dropped as redundant, which leaves gaps in the
    research IDs (Q01, Q04 and Q07); the app numbers the kept ones 1 to 10 in the same order.
    """
    rows = [
        f"| Q{i} | {qid} | {str(v['type']).replace('_', '-')} |"
        for i, (qid, v) in enumerate(per_question.items(), start=1)
    ]
    return [
        "## Question numbering",
        "",
        "The web app numbers the kept questions Q1-Q10; this report and the data use the original "
        "research IDs (Q01, Q04 and Q07 were dropped as redundant).",
        "",
        "| app | research ID | type |",
        "|---|---|---|",
        *rows,
        "",
    ]


def render_markdown(
    summary: dict[str, Any],
    meta: dict[str, Any],
    sensitivity: dict[str, Any] | None = None,
) -> str:
    S = summary["strategies"]
    names = [n for n in ALL_NAMES if n in S]
    n_ans = meta["n_answerable"]
    lines: list[str] = [
        "# Benchmark results",
        "",
        "## Design and limitations (read first)",
        "",
        f"- **{meta['n_questions']} questions** ({meta['composition']}), "
        f"**{meta['trials']} trials** per question and strategy; only trial 0 is judged "
        "(groundedness, coverage), because the free tier cannot fund the full design. "
        "The reduction from the planned 13 questions x 3 trials is documented in "
        "ARCHITECTURE_PLAN.md section 4.8.",
        f"- **Consistency rests on {meta['trials']} samples per question** (one pair of answers), "
        "not 3 (three pairs), so it is a noisier estimate. Treat consistency differences as "
        "indicative only.",
        f"- Quality metrics average over the **{n_ans} answerable questions**. Confidence "
        "intervals are 95% bootstrap intervals over questions, so they are wide with this few. "
        "Differences whose intervals overlap should not be called real.",
        "- The out-of-corpus question is reported only as abstention behaviour.",
        f"- Generator `{meta['generator']}`, judge `{meta['judge']}` (an LLM judge; "
        "validated against a human on a small sample, see the next bullet).",
        JUDGE_VALIDATION,
        "- Groundedness is measured against the retrieved top-4 passages for every strategy, "
        "including the no-retrieval control, so a correct answer from memory still counts as "
        "unsupported. FCS counts a refusal on an answerable question as 0.",
        "",
        *question_numbering(summary["per_question"]),
        "## Quality (answerable questions, mean [95% CI])",
        "",
        "| strategy | groundedness G | coverage C | faithful-coverage FCS |",
        "|---|---|---|---|",
    ]
    for n in names:
        r = S[n]
        lines.append(f"| {label(n)} | {fmt(r['G'])} | {fmt(r['C'])} | {fmt(r['FCS'])} |")

    lines += [
        "",
        "### FCS by question type",
        "",
        "| strategy | " + " | ".join(summary["by_type"]) + " |",
    ]
    lines.append("|---|" + "---|" * len(summary["by_type"]))
    for n in names:
        cells = [fmt(summary["by_type"][t][n]["FCS"]) for t in summary["by_type"]]
        lines.append(f"| {label(n)} | " + " | ".join(cells) + " |")

    lines += ["", "## Paired differences in FCS (strategy minus reference)", ""]
    for ref_key, ref_label in (("vs_baseline", "Baseline"), ("vs_control", "no-retrieval control")):
        lines += [
            f"**vs {ref_label}** (per-question differences; Wilcoxon p is exploratory with this n)",
            "",
            "| strategy | dG | dC | dFCS | Wilcoxon p (FCS) |",
            "|---|---|---|---|---|",
        ]
        for n, row in summary["paired"][ref_key].items():
            p = row["FCS"]["wilcoxon_p"] if row.get("FCS") else None
            diffs = " | ".join(_diff(row, m) for m in ("G", "C", "FCS"))
            lines.append(f"| {label(n)} | {diffs} | {_num(p, 3)} |")
        lines.append("")

    A = summary["abstention"]
    lines += [
        "## Abstention",
        "",
        "Out-of-corpus question (should abstain) and false refusals on answerable questions "
        "(should not). An answer counts as an abstention when it contains the refusal sentence.",
        "",
        "| strategy | abstained on out-of-corpus | n | refused an answerable question | n |",
        "|---|---|---|---|---|",
    ]
    for n in names:
        a = A[n]
        lines.append(
            f"| {label(n)} | {_num(a['out_of_corpus_abstain_rate'])} | {a['out_of_corpus_n']} | "
            f"{_num(a['false_abstain_rate_answerable'])} | {a['answerable_n']} |"
        )

    lines += [
        "",
        "## Length, diversity and consistency (answerable questions)",
        "",
        "| strategy | words | sentences | MATTR | semantic consistency | structural consistency |",
        "|---|---|---|---|---|---|",
    ]
    for n in names:
        r = S[n]
        lines.append(
            f"| {label(n)} | {fmt(r['words'], 0)} | {fmt(r['sentences'], 1)} | {fmt(r['mattr'])} | "
            f"{fmt(r['consistency_semantic'])} | {fmt(r['consistency_structural'])} |"
        )
    lines += [
        "",
        "MATTR is undefined (excluded) for answers shorter than its 25-word window, which removes "
        "the original report's artefact of very short answers scoring a perfect diversity of 1.0.",
        "",
        "## Efficiency (per answer, generation only, list prices)",
        "",
        "| strategy | model calls | latency (s) | tokens | list-price cost (USD) |",
        "|---|---|---|---|---|",
    ]
    for n in names:
        r = S[n]
        cost = r.get("cost_usd")
        cost_s = "n/a" if not cost else f"{cost['mean']:.5f}"
        lines.append(
            f"| {label(n)} | {fmt(r['calls'], 1)} | {fmt(r['latency_s'], 1)} | "
            f"{fmt(r['tokens'], 0)} | {cost_s} |"
        )
    lines += [
        "",
        "Free-tier usage costs $0; the figures are list prices from `configs/pricing.yaml`. "
        "Latency is the sum over a strategy's sequential calls.",
    ]

    R = summary["retrieval"]
    lines += [
        "",
        "## Retrieval diagnostics",
        "",
        "- Mean cosine similarity by rank: "
        + ", ".join(f"rank {k}: {v:.2f}" for k, v in R["mean_similarity_by_rank"].items()),
        f"- Distractor-paper chunks in any top-4 list: **{R['distractor_chunks_in_top4']}** "
        f"(over {R['questions']} questions).",
        "- **Context recall** is the share of a question's gold facts that the retrieved top-4 "
        "actually contains, i.e. a ceiling on achievable coverage. Two measures: *judge-based* "
        "(the judge decides whether the passages contain each fact, paraphrase allowed; "
        f"mean **{_num(R.get('mean_judge_context_recall'))}**) and *quote match* (the gold quote's "
        f"words appear in one chunk; mean {_num(R['mean_context_recall'])}, a strict lower bound).",
        "- Low coverage on a question with low context recall is a **retrieval** failure, not a "
        "generation failure. A refusal there is the correct behaviour for the context given.",
        "",
        "| question | context recall (judge) | context recall (quote match) |",
        "|---|---|---|",
    ]
    judge_r = R.get("judge_context_recall_by_question", {})
    for qid, v in R["context_recall_by_question"].items():
        lines.append(f"| {qid} | {_num(judge_r.get(qid))} | {_num(v)} |")

    if sensitivity is not None:
        lines += [""] + render_sensitivity(
            summary,
            sensitivity["summary"],
            sensitivity["excluded"],
            meta["n_answerable"],
            sensitivity.get("note"),
        )

    T = summary["totals"]
    lines += [
        "",
        "## Run totals",
        "",
        f"- {T['answers']} answers ({T['judged_answers']} judged), {T['model_calls']} model calls "
        f"({T['cached_calls']} served from cache), {T['rate_limit_retries']} rate-limit retries.",
        f"- Tokens: generator {T['generator_tokens']:,}, judge {T['judge_tokens']:,}.",
    ]
    if meta.get("log"):
        lg = meta["log"]
        lines.append(
            f"- {lg['invocations']} run invocation(s), {lg['active_minutes']:.1f} active minutes, "
            f"{lg['quota_stops']} quota stop(s), {lg['judge_errors']} judge error(s)."
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    summary: dict[str, Any],
    meta: dict[str, Any],
    out_dir: Path,
    sensitivity: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"meta": meta, **summary}
    if sensitivity is not None:
        payload["sensitivity"] = {
            "excluded": sensitivity["excluded"],
            **{k: v for k, v in sensitivity["summary"].items() if k != "per_question"},
        }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(
        render_markdown(summary, meta, sensitivity), encoding="utf-8"
    )


# ---- figures ------------------------------------------------------------------------------------


def _errbars(entries: list[dict[str, Any] | None]) -> tuple[list[float], list[list[float]]]:
    means = [e["mean"] if e else 0.0 for e in entries]
    lo = [(e["mean"] - e["lo"]) if e else 0.0 for e in entries]
    hi = [(e["hi"] - e["mean"]) if e else 0.0 for e in entries]
    return means, [lo, hi]


def make_figures(summary: dict[str, Any], meta: dict[str, Any], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    S = summary["strategies"]
    names = [n for n in ALL_NAMES if n in S]
    labels = [label(n).replace(" (", "\n(") for n in names]
    colors = [COLORS[n] for n in names]
    note = (
        f"n = {meta['n_answerable']} answerable questions, {meta['trials']} trials "
        f"(judged: trial 0). Bars: mean; whiskers: 95% bootstrap CI over questions."
    )
    paths: list[Path] = []

    def save(fig: Figure, name: str) -> None:
        fig.text(0.01, 0.005, note, fontsize=7, color="#555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        p = fig_dir / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # 1. quality
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (m, title) in zip(
        axes,
        [("G", "Groundedness"), ("C", "Coverage"), ("FCS", "Faithful coverage (FCS)")],
        strict=True,
    ):
        means, err = _errbars([S[n][m] for n in names])
        ax.bar(range(len(names)), means, yerr=err, color=colors, capsize=4)
        ax.set_xticks(range(len(names)), labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("score (0-1)")
    save(fig, "fig1_quality.png")

    # 2. lookup vs multi-hop
    types = list(summary["by_type"])
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        ents = [summary["by_type"][t][n]["FCS"] for t in types]
        means, err = _errbars(ents)
        xs = [j + i * width for j in range(len(types))]
        ax.bar(xs, means, width, yerr=err, color=COLORS[n], capsize=3, label=label(n))
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(types))], types)
    ax.set_ylabel("FCS")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("FCS by question type")
    save(fig, "fig2_by_question_type.png")

    # 3. retrieval impact (paired vs control)
    pc = summary["paired"]["vs_control"]
    keep = [n for n in names if n in pc]
    fig, ax = plt.subplots(figsize=(8, 4))
    ents = [pc[n]["FCS"]["diff"] if pc[n].get("FCS") else None for n in keep]
    means, err = _errbars(ents)
    ax.bar(range(len(keep)), means, yerr=err, color=[COLORS[n] for n in keep], capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(keep)), [label(n).replace(" (", "\n(") for n in keep], fontsize=8)
    ax.set_ylabel("FCS(strategy) - FCS(no retrieval)")
    ax.set_title("Impact of retrieval: paired difference vs the no-retrieval control")
    save(fig, "fig3_retrieval_impact.png")

    # 4. cost / latency
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lat = [S[n]["latency_s"] for n in names]
    means, err = _errbars(lat)
    axes[0].bar(range(len(names)), means, yerr=err, color=colors, capsize=4)
    axes[0].set_xticks(range(len(names)), labels, fontsize=8)
    axes[0].set_ylabel("generation latency per answer (s)")
    axes[0].set_title("Latency")
    for n in names:
        c, f = S[n].get("cost_usd"), S[n]["FCS"]
        if c and f:
            axes[1].scatter(c["mean"], f["mean"], s=90, color=COLORS[n])
            axes[1].annotate(
                label(n),
                (c["mean"], f["mean"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    axes[1].set_xlabel("list-price cost per answer (USD)")
    axes[1].set_ylabel("FCS")
    axes[1].set_title("Cost vs quality")
    save(fig, "fig4_cost_latency.png")

    # 5. consistency / diversity
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (m, title) in zip(
        axes,
        [
            ("consistency_semantic", "Semantic consistency (2 samples/question)"),
            ("mattr", "Lexical diversity (MATTR)"),
            ("words", "Answer length (words)"),
        ],
        strict=True,
    ):
        means, err = _errbars([S[n][m] for n in names])
        ax.bar(range(len(names)), means, yerr=err, color=colors, capsize=4)
        ax.set_xticks(range(len(names)), labels, fontsize=8)
        ax.set_title(title, fontsize=10)
    save(fig, "fig5_consistency_diversity.png")

    # 6. heatmap of FCS per question
    qids = [q for q, v in summary["per_question"].items() if v["type"] != "out_of_corpus"]
    grid = [[summary["per_question"][q]["strategies"][n]["FCS"] for q in qids] for n in names]
    fig, ax = plt.subplots(figsize=(1 + 0.9 * len(qids), 3.2))
    data = [[v if v is not None else float("nan") for v in row] for row in grid]
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(qids)), qids)
    ax.set_yticks(range(len(names)), [label(n) for n in names], fontsize=8)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            if v == v:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.6 else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, label="FCS")
    ax.set_title("FCS by strategy and question (judged trial)")
    save(fig, "fig6_fcs_heatmap.png")

    # 7. retrieval diagnostics
    R = summary["retrieval"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ranks = list(R["mean_similarity_by_rank"])
    axes[0].plot(ranks, [R["mean_similarity_by_rank"][r] for r in ranks], marker="o")
    axes[0].set_xlabel("retrieval rank")
    axes[0].set_ylabel("mean cosine similarity")
    axes[0].set_title("Similarity by rank (all questions)")
    judge_r = R.get("judge_context_recall_by_question") or {}
    rec = judge_r or {q: v for q, v in R["context_recall_by_question"].items() if v is not None}
    axes[1].bar(list(rec), list(rec.values()), color="#4C78A8")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("context recall")
    axes[1].set_title("Gold facts contained in retrieved context")
    save(fig, "fig7_retrieval.png")
    return paths
