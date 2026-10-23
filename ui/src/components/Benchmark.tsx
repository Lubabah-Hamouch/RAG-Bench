import clsx from "clsx";
import { useMemo, useState, type ReactNode } from "react";
import { ci, compact, excludesZero, num, seconds, signed, usd } from "@/lib/format";
import { color, strategyLabel, STRATEGY_ORDER, strategyShort } from "@/lib/strategies";
import type { Est, Summary, SummaryBlock } from "@/lib/types";
import { qLabel } from "@/lib/qlabel";
import { SectionTitle, StatRow, Switch } from "./ui";
import { CiBars, CostScatter, Forest, Heatmap, RecallBars } from "./charts/BenchmarkCharts";

type Metric = "FCS" | "G" | "C";
const RETRIEVAL = STRATEGY_ORDER.filter((s) => s !== "no_retrieval");

function overlap(a: Est, b: Est): boolean {
  return a.lo <= b.hi && b.lo <= a.hi;
}

function Panel({ eyebrow, title, note, children }: { eyebrow: string; title: string; note?: ReactNode; children: ReactNode }) {
  return (
    <section className="section-rule">
      <SectionTitle eyebrow={eyebrow} title={title} />
      {note && <p className="-mt-2 mb-4 max-w-3xl text-[15px] leading-relaxed text-dim">{note}</p>}
      {children}
    </section>
  );
}

export function Benchmark({ summary, onPick }: { summary: Summary; onPick: (qid: string) => void }) {
  const [excludeSens, setExcludeSens] = useState(false);
  const [metric, setMetric] = useState<Metric>("FCS");
  const sens = summary.sensitivity;
  const block: SummaryBlock = excludeSens && sens ? sens : summary;
  const nQ = excludeSens && sens ? summary.meta.n_answerable - sens.excluded.length : summary.meta.n_answerable;

  const findings = useMemo(() => {
    const vc = block.paired.vs_control;
    const gains = RETRIEVAL.map((s) => ({ s, e: vc[s]?.FCS?.diff ?? null })).filter((x): x is { s: (typeof RETRIEVAL)[number]; e: Est } => !!x.e);
    const best = gains.sort((a, b) => b.e.mean - a.e.mean)[0];
    const distinct = gains.filter((g) => excludesZero(g.e)).length;
    const fcs = RETRIEVAL.map((s) => block.strategies[s]?.FCS).filter((e): e is Est => !!e);
    const allOverlap = fcs.every((a) => fcs.every((b) => overlap(a, b)));
    const topFcs = [...RETRIEVAL].sort((a, b) => (block.strategies[b].FCS?.mean ?? 0) - (block.strategies[a].FCS?.mean ?? 0))[0];
    const rrrBase = block.paired.vs_baseline.rrr?.FCS;
    const recall = block.retrieval.mean_judge_context_recall ?? null;
    const tokBase = block.strategies.baseline?.tokens?.mean ?? 0;
    const tokSelf = block.strategies.self_rag?.tokens?.mean ?? 0;
    return { best, distinct, allOverlap, topFcs, rrrBase, recall, ratio: tokBase ? tokSelf / tokBase : null };
  }, [block]);

  const meta = summary.meta;
  const types = Object.fromEntries(Object.entries(summary.per_question).map(([q, v]) => [q, v.type]));
  const recallMap = block.retrieval.judge_context_recall_by_question ?? {};

  return (
    <div className="space-y-10">
      <div className="max-w-4xl">
        <div className="eyebrow mb-2">Aggregate results</div>
        <h1 className="text-[32px] font-semibold leading-[1.15] sm:text-[42px]">What {summary.totals.answers} answers say</h1>
        <p className="mt-4 max-w-3xl text-[17px] leading-relaxed text-dim">
          <span className="num text-[15px]">{meta.generator}</span> wrote the answers and <span className="num text-[15px]">{meta.judge}</span> checked every claim.
          Only trial {meta.judge_trials} of {meta.trials} was judged.
        </p>
        <div className="mt-5">
          <StatRow
            items={[
              { value: String(meta.n_questions), label: "questions" },
              { value: String(meta.trials), label: "trials each" },
              { value: "5", label: "conditions" },
              { value: String(summary.totals.answers), label: "answers" },
            ]}
          />
        </div>
      </div>

      <div role="note" className="max-w-4xl border-l-2 border-ink pl-4 text-[15.5px] leading-relaxed">
        <p>
          <b className="font-semibold">Read the intervals, not just the bars.</b> This is a small sample: {meta.n_answerable} answerable questions,{" "}
          {meta.trials} trials, one judged trial. Bars show the mean; whiskers are 95% bootstrap intervals over questions. Where whiskers overlap, no
          difference is claimed. Consistency rests on only {meta.trials} samples per question, and the judge was checked against a human on only 30 claims (see Method).
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border hairline px-4 py-2.5">
        <div className="text-[15px] text-dim">
          Showing <b className="num text-ink">{nQ}</b> answerable questions
          {excludeSens && sens ? ` (excluding ${sens.excluded.map(qLabel).join(", ")})` : ""}
        </div>
        {sens && (
          <label className="flex cursor-pointer items-center gap-2.5 text-[15px] text-dim">
            Exclude {sens.excluded.map(qLabel).join(", ")} <span className="font-normal text-faint">(its context held 1 of 3 gold facts)</span>
            <Switch checked={excludeSens} onChange={setExcludeSens} label={`Exclude ${sens.excluded.map(qLabel).join(", ")} (sensitivity analysis)`} />
          </label>
        )}
      </div>

      <section className="section-rule" aria-label="Main findings">
        <SectionTitle eyebrow="Summary" title="Four findings" />
        <div className="overflow-x-auto">
          <table className="ledger min-w-[640px]">
            <thead>
              <tr>
                <th>Finding</th>
                <th className="text-right">Figure</th>
                <th>What the data say</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="font-semibold">Retrieval clearly helps</td>
                <td className="n text-[18px] text-accent">{findings.best ? signed(findings.best.e.mean) : "–"}</td>
                <td className="text-dim">{`${findings.best ? strategyLabel(findings.best.s) : "Best strategy"} gains ${ci(findings.best?.e)} FCS over no retrieval. ${findings.distinct} of 4 strategies are distinguishable from the control.`}</td>
              </tr>
              <tr>
                <td className="font-semibold">{findings.allOverlap ? "No strategy wins" : "Highest FCS"}</td>
                <td className="n text-[18px]">{findings.allOverlap ? "n.s." : strategyShort(findings.topFcs)}</td>
                <td className="text-dim">{`All four intervals ${findings.allOverlap ? "overlap" : "don't all overlap"}. RRR vs Baseline: ${ci(findings.rrrBase?.diff)}${findings.rrrBase?.wilcoxon_p != null ? `, Wilcoxon p = ${num(findings.rrrBase.wilcoxon_p, 3)}` : ""}. Suggestive, not established.`}</td>
              </tr>
              <tr>
                <td className="font-semibold">Retrieval is the bottleneck</td>
                <td className="n text-[18px]">{findings.recall !== null ? `${Math.round(findings.recall * 100)}%` : "–"}</td>
                <td className="text-dim">Share of gold facts the retrieved top-4 actually contained. Cross-document questions often miss half their evidence, and no generation strategy can fix that.</td>
              </tr>
              <tr>
                <td className="font-semibold">What elaboration costs</td>
                <td className="n text-[18px]">{findings.ratio ? `${findings.ratio.toFixed(1)}×` : "–"}</td>
                <td className="text-dim">Self-RAG&apos;s token use relative to Baseline. RRR and Self-RAG resend the full context on each of three calls, for little reliable gain here.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <Panel eyebrow="Quality" title="How good is each strategy?" note="Grounded = share of claims the passages support. Coverage = share of gold facts stated. FCS = their harmonic mean, so an answer must be both faithful and complete. A refusal on an answerable question scores 0.">
        <CiBars rows={block.strategies} />
        {block.by_type && (
          <div className="mt-8 overflow-x-auto">
            <table className="ledger min-w-[520px] max-w-3xl">
              <caption>FCS by question type, with 95% interval</caption>
              <thead>
                <tr>
                  <th>Strategy</th>
                  {Object.keys(block.by_type).map((t) => (
                    <th key={t} className="text-right">
                      {t.replace("_", "-")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {STRATEGY_ORDER.map((s) => (
                  <tr key={s}>
                    <td>
                      <span className="mr-2 inline-block h-2 w-2" style={{ background: color(s) }} />
                      {strategyLabel(s)}
                    </td>
                    {Object.keys(block.by_type!).map((t) => (
                      <td key={t} className="n">
                        {ci(block.by_type![t][s]?.FCS)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="grid gap-10 xl:grid-cols-2">
        <Panel eyebrow="Paired differences" title="What retrieval adds" note="Strategy minus the no-retrieval control, question by question. A filled dot means the 95% interval excludes zero.">
          <MetricTabs metric={metric} onChange={setMetric} />
          <Forest
            referenceLabel="control"
            rows={RETRIEVAL.map((s) => ({ name: s, est: block.paired.vs_control[s]?.[metric]?.diff ?? null, p: block.paired.vs_control[s]?.[metric]?.wilcoxon_p }))}
          />
        </Panel>
        <Panel eyebrow="Paired differences" title="Do elaborate strategies beat Baseline?" note="Mostly not distinguishably. The Wilcoxon test is exploratory with this few questions.">
          <MetricTabs metric={metric} onChange={setMetric} />
          <Forest
            referenceLabel="Baseline"
            domain={[-0.4, 0.6]}
            rows={(["structured", "rrr", "self_rag"] as const).map((s) => ({ name: s, est: block.paired.vs_baseline[s]?.[metric]?.diff ?? null, p: block.paired.vs_baseline[s]?.[metric]?.wilcoxon_p }))}
          />
        </Panel>
      </div>

      <Panel eyebrow="Per question" title="Where each strategy wins and fails" note="Faithful coverage for every strategy on every answerable question. Click a cell to open that question.">
        <Heatmap perQuestion={summary.per_question} onPick={onPick} />
      </Panel>

      <div className="grid gap-10 xl:grid-cols-2">
        <Panel eyebrow="Efficiency" title="Quality versus cost" note="Bubble size is generation latency. Quality is only part of the story: the three-call strategies cost several times more for similar FCS.">
          <CostScatter rows={block.strategies} />
        </Panel>
        <Panel eyebrow="Retrieval diagnostics" title="Did the evidence reach the generator?" note="Share of each question's gold facts held by the retrieved top-4, as judged. Low bars are retrieval failures, not generation failures.">
          <RecallBars recall={recallMap} types={types} onPick={onPick} />
          <p className="mt-4 text-[14px] italic text-faint">
            Mean similarity by rank: <span className="num not-italic">{Object.entries(block.retrieval.mean_similarity_by_rank).map(([r, v]) => `#${r} ${num(v)}`).join(", ")}</span>.
            Distractor chunks retrieved: <span className="num not-italic">{block.retrieval.distractor_chunks_in_top4}</span>.
          </p>
        </Panel>
      </div>

      <Panel eyebrow="Style and stability" title="Length, diversity, consistency and abstention" note="Consistency is the mean embedding similarity between a question's repeated answers, from only two samples each, so treat differences as indicative. MATTR is a length-robust vocabulary diversity.">
        <div className="overflow-x-auto">
          <table className="ledger min-w-[760px]">
            <thead>
              <tr>
                {["Strategy", "Words", "MATTR", "Consistency", "Latency", "Tokens", "List cost", "Declined (answerable)", "Declined (out of corpus)"].map((h, i) => (
                  <th key={h} className={i ? "text-right" : undefined}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {STRATEGY_ORDER.map((s) => {
                const r = block.strategies[s];
                const a = block.abstention[s];
                return (
                  <tr key={s}>
                    <td>
                      <span className="mr-2 inline-block h-2 w-2" style={{ background: color(s) }} />
                      {strategyLabel(s)}
                    </td>
                    <td className="n">{num(r.words?.mean, 0)}</td>
                    <td className="n">{num(r.mattr?.mean)}</td>
                    <td className="n">{ci(r.consistency_semantic)}</td>
                    <td className="n">{seconds(r.latency_s?.mean)}</td>
                    <td className="n">{compact(r.tokens?.mean)}</td>
                    <td className="n">{usd(r.cost_usd?.mean)}</td>
                    <td className="n">{a?.false_abstain_rate_answerable != null ? `${Math.round(a.false_abstain_rate_answerable * 100)}%` : "–"}</td>
                    <td className="n">{a?.out_of_corpus_abstain_rate != null ? `${Math.round(a.out_of_corpus_abstain_rate * 100)}%` : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      <p className="border-t hairline pb-4 pt-3 text-[14px] italic text-faint">
        The run made <span className="num not-italic">{summary.totals.model_calls}</span> model calls and used{" "}
        <span className="num not-italic">{compact(summary.totals.generator_tokens)}</span> generator and{" "}
        <span className="num not-italic">{compact(summary.totals.judge_tokens)}</span> judge tokens
        {meta.log ? <>, over <span className="num not-italic">{meta.log.active_minutes.toFixed(0)}</span> minutes</> : null}.
      </p>
    </div>
  );
}

function MetricTabs({ metric, onChange }: { metric: Metric; onChange: (m: Metric) => void }) {
  const items: { m: Metric; label: string }[] = [
    { m: "FCS", label: "FCS" },
    { m: "G", label: "Grounded" },
    { m: "C", label: "Coverage" },
  ];
  return (
    <div role="tablist" aria-label="Metric" className="mb-3 inline-flex border hairline">
      {items.map((i) => (
        <button
          key={i.m}
          role="tab"
          aria-selected={metric === i.m}
          type="button"
          onClick={() => onChange(i.m)}
          className={clsx("focus-ring px-3 py-1 text-[14px] transition-colors", metric === i.m ? "bg-ink text-bg" : "text-dim hover:text-ink")}
        >
          {i.label}
        </button>
      ))}
    </div>
  );
}
