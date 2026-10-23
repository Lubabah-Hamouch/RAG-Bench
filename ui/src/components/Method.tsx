import type { ReactNode } from "react";
import { DOC_LABELS } from "@/lib/docs";
import { color, STRATEGIES, STRATEGY_ORDER } from "@/lib/strategies";
import { Badge, SectionTitle } from "./ui";

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <div className="flex-1 px-4 py-3 first:pl-0 lg:border-l lg:first:border-l-0 hairline">
      <div className="mb-1 flex items-baseline gap-2">
        <span className="num text-[14px] text-faint">{n}</span>
        <span className="text-[18px] font-semibold">{title}</span>
      </div>
      {children}
    </div>
  );
}

const METRICS: { name: string; formula: string; body: string }[] = [
  { name: "Grounded (G)", formula: "G = supported claims / all claims", body: "Each atomic claim is labelled supported, unsupported (not in the passages, even if true) or contradicted. Undefined when an answer makes no claims, such as a refusal." },
  { name: "Coverage (C)", formula: "C = gold facts stated / gold facts", body: "Each question has hand-checked gold facts, each with a verbatim source quote. Guards against short answers being grounded but useless." },
  { name: "Faithful coverage (FCS)", formula: "FCS = 2·G·C / (G + C)", body: "Harmonic mean: an answer must be both faithful and complete. A refusal on an answerable question counts as 0." },
  { name: "Consistency", formula: "mean cosine(answer_i, answer_j)", body: "Embedding similarity between repeated answers to the same question. Only two trials were affordable, so one pair per question." },
  { name: "Diversity (MATTR)", formula: "mean unique/25 over sliding windows", body: "A length-robust vocabulary measure. It fixes the original report's artefact where a six-word answer scored a perfect 1.0." },
  { name: "Cost and latency", formula: "cost = Σ tokens × list price", body: "Measured per call. Free-tier use costs nothing; the dollar figures are list prices." },
];

const LIMITATIONS = [
  "Small sample: 10 questions (9 answerable), 2 trials, and only the first trial judged. The planned 13 × 3 design did not fit the free tier.",
  "The judge is an LLM, checked against a human on 30 claims: 93% agreement (28/30), Cohen's kappa 0.879, 95% CI for accuracy [0.79, 0.98]. All three disagreements found ran against the retrieval-vs-control gap (two lenient toward the no-retrieval control, one harsh on an RRR claim), so the retrieval-helps finding is likely conservative. Limits: n=30, not randomly drawn (weighted toward Control and Structured claims, plus one targeted suspect case), one labeller who also wrote the gold facts, and coverage and FCS are validated far less than groundedness (one coverage case checked, and it went against the judge).",
  "Retrieval limits the cross-document questions (for example Q8): the evidence never reaches the generator. Decomposition, bigger k and reranking were all tested and none cleared a pre-set bar, so this is documented rather than fixed.",
  "Self-RAG here is prompt-emulated. It is not the fine-tuned model from the paper, which emits learned reflection tokens.",
  "Gold facts were written by the project's author from the corpus and each is verified against a verbatim quote.",
  "Groundedness is judged against the retrieved passages for every condition, so a correct answer from memory still counts as unsupported.",
];

export function Method() {
  return (
    <div className="space-y-12">
      <div className="max-w-4xl">
        <div className="eyebrow mb-2">Method</div>
        <h1 className="text-[32px] font-semibold leading-[1.15] sm:text-[42px]">How the comparison works</h1>
        <p className="mt-4 max-w-3xl text-[17px] leading-relaxed text-dim">
          The original course report compared generation strategies on surface statistics such as length and vocabulary. This rebuild
          keeps its question (does the way you prompt the generator matter?) but measures what matters: whether each claim in an answer
          is actually supported by the sources.
        </p>
      </div>

      <section className="section-rule">
        <SectionTitle eyebrow="Pipeline" title="One retrieval, five generators, one independent judge" />
        <div className="flex flex-col border-y hairline lg:flex-row">
          <Step n={1} title="Retrieve once">
            <p className="text-[15px] leading-snug text-dim">
              <span className="num text-[13.5px]">all-MiniLM-L6-v2</span> embeddings in ChromaDB, cosine similarity, top-4 chunks of about 600 characters. Every strategy gets the identical passages.
            </p>
          </Step>
          <Step n={2} title="Generate five ways">
            <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
              {STRATEGY_ORDER.map((s) => (
                <span key={s} className="inline-flex items-center gap-1.5 text-[14.5px]">
                  <span className="h-2 w-2" style={{ background: color(s) }} />
                  {STRATEGIES[s].short}
                </span>
              ))}
            </div>
            <p className="text-[15px] leading-snug text-dim">
              Generator: <span className="num text-[13.5px]">qwen3.8-27b</span>. Only the prompt differs between strategies.
            </p>
          </Step>
          <Step n={3} title="Judge claim by claim">
            <p className="text-[15px] leading-snug text-dim">
              A different model (<span className="num text-[13.5px]">gpt-oss-20b</span>) splits each answer into atomic claims and checks every one against the four passages, then checks coverage of the gold facts.
            </p>
          </Step>
        </div>
      </section>

      <section className="section-rule">
        <SectionTitle eyebrow="Strategies" title="The five conditions" />
        <div className="overflow-x-auto">
          <table className="ledger min-w-[640px]">
            <thead>
              <tr>
                <th>Condition</th>
                <th className="text-right">Calls</th>
                <th>Shape</th>
                <th>How it works</th>
              </tr>
            </thead>
            <tbody>
              {STRATEGY_ORDER.map((s) => (
                <tr key={s}>
                  <td className="whitespace-nowrap font-semibold">
                    <span className="mr-2 inline-block h-2.5 w-2.5" style={{ background: color(s) }} />
                    {STRATEGIES[s].label}
                  </td>
                  <td className="n">{STRATEGIES[s].calls}</td>
                  <td className="italic text-faint">{STRATEGIES[s].tagline}</td>
                  <td className="text-dim">{STRATEGIES[s].how}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-faint">
          Every retrieval strategy is told to reply with an exact refusal sentence when the context cannot answer, and to answer only the supported part when it covers part of the question.
        </p>
      </section>

      <section className="section-rule">
        <SectionTitle eyebrow="Metrics" title="What is measured, and how" />
        <div className="overflow-x-auto">
          <table className="ledger min-w-[680px]">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Formula</th>
                <th>What it tells you</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map((m) => (
                <tr key={m.name}>
                  <td className="whitespace-nowrap font-semibold">{m.name}</td>
                  <td className="num whitespace-nowrap text-[13px]">{m.formula}</td>
                  <td className="text-dim">{m.body}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="grid gap-12 lg:grid-cols-2">
        <div className="section-rule">
          <h2 className="mb-3 text-[21px] font-semibold">Honest limitations</h2>
          <ol className="border-t hairline">
            {LIMITATIONS.map((t, i) => (
              <li key={t} className="flex gap-3 border-b hairline py-2.5 text-[15.5px] leading-relaxed text-dim">
                <span className="num mt-0.5 shrink-0 text-[13px] text-faint">{i + 1}</span>
                <span>{t}</span>
              </li>
            ))}
          </ol>
        </div>
        <div className="section-rule">
          <h2 className="mb-3 text-[21px] font-semibold">The corpus</h2>
          <p className="mb-3 text-[15.5px] leading-relaxed text-dim">
            Seven PDFs. Six are about RAG and vector search; one is a deliberate off-topic distractor that retrieval should never surface (it never did).
          </p>
          <table className="ledger">
            <tbody>
              {Object.entries(DOC_LABELS).map(([id, label]) => (
                <tr key={id}>
                  <td>{label}</td>
                  <td className="text-right">{id.startsWith("zhu") && <Badge tone="warn">distractor</Badge>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
