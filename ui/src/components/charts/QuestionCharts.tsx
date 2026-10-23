import { motion } from "framer-motion";
import { color, STRATEGY_ORDER, strategyShort } from "@/lib/strategies";
import { compact, num, seconds, usd } from "@/lib/format";
import type { AnswerOut } from "@/lib/types";

type Metric = "G" | "C" | "FCS";
const GROUPS: { key: Metric; label: string; hint: string }[] = [
  { key: "G", label: "Grounded", hint: "claims supported by the passages" },
  { key: "C", label: "Coverage", hint: "gold facts stated" },
  { key: "FCS", label: "Faithful coverage", hint: "harmonic mean of the two" },
];

const valueOf = (a: AnswerOut | undefined, k: Metric): number | null => {
  const m = a?.metrics;
  if (!m) return null;
  return (k === "G" ? m.groundedness?.G : k === "C" ? m.coverage?.C : m.fcs) ?? null;
};

/** Grouped bars: every strategy on Grounded, Coverage and FCS for the selected question. */
export function QualityBars({ answers }: { answers: Record<string, AnswerOut> }) {
  const names = STRATEGY_ORDER.filter((s) => answers[s]);
  const W = 720;
  const H = 250;
  const m = { l: 38, r: 10, t: 14, b: 30 };
  const innerW = W - m.l - m.r;
  const innerH = H - m.t - m.b;
  const groupW = innerW / GROUPS.length;
  const bw = Math.min(30, (groupW * 0.8) / Math.max(names.length, 1));
  const y = (v: number) => m.t + innerH * (1 - v);

  return (
    <figure className="section-rule">
      <figcaption className="mb-2">
        <div className="eyebrow mb-1">This question</div>
        <h3 className="text-[21px] font-semibold">Quality at a glance</h3>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Grouped bar chart of grounded, coverage and faithful-coverage scores by strategy">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke={t === 0 ? "rgb(var(--ink))" : "rgb(var(--rule))"} strokeWidth={t === 0 ? 1.2 : 1} />
            <text x={m.l - 8} y={y(t) + 3.5} textAnchor="end" className="num fill-faint" fontSize="11">
              {t.toFixed(t === 0 || t === 1 ? 0 : 2)}
            </text>
          </g>
        ))}
        {GROUPS.map((g, gi) => {
          const gx = m.l + gi * groupW + groupW / 2;
          return (
            <g key={g.key}>
              {names.map((s, si) => {
                const v = valueOf(answers[s], g.key);
                const x = gx - (names.length * bw + (names.length - 1) * 5) / 2 + si * (bw + 5);
                return (
                  <g key={s}>
                    {v === null ? (
                      <text x={x + bw / 2} y={y(0) - 5} textAnchor="middle" fontSize="10" className="num fill-faint">
                        n/a
                      </text>
                    ) : (
                      <>
                        <motion.rect
                          x={x}
                          width={bw}
                          fill={color(s)}
                          initial={{ y: y(0), height: 0 }}
                          animate={{ y: y(Math.max(v, 0.012)), height: innerH * Math.max(v, 0.012) }}
                          transition={{ duration: 0.7, delay: si * 0.05, ease: "easeOut" }}
                        >
                          <title>{`${strategyShort(s)}, ${g.label}: ${num(v)}`}</title>
                        </motion.rect>
                        <text x={x + bw / 2} y={y(v) - 5} textAnchor="middle" fontSize="11" className="num fill-dim">
                          {v.toFixed(2)}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
              <text x={gx} y={H - 10} textAnchor="middle" fontSize="13" className="fill-dim">
                {g.label}
              </text>
            </g>
          );
        })}
      </svg>
      <Legend names={names} />
    </figure>
  );
}

export function Legend({ names }: { names: string[] }) {
  return (
    <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
      {names.map((s) => (
        <li key={s} className="flex items-center gap-1.5 text-[14px] text-dim">
          <span className="h-2 w-2" style={{ background: color(s) }} />
          {strategyShort(s)}
        </li>
      ))}
    </ul>
  );
}

function HBars({
  title,
  rows,
  format,
}: {
  title: string;
  rows: { name: string; value: number }[];
  format: (v: number) => string;
}) {
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  return (
    <div>
      <div className="eyebrow mb-2">{title}</div>
      <ul className="space-y-2">
        {rows.map((r) => (
          <li key={r.name} className="flex items-center gap-3">
            <span className="w-[76px] shrink-0 text-[14px] text-dim">{strategyShort(r.name)}</span>
            <div className="h-2 flex-1 overflow-hidden bg-ink/[0.08]">
              <motion.div
                className="h-full"
                style={{ background: color(r.name) }}
                initial={{ width: 0 }}
                animate={{ width: `${(r.value / max) * 100}%` }}
                transition={{ duration: 0.7, ease: "easeOut" }}
              />
            </div>
            <span className="num w-16 shrink-0 text-right text-[13px] text-ink">{format(r.value)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** What each strategy cost for this question: latency, tokens and list-price dollars. */
export function CostLatency({ answers }: { answers: Record<string, AnswerOut> }) {
  const names = STRATEGY_ORDER.filter((s) => answers[s]);
  const rows = (pick: (a: AnswerOut) => number) =>
    names.map((name) => ({ name, value: pick(answers[name]) }));
  return (
    <figure className="section-rule">
      <figcaption className="mb-4">
        <div className="eyebrow mb-1">This question</div>
        <h3 className="text-[21px] font-semibold">What each answer cost</h3>
        <p className="mt-0.5 text-[15px] text-dim">
          Generation only. RRR and Self-RAG resend the context on each of their three calls.
        </p>
      </figcaption>
      <div className="space-y-5">
        <HBars title="Latency" rows={rows((a) => a.metrics.latency_s)} format={seconds} />
        <HBars
          title="Tokens"
          rows={rows((a) => a.metrics.prompt_tokens + a.metrics.completion_tokens)}
          format={compact}
        />
        <HBars title="List-price cost" rows={rows((a) => a.metrics.cost_usd ?? 0)} format={usd} />
      </div>
    </figure>
  );
}
