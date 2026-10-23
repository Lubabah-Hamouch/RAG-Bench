import clsx from "clsx";
import { motion } from "framer-motion";
import { color, STRATEGY_ORDER, strategyLabel, strategyShort } from "@/lib/strategies";
import { ci, excludesZero, num, signed, usd } from "@/lib/format";
import type { Est, PerQuestionRow, StrategyRow } from "@/lib/types";
import { qLabel, qOriginal } from "@/lib/qlabel";
import { Tip } from "../ui";
import { Legend } from "./QuestionCharts";

type Metric = "G" | "C" | "FCS";
const GROUPS: { key: Metric; label: string }[] = [
  { key: "G", label: "Grounded" },
  { key: "C", label: "Coverage" },
  { key: "FCS", label: "Faithful coverage" },
];

/** Bars with 95% bootstrap intervals over questions. Overlapping whiskers = no claimed difference. */
export function CiBars({ rows }: { rows: Record<string, StrategyRow> }) {
  const names = STRATEGY_ORDER.filter((s) => rows[s]);
  const W = 760;
  const H = 300;
  const m = { l: 40, r: 10, t: 16, b: 34 };
  const innerW = W - m.l - m.r;
  const innerH = H - m.t - m.b;
  const groupW = innerW / GROUPS.length;
  const bw = Math.min(34, (groupW * 0.8) / names.length);
  const y = (v: number) => m.t + innerH * (1 - Math.max(0, Math.min(1, v)));
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Bars with confidence intervals for grounded, coverage and faithful coverage by strategy">
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
                const e = rows[s][g.key];
                const x = gx - (names.length * bw + (names.length - 1) * 6) / 2 + si * (bw + 6);
                if (!e) return null;
                const cx = x + bw / 2;
                return (
                  <g key={s}>
                    <motion.rect
                      x={x}
                      width={bw}
                      fill={color(s)}
                      initial={{ y: y(0), height: 0 }}
                      animate={{ y: y(e.mean), height: innerH * Math.max(e.mean, 0.008) }}
                      transition={{ duration: 0.8, delay: si * 0.05, ease: "easeOut" }}
                    >
                      <title>{`${strategyLabel(s)}, ${g.label}: ${ci(e)}`}</title>
                    </motion.rect>
                    <motion.g
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.7 + si * 0.05 }}
                      stroke="rgb(var(--ink))"
                      strokeWidth="1.4"
                    >
                      <line x1={cx} x2={cx} y1={y(e.lo)} y2={y(e.hi)} />
                      <line x1={cx - 4} x2={cx + 4} y1={y(e.lo)} y2={y(e.lo)} />
                      <line x1={cx - 4} x2={cx + 4} y1={y(e.hi)} y2={y(e.hi)} />
                    </motion.g>
                  </g>
                );
              })}
              <text x={gx} y={H - 10} textAnchor="middle" fontSize="13.5" className="fill-dim">
                {g.label}
              </text>
            </g>
          );
        })}
      </svg>
      <Legend names={names} />
    </div>
  );
}

/** Forest plot: paired difference (strategy minus reference) with its 95% interval. */
export function Forest({
  rows,
  referenceLabel,
  domain = [-0.3, 0.8],
}: {
  rows: { name: string; est: Est | null; p?: number | null }[];
  referenceLabel: string;
  domain?: [number, number];
}) {
  const W = 660;
  const rowH = 46;
  const m = { l: 104, r: 176, t: 12, b: 38 };
  const H = m.t + m.b + rows.length * rowH;
  const x = (v: number) => m.l + ((v - domain[0]) / (domain[1] - domain[0])) * (W - m.l - m.r);
  const ticks = [-0.2, 0, 0.2, 0.4, 0.6, 0.8].filter((t) => t >= domain[0] && t <= domain[1]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Paired differences versus ${referenceLabel}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} stroke={t === 0 ? "rgb(var(--ink))" : "rgb(var(--rule))"} strokeWidth={t === 0 ? 1.2 : 1} />
          <text x={x(t)} y={H - m.b + 16} textAnchor="middle" className="num fill-faint" fontSize="12">
            {t > 0 ? "+" : ""}
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={m.t} y2={H - m.b} stroke="rgb(var(--ink))" />
      {rows.map((r, i) => {
        const cy = m.t + i * rowH + rowH / 2;
        const e = r.est;
        const sig = excludesZero(e);
        return (
          <g key={r.name}>
            <text x={m.l - 12} y={cy + 4} textAnchor="end" fontSize="15" className="fill-dim">
              {strategyShort(r.name)}
            </text>
            {e && (
              <>
                <motion.line
                  y1={cy}
                  y2={cy}
                  stroke={color(r.name)}
                  strokeWidth="2"
                  initial={{ x1: x(e.mean), x2: x(e.mean), opacity: 0 }}
                  animate={{ x1: x(e.lo), x2: x(e.hi), opacity: 1 }}
                  transition={{ duration: 0.7, delay: i * 0.07, ease: "easeOut" }}
                />
                <motion.rect
                  y={cy - 6}
                  width={12}
                  height={12}
                  fill={sig ? color(r.name) : "rgb(var(--bg))"}
                  stroke={color(r.name)}
                  strokeWidth="2"
                  initial={{ x: x(0) - 6 }}
                  animate={{ x: x(e.mean) - 6 }}
                  transition={{ duration: 0.7, delay: i * 0.07, ease: "easeOut" }}
                >
                  <title>{`${strategyLabel(r.name)} minus ${referenceLabel}: ${ci(e)}${r.p != null ? ` (Wilcoxon p = ${num(r.p, 3)})` : ""}`}</title>
                </motion.rect>
                <text x={W - m.r + 12} y={cy + 4} className="num fill-ink" fontSize="13.5">
                  {signed(e.mean)}{" "}
                  <tspan className="fill-faint" fontSize="12">
                    [{num(e.lo)}, {num(e.hi)}]
                  </tspan>
                </text>
              </>
            )}
          </g>
        );
      })}
      <text x={x(0) + 6} y={H - 4} className="fill-faint" fontSize="13" fontStyle="italic">
        better than {referenceLabel} →
      </text>
    </svg>
  );
}

/** FCS for every strategy on every answerable question. Click a cell to open that question. */
export function Heatmap({
  perQuestion,
  onPick,
}: {
  perQuestion: Record<string, PerQuestionRow>;
  onPick: (qid: string) => void;
}) {
  const qids = Object.entries(perQuestion)
    .filter(([, v]) => v.type !== "out_of_corpus")
    .map(([id]) => id);
  const names = STRATEGY_ORDER;
  return (
    <div className="overflow-x-auto">
      <div
        className="grid min-w-[620px] gap-1"
        style={{ gridTemplateColumns: `112px repeat(${qids.length}, minmax(48px, 1fr))` }}
      >
        <div />
        {qids.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="focus-ring num py-1 text-center text-[13px] text-dim hover:text-ink"
            title={`${qOriginal(q) ?? ""}: ${perQuestion[q].question}`}
          >
            {qLabel(q)}
          </button>
        ))}
        {names.map((s) => (
          <div key={s} className="contents">
            <div className="flex items-center gap-2 pr-2 text-[14px] text-dim">
              <span className="h-2 w-2" style={{ background: color(s) }} />
              {strategyShort(s)}
            </div>
            {qids.map((q) => {
              const cell = perQuestion[q].strategies[s];
              const v = cell?.FCS ?? null;
              const refused = (cell?.abstain_rate ?? 0) > 0;
              return (
                <Tip
                  key={q}
                  text={
                    <span>
                      <b>{qLabel(q)}</b>, {strategyShort(s)}
                      <br />
                      FCS {num(v)}, G {num(cell?.G)}, C {num(cell?.C)}
                      {refused && (
                        <>
                          <br />
                          Declined to answer
                        </>
                      )}
                    </span>
                  }
                >
                  <button
                    type="button"
                    onClick={() => onPick(q)}
                    className={clsx(
                      "focus-ring num relative grid h-11 w-full place-items-center overflow-hidden text-[13px] outline-offset-0 hover:outline hover:outline-1 hover:outline-ink",
                      v === null ? "border border-rule text-faint" : "text-ink",
                    )}
                    style={v === null ? undefined : { background: `rgb(var(--accent) / ${0.08 + 0.62 * v})` }}
                    aria-label={`${qLabel(q)} ${strategyShort(s)} FCS ${num(v)}`}
                  >
                    {v === null ? "–" : v.toFixed(2)}
                    {refused && (
                      <span
                        className="absolute inset-0 opacity-40"
                        style={{
                          backgroundImage:
                            "repeating-linear-gradient(135deg, transparent 0 5px, rgb(var(--warn) / 0.55) 5px 6px)",
                        }}
                      />
                    )}
                  </button>
                </Tip>
              );
            })}
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-[14px] italic text-dim">
        <span className="flex items-center gap-2">
          0
          <span className="h-2 w-24" style={{ background: "linear-gradient(90deg, rgb(var(--accent)/0.08), rgb(var(--accent)/0.7))" }} />1
        </span>
        <span className="flex items-center gap-2">
          <span
            className="h-3 w-5 border hairline"
            style={{ backgroundImage: "repeating-linear-gradient(135deg, transparent 0 3px, rgb(var(--warn) / 0.7) 3px 4px)" }}
          />
          declined to answer (the context lacked the evidence)
        </span>
      </div>
    </div>
  );
}

/** Quality against cost: how much each strategy pays for its score. */
export function CostScatter({ rows }: { rows: Record<string, StrategyRow> }) {
  const names = STRATEGY_ORDER.filter((s) => rows[s]?.cost_usd && rows[s]?.FCS);
  const W = 720;
  const H = 300;
  const m = { l: 60, r: 24, t: 16, b: 44 };
  const maxCost = Math.max(...names.map((s) => rows[s].cost_usd!.mean)) * 1.15;
  const x = (v: number) => m.l + (v / maxCost) * (W - m.l - m.r);
  const y = (v: number) => m.t + (H - m.t - m.b) * (1 - v);
  const xt = [0, 0.25, 0.5, 0.75, 1].map((f) => f * maxCost);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Scatter of list-price cost against faithful coverage">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke={t === 0 ? "rgb(var(--ink))" : "rgb(var(--rule))"} strokeWidth={t === 0 ? 1.2 : 1} />
          <text x={m.l - 8} y={y(t) + 3.5} textAnchor="end" className="num fill-faint" fontSize="11">
            {t.toFixed(t === 0 || t === 1 ? 0 : 2)}
          </text>
        </g>
      ))}
      {xt.map((t) => (
        <text key={t} x={x(t)} y={H - m.b + 16} textAnchor="middle" className="num fill-faint" fontSize="11">
          {usd(t)}
        </text>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle" className="fill-dim" fontSize="13">
        list-price cost per answer (generation only)
      </text>
      <text transform={`translate(10 ${(m.t + H - m.b) / 2}) rotate(-90)`} textAnchor="middle" className="fill-dim" fontSize="13">
        FCS (95% CI)
      </text>
      {names.map((s, i) => {
        const cx = x(rows[s].cost_usd!.mean);
        const f = rows[s].FCS!;
        const r = 7 + Math.min(10, (rows[s].latency_s?.mean ?? 0) * 3);
        return (
          <motion.g key={s} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.08 }}>
            <line x1={cx} x2={cx} y1={y(f.lo)} y2={y(f.hi)} stroke={color(s)} strokeWidth="2" opacity="0.6" />
            <circle cx={cx} cy={y(f.mean)} r={r} fill={color(s, 0.85)} stroke={color(s)} strokeWidth="1.5">
              <title>{`${strategyLabel(s)}: FCS ${ci(f)} at ${usd(rows[s].cost_usd!.mean)} per answer`}</title>
            </circle>
            <text x={cx + r + 6} y={y(f.mean) + 4} fontSize="14" className="fill-ink">
              {strategyShort(s)}
            </text>
          </motion.g>
        );
      })}
    </svg>
  );
}

/** Share of each question's gold facts that the retrieved context actually held. */
export function RecallBars({
  recall,
  types,
  onPick,
}: {
  recall: Record<string, number>;
  types: Record<string, string>;
  onPick: (qid: string) => void;
}) {
  const entries = Object.entries(recall);
  return (
    <ul className="space-y-2">
      {entries.map(([q, v], i) => (
        <li key={q}>
          <button type="button" onClick={() => onPick(q)} className="focus-ring group flex w-full items-center gap-3 text-left">
            <span className="num w-9 shrink-0 text-[13px] text-dim group-hover:text-ink" title={qOriginal(q)}>
              {qLabel(q)}
            </span>
            <span className="h-2 flex-1 overflow-hidden bg-ink/[0.08]">
              <motion.span
                className="block h-full"
                style={{ background: v >= 0.99 ? "rgb(var(--good))" : v >= 0.5 ? "rgb(var(--s-structured))" : "rgb(var(--bad))" }}
                initial={{ width: 0 }}
                animate={{ width: `${v * 100}%` }}
                transition={{ duration: 0.7, delay: i * 0.04, ease: "easeOut" }}
              />
            </span>
            <span className="num w-10 shrink-0 text-right text-[13px] text-ink">{Math.round(v * 100)}%</span>
            <span className="w-20 shrink-0 text-[13px] italic text-faint">{(types[q] ?? "").replace("_", "-")}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
