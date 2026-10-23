import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import {
  Ban,
  Check,
  ChevronDown,
  Coins,
  Hash,
  Layers,
  Minus,
  ShieldAlert,
  Timer,
  X,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { color, STRATEGIES } from "@/lib/strategies";
import { compact, num, seconds, usd } from "@/lib/format";
import type { Phase } from "@/lib/liveState";
import type { AnswerOut, ClaimOut, QuestionOut, StrategyName } from "@/lib/types";
import { useHighlight } from "./highlight";
import { Markdown } from "./Markdown";
import { Badge, Chip, Meter, Skeleton, Tip } from "./ui";

function Disclosure({
  title,
  badge,
  children,
  defaultOpen = false,
}: {
  title: string;
  badge?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-t hairline">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="focus-ring flex w-full items-center justify-between gap-2 px-4 py-2.5 text-left text-[14.5px] text-dim transition-colors hover:text-ink"
      >
        <span className="flex items-center gap-2">
          {title}
          {badge}
        </span>
        <ChevronDown size={14} className={clsx("transition-transform", open && "rotate-180")} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-3.5">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

const CLAIM_STYLE = {
  supported: { icon: Check, cls: "text-bg bg-good", word: "Supported" },
  unsupported: { icon: Minus, cls: "text-bg bg-warn", word: "Not in the context" },
  contradicted: { icon: X, cls: "text-bg bg-bad", word: "Contradicts the context" },
} as const;

function ClaimRow({ claim, strategy }: { claim: ClaimOut; strategy: string }) {
  const hl = useHighlight();
  const st = CLAIM_STYLE[claim.label];
  const Icon = st.icon;
  const passage = claim.evidence_passage ?? null;
  return (
    <li
      className="group px-2 py-1.5 transition-colors hover:bg-ink/[0.05]"
      onMouseEnter={() => hl.set(strategy, passage)}
      onMouseLeave={() => hl.set(null, null)}
    >
      <div className="flex items-start gap-2">
        <span className={clsx("mt-0.5 grid h-4 w-4 shrink-0 place-items-center", st.cls)} title={st.word}>
          <Icon size={11} strokeWidth={3} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-[14px] leading-snug text-ink">{claim.text}</p>
          {claim.reason && <p className="mt-0.5 text-[13px] italic leading-snug text-faint">{claim.reason}</p>}
        </div>
        {passage !== null && (
          <span className="num shrink-0 border border-rule px-1.5 py-0.5 text-[11.5px] text-dim" title="Evidence passage">
            [{passage}]
          </span>
        )}
      </div>
    </li>
  );
}

function ClaimBar({ s, u, c }: { s: number; u: number; c: number }) {
  const total = s + u + c || 1;
  return (
    <span className="flex h-1.5 w-16 overflow-hidden bg-ink/10" aria-hidden="true">
      <span className="bg-good" style={{ width: `${(s / total) * 100}%` }} />
      <span className="bg-warn" style={{ width: `${(u / total) * 100}%` }} />
      <span className="bg-bad" style={{ width: `${(c / total) * 100}%` }} />
    </span>
  );
}

export function AnswerCard({
  strategy,
  answer,
  question,
  phase,
  live = false,
}: {
  strategy: StrategyName;
  answer?: AnswerOut;
  question: QuestionOut;
  phase?: Phase;
  live?: boolean;
}) {
  const meta = STRATEGIES[strategy];
  const c = color(strategy);
  const [expanded, setExpanded] = useState(false);
  const m = answer?.metrics;
  const g = m?.groundedness;
  const claims = g?.claims ?? [];
  const long = (answer?.answer.length ?? 0) > 520;
  const judged = !!m?.judged;
  // A free-text question has no gold facts, so coverage (and FCS, which needs it) cannot exist.
  const freeText = question.type === "free_text";
  const needsGold = "Needs gold facts, which only the 10 benchmark questions have.";

  return (
    <motion.article
      layout="position"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="surface relative flex min-w-0 flex-col overflow-hidden"
      aria-label={`${meta.label} answer`}
    >
      <div className="h-[3px] w-full" style={{ background: c }} />

      <header className="relative px-4 pb-2 pt-3.5">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 shrink-0" style={{ background: c }} />
              <h3 className="truncate text-[18px] font-semibold">{meta.label}</h3>
            </div>
            <p className="mt-0.5 text-[14px] italic text-dim">{meta.tagline}</p>
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1">
            {live && <Badge tone="accent">Live</Badge>}
          </div>
        </div>
      </header>

      {!answer ? (
        <PendingBody phase={phase} />
      ) : (
        <>
          <div className="relative flex items-end gap-4 px-4 pb-3 pt-1">
            <Meter label="Grounded" value={g?.G} color={c} hint="Share of the answer's claims the retrieved passages support" />
            <Meter label="Coverage" value={m?.coverage?.C} color={c} hint={freeText ? needsGold : "Share of the gold facts the answer states"} />
            <Meter label="FCS" value={m?.fcs} color={c} hint={freeText ? needsGold : "Faithful coverage: harmonic mean of grounded and coverage. A refusal on an answerable question scores 0."} />
          </div>
          {!judged && (
            <p className="px-4 pb-2 text-[13.5px] italic text-faint">
              {live ? "Scoring skipped or pending for this answer." : "This trial was not scored; scores come from trial 0."}
            </p>
          )}
          {judged && freeText && (
            <p className="px-4 pb-2 text-[13.5px] italic text-faint" data-testid="free-text-note">
              Coverage and FCS need gold facts, which only the 10 benchmark questions have.
              Grounded is still scored.
            </p>
          )}
          {judged && g?.abstained && m?.groundedness && (
            <p className="px-4 pb-2 text-[13.5px] italic text-faint">No factual claims to check, so groundedness is undefined.</p>
          )}

          {judged && strategy === "no_retrieval" && !answer.abstained && (
            <p className="px-4 pb-2 text-[13.5px] italic text-faint" data-testid="control-note">
              The control never saw the passages but is scored against them. A correct answer from
              memory can still count as supported.
            </p>
          )}

          <StatusRibbon strategy={strategy} answer={answer} question={question} />

          <div className="flex flex-wrap gap-1.5 px-4 pb-3">
            <Chip icon={<Layers size={11} />} label="calls" value={`${m?.calls ?? meta.calls}`} />
            <Chip icon={<Timer size={11} />} label="" value={seconds(m?.latency_s)} title="Generation latency (sum of sequential calls)" />
            <Chip icon={<Hash size={11} />} label="" value={compact((m?.prompt_tokens ?? 0) + (m?.completion_tokens ?? 0))} title="Generation tokens (prompt + completion)" />
            <Chip icon={<Coins size={11} />} label="" value={usd(m?.cost_usd)} title="List-price cost of generation (free tier costs $0)" />
          </div>

          <div className="relative px-4 pb-3">
            <div className={clsx("relative overflow-hidden", !expanded && long && "max-h-[238px]")}>
              <Markdown text={answer.answer} />
              {!expanded && long && (
                <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-[rgb(var(--bg))] to-transparent" />
              )}
            </div>
            {long && (
              <button
                type="button"
                onClick={() => setExpanded((e) => !e)}
                className="focus-ring mt-1 text-[14px] text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
              >
                {expanded ? "Show less" : "Show full answer"}
              </button>
            )}
          </div>

          <div className="mt-auto">
            {claims.length > 0 && (
              <Disclosure
                title="Claims checked"
                badge={
                  <span className="flex items-center gap-2">
                    <Badge>{claims.length}</Badge>
                    <ClaimBar s={g?.n_supported ?? 0} u={g?.n_unsupported ?? 0} c={g?.n_contradicted ?? 0} />
                  </span>
                }
              >
                <ul className="-mx-2 space-y-0.5">
                  {claims.map((cl, i) => (
                    <ClaimRow key={i} claim={cl} strategy={strategy} />
                  ))}
                </ul>
                <p className="mt-2 text-[13px] italic leading-snug text-faint">
                  Judged against the four retrieved passages only. Hover a claim to see its evidence.
                </p>
              </Disclosure>
            )}
            {answer.trace.length > 1 && (
              <Disclosure title="Pipeline stages" badge={<Badge>{answer.trace.length}</Badge>}>
                <ol className="space-y-3">
                  {answer.trace.map((s, i) => (
                    <li key={s.name} className="relative pl-5">
                      <span className="absolute left-0 top-1 num grid h-4 w-4 place-items-center text-[10px] text-bg" style={{ background: c }}>
                        {i + 1}
                      </span>
                      <div className="eyebrow mb-1">{s.name}</div>
                      <div className="border-l-2 border-rule pl-3">
                        <Markdown text={s.text} />
                      </div>
                    </li>
                  ))}
                </ol>
              </Disclosure>
            )}
            {answer.critique && (
              <Disclosure title="Self-critique" badge={<Badge tone="accent">visible reasoning</Badge>}>
                <pre className="whitespace-pre-wrap border-l-2 border-rule pl-3 font-mono text-[12.5px] leading-relaxed text-ink">
                  {answer.critique}
                </pre>
                <p className="mt-2 text-[13px] italic text-faint">Shown for transparency; it is never scored as part of the answer.</p>
              </Disclosure>
            )}
            {answer.other_trials.length > 0 && (
              <Disclosure title="Same question, another trial">
                {answer.other_trials.map((t) => (
                  <div key={t.trial} className="border-l-2 border-rule pl-3">
                    <div className="eyebrow mb-1">trial {t.trial + 1}</div>
                    <Markdown text={t.answer} />
                  </div>
                ))}
                {m?.consistency && (
                  <p className="mt-2 text-[13px] italic leading-snug text-faint">
                    Used for the consistency estimate (semantic {num(m.consistency.semantic)}, structural{" "}
                    {num(m.consistency.structural)}), from {m.consistency.n_samples} samples.
                  </p>
                )}
              </Disclosure>
            )}
          </div>
        </>
      )}
    </motion.article>
  );
}

function StatusRibbon({
  strategy,
  answer,
  question,
}: {
  strategy: StrategyName;
  answer: AnswerOut;
  question: QuestionOut;
}) {
  if (answer.abstained) {
    return question.expected_abstain ? (
      <RibbonBox tone="good" icon={<Check size={13} />} text="Correctly declined: the corpus does not cover this." />
    ) : (
      <RibbonBox
        tone="warn"
        icon={<ShieldAlert size={13} />}
        text="Declined to answer: the retrieved context lacked the information."
        tip="For a question the corpus can answer, this usually means retrieval missed the evidence, so declining was the right call for the context given."
      />
    );
  }
  if (strategy === "no_retrieval" && question.expected_abstain) {
    return <RibbonBox tone="bad" icon={<Ban size={13} />} text="Answered from memory on a question the corpus cannot answer." />;
  }
  return null;
}

function RibbonBox({
  tone,
  icon,
  text,
  tip,
}: {
  tone: "good" | "warn" | "bad";
  icon: ReactNode;
  text: string;
  tip?: string;
}) {
  const cls = { good: "border-good text-good", warn: "border-warn text-warn", bad: "border-bad text-bad" }[tone];
  const body = (
    <div className={clsx("mx-4 mb-3 flex items-start gap-2 border-l-2 py-0.5 pl-2.5 text-[14px] leading-snug", cls)}>
      <span className="mt-0.5 shrink-0">{icon}</span>
      <span>{text}</span>
    </div>
  );
  return tip ? <Tip text={tip}>{body}</Tip> : body;
}

function PendingBody({ phase }: { phase?: Phase }) {
  const text =
    phase === "running" ? "Generating…" : phase === "queued" ? "Waiting its turn" : "Preparing…";
  return (
    <div className="space-y-3 px-4 pb-5 pt-2" aria-busy="true" aria-label={text}>
      <div className="flex gap-4">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-[58px] flex-1" />
        ))}
      </div>
      <div className="flex items-center gap-2 text-[14px] italic text-dim">
        <span className={clsx("h-1.5 w-1.5 bg-accent", phase === "running" && "animate-pulse")} />
        {text}
      </div>
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-[92%]" />
      <Skeleton className="h-3 w-[78%]" />
      <Skeleton className="h-3 w-[60%]" />
    </div>
  );
}
