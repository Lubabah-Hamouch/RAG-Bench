import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, Check, Loader2, Play, RotateCcw, Square, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import type { KeyStatus } from "@/hooks/useLive";
import { friendly } from "@/lib/api";
import { liveProgress, type LiveState } from "@/lib/liveState";
import { color, STRATEGY_ORDER, strategyShort } from "@/lib/strategies";
import { Switch } from "./ui";

const MAX_CHARS = 500;

export function LiveBar({
  keyStatus,
  live,
  questionId,
  onRun,
  onCancel,
  onReset,
}: {
  keyStatus: KeyStatus;
  live: LiveState;
  questionId: string;
  onRun: (body: { question?: string; question_id?: string; score: boolean }) => void;
  onCancel: () => void;
  onReset: () => void;
}) {
  const [score, setScore] = useState(true);
  const [custom, setCustom] = useState(false);
  const [text, setText] = useState("");
  const ready = keyStatus.state === "valid";
  const running = live.status === "running";

  const elapsed = useElapsed(live.startedAt, running);
  const progress = liveProgress(live);

  const run = () => {
    if (custom && text.trim()) onRun({ question: text.trim(), score });
    else onRun({ question_id: questionId, score });
  };

  return (
    <section className="surface relative border-accent p-4 sm:p-5" aria-label="Live run controls">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
        <div className="flex items-center gap-2">
          <span className="grid h-7 w-7 place-items-center border border-accent text-accent">
            <Sparkles size={15} />
          </span>
          <div>
            <div className="text-[18px] font-semibold leading-tight">Run it live</div>
            <div className="text-[14px] italic text-faint">{ready ? "Your key is verified" : "Verify your key above to enable"}</div>
          </div>
        </div>

        <label className="flex items-center gap-2 text-[14.5px] text-dim">
          <Switch checked={score} onChange={setScore} label="Score the answers with the judge model" />
          Score with the judge
          <span className="text-faint">(about 15 extra calls)</span>
        </label>

        <button type="button" onClick={() => setCustom((c) => !c)} className={clsx("btn-ghost !py-1.5 text-[14.5px]", custom && "border-accent text-accent")}>
          {custom ? "Use the selected question" : "Ask your own question"}
        </button>

        <div className="ml-auto flex items-center gap-2">
          {live.status !== "idle" && !running && (
            <button type="button" onClick={onReset} className="btn-ghost">
              <RotateCcw size={14} /> Back to benchmark result
            </button>
          )}
          {running ? (
            <button type="button" onClick={onCancel} className="btn-ghost">
              <Square size={13} /> Cancel
            </button>
          ) : (
            <button type="button" onClick={run} disabled={!ready || (custom && !text.trim())} className="btn-primary">
              <Play size={14} /> {custom ? "Run my question" : "Run this question"}
            </button>
          )}
        </div>
      </div>

      <AnimatePresence initial={false}>
        {custom && !running && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="pt-3">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
                rows={2}
                placeholder="Ask something about RAG, Self-RAG, DSPy Assertions, vector databases or LangChain…"
                aria-label="Your question"
                className="focus-ring w-full resize-none border bg-bg px-3 py-2 text-[15.5px] hairline placeholder:text-faint"
              />
              <div className="mt-1 flex justify-between text-[13.5px] italic text-faint">
                <span>Free-text questions are scored for groundedness only (there are no gold facts to measure coverage against).</span>
                <span className="num not-italic">
                  {text.length}/{MAX_CHARS}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {live.status !== "idle" && (
        <div className="mt-4">
          <div className="mb-2 flex items-center justify-between text-[14.5px] text-dim">
            <span className="flex items-center gap-2">
              {running ? <Loader2 size={13} className="animate-spin text-accent" /> : live.status === "done" ? <Check size={14} className="text-good" /> : <AlertTriangle size={14} className="text-bad" />}
              {running ? "Running all five conditions…" : live.status === "done" ? "Finished" : "Stopped"}
            </span>
            <span className="num">{elapsed}s</span>
          </div>
          <div className="h-1.5 overflow-hidden bg-ink/10">
            <motion.div className="h-full bg-accent" animate={{ width: `${progress * 100}%` }} transition={{ duration: 0.5, ease: "easeOut" }} />
          </div>
          <ol className="mt-3 flex flex-wrap gap-2">
            {STRATEGY_ORDER.map((s) => {
              const p = live.phases[s] ?? "queued";
              const done = p === "scored" || p === "unscored" || (p === "generated" && !live.scoring);
              return (
                <li key={s} className={clsx("flex items-center gap-1.5 border px-2 py-0.5 text-[14px]", done ? "border-good text-good" : p === "queued" ? "hairline text-faint" : "border-accent text-accent")}>
                  <span className="h-1.5 w-1.5" style={{ background: color(s) }} />
                  {strategyShort(s)}
                  <span className="text-[12.5px] italic opacity-80">{labelFor(p, live.scoring)}</span>
                </li>
              );
            })}
          </ol>
          {live.status === "error" && live.error && <ErrorNote error={live.error} />}
        </div>
      )}
    </section>
  );
}

function labelFor(p: string, scoring: boolean): string {
  if (p === "queued") return "waiting";
  if (p === "running") return "generating";
  if (p === "generated") return scoring ? "scoring" : "done";
  if (p === "unscored") return "unscored";
  return "done";
}

function ErrorNote({ error }: { error: NonNullable<LiveState["error"]> }) {
  const f = friendly(error.error, error.message);
  return (
    <div role="alert" className="mt-3 flex items-start gap-2 border-l-2 border-bad py-0.5 pl-3 text-[14.5px] text-bad">
      <AlertTriangle size={15} className="mt-0.5 shrink-0" />
      <span>
        <b className="font-semibold">{f.title}.</b> {f.hint}
        {error.retry_after_s ? ` Retry in about ${Math.ceil(error.retry_after_s)}s.` : ""}
      </span>
    </div>
  );
}

function useElapsed(startedAt: number | null, running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [running]);
  return startedAt ? Math.max(0, Math.round((now - startedAt) / 1000)) : 0;
}
