import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { useState } from "react";
import type { Backend } from "@/hooks/useLive";
import { useLiveSession } from "@/hooks/useLive";
import { useReplayResult } from "@/hooks/useReplay";
import type { ReplayBase } from "@/lib/api";
import { docLabel } from "@/lib/docs";
import { DROPPED_QUESTIONS, qLabel, qOriginal } from "@/lib/qlabel";
import { STRATEGY_ORDER, TYPE_LABEL } from "@/lib/strategies";
import type { AnswerOut, ChunkOut, QuestionOut, StrategyName } from "@/lib/types";
import { AnswerCard } from "./AnswerCard";
import { ContextPanel } from "./ContextPanel";
import { CoverageMatrix } from "./CoverageMatrix";
import { HighlightProvider } from "./highlight";
import { LiveBar } from "./LiveBar";
import { QuestionBar, typeTone } from "./QuestionBar";
import { CostLatency, QualityBars } from "./charts/QuestionCharts";
import { Badge, Skeleton, StatRow } from "./ui";

type Session = ReturnType<typeof useLiveSession>;

export function Explorer({
  base,
  qid,
  onSelect,
  liveEnabled,
  session,
  backend,
}: {
  base: ReplayBase;
  qid: string;
  onSelect: (id: string) => void;
  liveEnabled: boolean;
  session: Session;
  backend: Backend;
}) {
  const replay = useReplayResult(base, qid);
  const { live } = session;
  const benchmarkQuestion = base.questions.find((q) => q.id === qid) ?? base.questions[0];

  // A live run replaces the recorded result while it is shown.
  const showingLive = liveEnabled && live.status !== "idle";
  const liveQuestion = live.question;
  const question: QuestionOut = showingLive && liveQuestion ? liveQuestion : benchmarkQuestion;

  let answers: Record<string, AnswerOut> = {};
  let chunks: ChunkOut[] | null = null;
  let recall: number | null | undefined;
  if (showingLive) {
    answers = live.status === "done" && live.result ? live.result.answers : live.answers;
    chunks = live.status === "done" && live.result ? live.result.retrieval.chunks : live.chunks;
  } else if (replay.result) {
    answers = replay.result.answers;
    chunks = replay.result.retrieval.chunks;
    recall = replay.result.retrieval.context_recall_judge;
  }
  const hasAnswers = Object.keys(answers).length > 0;
  const live_running = showingLive && live.status === "running";

  return (
    <HighlightProvider>
      <div className="space-y-10">
        <Hero count={base.questions.length} />

        <QuestionBar
          questions={base.questions}
          selected={showingLive && !live.question?.id.startsWith("Q") ? null : qid}
          onSelect={(id) => {
            if (showingLive) session.reset();
            onSelect(id);
          }}
        />
        <p className="-mt-8 text-[14px] italic text-faint" data-testid="question-note">
          {base.questions.length} of {base.questions.length + DROPPED_QUESTIONS} candidate questions; the other{" "}
          {DROPPED_QUESTIONS} were dropped as redundant. Hover a question for its original research ID.
        </p>

        <motion.div
          key={showingLive ? `live-${question.id}-${question.question.slice(0, 12)}` : question.id}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
        >
          <QuestionHeader
            question={question}
            live={showingLive}
            models={showingLive ? live.result?.models : undefined}
          />
        </motion.div>

        {liveEnabled && (
          <LiveBar
            keyStatus={session.keyStatus}
            live={live}
            questionId={benchmarkQuestion.id}
            onRun={(body) => void session.run(body)}
            onCancel={session.cancel}
            onReset={session.reset}
          />
        )}
        {liveEnabled && backend.state === "down" && live.status === "idle" && (
          <p className="-mt-3 text-[12.5px] text-faint">The backend isn't running, so live runs are unavailable. Replay below still works.</p>
        )}

        {replay.error && !showingLive ? (
          <div role="alert" className="surface p-6 text-[14px] text-bad">
            Couldn't load this question's results. {replay.error.message}
          </div>
        ) : !chunks && !hasAnswers && !live_running ? (
          <LoadingGrid />
        ) : (
          <>
            {chunks && <ContextPanel chunks={chunks} recall={recall} />}

            <AnswerGrid
              question={question}
              answers={answers}
              live={showingLive}
              phases={live.phases}
              running={live_running}
            />

            {hasAnswers && !live_running && (
              <>
                <CoverageMatrix facts={question.gold_facts.map((g) => g.text)} answers={answers} />
                <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
                  <QualityBars answers={answers} />
                  <CostLatency answers={answers} />
                </div>
              </>
            )}
          </>
        )}
      </div>
    </HighlightProvider>
  );
}

function AnswerGrid({
  question,
  answers,
  live,
  phases,
  running,
}: {
  question: QuestionOut;
  answers: Record<string, AnswerOut>;
  live: boolean;
  phases: Record<string, string>;
  running: boolean;
}) {
  return (
    <section aria-label="Answers from the five conditions">
      <div className="mb-3 flex items-end justify-between gap-3">
        <div>
          <div className="eyebrow mb-1">Five conditions</div>
          <h2 className="text-[21px] font-semibold">Same question, five ways to answer</h2>
        </div>
        <p className="hidden max-w-sm text-right text-[13.5px] italic text-faint md:block">
          Hover a claim inside a card to see which retrieved passage it rests on.
        </p>
      </div>
      <div className="grid items-stretch gap-4 [grid-template-columns:repeat(auto-fit,minmax(252px,1fr))]">
        {STRATEGY_ORDER.map((s: StrategyName) => (
          <AnswerCard
            key={s}
            strategy={s}
            answer={answers[s]}
            question={question}
            live={live}
            phase={running ? (phases[s] as never) : undefined}
          />
        ))}
      </div>
    </section>
  );
}

function Hero({ count }: { count: number }) {
  return (
    <div className="max-w-4xl">
      <div className="eyebrow mb-2">A controlled comparison</div>
      <h1 className="text-[32px] font-semibold leading-[1.15] sm:text-[42px]">
        Same retrieval, five ways to answer.
      </h1>
      <p className="mt-4 max-w-3xl text-[17px] leading-relaxed text-dim">
        Each question is answered from the <em>same</em> four retrieved passages by four prompting strategies, plus a control with no retrieval.
        A second model then checks every claim against those passages, so you can see what each strategy gains and what it invents.
      </p>
      <div className="mt-5">
        <StatRow
          items={[
            { value: String(count), label: "questions" },
            { value: "5", label: "conditions" },
            { value: "2", label: "trials per question" },
            { value: "4", label: "shared passages" },
          ]}
        />
      </div>
      <p className="mt-3 text-[14px] italic text-faint">These are recorded results; viewing them makes no API calls.</p>
    </div>
  );
}

function QuestionHeader({
  question,
  live,
  models,
}: {
  question: QuestionOut;
  live: boolean;
  models?: Record<string, string>;
}) {
  const [open, setOpen] = useState(false);
  return (
    <header>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="num border border-ink/40 px-1.5 py-0.5 text-[13px]" title={qOriginal(question.id)}>
          {question.id === "live" ? "Live" : qLabel(question.id)}
        </span>
        <span className={clsx("text-[14px] italic", typeTone(question.type))}>{TYPE_LABEL[question.type]}</span>
        {question.expected_abstain && <Badge tone="warn">The corpus cannot answer this: declining is correct</Badge>}
        {live && <Badge tone="accent">Live result</Badge>}
      </div>
      <h2 className="max-w-4xl text-[26px] font-semibold leading-snug sm:text-[30px]">{question.question}</h2>
      {live && models?.judge && (
        <p className="mt-2 text-[15px] italic text-faint" data-testid="live-models">
          Written by <span className="num not-italic">{models.generator}</span> and scored by{" "}
          <span className="num not-italic">{models.judge}</span>, one trial.
          {/^mock/i.test(models.judge) && (
            <Badge tone="warn" className="ml-2 not-italic">
              Demo judge: these scores are not real
            </Badge>
          )}
        </p>
      )}
      {question.gold_facts.length > 0 && (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="focus-ring inline-flex items-center gap-1.5 text-[15px] text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
          >
            What a complete answer contains ({question.gold_facts.length} gold facts)
            <ChevronDown size={14} className={clsx("transition-transform", open && "rotate-180")} />
          </button>
          <AnimatePresence initial={false}>
            {open && (
              <motion.ol
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="mt-2 max-w-4xl overflow-hidden border-t hairline"
              >
                {question.gold_facts.map((g, i) => (
                  <li key={i} className="flex gap-3 border-b hairline px-1 py-2 text-[14.5px] leading-snug text-ink">
                    <span className="num text-faint">{i + 1}</span>
                    <span>
                      {g.text}{" "}
                      <span className="text-faint">
                        ({docLabel(g.doc_id)}, p.{g.page})
                      </span>
                    </span>
                  </li>
                ))}
              </motion.ol>
            )}
          </AnimatePresence>
        </div>
      )}
    </header>
  );
}

function LoadingGrid() {
  return (
    <div className="space-y-6" aria-busy="true" aria-label="Loading results">
      <Skeleton className="h-48 w-full " />
      <div className="grid gap-4 [grid-template-columns:repeat(auto-fit,minmax(252px,1fr))]">
        {[0, 1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-80 " />
        ))}
      </div>
    </div>
  );
}
