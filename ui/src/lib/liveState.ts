import type {
  AnswerOut,
  ChunkOut,
  LiveError,
  LiveEvent,
  QuestionOut,
  QuestionResultOut,
} from "./types";

export type Phase = "queued" | "running" | "generated" | "scored" | "unscored";

export interface LiveState {
  status: "idle" | "running" | "done" | "error";
  question: QuestionOut | null;
  chunks: ChunkOut[] | null;
  scoring: boolean;
  phases: Record<string, Phase>;
  answers: Record<string, AnswerOut>;
  result: QuestionResultOut | null;
  error: LiveError | null;
  startedAt: number | null;
}

export const initialLive: LiveState = {
  status: "idle",
  question: null,
  chunks: null,
  scoring: true,
  phases: {},
  answers: {},
  result: null,
  error: null,
  startedAt: null,
};

export type LiveAction =
  | { type: "start"; scoring: boolean; at: number }
  | { type: "event"; event: LiveEvent }
  | { type: "fail"; error: LiveError }
  | { type: "reset" };

const ORDER = ["baseline", "structured", "rrr", "self_rag", "no_retrieval"];

export function liveReducer(state: LiveState, action: LiveAction): LiveState {
  switch (action.type) {
    case "reset":
      return initialLive;
    case "start":
      return {
        ...initialLive,
        status: "running",
        scoring: action.scoring,
        startedAt: action.at,
        phases: Object.fromEntries(ORDER.map((s) => [s, "queued" as Phase])),
      };
    case "fail":
      return { ...state, status: "error", error: action.error };
    case "event": {
      const ev = action.event;
      switch (ev.event) {
        case "retrieval":
          return { ...state, question: ev.data.question, chunks: ev.data.chunks };
        case "strategy_started":
          return { ...state, phases: { ...state.phases, [ev.data.strategy]: "running" } };
        case "strategy_done":
          return {
            ...state,
            answers: { ...state.answers, [ev.data.strategy]: ev.data },
            phases: { ...state.phases, [ev.data.strategy]: "generated" },
          };
        case "metrics": {
          const { strategy, metrics } = ev.data;
          const current = state.answers[strategy];
          if (!metrics || !current) {
            // the judge failed for this answer: keep it, unscored
            return { ...state, phases: { ...state.phases, [strategy]: "unscored" } };
          }
          return {
            ...state,
            answers: { ...state.answers, [strategy]: { ...current, metrics } },
            phases: { ...state.phases, [strategy]: "scored" },
          };
        }
        case "done":
          return { ...state, status: "done", result: ev.data };
        case "error":
          return { ...state, status: "error", error: ev.data };
      }
    }
  }
}

/** Progress 0..1 across the five conditions (generation counts half, scoring the other half). */
export function liveProgress(state: LiveState): number {
  if (state.status === "done") return 1;
  const per = 1 / ORDER.length;
  return ORDER.reduce((sum, s) => {
    const p = state.phases[s];
    const gen = p === "generated" || p === "scored" || p === "unscored" ? 1 : 0;
    const scored = p === "scored" || p === "unscored" ? 1 : 0;
    return sum + per * (state.scoring ? 0.5 * gen + 0.5 * scored : gen);
  }, 0);
}
