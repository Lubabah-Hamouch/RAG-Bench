import { describe, expect, it } from "vitest";
import { initialLive, liveProgress, liveReducer, type LiveState } from "../liveState";
import type { AnswerOut, LiveEvent, MetricsOut, QuestionResultOut } from "../types";

const metrics = (judged: boolean): MetricsOut => ({
  judged,
  lexical: { chars: 1, words: 1, sentences: 1, unique_vocab: 1 },
  latency_s: 1,
  prompt_tokens: 1,
  completion_tokens: 1,
  calls: 1,
});

const answer = (strategy: string): AnswerOut => ({
  strategy,
  label: strategy,
  uses_retrieval: true,
  answer: "text",
  trace: [],
  abstained: false,
  trial: 0,
  metrics: metrics(false),
  other_trials: [],
});

const ev = (e: LiveEvent) => ({ type: "event" as const, event: e });
const start = (scoring = true): LiveState =>
  liveReducer(initialLive, { type: "start", scoring, at: 1000 });

describe("liveReducer", () => {
  it("starts with all five conditions queued", () => {
    const s = start();
    expect(s.status).toBe("running");
    expect(Object.values(s.phases)).toEqual(["queued", "queued", "queued", "queued", "queued"]);
  });

  it("walks a strategy through running, generated and scored", () => {
    let s = start();
    s = liveReducer(s, ev({ event: "strategy_started", data: { strategy: "baseline", label: "Baseline" } }));
    expect(s.phases.baseline).toBe("running");
    s = liveReducer(s, ev({ event: "strategy_done", data: answer("baseline") }));
    expect(s.phases.baseline).toBe("generated");
    expect(s.answers.baseline.metrics.judged).toBe(false);
    s = liveReducer(s, ev({ event: "metrics", data: { strategy: "baseline", metrics: metrics(true) } }));
    expect(s.phases.baseline).toBe("scored");
    expect(s.answers.baseline.metrics.judged).toBe(true);
  });

  it("keeps an answer, unscored, when the judge failed for it", () => {
    let s = start();
    s = liveReducer(s, ev({ event: "strategy_done", data: answer("rrr") }));
    s = liveReducer(
      s,
      ev({ event: "metrics", data: { strategy: "rrr", error: "judge_failed", message: "x" } }),
    );
    expect(s.phases.rrr).toBe("unscored");
    expect(s.answers.rrr.answer).toBe("text");
    expect(s.status).toBe("running");
  });

  it("finishes on done and fails on error events", () => {
    const done = liveReducer(
      start(),
      ev({ event: "done", data: { mode: "live" } as unknown as QuestionResultOut }),
    );
    expect(done.status).toBe("done");
    const failed = liveReducer(
      start(),
      ev({ event: "error", data: { error: "invalid_key", message: "no", status: 401 } }),
    );
    expect(failed.status).toBe("error");
    expect(failed.error?.error).toBe("invalid_key");
  });

  it("start discards any previous run and reset returns to idle", () => {
    let s = liveReducer(start(), ev({ event: "strategy_done", data: answer("baseline") }));
    s = liveReducer(s, { type: "start", scoring: false, at: 5 });
    expect(Object.keys(s.answers)).toHaveLength(0);
    expect(liveReducer(s, { type: "reset" })).toEqual(initialLive);
  });

  it("an HTTP-level failure keeps what had already arrived", () => {
    let s = liveReducer(start(), ev({ event: "strategy_done", data: answer("baseline") }));
    s = liveReducer(s, { type: "fail", error: { error: "network", message: "x", status: 0 } });
    expect(s.status).toBe("error");
    expect(s.answers.baseline).toBeDefined();
  });
});

describe("liveProgress", () => {
  it("counts generation and scoring as halves of each condition when scoring", () => {
    let s = start(true);
    expect(liveProgress(s)).toBe(0);
    s = liveReducer(s, ev({ event: "strategy_done", data: answer("baseline") }));
    expect(liveProgress(s)).toBeCloseTo(0.1);
    s = liveReducer(s, ev({ event: "metrics", data: { strategy: "baseline", metrics: metrics(true) } }));
    expect(liveProgress(s)).toBeCloseTo(0.2);
  });

  it("counts generation alone when scoring is off, and is 1 when done", () => {
    let s = start(false);
    s = liveReducer(s, ev({ event: "strategy_done", data: answer("baseline") }));
    expect(liveProgress(s)).toBeCloseTo(0.2);
    expect(liveProgress({ ...s, status: "done" })).toBe(1);
  });
});
