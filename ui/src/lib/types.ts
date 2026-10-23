import type { components } from "./api-schema";

type S = components["schemas"];

// Generated from the backend's OpenAPI schema (npm run gen:api), so the contract cannot drift.
export type QuestionOut = S["QuestionOut"];
export type QuestionResultOut = S["QuestionResultOut"];
export type AnswerOut = S["AnswerOut"];
export type MetricsOut = S["MetricsOut"];
export type ChunkOut = S["ChunkOut"];
export type ClaimOut = S["ClaimOut"];
export type GroundednessOut = S["GroundednessOut"];
export type CoverageOut = S["CoverageOut"];
export type ConsistencyOut = S["ConsistencyOut"];
export type StageOut = S["StageOut"];
export type StrategyInfo = S["StrategyInfo"];
export type HealthOut = S["HealthOut"];
export type ValidateOut = S["ValidateOut"];

export type StrategyName = "baseline" | "structured" | "rrr" | "self_rag" | "no_retrieval";

// ---- summary.json (the aggregate benchmark report) ---------------------------------------------

export interface Est {
  mean: number;
  lo: number;
  hi: number;
  n: number;
}

export interface StrategyRow {
  G: Est | null;
  C: Est | null;
  FCS: Est | null;
  X: Est | null;
  words: Est | null;
  sentences: Est | null;
  mattr: Est | null;
  ttr: Est | null;
  avg_word_length: Est | null;
  latency_s: Est | null;
  tokens: Est | null;
  cost_usd: Est | null;
  calls: Est | null;
  consistency_semantic: Est | null;
  consistency_structural: Est | null;
}

export interface PairedCell {
  diff: Est;
  wilcoxon_p: number | null;
}

export type PairedTable = Record<string, Partial<Record<"G" | "C" | "FCS", PairedCell | null>>>;

export interface AbstentionRow {
  out_of_corpus_abstain_rate: number | null;
  out_of_corpus_n: number;
  false_abstain_rate_answerable: number | null;
  answerable_n: number;
}

export interface SummaryBlock {
  strategies: Record<string, StrategyRow>;
  by_type?: Record<string, Record<string, { G: Est | null; C: Est | null; FCS: Est | null }>>;
  paired: { vs_baseline: PairedTable; vs_control: PairedTable };
  abstention: Record<string, AbstentionRow>;
  retrieval: {
    mean_similarity_by_rank: Record<string, number>;
    distractor_chunks_in_top4: number;
    questions: number;
    context_recall_by_question: Record<string, number | null>;
    judge_context_recall_by_question?: Record<string, number>;
    mean_judge_context_recall?: number | null;
    mean_context_recall: number | null;
  };
}

export interface PerQuestionRow {
  type: string;
  question: string;
  strategies: Record<
    string,
    {
      G: number | null;
      C: number | null;
      FCS: number | null;
      consistency_semantic: number | null;
      context_recall_judge: number | null;
      abstain_rate: number | null;
    }
  >;
}

export interface Summary extends SummaryBlock {
  meta: {
    n_questions: number;
    n_answerable: number;
    composition: string;
    trials: number;
    judge_trials: number;
    generator: string;
    judge: string;
    generated_at: string;
    log?: { invocations: number; active_minutes: number; quota_stops: number; judge_errors: number };
  };
  per_question: Record<string, PerQuestionRow>;
  totals: {
    answers: number;
    judged_answers: number;
    model_calls: number;
    cached_calls: number;
    rate_limit_retries: number;
    generator_tokens: number;
    judge_tokens: number;
  };
  sensitivity?: { excluded: string[] } & SummaryBlock;
}

// ---- live run events (server-sent) ----------------------------------------------------------------

export interface LiveError {
  error: string;
  message: string;
  status: number;
  retry_after_s?: number;
}

export type LiveEvent =
  | { event: "retrieval"; data: { question: QuestionOut; chunks: ChunkOut[] } }
  | { event: "strategy_started"; data: { strategy: string; label: string } }
  | { event: "strategy_done"; data: AnswerOut }
  | {
      event: "metrics";
      data: { strategy: string; metrics?: MetricsOut; error?: string; message?: string };
    }
  | { event: "done"; data: QuestionResultOut }
  | { event: "error"; data: LiveError };
