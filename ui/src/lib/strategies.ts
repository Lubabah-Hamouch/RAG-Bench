import type { StrategyName } from "./types";

export const STRATEGY_ORDER: StrategyName[] = [
  "baseline",
  "structured",
  "rrr",
  "self_rag",
  "no_retrieval",
];

export interface StrategyMeta {
  name: StrategyName;
  label: string;
  short: string;
  tagline: string;
  how: string;
  calls: number;
  /** CSS custom property holding this strategy's colour channels. */
  cssVar: string;
}

export const STRATEGIES: Record<StrategyName, StrategyMeta> = {
  baseline: {
    name: "baseline",
    label: "Baseline",
    short: "Baseline",
    tagline: "Direct extractive answer",
    how: "One call. Answers in a sentence or two, staying close to the source wording.",
    calls: 1,
    cssVar: "--s-baseline",
  },
  structured: {
    name: "structured",
    label: "Structured",
    short: "Structured",
    tagline: "Summary, key points, elaboration",
    how: "One call that must fill a fixed three-section template.",
    calls: 1,
    cssVar: "--s-structured",
  },
  rrr: {
    name: "rrr",
    label: "RRR",
    short: "RRR",
    tagline: "Read, synthesize, refine",
    how: "Three calls: extract the relevant facts, write a draft, then edit it for coherence.",
    calls: 3,
    cssVar: "--s-rrr",
  },
  self_rag: {
    name: "self_rag",
    label: "Self-RAG (prompted)",
    short: "Self-RAG",
    tagline: "Draft, critique, revise",
    how: "Three calls: answer, critique the answer against the sources, then revise. Emulated with prompts; it is not the trained Self-RAG model.",
    calls: 3,
    cssVar: "--s-self-rag",
  },
  no_retrieval: {
    name: "no_retrieval",
    label: "No retrieval",
    short: "Control",
    tagline: "Control: answers from memory",
    how: "One call with no retrieved context at all. Shows what retrieval actually adds.",
    calls: 1,
    cssVar: "--s-control",
  },
};

export const color = (name: string, alpha = 1): string =>
  `rgb(var(${STRATEGIES[name as StrategyName]?.cssVar ?? "--s-control"}) / ${alpha})`;

export const strategyLabel = (name: string): string =>
  STRATEGIES[name as StrategyName]?.label ?? name;

export const strategyShort = (name: string): string =>
  STRATEGIES[name as StrategyName]?.short ?? name;

export const TYPE_LABEL: Record<string, string> = {
  lookup: "Lookup",
  multi_hop: "Multi-hop",
  out_of_corpus: "Out of corpus",
  free_text: "Your question",
};
