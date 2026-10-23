import type { Est } from "./types";

export const num = (x: number | null | undefined, digits = 2): string =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(digits);

export const signed = (x: number | null | undefined, digits = 2): string => {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
};

export const pct = (x: number | null | undefined, digits = 0): string =>
  x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`;

export const usd = (x: number | null | undefined): string => {
  if (x === null || x === undefined) return "–";
  if (x === 0) return "$0";
  if (x < 0.0001) return "<$0.0001";
  return `$${x.toFixed(x < 0.01 ? 4 : 3)}`;
};

export const seconds = (x: number | null | undefined): string =>
  x === null || x === undefined ? "–" : x < 10 ? `${x.toFixed(1)}s` : `${Math.round(x)}s`;

export const compact = (x: number | null | undefined): string => {
  if (x === null || x === undefined) return "–";
  if (x >= 1_000_000) return `${(x / 1_000_000).toFixed(1)}M`;
  if (x >= 1_000) return `${(x / 1_000).toFixed(x >= 10_000 ? 0 : 1)}k`;
  return `${Math.round(x)}`;
};

/** "0.53 [0.26, 0.78]" */
export const ci = (e: Est | null | undefined, digits = 2): string =>
  e ? `${num(e.mean, digits)} [${num(e.lo, digits)}, ${num(e.hi, digits)}]` : "–";

/** True when a paired difference's interval excludes zero. */
export const excludesZero = (e: Est | null | undefined): boolean =>
  !!e && (e.lo > 0 || e.hi < 0);
