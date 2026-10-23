/**
 * Display numbering for the benchmark questions.
 *
 * The data, URLs and summary.md keep the original research IDs (Q02, Q03, Q05, ...): three of the
 * 13 candidate questions were dropped as redundant, which leaves gaps. Visitors see a gap-free
 * Q1..Q10 instead, in the same order; the original ID stays available as a tooltip.
 */
let order: string[] = [];

/** Candidate questions that were dropped as redundant (see ARCHITECTURE_PLAN.md section 4.8). */
export const DROPPED_QUESTIONS = 3;

export function registerQuestionOrder(ids: string[]): void {
  order = ids;
}

/** "Q09" -> "Q7". Anything that is not a benchmark question (e.g. "live") is returned unchanged. */
export function qLabel(id: string): string {
  const i = order.indexOf(id);
  return i < 0 ? id : `Q${i + 1}`;
}

/** Tooltip text that traces a display label back to the research ID. */
export function qOriginal(id: string): string | undefined {
  return order.includes(id) ? `Research ID ${id}` : undefined;
}
