import type {
  HealthOut,
  QuestionOut,
  QuestionResultOut,
  StrategyInfo,
  Summary,
  ValidateOut,
} from "./types";

export const API_URL: string = (
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export const KEY_HEADER = "X-Groq-Api-Key";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string,
    public retryAfterS?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Plain-language text for each backend error code. */
const FRIENDLY: Record<string, { title: string; hint: string }> = {
  invalid_key: {
    title: "Key rejected",
    hint: "Groq did not accept this key. Check it was copied completely and has not been revoked.",
  },
  invalid_key_format: {
    title: "That doesn't look like a Groq key",
    hint: "Keys are one long token starting with gsk_ with no spaces.",
  },
  key_required: { title: "Key needed", hint: "Paste your Groq key and verify it first." },
  https_required: {
    title: "Secure connection required",
    hint: "This server only accepts keys over HTTPS.",
  },
  provider_quota: {
    title: "Groq limit reached",
    hint: "Your Groq rate limit or daily quota is used up. Wait a little, or turn off scoring to use fewer calls.",
  },
  rate_limited: { title: "Slow down a little", hint: "Too many requests from this connection." },
  run_in_progress: {
    title: "A run is already in progress",
    hint: "Wait for the current run to finish.",
  },
  server_busy: { title: "Server is busy", hint: "Try again in a moment." },
  timeout: {
    title: "The run took too long",
    hint: "Groq's per-minute limits can slow a run. Try again, or turn off scoring.",
  },
  model_unavailable: {
    title: "Model unavailable",
    hint: "One of the benchmark's models is not offered to this key.",
  },
  judge_failed: { title: "Scoring failed", hint: "The judge model could not score an answer." },
  question_too_long: { title: "Question too long", hint: "Keep it under 500 characters." },
  invalid_request: { title: "Nothing to ask", hint: "Type a question first." },
  unknown_question: { title: "Unknown question", hint: "That benchmark question does not exist." },
  upstream_error: { title: "The model provider failed", hint: "Try again in a moment." },
  internal_error: { title: "Unexpected server error", hint: "Try again; if it persists, check the server log." },
  network: {
    title: "Can't reach the backend",
    hint: "Start it with `ragbench serve` (default http://127.0.0.1:8000).",
  },
};

export const friendly = (code: string, fallback?: string): { title: string; hint: string } =>
  FRIENDLY[code] ?? { title: "Something went wrong", hint: fallback ?? "Please try again." };

export async function errorFromResponse(res: Response): Promise<ApiError> {
  let code = `http_${res.status}`;
  let message = res.statusText;
  let retry: number | undefined;
  try {
    const body = (await res.json()) as { error?: string; message?: string; retry_after_s?: number };
    code = body.error ?? code;
    message = body.message ?? message;
    retry = body.retry_after_s;
  } catch {
    /* non-JSON error body */
  }
  const headerRetry = Number(res.headers.get("Retry-After"));
  return new ApiError(message, res.status, code, retry ?? (headerRetry > 0 ? headerRetry : undefined));
}

async function getJson<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError("Network error", 0, "network");
  }
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as T;
}

// ---- replay data: static files when exported, else the API ------------------------------------------

export type ReplaySource = "static" | "api";

export interface ReplayBase {
  source: ReplaySource;
  questions: QuestionOut[];
  strategies: StrategyInfo[];
  summary: Summary | null;
}

/** The Vite dev server answers unknown paths with index.html, so verify we really got JSON. */
async function tryStatic<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`/replay/${path}`, { headers: { Accept: "application/json" } });
    if (!res.ok || !(res.headers.get("content-type") ?? "").includes("json")) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export async function loadReplayBase(): Promise<ReplayBase> {
  const staticQuestions = await tryStatic<QuestionOut[]>("questions.json");
  if (staticQuestions && Array.isArray(staticQuestions)) {
    const [strategies, summary] = await Promise.all([
      tryStatic<StrategyInfo[]>("strategies.json"),
      tryStatic<Summary>("summary.json"),
    ]);
    return { source: "static", questions: staticQuestions, strategies: strategies ?? [], summary };
  }
  const [questions, strategies, summary] = await Promise.all([
    getJson<QuestionOut[]>(`${API_URL}/api/questions`),
    getJson<StrategyInfo[]>(`${API_URL}/api/strategies`),
    getJson<Summary>(`${API_URL}/api/results/summary`).catch(() => null),
  ]);
  return { source: "api", questions, strategies, summary };
}

export async function loadResult(source: ReplaySource, qid: string): Promise<QuestionResultOut> {
  if (source === "static") {
    const r = await tryStatic<QuestionResultOut>(`results/${qid}.json`);
    if (r) return r;
  }
  return getJson<QuestionResultOut>(`${API_URL}/api/results/${encodeURIComponent(qid)}`);
}

// ---- backend health and key validation ------------------------------------------------------------------

export async function getHealth(): Promise<HealthOut | null> {
  try {
    return await getJson<HealthOut>(`${API_URL}/api/health`);
  } catch {
    return null;
  }
}

export async function validateKey(key: string, signal?: AbortSignal): Promise<ValidateOut> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/live/validate`, {
      method: "POST",
      headers: { [KEY_HEADER]: key },
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Network error", 0, "network");
  }
  if (!res.ok) throw await errorFromResponse(res);
  return (await res.json()) as ValidateOut;
}
