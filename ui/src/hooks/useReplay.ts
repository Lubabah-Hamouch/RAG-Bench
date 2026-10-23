import { useEffect, useRef, useState } from "react";
import { ApiError, loadReplayBase, loadResult, type ReplayBase } from "@/lib/api";
import type { QuestionResultOut } from "@/lib/types";

type BaseState =
  | { status: "loading" }
  | { status: "error"; error: ApiError }
  | { status: "ready"; base: ReplayBase };

export function useReplayBase(): BaseState {
  const [state, setState] = useState<BaseState>({ status: "loading" });
  useEffect(() => {
    let alive = true;
    loadReplayBase()
      .then((base) => alive && setState({ status: "ready", base }))
      .catch((e: unknown) =>
        alive &&
        setState({
          status: "error",
          error: e instanceof ApiError ? e : new ApiError(String(e), 0, "network"),
        }),
      );
    return () => {
      alive = false;
    };
  }, []);
  return state;
}

/** The recorded result for one question. All results are prefetched in the background (about
 * 30 KB each), so moving between questions is instant after the first load. */
export function useReplayResult(base: ReplayBase | null, qid: string | null) {
  const cache = useRef(new Map<string, QuestionResultOut>());
  const [, bump] = useState(0);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    if (!base) return;
    let alive = true;
    const fetchOne = async (id: string) => {
      if (cache.current.has(id)) return;
      try {
        cache.current.set(id, await loadResult(base.source, id));
        if (alive) bump((n) => n + 1);
      } catch (e) {
        if (alive && id === qid) {
          setError(e instanceof ApiError ? e : new ApiError(String(e), 0, "network"));
        }
      }
    };
    void (async () => {
      if (qid) await fetchOne(qid);
      for (const q of base.questions) await fetchOne(q.id);
    })();
    return () => {
      alive = false;
    };
  }, [base, qid]);

  const result = qid ? (cache.current.get(qid) ?? null) : null;
  return { result, loading: !!qid && !result && !error, error };
}
