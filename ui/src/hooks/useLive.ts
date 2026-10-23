import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { ApiError, friendly, getHealth, validateKey } from "@/lib/api";
import { initialLive, liveReducer } from "@/lib/liveState";
import { streamLiveRun, type RunRequest } from "@/lib/sse";
import type { HealthOut } from "@/lib/types";

export type KeyStatus =
  | { state: "idle" }
  | { state: "verifying" }
  | { state: "valid"; message: string; generator: string; judge: string }
  | { state: "problem"; title: string; hint: string; retryAfterS?: number };

export type Backend =
  | { state: "checking" }
  | { state: "down" }
  | { state: "up"; liveReady: boolean; liveError: string | null; health: HealthOut };

/** Is the API reachable, and has its embedding model finished warming up? */
export function useBackend(enabled: boolean): Backend {
  const [backend, setBackend] = useState<Backend>({ state: "checking" });
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      const health = await getHealth();
      if (!alive) return;
      if (!health) {
        setBackend({ state: "down" });
        timer = setTimeout(poll, 4000);
      } else {
        const liveError = health.live_error ?? null;
        setBackend({ state: "up", liveReady: health.live_ready, liveError, health });
        // keep polling while it warms up; retry slowly if loading failed
        if (!health.live_ready) timer = setTimeout(poll, liveError ? 10000 : 2000);
      }
    };
    setBackend({ state: "checking" });
    void poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [enabled]);
  return backend;
}

/**
 * The bring-your-own-key session. The key is held in a ref (memory only): it is never written to
 * localStorage, sessionStorage, cookies or the URL, and `wipe` discards it.
 */
export function useLiveSession() {
  const keyRef = useRef("");
  const abortRef = useRef<AbortController | null>(null);
  const [keyStatus, setKeyStatus] = useState<KeyStatus>({ state: "idle" });
  const [live, dispatch] = useReducer(liveReducer, initialLive);

  const setKey = useCallback((value: string) => {
    keyRef.current = value.trim();
    setKeyStatus((s) => (s.state === "idle" ? s : { state: "idle" }));
  }, []);

  const verify = useCallback(async () => {
    if (!keyRef.current) return;
    setKeyStatus({ state: "verifying" });
    try {
      const out = await validateKey(keyRef.current);
      if (out.valid) {
        setKeyStatus({
          state: "valid",
          message: out.message,
          generator: out.generator_model,
          judge: out.judge_model,
        });
      } else {
        setKeyStatus({
          state: "problem",
          title: "Models not available to this key",
          hint: out.message,
        });
      }
    } catch (e) {
      const err = e instanceof ApiError ? e : new ApiError(String(e), 0, "network");
      const f = friendly(err.code === "provider_quota" ? "provider_quota" : err.code, err.message);
      setKeyStatus({ state: "problem", title: f.title, hint: f.hint, retryAfterS: err.retryAfterS });
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const run = useCallback(
    async (body: RunRequest) => {
      if (!keyRef.current) return;
      cancel();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      dispatch({ type: "start", scoring: body.score, at: Date.now() });
      try {
        await streamLiveRun(keyRef.current, body, (event) => dispatch({ type: "event", event }), ctrl.signal);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        const err = e instanceof ApiError ? e : new ApiError(String(e), 0, "network");
        dispatch({
          type: "fail",
          error: {
            error: err.code,
            message: err.message,
            status: err.status,
            retry_after_s: err.retryAfterS,
          },
        });
      }
    },
    [cancel],
  );

  /** Abort anything in flight and forget the key. */
  const wipe = useCallback(() => {
    cancel();
    keyRef.current = "";
    setKeyStatus({ state: "idle" });
    dispatch({ type: "reset" });
  }, [cancel]);

  const reset = useCallback(() => dispatch({ type: "reset" }), []);

  useEffect(() => wipe, [wipe]);

  return { keyStatus, setKey, verify, run, cancel, wipe, reset, live };
}
