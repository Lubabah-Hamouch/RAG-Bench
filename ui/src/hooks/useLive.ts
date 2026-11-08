import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { ApiError, IS_LOCAL_API, friendly, getHealth, validateKey } from "@/lib/api";
import { initialLive, liveReducer } from "@/lib/liveState";
import { streamLiveRun, type RunRequest } from "@/lib/sse";
import type { HealthOut } from "@/lib/types";

export type KeyStatus =
  | { state: "idle" }
  | { state: "verifying" }
  | { state: "valid"; message: string; generator: string; judge: string }
  | { state: "problem"; title: string; hint: string; retryAfterS?: number };

/**
 * `state` is reachability: did the health endpoint answer at all? `liveReady` is separate and only
 * says whether the embedding model is loaded, which the first live run does.
 */
export type Backend =
  | { state: "checking" }
  | { state: "waking" }
  | { state: "down" }
  | { state: "up"; liveReady: boolean; liveError: string | null; health: HealthOut };

const RETRY_MS = 4000;
// A hosted server asleep on a free tier takes about a minute to answer; allow for a slow wake.
const WAKE_WINDOW_MS = 120_000;

/**
 * Tracks whether the API is reachable, only while live mode is on, so replay-only visitors never
 * wake the server. Polling stops once it answers: further polls would keep it from going idle.
 * `ensureReachable` checks again on demand, and waits while a sleeping server wakes.
 */
export function useBackend(enabled: boolean) {
  const [backend, setBackend] = useState<Backend>({ state: "checking" });
  const session = useRef(0); // changes whenever live mode is switched, abandoning older polls
  const polling = useRef<{ result: Promise<boolean>; settle: (ok: boolean) => void } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();

  const publish = useCallback((health: HealthOut) => {
    const liveError = health.live_error ?? null;
    setBackend({ state: "up", liveReady: health.live_ready, liveError, health });
  }, []);

  /** Poll until the server answers. Resolves false on giving up, but keeps retrying regardless. */
  const poll = useCallback((): Promise<boolean> => {
    if (polling.current) return polling.current.result;
    const id = session.current;
    const startedAt = Date.now();
    let settle: (ok: boolean) => void = () => {};
    const result = new Promise<boolean>((resolve) => {
      settle = resolve;
    });
    polling.current = { result, settle };
    const attempt = async () => {
      const health = await getHealth();
      if (id !== session.current) return;
      if (health) {
        polling.current = null;
        publish(health);
        settle(true);
        return;
      }
      if (IS_LOCAL_API || Date.now() - startedAt >= WAKE_WINDOW_MS) {
        setBackend({ state: "down" });
        settle(false);
      } else {
        setBackend({ state: "waking" });
      }
      timer.current = setTimeout(attempt, RETRY_MS);
    };
    void attempt();
    return result;
  }, [publish]);

  const ensureReachable = useCallback(async (): Promise<boolean> => {
    if (polling.current) return polling.current.result;
    const id = session.current;
    const health = await getHealth();
    if (id !== session.current) return false;
    if (health) {
      publish(health);
      return true;
    }
    setBackend({ state: IS_LOCAL_API ? "down" : "waking" });
    return poll();
  }, [poll, publish]);

  useEffect(() => {
    if (!enabled) return;
    session.current += 1;
    setBackend({ state: "checking" });
    void poll();
    return () => {
      session.current += 1;
      clearTimeout(timer.current);
      polling.current?.settle(false);
      polling.current = null;
    };
  }, [enabled, poll]);

  return { backend, ensureReachable };
}

/**
 * The bring-your-own-key session. The key is held in a ref (memory only): it is never written to
 * localStorage, sessionStorage, cookies or the URL, and `wipe` discards it.
 */
export function useLiveSession(ensureReachable: () => Promise<boolean>) {
  const keyRef = useRef("");
  const abortRef = useRef<AbortController | null>(null);
  const [keyStatus, setKeyStatus] = useState<KeyStatus>({ state: "idle" });
  const [live, dispatch] = useReducer(liveReducer, initialLive);

  const setKey = useCallback((value: string) => {
    keyRef.current = value.trim();
    setKeyStatus((s) => (s.state === "idle" ? s : { state: "idle" }));
  }, []);

  const verify = useCallback(async () => {
    const key = keyRef.current;
    if (!key) return;
    setKeyStatus({ state: "verifying" });
    const reachable = await ensureReachable();
    if (keyRef.current !== key) return; // edited or discarded while the server woke
    if (!reachable) {
      setKeyStatus({ state: "problem", ...friendly("network") });
      return;
    }
    try {
      const out = await validateKey(key);
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
  }, [ensureReachable]);

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
        const reachable = await ensureReachable();
        if (ctrl.signal.aborted) return;
        if (!reachable) throw new ApiError("Network error", 0, "network");
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
      // the run has loaded the model on the server, so pick up the new live_ready
      if (!ctrl.signal.aborted) void ensureReachable();
    },
    [cancel, ensureReachable],
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
