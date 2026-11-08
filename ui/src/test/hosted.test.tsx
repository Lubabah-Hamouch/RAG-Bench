import { act, render, renderHook, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import questions from "../../public/replay/questions.json";
import App from "../App";
import { KeyPanel } from "../components/KeyPanel";
import { useBackend, type Backend } from "../hooks/useLive";
import { friendly, getHealth, IS_LOCAL_API, isLocalApiUrl } from "../lib/api";

// Runs before the imports above, so the modules see the API URL a hosted build is given.
vi.hoisted(() => {
  vi.stubEnv("VITE_API_URL", "https://api.example");
});

const healthy = (live_ready: boolean) =>
  new Response(
    JSON.stringify({ status: "ok", replay_available: true, questions: 10, models: {}, require_https: true, live_ready, live_error: null }),
    { status: 200, headers: { "content-type": "application/json" } },
  );

/** A request that never answers until it is aborted, as a sleeping server's can. */
const hang = (_url: unknown, init?: RequestInit) =>
  new Promise<Response>((_, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
  });

const down = async () => new Response("down", { status: 503 });

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("telling local from hosted", () => {
  it("goes by the API's host", () => {
    expect(IS_LOCAL_API).toBe(false);
    expect(isLocalApiUrl("http://127.0.0.1:8001")).toBe(true);
    expect(isLocalApiUrl("http://localhost:8000")).toBe(true);
    expect(isLocalApiUrl("https://rag-bench.onrender.com")).toBe(false);
  });
});

describe("hosted copy", () => {
  const panel = (backend: Backend) =>
    render(<KeyPanel backend={backend} status={{ state: "idle" }} onKey={() => {}} onVerify={() => {}} />);
  const up: Backend = {
    state: "up",
    liveReady: false,
    liveError: "RuntimeError",
    health: { status: "ok", replay_available: true, questions: 10, models: {}, require_https: true, live_ready: false, live_error: "RuntimeError" },
  };

  it("shows the hosted warning", () => {
    panel({ state: "checking" });
    expect(screen.getByText(/On the hosted demo/)).toBeInTheDocument();
  });

  it.each<[string, Backend, RegExp]>([
    ["waking", { state: "waking" }, /Waking the server/],
    ["down", { state: "down" }, /Try again shortly/],
    ["load failure", up, /couldn't load its retrieval index \(RuntimeError\).*Try again shortly/],
  ])("gives no local-only instructions when %s", (_, backend, message) => {
    panel(backend);
    expect(screen.getByText(message)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("ragbench");
  });

  it("gives no local-only instructions in the network or server errors", () => {
    expect(friendly("network").hint).not.toContain("ragbench");
    expect(friendly("internal_error").hint).not.toMatch(/ragbench|server log/);
  });

  it("gives no local-only instructions when the benchmark data cannot load", async () => {
    vi.stubGlobal("fetch", vi.fn(down));
    render(<App />);
    expect(await screen.findByText(/Couldn't load the benchmark data/)).toBeInTheDocument();
    expect(screen.getByText(/Reload the page/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("ragbench");
  });

  it("gives no local-only instructions when the aggregate report is missing", async () => {
    window.location.hash = "#/benchmark";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) =>
        String(input).endsWith("/replay/questions.json")
          ? new Response(JSON.stringify(questions), { status: 200, headers: { "content-type": "application/json" } })
          : down(),
      ),
    );
    render(<App />);
    expect(await screen.findByText(/aggregate report isn't available/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("ragbench");
    window.location.hash = "";
  });
});

describe("getHealth", () => {
  it("gives up on a request that never answers", async () => {
    vi.stubGlobal("fetch", vi.fn(hang));
    await expect(getHealth(10)).resolves.toBeNull();
  });

  it("bypasses the HTTP cache", async () => {
    const fetchMock = vi.fn(async () => healthy(true));
    vi.stubGlobal("fetch", fetchMock);
    await getHealth();
    expect(fetchMock.mock.calls[0]).toEqual([
      "https://api.example/api/health",
      expect.objectContaining({ cache: "no-store" }),
    ]);
  });
});

describe("useBackend", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  const tick = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

  it("makes no request until live mode is switched on", async () => {
    const fetchMock = vi.fn(async () => healthy(false));
    vi.stubGlobal("fetch", fetchMock);
    const { result, rerender } = renderHook(({ on }) => useBackend(on), { initialProps: { on: false } });
    await tick(60_000);
    expect(fetchMock).not.toHaveBeenCalled();
    rerender({ on: true });
    await tick(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.current.backend.state).toBe("up");
  });

  it("counts any answer as reachable and stops polling, whatever live_ready says", async () => {
    const fetchMock = vi.fn(async () => healthy(false));
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() => useBackend(true));
    await tick(0);
    expect(result.current.backend).toMatchObject({ state: "up", liveReady: false });
    await tick(30 * 60_000); // polling here would stop the server ever going idle
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("moves from contacting to waking when a request hangs, then to up", async () => {
    let asleep = true;
    vi.stubGlobal("fetch", vi.fn((url: unknown, init?: RequestInit) => (asleep ? hang(url, init) : Promise.resolve(healthy(false)))));
    const { result } = renderHook(() => useBackend(true));
    expect(result.current.backend.state).toBe("checking");
    await tick(8_000);
    expect(result.current.backend.state).toBe("waking");
    asleep = false;
    await tick(4_000);
    expect(result.current.backend.state).toBe("up");
  });

  it("calls it down only after the wake window, and still recovers", async () => {
    let answer = down;
    vi.stubGlobal("fetch", vi.fn(() => answer()));
    const { result } = renderHook(() => useBackend(true));
    await tick(110_000);
    expect(result.current.backend.state).toBe("waking");
    await tick(20_000);
    expect(result.current.backend.state).toBe("down");
    answer = async () => healthy(true);
    await tick(4_000);
    expect(result.current.backend).toMatchObject({ state: "up", liveReady: true });
  });

  it("notices a server that went back to sleep, and waits for it to wake", async () => {
    let answer = async () => healthy(true);
    vi.stubGlobal("fetch", vi.fn(() => answer()));
    const { result } = renderHook(() => useBackend(true));
    await tick(0);
    expect(result.current.backend.state).toBe("up");

    answer = down;
    let reachable: boolean | undefined;
    void result.current.ensureReachable().then((ok) => {
      reachable = ok;
    });
    await tick(5_000);
    expect(result.current.backend.state).toBe("waking");
    expect(reachable).toBeUndefined();

    answer = async () => healthy(false);
    await tick(4_000);
    expect(reachable).toBe(true);
    expect(result.current.backend).toMatchObject({ state: "up", liveReady: false });
  });

  it("reports unreachable after the wake window instead of waiting forever", async () => {
    vi.stubGlobal("fetch", vi.fn(down));
    const { result } = renderHook(() => useBackend(true));
    await tick(0);
    let reachable: boolean | undefined;
    void result.current.ensureReachable().then((ok) => {
      reachable = ok;
    });
    await tick(130_000);
    expect(reachable).toBe(false);
  });

  it("stops polling when live mode is switched off", async () => {
    const fetchMock = vi.fn(down);
    vi.stubGlobal("fetch", fetchMock);
    const { rerender } = renderHook(({ on }) => useBackend(on), { initialProps: { on: true } });
    await tick(0);
    rerender({ on: false });
    const made = fetchMock.mock.calls.length;
    await tick(60_000);
    expect(fetchMock).toHaveBeenCalledTimes(made);
  });
});
