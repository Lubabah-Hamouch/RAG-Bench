import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api";
import { parseSse, streamLiveRun } from "../sse";
import type { LiveEvent } from "../types";

// Built at runtime so no key-shaped literal sits in the source (see tests/unit/test_no_secrets.py).
const KEY = ["gsk", "TESTKEYabcdefghij0123456789ABCDEFGHIJ"].join("_");

const sse = (event: string, data: unknown, eol = "\r\n") =>
  `event: ${event}${eol}data: ${JSON.stringify(data)}${eol}${eol}`;

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("parseSse", () => {
  it("parses complete events and keeps the unfinished remainder", () => {
    const text = sse("retrieval", { a: 1 }) + 'event: done\r\ndata: {"x"';
    const { events, rest } = parseSse(text);
    expect(events).toEqual([{ event: "retrieval", data: { a: 1 } }]);
    expect(rest).toContain("event: done");
  });

  it("handles LF line endings, comments and multi-line data", () => {
    const text = ': ping\n\nevent: metrics\ndata: {"a":\ndata: 2}\n\n';
    expect(parseSse(text).events).toEqual([{ event: "metrics", data: { a: 2 } }]);
  });

  it("skips malformed events instead of aborting the stream", () => {
    const { events } = parseSse("event: a\ndata: {not json}\n\n" + sse("b", { ok: true }, "\n"));
    expect(events).toEqual([{ event: "b", data: { ok: true } }]);
  });

  it("is incremental: a split event completes once the rest arrives", () => {
    const whole = sse("strategy_done", { strategy: "baseline" });
    const first = parseSse(whole.slice(0, 20));
    expect(first.events).toEqual([]);
    const second = parseSse(first.rest + whole.slice(20));
    expect(second.events).toEqual([{ event: "strategy_done", data: { strategy: "baseline" } }]);
  });
});

describe("streamLiveRun", () => {
  it("delivers events in order, even when chunks split them mid-event", async () => {
    const body =
      sse("retrieval", { chunks: [] }) +
      sse("strategy_started", { strategy: "baseline", label: "Baseline" }) +
      sse("done", { mode: "live" });
    const cut = Math.floor(body.length / 3);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(streamOf([body.slice(0, cut), body.slice(cut, 2 * cut), body.slice(2 * cut)]), {
          status: 200,
        }),
      ),
    );
    const seen: string[] = [];
    await streamLiveRun(KEY, { question_id: "Q05", score: true }, (e: LiveEvent) => seen.push(e.event));
    expect(seen).toEqual(["retrieval", "strategy_started", "done"]);
  });

  it("sends the key only in a header, never in the URL or the body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(streamOf([]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await streamLiveRun(KEY, { question: "What is RAG?", score: false }, () => {});
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).not.toContain("gsk_");
    expect(String(init.body)).not.toContain("gsk_");
    expect((init.headers as Record<string, string>)["X-Groq-Api-Key"]).toBe(KEY);
    expect(init.method).toBe("POST");
  });

  it("turns an HTTP error into an ApiError with the server code and retry hint", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: "rate_limited", message: "slow down", retry_after_s: 12 }), {
          status: 429,
          headers: { "Retry-After": "12" },
        }),
      ),
    );
    await expect(streamLiveRun(KEY, { question_id: "Q05", score: true }, () => {})).rejects.toMatchObject({
      status: 429,
      code: "rate_limited",
      retryAfterS: 12,
    });
  });

  it("reports an unreachable backend as a network error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const err = await streamLiveRun(KEY, { question_id: "Q05", score: true }, () => {}).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("network");
  });

  it("lets an abort propagate as an AbortError, not an API error", async () => {
    const abort = Object.assign(new Error("aborted"), { name: "AbortError" });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(abort));
    await expect(
      streamLiveRun(KEY, { question_id: "Q05", score: true }, () => {}),
    ).rejects.toMatchObject({ name: "AbortError" });
  });
});
