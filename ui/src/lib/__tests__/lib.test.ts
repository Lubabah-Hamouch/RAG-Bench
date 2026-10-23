import { describe, expect, it } from "vitest";
import { ApiError, errorFromResponse, friendly } from "../api";
import { ci, compact, excludesZero, num, pct, seconds, signed, usd } from "../format";
import { parseHash, toHash } from "../route";
import { color, STRATEGIES, STRATEGY_ORDER } from "../strategies";

describe("format", () => {
  it("formats numbers, signs and nulls", () => {
    expect(num(0.5333)).toBe("0.53");
    expect(num(null)).toBe("–");
    expect(signed(0.2)).toBe("+0.20");
    expect(signed(-0.07)).toBe("−0.07");
    expect(pct(0.333)).toBe("33%");
  });

  it("formats money, time and counts", () => {
    expect(usd(0)).toBe("$0");
    expect(usd(0.00004)).toBe("<$0.0001");
    expect(usd(0.0042)).toBe("$0.0042");
    expect(usd(0.5)).toBe("$0.500");
    expect(seconds(2.34)).toBe("2.3s");
    expect(seconds(120)).toBe("120s");
    expect(compact(950)).toBe("950");
    expect(compact(2631)).toBe("2.6k");
    expect(compact(145952)).toBe("146k");
  });

  it("formats intervals and detects intervals that exclude zero", () => {
    const e = { mean: 0.39, lo: 0.17, hi: 0.62, n: 9 };
    expect(ci(e)).toBe("0.39 [0.17, 0.62]");
    expect(excludesZero(e)).toBe(true);
    expect(excludesZero({ mean: 0.1, lo: -0.1, hi: 0.3, n: 9 })).toBe(false);
    expect(excludesZero(null)).toBe(false);
  });
});

describe("routes", () => {
  it("parses hashes", () => {
    expect(parseHash("#/explore/Q05")).toEqual({ view: "explore", qid: "Q05" });
    expect(parseHash("#/benchmark")).toEqual({ view: "benchmark", qid: null });
    expect(parseHash("#/method")).toEqual({ view: "method", qid: null });
    expect(parseHash("")).toEqual({ view: "explore", qid: null });
    expect(parseHash("#/nonsense")).toEqual({ view: "explore", qid: null });
  });

  it("round-trips", () => {
    const routes = [
      { view: "explore" as const, qid: "Q09" },
      { view: "benchmark" as const, qid: null },
      { view: "method" as const, qid: null },
    ];
    for (const r of routes) expect(parseHash(toHash(r))).toEqual(r);
  });
});

describe("strategies", () => {
  it("has five ordered strategies with distinct colours", () => {
    expect(STRATEGY_ORDER).toEqual(["baseline", "structured", "rrr", "self_rag", "no_retrieval"]);
    expect(new Set(STRATEGY_ORDER.map((s) => STRATEGIES[s].cssVar)).size).toBe(5);
    expect(color("rrr", 0.5)).toBe("rgb(var(--s-rrr) / 0.5)");
    expect(STRATEGIES.self_rag.label).toContain("prompted");
  });
});

describe("api errors", () => {
  it("maps every backend error code to plain language", () => {
    const codes = [
      "invalid_key",
      "provider_quota",
      "rate_limited",
      "run_in_progress",
      "timeout",
      "https_required",
      "network",
      "model_unavailable",
    ];
    for (const code of codes) {
      const f = friendly(code);
      expect(f.title.length).toBeGreaterThan(3);
      expect(f.hint.length).toBeGreaterThan(10);
    }
    expect(friendly("something_new", "raw message").hint).toBe("raw message");
  });

  it("reads the code and retry hint out of an error response", async () => {
    const res = new Response(
      JSON.stringify({ error: "provider_quota", message: "m", retry_after_s: 30 }),
      { status: 429 },
    );
    const err = await errorFromResponse(res);
    expect(err).toBeInstanceOf(ApiError);
    expect([err.status, err.code, err.retryAfterS]).toEqual([429, "provider_quota", 30]);
  });

  it("copes with a non-JSON error body", async () => {
    const err = await errorFromResponse(
      new Response("Bad Gateway", { status: 502, statusText: "Bad Gateway" }),
    );
    expect(err.status).toBe(502);
    expect(err.code).toBe("http_502");
  });
});
