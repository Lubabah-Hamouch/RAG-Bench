import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import questions from "../../public/replay/questions.json";
import strategies from "../../public/replay/strategies.json";
import summary from "../../public/replay/summary.json";
import { STRATEGIES } from "../lib/strategies";
import type { AnswerOut, QuestionResultOut, StrategyName } from "../lib/types";

const results = import.meta.glob("../../public/replay/results/*.json", {
  eager: true,
  import: "default",
}) as Record<string, QuestionResultOut>;
const byId = Object.fromEntries(
  Object.entries(results).map(([path, r]) => [path.split("/").pop()!.replace(".json", ""), r]),
);

// Built at runtime so no key-shaped literal sits in the source (see tests/unit/test_no_secrets.py).
const KEY = ["gsk", "UITESTKEY0123456789abcdefghijklmnop"].join("_");
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

function liveStream(
  qid: string,
  opts: {
    models?: Record<string, string>;
    mutate?: (a: AnswerOut) => AnswerOut;
    question?: QuestionResultOut["question"];
  } = {},
): Response {
  const r = byId[qid];
  const answers = Object.fromEntries(
    Object.entries(r.answers).map(([name, a]) => [name, opts.mutate ? opts.mutate(a) : a]),
  ) as Record<string, AnswerOut>;
  const sse = (event: string, data: unknown) => `event: ${event}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`;
  const question = opts.question ?? r.question;
  let body = sse("retrieval", { question, chunks: r.retrieval.chunks });
  for (const [name, a] of Object.entries(answers)) {
    body += sse("strategy_started", { strategy: name, label: a.label });
    body += sse("strategy_done", a);
    body += sse("metrics", { strategy: name, metrics: a.metrics });
  }
  body += sse("done", {
    ...r,
    question,
    answers,
    models: opts.models ?? r.models,
    mode: "live",
    n_trials: 1,
    judged_trials: 1,
  });
  return new Response(new TextEncoder().encode(body), {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

let calls: { url: string; init?: RequestInit }[] = [];

function installFetch(overrides: Partial<Record<string, () => Response>> = {}) {
  calls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      const hit = Object.keys(overrides).find((k) => url.includes(k));
      if (hit) return overrides[hit]!();
      if (url.endsWith("/replay/questions.json")) return json(questions);
      if (url.endsWith("/replay/strategies.json")) return json(strategies);
      if (url.endsWith("/replay/summary.json")) return json(summary);
      const m = url.match(/\/replay\/results\/(Q\d+)\.json$/);
      if (m) return json(byId[m[1]]);
      if (url.endsWith("/api/health"))
        return json({ status: "ok", replay_available: true, questions: 10, models: {}, require_https: false, live_ready: true, live_error: null });
      if (url.endsWith("/api/live/validate"))
        return json({ valid: true, generator_model: "gen-model", generator_available: true, judge_model: "judge-model", judge_available: true, message: "ok" });
      if (url.includes("/api/run")) return liveStream("Q05");
      return new Response("not found", { status: 404 });
    }),
  );
}

beforeEach(() => {
  window.location.hash = "";
  installFetch();
});
afterEach(() => vi.unstubAllGlobals());

async function loaded() {
  render(<App />);
  await screen.findAllByRole("article");
}

describe("replay explorer", () => {
  it("shows the ten questions and the five conditions for the first one", async () => {
    await loaded();
    expect(screen.getAllByRole("tab")).toHaveLength(10);
    const cards = screen.getAllByRole("article");
    expect(cards).toHaveLength(5);
    for (const name of ["Baseline", "Structured", "RRR", "Self-RAG (prompted)", "No retrieval"]) {
      expect(screen.getByRole("article", { name: `${name} answer` })).toBeInTheDocument();
    }
    expect(screen.getAllByLabelText(/^Passage \d$/)).toHaveLength(4);
  });

  it("numbers the ten questions Q1 to Q10 with no gaps, keeping the research IDs traceable", async () => {
    await loaded();
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.querySelector(".num")!.textContent)).toEqual(
      Array.from({ length: 10 }, (_, i) => `Q${i + 1}`),
    );
    // the research IDs stay in the tooltip, the URL and the data
    expect(tabs.map((t) => t.getAttribute("title"))).toEqual(
      questions.map((q: { id: string }) => `Research ID ${q.id}`),
    );
    expect(screen.queryByText("Q02")).not.toBeInTheDocument();
  });

  it("says in one visible line that 10 of 13 candidate questions were kept", async () => {
    await loaded();
    const note = screen.getByTestId("question-note");
    expect(note).toHaveTextContent("10 of 13 candidate questions");
    expect(note).toHaveTextContent("3 were dropped as redundant");
  });

  it("makes no live API call in replay mode (only static files)", async () => {
    await loaded();
    expect(calls.every((c) => c.url.includes("/replay/"))).toBe(true);
  });

  it("explains that the control is scored against passages it never saw", async () => {
    await loaded();
    const control = screen.getByRole("article", { name: "No retrieval answer" });
    expect(within(control).getByTestId("control-note")).toHaveTextContent(/never saw the passages/);
    const baseline = screen.getByRole("article", { name: "Baseline answer" });
    expect(within(baseline).queryByTestId("control-note")).not.toBeInTheDocument();
  });

  it("selecting a question changes the answers and the deep link", async () => {
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("tab", { name: /^Q6/ }));
    await waitFor(() => expect(window.location.hash).toBe("#/explore/Q09"));
    expect(await screen.findByRole("heading", { name: /original RAG model and Self-RAG/ })).toBeInTheDocument();
  });

  it("explains a refusal on an answerable question as a context gap", async () => {
    window.location.hash = "#/explore/Q09";
    await loaded();
    const baseline = await screen.findByRole("article", { name: "Baseline answer" });
    expect(within(baseline).getByText(/Declined to answer/)).toBeInTheDocument();
  });

  it("marks correct abstention on the out-of-corpus question", async () => {
    window.location.hash = "#/explore/Q13";
    await loaded();
    const rrr = await screen.findByRole("article", { name: "RRR answer" });
    await waitFor(() => expect(within(rrr).getByText(/Correctly declined/)).toBeInTheDocument());
    const control = screen.getByRole("article", { name: "No retrieval answer" });
    expect(within(control).getByText(/Answered from memory/)).toBeInTheDocument();
  });

  it("shows the claim-level evidence and the pipeline stages when expanded", async () => {
    const user = userEvent.setup();
    window.location.hash = "#/explore/Q05";
    await loaded();
    const rrr = await screen.findByRole("article", { name: "RRR answer" });
    await user.click(within(rrr).getByRole("button", { name: /Pipeline stages/ }));
    expect(within(rrr).getByText("identify")).toBeInTheDocument();
    await user.click(within(rrr).getByRole("button", { name: /Claims checked/ }));
    expect(within(rrr).getAllByText(/Judged against the four retrieved passages/).length).toBe(1);
  });
});

describe("benchmark and method tabs", () => {
  it("renders the aggregate report with its small-sample warning", async () => {
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("button", { name: "Benchmark" }));
    expect(await screen.findByRole("heading", { name: /What 100 answers say/ })).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent(/small sample/i);
    expect(screen.getByText(/Showing/)).toHaveTextContent("9");
  });

  it("the Q8 sensitivity switch changes the sample shown", async () => {
    const user = userEvent.setup();
    window.location.hash = "#/benchmark";
    await loaded().catch(() => undefined);
    const toggle = await screen.findByRole("switch", { name: /Exclude Q8/ });
    await user.click(toggle);
    expect(screen.getByText(/Showing/)).toHaveTextContent("8");
    expect(screen.getByText(/excluding Q8/)).toBeInTheDocument();
  });

  it("renders the method page with its limitations", async () => {
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("button", { name: "Method" }));
    expect(await screen.findByRole("heading", { name: /How the comparison works/ })).toBeInTheDocument();
    expect(screen.getByText(/Honest limitations/)).toBeInTheDocument();
  });
});

describe("bring-your-own-key live mode", () => {
  it("is off by default and reveals the key panel only when switched on", async () => {
    const user = userEvent.setup();
    await loaded();
    expect(screen.queryByLabelText("Groq API key")).not.toBeInTheDocument();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    const input = await screen.findByLabelText("Groq API key");
    expect(input).toHaveAttribute("type", "password");
    expect(input).toHaveAttribute("autocomplete", "off");
    expect(screen.getByRole("button", { name: "Verify key" })).toBeDisabled();
  });

  it("verifies a key through the validate endpoint, sending it only in a header", async () => {
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(screen.getByRole("button", { name: "Verify key" }));
    expect(await screen.findByText(/Key accepted/)).toBeInTheDocument();
    const validate = calls.find((c) => c.url.endsWith("/api/live/validate"))!;
    expect((validate.init!.headers as Record<string, string>)["X-Groq-Api-Key"]).toBe(KEY);
    expect(validate.url).not.toContain(KEY);
    expect(String(validate.init!.body ?? "")).not.toContain(KEY);
  });

  it("never writes the key to storage, cookies or the URL", async () => {
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(screen.getByRole("button", { name: "Verify key" }));
    await screen.findByText(/Key accepted/);
    await user.click(screen.getByRole("button", { name: /Run this question/ }));
    await screen.findByText("Finished");
    for (const call of setItem.mock.calls) expect(call.join("|")).not.toContain("gsk_");
    expect(JSON.stringify({ ...localStorage })).not.toContain("gsk_");
    expect(JSON.stringify({ ...sessionStorage })).not.toContain("gsk_");
    expect(document.cookie).not.toContain("gsk_");
    expect(window.location.href).not.toContain("gsk_");
    expect(document.body.textContent).not.toContain(KEY);
    setItem.mockRestore();
  });

  it("runs a question live, labels the result Live and streams scores in", async () => {
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(screen.getByRole("button", { name: "Verify key" }));
    await screen.findByText(/Key accepted/);
    await user.click(screen.getByRole("button", { name: /Run this question/ }));
    await screen.findByText("Finished");
    expect(screen.getAllByText("Live").length).toBeGreaterThanOrEqual(5);
    const run = calls.find((c) => c.url.includes("/api/run"))!;
    expect(JSON.parse(String(run.init!.body))).toEqual({ question_id: "Q02", score: true });
    expect((run.init!.headers as Record<string, string>)["X-Groq-Api-Key"]).toBe(KEY);
    expect(run.url).not.toContain(KEY);
    await user.click(screen.getByRole("button", { name: /Back to benchmark result/ }));
    await waitFor(() => expect(screen.queryByText("Live result")).not.toBeInTheDocument());
  });

  // Regression: after a live run started, the Benchmark and Method tabs stopped working. The route
  // switch waited (AnimatePresence mode="wait") for the live Explorer's exit animation, which never
  // finished, so the new page was never mounted and the old one faded out: a blank screen.
  it.each([
    ["Benchmark", /What 100 answers say/],
    ["Method", /How the comparison works/],
  ])("the %s tab still opens after a live run", async (tab, heading) => {
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(screen.getByRole("button", { name: "Verify key" }));
    await screen.findByText(/Key accepted/);
    await user.click(screen.getByRole("button", { name: /Run this question/ }));
    await screen.findByText("Finished");
    await user.click(screen.getByRole("button", { name: tab }));
    expect(await screen.findByRole("heading", { name: heading }, { timeout: 3000 })).toBeVisible();
    expect(screen.queryByRole("heading", { name: /Same retrieval, five ways/ })).not.toBeInTheDocument();
  }, 15000);

  it("a tab clicked while a live run is still streaming opens straight away", async () => {
    // a stream that never finishes: the run stays "running" while the visitor navigates
    installFetch({
      "/api/run": () => {
        const enc = new TextEncoder();
        const r = byId.Q05;
        const first = `event: retrieval
data: ${JSON.stringify({ question: r.question, chunks: r.retrieval.chunks })}

`;
        const body = new ReadableStream({
          start(c) {
            c.enqueue(enc.encode(first));
            c.enqueue(
              enc.encode(`event: strategy_started
data: ${JSON.stringify({ strategy: "baseline", label: "Baseline" })}

`),
            );
          },
        });
        return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
      },
    });
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(screen.getByRole("button", { name: "Verify key" }));
    await screen.findByText(/Key accepted/);
    await user.click(screen.getByRole("button", { name: /Run this question/ }));
    await screen.findByText(/Running all five conditions/);
    await user.click(screen.getByRole("button", { name: "Method" }));
    expect(await screen.findByRole("heading", { name: /How the comparison works/ }, { timeout: 3000 })).toBeVisible();
  }, 15000);

  async function runLive(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(screen.getByRole("button", { name: "Verify key" }));
    await screen.findByText(/Key accepted/);
    await user.click(screen.getByRole("button", { name: /Run this question/ }));
    await screen.findByText("Finished");
  }

  const meterText = (card: HTMLElement, label: string) =>
    within(card)
      .getAllByLabelText(new RegExp(`^${label} `))
      .map((e) => e.getAttribute("aria-label"))[0];
  const fmt = (v: number | null | undefined) => (v == null ? "not available" : v.toFixed(2));

  // Regression for "every strategy shows 1.00": each card must show exactly the scores in its own
  // payload, so five different answers can never render as one set of identical numbers.
  it("each live card shows exactly the scores of its own answer", async () => {
    const user = userEvent.setup();
    await loaded();
    await runLive(user);
    const shown: string[] = [];
    for (const [name, a] of Object.entries(byId.Q05.answers)) {
      const card = screen.getByRole("article", { name: `${STRATEGIES[name as StrategyName].label} answer` });
      const m = a.metrics;
      expect(meterText(card, "Grounded")).toBe(`Grounded ${fmt(m.groundedness?.G)}`);
      expect(meterText(card, "Coverage")).toBe(`Coverage ${fmt(m.coverage?.C)}`);
      expect(meterText(card, "FCS")).toBe(`FCS ${fmt(m.fcs)}`);
      // what a person actually sees must match too (a 0.00 once rendered as a dash)
      const visible = (v: number | null | undefined) => (v == null ? "–" : v.toFixed(2));
      expect(within(card).getAllByTestId("meter-value").map((e) => e.textContent)).toEqual([
        visible(m.groundedness?.G),
        visible(m.coverage?.C),
        visible(m.fcs),
      ]);
      shown.push(`${name}:${meterText(card, "Grounded")}|${meterText(card, "Coverage")}|${meterText(card, "FCS")}`);
    }
    expect(new Set(shown.map((s) => s.split(":")[1])).size).toBeGreaterThan(1); // not one constant
  });

  it("a missing score renders as not available, never as 1.00", async () => {
    installFetch({
      "/api/run": () =>
        liveStream("Q05", {
          mutate: (a) => ({
            ...a,
            metrics: { ...a.metrics, fcs: null, coverage: null, groundedness: null },
          }),
        }),
    });
    const user = userEvent.setup();
    await loaded();
    await runLive(user);
    for (const name of Object.keys(byId.Q05.answers)) {
      const card = screen.getByRole("article", { name: `${STRATEGIES[name as StrategyName].label} answer` });
      expect(meterText(card, "Grounded")).toBe("Grounded not available");
      expect(meterText(card, "Coverage")).toBe("Coverage not available");
      expect(meterText(card, "FCS")).toBe("FCS not available");
      expect(within(card).queryByText("1.00")).not.toBeInTheDocument();
    }
  });

  it("explains why Coverage and FCS are unavailable for a free-text question, and only then", async () => {
    installFetch({
      "/api/run": () =>
        liveStream("Q05", {
          question: { id: "live", type: "free_text", question: "What is a vector database?", expected_abstain: false, gold_facts: [] },
          mutate: (a) => ({ ...a, metrics: { ...a.metrics, fcs: null, coverage: null } }),
        }),
    });
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(screen.getByRole("button", { name: "Verify key" }));
    await screen.findByText(/Key accepted/);
    await user.click(screen.getByRole("button", { name: "Ask your own question" }));
    await user.type(screen.getByLabelText("Your question"), "What is a vector database?");
    await user.click(screen.getByRole("button", { name: /Run my question/ }));
    await screen.findByText("Finished");
    const notes = await screen.findAllByTestId("free-text-note");
    expect(notes).toHaveLength(5);
    expect(notes[0]).toHaveTextContent(/need gold facts, which only the 10 benchmark questions have/);
  });

  it("shows no free-text explanation on the benchmark questions", async () => {
    const user = userEvent.setup();
    await loaded();
    await runLive(user);
    expect(screen.queryByTestId("free-text-note")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Back to benchmark result/ }));
    expect(screen.queryByTestId("free-text-note")).not.toBeInTheDocument();
  });

  it("says which models wrote and scored a live result", async () => {
    installFetch({ "/api/run": () => liveStream("Q05", { models: { generator: "gen-x", judge: "judge-y" } }) });
    const user = userEvent.setup();
    await loaded();
    await runLive(user);
    const line = await screen.findByTestId("live-models");
    expect(line).toHaveTextContent("gen-x");
    expect(line).toHaveTextContent("judge-y");
    expect(screen.queryByText(/Demo judge/)).not.toBeInTheDocument();
  });

  it("warns when a live result was scored by a demo (mock) judge", async () => {
    installFetch({ "/api/run": () => liveStream("Q05", { models: { generator: "mock-generator", judge: "mock-judge" } }) });
    const user = userEvent.setup();
    await loaded();
    await runLive(user);
    expect(await screen.findByText(/Demo judge: these scores are not real/)).toBeInTheDocument();
  });

  it("switching live mode off discards the key and the panel", async () => {
    const user = userEvent.setup();
    await loaded();
    const toggle = screen.getByRole("switch", { name: /Try it live/ });
    await user.click(toggle);
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(toggle);
    await waitFor(() => expect(screen.queryByLabelText("Groq API key")).not.toBeInTheDocument());
    await user.click(toggle);
    expect(await screen.findByLabelText("Groq API key")).toHaveValue("");
    expect(screen.getByRole("button", { name: "Verify key" })).toBeDisabled();
  });

  it("explains a rejected key in plain language", async () => {
    installFetch({
      "/api/live/validate": () => json({ error: "invalid_key", message: "rejected" }, 401),
    });
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    await user.type(await screen.findByLabelText("Groq API key"), KEY);
    await user.click(screen.getByRole("button", { name: "Verify key" }));
    expect(await screen.findByText(/Key rejected/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain(KEY);
  });

  it("says so when the backend is not running, and leaves replay working", async () => {
    installFetch({ "/api/health": () => new Response("nope", { status: 503 }) });
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    expect(await screen.findByText(/Can't reach the backend/)).toBeInTheDocument();
    expect(screen.getAllByRole("article")).toHaveLength(5);
  });

  it("reports a warm-up failure instead of waiting forever", async () => {
    installFetch({
      "/api/health": () =>
        json({ status: "ok", replay_available: true, questions: 10, models: {}, require_https: false, live_ready: false, live_error: "RuntimeError" }),
    });
    const user = userEvent.setup();
    await loaded();
    await user.click(screen.getByRole("switch", { name: /Try it live/ }));
    expect(await screen.findByText(/couldn't load its retrieval index/)).toBeInTheDocument();
  });
});
