# ARCHITECTURE_PLAN: RAG Generation-Strategy Evaluation Dashboard

Working name: **RAGBench** (Python package `ragbench`).
Status: **Phases 0-5 built (backend API done; frontend is Phase 6); the benchmark ran on the reduced design below (see Results). The retrieval investigation is closed (section 4.10): no retrieval change adopted, no rerun needed.** Decisions D1, D3 (BYOK), D7 and D8 are finalized (section 0, section 11).
Project root: the repository root. The original M2DS coursework files this project was rebuilt from are no longer included.

**Deviations from the plan so far (Phases 0-1)**
- No Makefile (Windows); commands are in the README. Heavy dependencies are optional extras installed per phase. `data/corpus/manifest.yaml` is the corpus source of truth, and `ragbench audit` generates `docs/corpus_audit.md`.
- **Location:** everything lives inside the project folder: the virtualenv (`.venv/`), the Chroma index (`data/chroma/`), the LLM cache (`.cache/llm_cache/`) and `ui/node_modules/`, all gitignored and rebuildable. (An earlier setup kept these outside the folder to avoid OneDrive sync; sync was never active, so that was removed. Override the first two with `RAGBENCH_CHROMA_DIR` and `RAGBENCH_CACHE_DIR`.)
- **Chunking:** chunks are built over the whole document with **sentence-level** packing, each tagged with the page it *starts* on (not page-by-page, not paragraph-packed). `chunk_chars` is a hard cap, raised from 550 to **600** at your request. Result (627 chunks): median 527, mean 499, p10 380, p90 586, max 599 chars, inside the ~500-600 target. (At 550 it was median 471 / mean 450; paragraph packing gave only ~400.)
- **Retrieval results** (draft questions Q01-Q13, real model): all 8 lookup questions rank the correct document first; multi-hop questions retrieve 7 of the 9 expected document slots (Q12 misses the original RAG paper); the distractor survey appears in none of the top-4 lists. Q13 (out-of-corpus) retrieves Self-RAG chunks with similarity 0.43-0.53, overlapping the multi-hop range (0.47-0.56), so **a similarity threshold cannot gate abstention; the generator has to abstain.**

**Phase 2 findings and deviations**
- **RESOLVED (you chose qwen3.8-27b + gpt-oss-20b): the originally configured Groq models were no longer available.** A live `models.list()` on this account (checked 2026-10-23) offers only: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.8-27b`, `allam-2-7b` (4k context), two Llama *prompt-guard* classifiers, TTS and Whisper. Neither `llama-3.3-70b-versatile` (D-generator) nor `meta-llama/llama-4-scout-17b-16e-instruct` (D1, judge) exists. Defaults in `.env.example`/`config.py` are unchanged until you choose. My recommendation: generator `qwen/qwen3.8-27b` (a plain instruct model; its output budget is not spent on hidden reasoning) and judge `openai/gpt-oss-120b` or `-20b` (different family, which is the point of D1). Caveat: the gpt-oss models are reasoning models, so `max_tokens` is shared with hidden reasoning (a one-word reply used 32 completion tokens, 30 of them reasoning); the judge calls would need a larger `max_tokens`.
- **Groq free-tier limits shape the benchmark.** Observed headers: 8,000 tokens/minute and 1,000 requests/day for each candidate model, and a 200,000 tokens/day cap on `gpt-oss-120b` (it was almost used up when I tried it). The Phase 4 benchmark is ~1,550 calls, roughly 2M tokens, so it needs a paid tier, several days of resumable runs, or fewer trials. The client now honours `Retry-After` for short waits and fails fast (instead of burning retries) when the provider asks for more than 90 s, which is how a daily quota looks.
- **Prompts** follow section 3 with abstention added, as requested: every answering stage carries the exact refusal sentence ("The provided context does not contain this information.") plus a partial-answer rule ("answer the part the context supports and state what it does not cover"); the RRR refine stage must leave a refusal unchanged; the Self-RAG critique gained an `Answerable from context: Yes | Partially | No` line, and its refine stage must reply with the refusal when that is `No`. `StrategyResult.abstained` flags answers containing the refusal sentence.
- **No `tenacity` dependency:** the retry loop is ~20 lines in the client, which keeps latency measurement exact (it times only the successful attempt) and is easy to test; `tenacity` was removed from the `llm` extra.
- **Cache location:** the LLM disk cache lives in `.cache/llm_cache` (gitignored; override `RAGBENCH_CACHE_DIR`) and is disabled for BYOK clients.
**Phase 3 findings and deviations**
- **Decisions applied:** `.env.example`, `config.py` defaults and `settings.yaml` now use `qwen/qwen3.8-27b` (generator, `reasoning_effort=none` = plain instruct mode) and `openai/gpt-oss-20b` (judge, `reasoning_effort=low`), and `n_trials` is 3. Your own `.env` still holds the old model names, which override the defaults, so it needs updating by you.
- **Judge reasoning overhead (measured):** at `low` effort gpt-oss-20b used ~430 hidden reasoning tokens over the 3 judge calls for a 9-claim answer, versus 2,326 at `medium` for identical scores, so `low` is kept. Verification is the expensive call (438 of 704 completion tokens). The judge starts at `max_tokens=1500` and, if a reply is cut off (`finish_reason=length` or empty), retries with double the budget up to 6,000. It also treats Groq's HTTP 400 `json_validate_failed` as retryable, salvaging the model's partial text when it parses.
- **Judge sanity (live):** claim extraction, verification and coverage returned valid JSON on every real call; short refusals skip the judge entirely (0 calls); a grounded versus a deliberately fabricated answer is checked by a live test (`test_judge_scores_a_grounded_answer_high_and_a_fabricated_one_low`). Limitations seen: groundedness is 1.0 for all strategies on an easy lookup question, so differences should show on multi-hop questions; and negative claims about the corpus ("Self-RAG did not report Llama 3 results") are labelled `unsupported`.
- **Pricing** (`configs/pricing.yaml`, from Groq's model pages on 2026-10-23): gpt-oss-20b $0.075 in / $0.30 out per 1M tokens; qwen3.8-27b (preview) $0.80 / $4.00. Unknown models give `cost = null`, never a guess. Free-tier usage costs $0; these are list prices.

**Phase 4 results (the real run: 10 questions x 2 trials, trial 0 judged)** - full tables in `evaluation/results/summary.md`, figures in `evaluation/results/figures/`, raw records in `runs.jsonl`.
- **Completed in one go:** 100 answers (50 judged) in 27.6 minutes, 0 judge errors, 0 quota stops, 166 per-minute rate-limit retries (absorbed by back-off). Actual generator tokens were 145,952 against a raw projection of ~146.6k (the 25% margin was not needed); judge tokens were 85,511 against ~102.8k projected. A later context-support check added ~12k judge tokens.
- **Retrieval matters most.** Against the no-retrieval control, FCS improves by +0.32 [0.09, 0.55] (Structured), +0.39 [0.17, 0.62] (RRR) and +0.30 [0.08, 0.53] (Self-RAG), and groundedness by +0.31 to +0.41 with intervals excluding zero. Baseline's FCS gain (+0.20 [-0.09, 0.51]) is not distinguishable from zero.
- **Among the four strategies, no ranking is established.** FCS: RRR 0.72 [0.49, 0.90], Structured 0.65 [0.39, 0.87], Self-RAG 0.63 [0.41, 0.83], Baseline 0.53 [0.26, 0.78]: all intervals overlap. The most suggestive contrast is RRR vs Baseline (dFCS +0.19 [0.03, 0.39], Wilcoxon p = 0.125). The elaborate strategies reach higher coverage than Baseline at a small groundedness cost (0.91-0.95 vs 1.00), and cost 1.5x (Structured) to 4.9x (Self-RAG) the tokens and 2.7x-7x the latency. Self-RAG (prompted) did not beat the simpler strategies.
- **Retrieval, not generation, limits the multi-hop questions.** Judge-based context recall (the share of gold facts the retrieved top-4 contains) averages 0.74, but is 0.33 on Q11 and 0.50 on Q03 and Q09. All four retrieval strategies refused Q11 on both trials (correctly, given that its top-4 contained no Pinecone chunk), and Baseline also refused Q03 and Q09. A single query embedding retrieves from one document, so cross-document questions often miss half their evidence. Multi-hop FCS is therefore a mix of retrieval and generation quality. This is the main finding to act on next (e.g. query decomposition or per-document retrieval).
- **Abstention:** all four retrieval strategies refused the out-of-corpus question on both trials (2/2); the control never refused and made unsupported claims. False refusals on answerable questions: Baseline 33% (Q03, Q09, Q11), the others 11% (Q11 only).
- **Consistency and diversity** barely separate the strategies: semantic consistency 0.92-0.97 with overlapping intervals (and from only one answer pair per question), MATTR 0.84-0.91. The original report's finding that Self-RAG gives the longest answers was not reproduced: Structured is longest here (211 words), then Self-RAG (184), RRR (117), Baseline (30).
- **Judge reliability was checked against a human on a small sample.** On 30 claims (`evaluation/results/claims_to_label.csv`, labelled by me) the judge agreed 28 times (93%, 95% CI for accuracy [0.79, 0.98]; Cohen's kappa 0.879 three-way, 0.864 supported-vs-not). The two claim-level disagreements were one lenient toward the control (a generic vector-database claim judged supported) and one harsh against RRR (the claim 'logs a SuggestionError silently' flagged contradicted although the passage says exactly that). One extra coverage case (Q02 control, checked because it looked lenient) was also judged covered when the human said not covered. All three disagreements run against the retrieval-vs-control gap, so the retrieval-helps finding is likely conservative. Limits: n=30; not randomly drawn (weighted toward control and Structured claims, plus the one targeted case); a single labeller who also wrote the gold facts; three labels were set from guided examples; and coverage and FCS are validated far less than groundedness (one coverage case, which went against the judge).
- **Quote-match context recall was too strict** (mean 0.22): gold quotes are specific sentences, while the context often holds the same fact in other words. It is kept in the report as a lower bound next to the judge-based measure.

**Phase 5 (backend API) - built; adaptations from section 7** (`src/ragbench/api/`, tests in `tests/unit/test_api_*.py`)
- **Endpoints (7, one more than planned):** `GET /api/health`, `/api/strategies` (added), `/api/questions`, `/api/results/summary`, `/api/results/{qid}`, `POST /api/live/validate`, `POST /api/run`. Replay needs no key and makes no model calls; both live endpoints need the visitor's key in `X-Groq-Api-Key`.
- **`/api/run` has two response modes.** The default is server-sent events (`retrieval`, then per strategy `strategy_started`, `strategy_done`, `metrics`, finally `done`, or `error`); `?stream=false` returns one JSON object (added for tests and simple clients). There are no separate `stage` events: each strategy's stages arrive in `strategy_done.trace`. In stream mode errors arrive as an `error` event carrying an HTTP-style `status`; in JSON mode they are real HTTP statuses (401 invalid key, 429 provider quota or our rate limit, 422 bad question, 504 timeout, 502 upstream).
- **Same shape for replay and live.** One code path (`api/assemble.py`) turns a benchmark record into the response for both, so they cannot drift. Live runs are single-trial: `consistency` is always null and there are no confidence intervals, as planned. Free-text live questions are scored for groundedness only (no gold facts, so no coverage or FCS); benchmark questions run via `question_id` also get coverage and FCS. FCS counts a refusal on an answerable question as 0, as in the report.
- **Key handling (all covered by tests, with three mutation checks that confirmed the tests can fail):** the key is read per request and passed only to a per-request client factory; clients built with a key bypass the disk cache; error text is scrubbed of the key; unexpected errors return a generic message (type only is logged, never the message or traceback); logs are redacted at record creation (a `LogRecord` factory, stronger than the planned per-handler filter, so it also covers uvicorn); a malformed key is rejected without being echoed. There is **no server-side key fallback at all**, even locally: live runs always need the header.
- **Abuse protection:** per-IP sliding-window rate limits (6 live runs per 10 minutes, 10 validations per minute), one run in flight per IP and a global cap of 4, a 500-character question limit, a 240 s run budget checked between steps (a call already in flight is bounded only by the client's 60 s timeout), and `Retry-After` headers. Limits are in-memory and per process, so a multi-worker deployment would need a shared store. Rate limiting is deliberately by IP only, so no key or key hash is ever held. `RAGBENCH_TRUST_PROXY_HEADERS` must be set to honour `X-Forwarded-For/Proto` behind a proxy.
- **HTTPS and CORS:** `RAGBENCH_REQUIRE_HTTPS=1` (hosted mode) makes key endpoints refuse plain HTTP; off by default for local use. CORS allows only the local dev origins (`localhost`/`127.0.0.1` on ports 5173 and 3000) and the key header, with no credentials; set `RAGBENCH_CORS_ORIGINS` (a JSON list) for the real domain. Responses carry `nosniff` and a restrictive CSP, and live responses are `no-store`.
- **Supporting changes:** `LLMClient` now sets error codes (`auth_failed`, `model_not_found`, `rate_limited`) and has `list_models()` (used by key validation, which costs no tokens); `ragbench serve` runs the API and `ragbench export-replay` writes the same JSON as static files (13 files, about 350 KB) for the keyless hosted demo. The embedder and Chroma store load lazily on the first live run, so a replay-only server starts instantly.
- **Not done, by design:** no call to the real Groq API was made to test the live path; every live test uses a fake client. A real end-to-end live run with your own key is left for you or for Phase 6.

**Stage 0: retrieval-variant comparison (2026-10-23, `evaluation/stage0_retrieval_comparison.py`, results in `evaluation/results/stage0/`)** - judge-based context recall on Q03, Q08 (lookups) and Q09, Q11, Q12 (multi-hop); nothing was changed in the main pipeline.

| variant | Q03 | Q08 | Q09 | Q11 | Q12 | multi-hop mean | lookup mean | max chunks |
|---|---|---|---|---|---|---|---|---|
| current k=4 | 0.50 | 0.67 | 0.50 | 0.33 | 0.67 | 0.50 | 0.58 | 4 |
| flat k=8 | 0.75 | 1.00 | 0.75 | 0.33 | 0.67 | 0.58 | 0.88 | 8 |
| LLM query decomposition (merged, <=6) | 0.50 | 0.67 | 0.50 | 0.67 | 0.67 | 0.61 | 0.58 | 6 |
| document-balanced k=6 | 0.75 | 0.67 | 0.75 | 0.00 | 0.67 | 0.47 | 0.71 | 6 |

- **Adoption bar for decomposition NOT cleared.** Multi-hop gain over k=8 was +0.03 (needed >= +0.15). The other three conditions passed (<= 8 chunks; no lookup regression, because the decomposer did not split Q03/Q08 so their retrieval is unchanged; Q13 not split, retrieval unchanged, all four retrieval strategies still abstain). Decomposition is **not adopted**.
- **Two different failure modes, with different fixes.** (a) Within-document ranking gaps (Q03, Q08, Q09): the right paper is retrieved but not the right chunk; deeper retrieval fixes them (k=8 gains one gold fact on each). (b) Cross-document topic domination (Q11): decomposition fixed it (+1 of 3 facts, via a clean 'Pinecone vector search' sub-query), whereas k=8 even added a Pinecone chunk to the context but not the one holding the fact. On Q12 the decomposer wrote two near-duplicate generic sub-queries, so it swapped one gold fact for another (net 0).
- **Document-balanced retrieval is rejected:** it scored 0.00 on Q11 and did not help Q08.
- **Caveat:** each question has only 3-4 gold facts, so every difference above is one fact. The comparison cannot separate small effects, which is why the pre-set bar was a large one.
- **Cost note:** k=8 doubles the context sent on every call (about +64% generation tokens, no longer fits one day of free-tier quota); decomposition only enlarges context (to 6 chunks) for questions it splits.

**Stage 0b: candidate pool + local cross-encoder reranker (`evaluation/stage0b_rerank.py`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, pool of 20)** - same questions and metric; criteria were fixed before running; the main pipeline was not changed.

| variant | Q03 | Q08 | Q09 | Q11 | Q12 | multi-hop | lookup | within-doc (Q03, Q08, Q09) | max chunks | context chars |
|---|---|---|---|---|---|---|---|---|---|---|
| current k=4 | 0.50 | 0.67 | 0.50 | 0.33 | 0.67 | 0.50 | 0.58 | 0.56 | 4 | 1,901 |
| flat k=8 | 0.75 | 1.00 | 0.75 | 0.33 | 0.67 | 0.58 | 0.88 | 0.83 | 8 | 3,814 |
| LLM decomposition | 0.50 | 0.67 | 0.50 | 0.67 | 0.67 | 0.61 | 0.58 | 0.56 | 6 | 2,503 |
| rerank pool20 -> 4 | 0.50 | 1.00 | 0.50 | 0.33 | 0.67 | 0.50 | 0.75 | 0.67 | 4 | 2,064 |
| rerank pool20 -> 6 | 0.50 | 1.00 | 0.75 | 0.33 | 0.67 | 0.58 | 0.75 | 0.75 | 6 | 3,105 |

- **Pre-set bar not cleared by either reranker variant** (needed within-document mean >= 0.765, i.e. 75% of k=8's gain): 0.67 for 20 -> 4 and 0.75 for 20 -> 6 (the latter misses by about one judged fact, recovering 70% of k=8's gain). Nothing is adopted.
- **What it did and did not capture.** Q08 reached k=8's level (1.00) even at 4 chunks; Q09 improved only at 6 chunks; **Q03 never improved** (the ms-marco cross-encoder does not surface the table-like chunk that defines the reflection tokens). **Q11 stayed unsolved (0.33), as expected:** it is a cross-document problem that a reranker over a dense pool cannot fix.
- **Token cost (rough estimate for the 10 x 2 benchmark; baseline run was 146.6k raw generator tokens, daily limit 200k):** rerank 20 -> 4 about 152k (+9% context, fits); rerank 20 -> 6 about 190k (+63% context, no margin, does not fit with a safety margin); flat k=8 about 215k (does not fit). So reranking to 6 costs about 40% of k=8's extra context less than k=8, but still too much for one free-tier day.
- **Missing control:** flat dense k=6 was not judged, so it is not shown that reranking adds value beyond simply retrieving more (cost: ~7k judge tokens).
- **Decision:** do not layer further fixes. Q11-type cross-document questions and Q03-type chunk-ranking gaps are accepted as **documented retrieval limitations** (context recall per question is already reported in `summary.md`). Because no retrieval change was adopted, **no benchmark rerun is needed**; the existing results stand. Q13 abstention under any reranked retrieval was not tested (it needs generator tokens).
- **Dependency/scope notes:** `scipy` is used only for the Wilcoxon test; the `eval` extra also installed pandas, scikit-learn and matplotlib for later phases.

---

## 0. Summary and decisions to confirm

**What it does.** One question goes through five conditions with *identical retrieval*:
four generation strategies (Baseline, Structured, RRR, Self-RAG) plus a no-retrieval control. Each answer is scored for groundedness (claim by claim), coverage of gold facts, lexical diversity, run-to-run semantic consistency, latency and cost. A web dashboard shows the results side by side, backed by a fixed benchmark of 13 questions x 3 trials.

**Decisions I made (flagged so you can overrule them).**

| # | Decision | Why |
|---|---|---|
| D1 | **REVISED (2026-10-23).** Generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-20b` (reasoning effort `low`), both on Groq | The originally planned Llama models are no longer offered. Different model families for generator and judge, as D1 intended. The judge is a reasoning model, so its token budget is escalated on truncation (section 4.8). Groq rotates its catalogue, so a live test checks the configured ids exist. |
| D2 | One OpenAI-compatible client for generator and judge (no LiteLLM, no vendor SDK) | Groq exposes an OpenAI-compatible endpoint, so swapping the judge to GPT-4o-mini is only env vars (section 6.1). Fewer dependencies. |
| D3 | RRR and Self-RAG are **true multi-call pipelines** (3 sequential calls each), not one big prompt | The original report describes them as "within the prompt". Multi-call makes each stage inspectable in the UI and makes latency/cost differences real. The cost is higher latency, and that cost is one of the findings. |
| D4 | Self-RAG here is **prompt-emulated** (labelled "Self-RAG (prompted)" in the UI) | The real Self-RAG fine-tunes a model to emit reflection tokens. We borrow its vocabulary (relevance, support, utility), not its training. Claiming otherwise would be inaccurate. |
| D5 | Groundedness is always scored **against the retrieved top-4 context**, including for the no-retrieval control | Retrieval is deterministic, so the control can be judged against the same passages. The control's low score then means "says things the corpus does not support", which is not strictly "false". Gold-fact coverage supplies the correctness view. |
| D6 | Build **FastAPI + React/Vite** (section 7), with pre-computed "replay" data as the **default view** | Portfolio impact is the stated goal. |
| D7 | **SUPERSEDED.** Originally: develop outside OneDrive so sync cannot corrupt ChromaDB's SQLite files. Later: project in OneDrive with venv and index kept outside it. Now: everything is inside the project folder (see Location above), because sync was never active | Simplest setup, with nothing outside the repository. |
| D8 | **FINAL.** Keep `2308.12113v5.pdf` as a **deliberate retrieval distractor** | It is a point-cloud data-augmentation survey, unrelated to RAG. Indexing it tests whether retrieval stays on-topic. Say so in the docs. |
| D9 | **FINAL.** Hosted demo is **replay by default, with a "Try it live" toggle that reveals a bring-your-own-key (BYOK) input** for the visitor's own Groq key | The operator pays nothing and holds no secret. The key handling rules are in section 7.4. |

**Corpus audit findings that change the plan** (from reading the folder):
1. `2308.12113v5.pdf` is off-topic (see D8).
2. `Vector_embeddings.pdf` and `vector_db_process.pdf` were one-page image exports with **no text layer**. They were excluded from the corpus and are no longer in the repository.
3. **Correction (Phase 0 measured result):** my earlier claim that the LangChain and Weaviate PDFs were garbled was only partly right. The dropped letters ("LngChin") come from **pypdf** mis-decoding the font (55 private-use glyphs in LangChain, 12 in Weaviate); **PyMuPDF extracts both cleanly** and the letter-spacing did not reproduce. PyMuPDF is now the primary extractor, with pypdf as fallback. Residual issue: the LangChain PDF has no text for hyperlink labels (blank slots like "and ."), which is a property of the PDF, not the extractor. No replacement of either file was needed.
4. Real noise found and handled by `ingest/cleaning.py`: running headers, a cookie banner repeated on all 22 Pinecone pages, bibliographies (cut per document via `data/corpus/manifest.yaml`, appendices kept), line-end hyphenation, icon-font glyphs. The distractor survey is cut from its first "Reference" heading to the end, because its appendix holds tables plus a second bibliography.
5. **Security:** `APIKEY.txt` contains live keys. It was never read during this project. It must be revoked and deleted by you, and the new repo does not contain it (a test enforces that).

---

## 1. Folder structure

```
ragbench/
├── README.md
├── ARCHITECTURE_PLAN.md
├── LICENSE
├── pyproject.toml                 # package + deps + ruff/pytest/mypy config
├── .env.example                   # template only; real .env is gitignored
├── .gitignore
│
├── configs/
│   ├── settings.yaml              # chunk size, k, temperatures, trials, seeds
│   ├── pricing.yaml               # $/1M tokens per model (versioned, with date)
│   └── budget.yaml                # free-tier limits used by `ragbench budget`
│
├── data/
│   ├── corpus/                    # the 7 source PDFs + manifest.yaml (cleaning rules)
│   ├── processed/                 # generated: cleaned text + chunks.jsonl (gitignored)
│   └── questions.yaml             # the 10 benchmark questions with gold facts (+ dropped ones)
│
├── src/ragbench/
│   ├── config.py                  # pydantic-settings: env + settings.yaml
│   ├── types.py                   # Chunk, RetrievedChunk
│   ├── questions.py               # load/validate questions.yaml
│   ├── pipeline.py                # run_question: one retrieval, every condition
│   ├── benchmark.py               # resumable benchmark runner -> runs.jsonl
│   ├── analysis.py                # aggregate results, bootstrap CIs, claim sampling
│   ├── report.py                  # summary.md and figures
│   ├── budget.py                  # call/token volume vs free-tier limits
│   ├── cli.py                     # audit | ingest | retrieve | bench | context-support |
│   │                              #   analyze | budget | ask | serve | export-replay
│   ├── ingest/
│   │   ├── manifest.py            # corpus manifest schema
│   │   ├── loaders.py             # PDF -> text (PyMuPDF primary, pypdf fallback)
│   │   ├── cleaning.py            # de-spacing, boilerplate, reference stripping
│   │   ├── chunking.py            # <=600 char chunks, 50 overlap, metadata (doc, page)
│   │   ├── corpus.py              # corpus -> chunks
│   │   └── audit.py               # `ragbench audit` -> docs/corpus_audit.md
│   ├── retrieval/
│   │   ├── embedder.py            # all-MiniLM-L6-v2 wrapper, lazy load + warm-up
│   │   └── store.py               # Chroma (cosine), query(top_k=4) -> RetrievedChunk[]
│   ├── llm/
│   │   ├── client.py              # OpenAI-compatible client: retry, timing, usage, disk cache
│   │   ├── roles.py               # generator / judge clients from env (or a visitor's key)
│   │   └── fake.py                # deterministic fake model for tests and offline use
│   ├── strategies/
│   │   ├── base.py                # Strategy protocol, StrategyResult, call recording
│   │   ├── baseline.py
│   │   ├── structured.py
│   │   ├── rrr.py
│   │   ├── self_rag.py
│   │   ├── no_retrieval.py        # control condition
│   │   └── registry.py            # name -> strategy
│   ├── prompts/                   # exact templates as .txt files, never inline in Python
│   │   ├── baseline.{system,user}.txt, structured.{system,user}.txt, control.{system,user}.txt
│   │   ├── rrr.{identify,synthesize,refine}.txt, selfrag.{draft,critique,refine}.txt
│   │   └── judge.{extract_claims,verify_claims,coverage,context_support}.txt
│   ├── metrics/
│   │   ├── lexical.py             # length, MATTR, TTR
│   │   ├── consistency.py         # semantic + structural consistency across trials
│   │   ├── judge.py               # JSON-validated judge calls with retry
│   │   ├── groundedness.py        # claim extraction + verification
│   │   ├── coverage.py            # gold-fact coverage, context support
│   │   ├── scoring.py             # one answer -> all metrics
│   │   ├── cost.py                # tokens, latency, $ from pricing.yaml
│   │   ├── stats.py               # bootstrap CIs, paired differences, Wilcoxon
│   │   └── validation.py          # judge-vs-human agreement
│   └── api/
│       ├── main.py                # FastAPI app factory, warm-up, CORS
│       ├── routes.py              # /health /strategies /questions /results /live/validate /run
│       ├── schemas.py             # response models (the contract the UI types are generated from)
│       ├── replay.py              # recorded results + static export
│       ├── live.py                # bring-your-own-key live runs (SSE)
│       ├── assemble.py            # records -> API shapes
│       ├── security.py            # key redaction, rate limits, one run per client
│       └── settings.py            # RAGBENCH_* settings
│
├── evaluation/
│   ├── validate_judge.py          # judge vs. human label agreement
│   ├── stage0_retrieval_comparison.py, stage0b_rerank.py   # retrieval investigation (plan 4.10)
│   └── results/                   # committed, so replay works without rerunning
│       ├── runs.jsonl, retrieval.jsonl, context_support.jsonl, run_log.jsonl   # raw
│       ├── summary.json, summary.md, sensitivity_note.md, figures/             # analysis
│       ├── claims_to_label.csv    # claims (and one coverage case) for human labelling
│       └── stage0/                # retrieval-investigation outputs
│
├── scripts/
│   └── mock_api.py                # the real API with a fake model, for trying live mode offline
│
├── ui/                            # React + Vite + TypeScript (section 7)
│   ├── src/{components,hooks,lib,test}/
│   ├── public/replay/             # frozen results for the keyless demo (ragbench export-replay)
│   ├── openapi.json               # API schema; `npm run gen:api` generates src/lib/api-schema.d.ts
│   └── package.json, vite.config.ts, tailwind.config.js, tsconfig.json
│
├── tests/
│   ├── unit/                      # offline, fake models
│   └── integration/               # real embedding model; live Groq checks are opt-in
│
└── docs/
    ├── corpus_audit.md            # generated by `ragbench audit`
    └── screenshots/               # UI screenshots (README and reference)
```

Rules: `src/ragbench` has **no** dependency on `ui/` or `api/`, so the CLI, benchmark and API all share one pipeline. Prompt text lives only in `src/ragbench/prompts/`, never inline in Python. Every LLM call goes through `llm/client.py`, which is the single place that measures latency and tokens.

---

## 2. Pipeline and data flow

```
question
  -> Retriever (embed, Chroma cosine, top-4)      [ONE call, shared by all conditions]
        -> Chunk[4] {text, doc, page, similarity}
  -> for each Strategy in {baseline, structured, rrr, self_rag}:
        Strategy.run(question, chunks, trial_seed) -> StrategyResult
  -> NoRetrieval.run(question) -> StrategyResult            (control)
  -> Metrics: lexical + cost (local), groundedness + coverage (judge LLM)
  -> consistency computed across the N trials of the same (question, strategy)
```

```python
# strategies/base.py (interface only)
@dataclass
class StrategyResult:
    strategy: str
    answer: str  # final answer text. Metrics are computed on this.
    trace: list[Stage]  # intermediate stages (RRR notes/draft, Self-RAG draft/critique)
    calls: list[CallRecord]  # per-call latency_s, prompt_tokens, completion_tokens, model
```

A strategy is a class with `name` and `run(question, chunks, *, temperature, seed) -> StrategyResult`. Adding a fifth strategy means adding one file and one registry line.

**Self-RAG output contract.** The user-visible output is `answer` (from the refine stage) followed by the critique. The `StrategyResult.answer` that metrics score is the **refined answer only**. The critique is kept in `trace` and shown in the UI, because scoring claims inside the critique would measure the critique, not the answer.

**Generation settings** (`settings.yaml`): `top_k=4`, `chunk_chars=600`, `chunk_overlap=50`, generator `qwen/qwen3.8-27b` (`reasoning_effort=none`), temperature `0.5` for all benchmark trials, `max_tokens=700` per call, trials `N=3`. Temperature must be above 0 for the consistency metric to be meaningful. Judge temperature is `0`.

---

## 3. Prompt templates (exact text)

Shared context formatting (all retrieval strategies):

```
[1] {chunk_1_text}

[2] {chunk_2_text}

[3] {chunk_3_text}

[4] {chunk_4_text}
```

Chunks are numbered for reference only. No strategy is required to cite, so citation behaviour does not confound the comparison.

### 3.1 Baseline (direct extractive answer): 1 call

`baseline.system.txt`
```
You are a question-answering assistant. Answer the question using ONLY the provided context.
Give a direct, concise answer in one or two sentences, taken as closely as possible from the context.
Do not add background, explanation, or formatting.
If the context does not contain the answer, reply exactly: "The provided context does not contain this information."
```
`baseline.user.txt`
```
Context:
{context}

Question: {question}

Answer:
```

### 3.2 Structured (headed sections): 1 call

`structured.system.txt`
```
You are a question-answering assistant. Answer the question using ONLY the provided context.
Do not use outside knowledge. Format your response in Markdown with exactly these three sections, in this order:

## Summary
One or two sentences that directly answer the question.

## Key Points
3 to 5 bullet points. Each bullet states one distinct fact from the context.

## Elaboration
One short paragraph that connects the key points and adds relevant detail from the context.

If the context does not contain the answer, write under "## Summary": "The provided context does not contain this information." and leave the other two sections as "N/A".
```
`structured.user.txt`
```
Context:
{context}

Question: {question}
```

### 3.3 RRR (Retrieve-Read-Refine): 3 calls

Stage 1 ("Read: identify key information"), `rrr.identify.txt`
```
You are reading retrieved passages in order to answer a question.
Using ONLY the passages below, list the facts that are relevant to the question.
For each fact, give the passage number in brackets, e.g. "- Fact text [2]".
Ignore passages or facts that are irrelevant. Do not answer the question yet.
If no passage is relevant, output exactly: NO RELEVANT INFORMATION

Passages:
{context}

Question: {question}

Relevant facts:
```

Stage 2 (synthesize across passages), `rrr.synthesize.txt`
```
You are synthesizing an answer from extracted facts.
Write a draft answer to the question by combining the relevant facts below across passages.
Merge facts that overlap, and note where passages agree or where information is missing.
Use ONLY information found in the facts or the passages. Do not add outside knowledge.
If the facts say NO RELEVANT INFORMATION, reply exactly: "The provided context does not contain this information."

Question: {question}

Relevant facts:
{stage1_output}

Passages (for verification):
{context}

Draft answer:
```

Stage 3 (refine for coherence; the final answer), `rrr.refine.txt`
```
You are editing a draft answer for coherence.
Rewrite the draft so it reads as one natural, well-organized answer: remove repetition,
fix awkward transitions, and order the ideas logically.
Do NOT add any claim that is not supported by the passages. Remove any claim in the draft that the passages do not support.
Output only the final answer, with no preamble and no commentary.

Question: {question}

Passages:
{context}

Draft answer:
{stage2_output}

Final answer:
```

### 3.4 Self-RAG (prompted): generate -> self-critique -> refine, 3 calls

The critique vocabulary is borrowed from the Self-RAG paper's reflection tokens (IsRel, IsSup, IsUse), emulated in text.

Stage 1 (generate), `selfrag.draft.txt`
```
Answer the question using ONLY the provided context. Be thorough: explain the reasoning that leads from
the context to your answer. Do not use outside knowledge.
If the context does not contain the answer, reply exactly: "The provided context does not contain this information."

Context:
{context}

Question: {question}

Initial answer:
```

Stage 2 (self-critique), `selfrag.critique.txt`
```
You are a strict reviewer. Critique the initial answer against the context. Be specific and brief.

Context:
{context}

Question: {question}

Initial answer:
{stage1_output}

Produce the critique in exactly this format:

Passage relevance:
[1] Relevant | Irrelevant
[2] Relevant | Irrelevant
[3] Relevant | Irrelevant
[4] Relevant | Irrelevant

Factual accuracy: Fully supported | Partially supported | Not supported
Unsupported or incorrect claims: <quote each, or "None">
Completeness: Complete | Incomplete
Missing information from the context: <list, or "None">
Utility (1-5): <integer>
Revision instructions: <one to three concrete edits, or "No changes needed">
```

Stage 3 (refine), `selfrag.refine.txt`
```
Revise the initial answer by applying the critique.
Remove or correct every claim flagged as unsupported or incorrect, add any missing information that is present in the context,
and keep everything that was supported. Use ONLY the context. Output only the revised answer, with no preamble.
If the critique says "No changes needed", repeat the initial answer.

Context:
{context}

Question: {question}

Initial answer:
{stage1_output}

Critique:
{stage2_output}

Revised answer:
```
The UI renders the final output as: **Answer** (stage 3), then a collapsible **Self-critique** (stage 2), then the **Initial answer** (stage 1).

### 3.5 Control: no retrieval: 1 call

`control.system.txt`
```
You are a question-answering assistant. No reference documents are provided.
Answer the question from your own knowledge in one or two sentences.
If you are not sure, say so rather than guessing.
```
`control.user.txt`
```
Question: {question}

Answer:
```
It mirrors Baseline's brevity so the only difference is the absence of context.

### 3.6 Judge prompts (used by `metrics/`)

`judge.extract_claims.txt`
```
Decompose the answer into atomic factual claims.
Rules:
- Each claim is one short, self-contained sentence. Resolve pronouns so it makes sense alone.
- Include only factual assertions about the world or the documents.
- Exclude opinions, meta-commentary about the answer or the context, hedges ("the context does not say..."), and restatements of the question.
- If the answer makes no factual claims (for example it declines to answer), return an empty list.

Question: {question}
Answer: {answer}

Return JSON only: {"claims": ["...", "..."]}
```

`judge.verify_claims.txt`
```
You are checking claims against source passages. Use ONLY the passages. Ignore your own world knowledge.

Passages:
{context}

Claims:
{numbered_claims}

For each claim choose one label:
- "supported": the passages state it explicitly or it follows directly from them.
- "contradicted": the passages state something incompatible with it.
- "unsupported": the passages neither state nor imply it (even if it is true in the real world).

Return JSON only:
{"verdicts": [{"claim_id": 1, "label": "supported|contradicted|unsupported", "evidence_passage": <1-4 or null>, "reason": "<=25 words"}]}
```

`judge.coverage.txt`
```
Decide whether the answer states each gold fact. A paraphrase counts; a vague or partial mention does not.

Question: {question}
Answer: {answer}

Gold facts:
{numbered_gold_facts}

Return JSON only: {"coverage": [{"fact_id": 1, "covered": true|false}]}
```

All judge calls use `response_format={"type":"json_object"}`, temperature 0, pydantic validation, and up to 2 repair retries.

---

## 4. Metric definitions

Notation: for one answer `a` (final answer text only), `W(a)` is its word list (lowercased, tokenised on `\w+`), `S(a)` its sentence list (`pysbd`), `n = |W(a)|`.

### 4.1 Lexical metrics (carried over from the report)
- **Length:** `chars = len(a)`, `words = n`, `sentences = |S(a)|`.
- **Avg word length** = `sum(len(w)) / n`. **Words per sentence** = `n / |S(a)|`.
- **Unique vocabulary** `U = |set(W(a))|`.
- **TTR** = `U / n` (reported for comparison with the original, which called it "diversity").
- **MATTR (headline diversity)** = mean of `U_window / L` over all sliding windows of `L = 25` words. If `n < L`, return `null` and mark "too short". MATTR fixes the report's artefact where the 6-word Baseline scored a perfect 1.0.

### 4.2 Groundedness (the primary quality metric)
Per answer, with the judge:
1. **Extract** claims `C = {c_1..c_m}` (`judge.extract_claims`).
2. **Verify** each claim against the 4 retrieved passages (`judge.verify_claims`) giving labels in `{supported, unsupported, contradicted}`.
3. Scores:
   - **Groundedness** `G = |supported| / m`   (range 0 to 1)
   - **Hallucination rate** `H = (|unsupported| + |contradicted|) / m = 1 - G`
   - **Contradiction rate** `X = |contradicted| / m` (reported separately because it is the worst failure)
4. **Abstention:** if `m = 0` the answer is flagged `abstained = true`, `G` is undefined and excluded from the mean. Abstention rate is reported per strategy. For the out-of-corpus question (Q13), a correct abstention counts as success (section 5).

### 4.3 Coverage (guards against brevity gaming)
Groundedness alone rewards short answers, since a 6-word answer can be perfectly grounded and useless. So each question has gold facts `F = {f_1..f_p}` (section 5) and
- **Coverage** `C_f = |covered| / p` via `judge.coverage`.
- **Faithful-Coverage score** (headline composite) `FCS = 2 * G * C_f / (G + C_f)` (harmonic mean; `0` if either is 0). A strategy needs to be both grounded and complete to score well.

### 4.4 Consistency (semantic, replacing the report's undefined metric)
For a fixed (question, strategy), take the `N = 3` trial answers `a_1..a_N`, embed each with all-MiniLM-L6-v2 (L2-normalised), and compute
`Consistency(q, s) = mean over i<j of cos(e_i, e_j)`   (3 pairs for N=3; **the final benchmark uses N=2, i.e. a single pair per question**, see section 4.9).
The strategy-level score is the mean over questions, with a bootstrap CI. A secondary **structural consistency** `SC = 1 - CV(word_count across trials)` (clipped to 0 to 1) is reported next to it, because the original report's consistency appeared to track format stability. Note: very short answers naturally score high, so interpret next to coverage.

### 4.5 Latency and cost
Recorded per call (and summed per strategy run), by `llm/client.py`:
- `latency_s` = `time.perf_counter()` around the HTTP call (excluding retry back-off; `n_retries` is stored separately). Strategy latency = sum of its sequential calls.
- `prompt_tokens`, `completion_tokens` from the API `usage` field.
- `cost_usd = (prompt_tokens * p_in + completion_tokens * p_out) / 1e6`, with prices from `configs/pricing.yaml` (stamped with a date; Groq prices to be confirmed at build time).
- Derived: tokens per call, calls per strategy, and **cost per grounded claim** = `cost_usd / |supported|`.
- Judge cost is tracked separately and never added to a strategy's cost.

### 4.6 Aggregation and statistics
- The unit of analysis is the **question** (13 of them), not the individual trial, since trials of one question are correlated.
- Per (strategy, metric): average trials within a question, then average across questions.
- **95% CIs:** cluster bootstrap, `B = 10,000` resamples of questions with replacement.
- **Paired differences vs. Baseline and vs. Control:** bootstrap CI of the per-question differences. Add a Wilcoxon signed-rank test as a secondary check. State plainly that n = 13 limits power, so CIs are the main evidence.
- **Retrieval diagnostics** (from the report): similarity by rank and chunk lengths, but computed per question across all 13, not from a single query.

### 4.7 Judge validation (required before trusting the numbers)
Hand-label ~30 claims across strategies (supported/unsupported/contradicted) in `evaluation/results/claims_to_label.csv`. `validate_judge.py` reports agreement (accuracy and Cohen's kappa) between the judge and the human labels, and the result is published in `docs/methodology.md`. Also report the judge-model-swap check: rerun the verification on a 50-claim subset with the alternate judge (e.g. 8B) and report agreement.

---

### 4.8 Projected benchmark load against free-tier limits (`ragbench budget`)

Limits (per model, Groq docs and observed headers): 30 requests/min, 1,000 requests/day, **8,000 tokens/min, 200,000 tokens/day**. Token profiles are measured from live calls (`configs/budget.yaml`); a 25% safety margin covers longer multi-hop answers and retried judge calls. Judging fewer trials than are generated is a real saving, because consistency needs all trials but no judge.

| scenario | generator tokens (% of daily) | judge tokens (% of daily) | days needed |
|---|---|---|---|
| 13 questions x 3 trials, judge all 3 | 359,539 (180%) | 511,256 (256%) | 3 |
| 13 x 3, judge 2 of 3 trials | 359,539 (180%) | 340,838 (170%) | 2 |
| 13 x 3, judge 1 of 3 trials | 359,539 (180%) | 170,419 (85%) | 2 |
| 13 x 2, judge 1 of 2 trials | 239,692 (120%) | 170,419 (85%) | 2 |
| **10 x 2, judge 1 of 2 trials** | 183,285 (92%) | 128,512 (64%) | **1** |
| **7 x 3, judge 2 of 3 trials** | 190,316 (95%) | 173,212 (87%) | **1** |

Generation alone is the binding constraint: even with **no** margin it is ~288k tokens (144% of the daily cap) for 13 questions x 3 trials, because RRR and Self-RAG resend the full context on each of their 3 calls (77% of generation tokens). The full design therefore needs two days of quota (resumable, since every call is cached) or a reduced scope. Always confirm the real limits at console.groq.com/settings/limits; the docs page labels its table "Developer plan" base limits and the free-plan tab was not readable.

### 4.9 Final benchmark design and its statistical implications (decided 2026-10-23)

The planned 13 questions x 3 trials does not fit the free tier (section 4.8). The benchmark that was actually run is **10 questions x 2 trials, with only trial 0 judged**:

| | planned | **run** |
|---|---|---|
| questions | 13 (8 lookup, 4 multi-hop, 1 out-of-corpus) | **10** (5 lookup, 4 multi-hop, 1 out-of-corpus) |
| trials per question and strategy | 3 (originally 5) | **2** |
| trials judged (groundedness, coverage) | all | **1 of 2** |
| answers generated / judged | 195 / 195 | **100 / 50** |

**Which questions.** All 4 multi-hop questions and the out-of-corpus question were kept, because a live calibration showed lookup questions saturate at groundedness 1.0 for every strategy, so the multi-hop questions are where strategies are expected to differ. The 8 lookup questions were cut to 5 by dropping the 3 most redundant, using question-embedding similarity and top-4 chunk overlap as evidence:

- **Q01** (RAG memory components): same paper as kept Q02, its top-4 mixes in Self-RAG chunks, and its content is covered by kept multi-hop Q09/Q12.
- **Q04** (why Self-RAG retrieves on demand): same paper as kept Q03, its top-4 mixes in Lewis chunks and shares a chunk with Q09, and it is the core of multi-hop Q09.
- **Q07** (What is Weaviate?): the most definitional question over a 6-page document, with essentially one gold fact, so coverage would be binary and uninformative.

Kept lookups: Q02, Q03, Q05, Q06, Q08 (one per source document except Weaviate, which is no longer a question target but stays in the corpus). The measured data showed no lookup pair sharing retrieved chunks, so the redundancy is by paper and question type, not literal overlap. `data/questions.yaml` records the dropped questions and reasons.

**Statistical implications (explicit limitations).**
- **Consistency rests on 2 samples per question**, i.e. one pair of answers, instead of 3 samples (three pairs). Each per-question value is therefore noisier, and consistency differences between strategies are indicative only.
- **Groundedness, coverage and FCS rest on one judged answer per question and strategy**, so within-question sampling variance is not averaged out.
- **9 answerable questions** (5 lookup, 4 multi-hop) feed the quality metrics, so bootstrap intervals are wide, per-type comparisons use only 5 and 4 questions, and the Wilcoxon test is exploratory. Differences with overlapping intervals should not be called real.
- Only one question tests abstention, with 2 samples per strategy, so abstention rates are coarse (0, 0.5 or 1).
- The Weaviate document is never a question target, so nothing tests retrieval from it.
These limitations are repeated in `evaluation/results/summary.md`, which is generated with them stated at the top.

### 4.10 Retrieval investigation: closed (2026-10-23)

**Why it was opened.** The benchmark showed that cross-document questions were limited by retrieval, not generation: the retrieved top-4 held on average 0.74 of the gold facts (judge-based), only 0.33 on Q11 and 0.50 on Q03 and Q09. Every candidate fix was tested against criteria fixed in advance, using judge-based context recall on Q03, Q08, Q09, Q11 and Q12 (3 to 4 gold facts each).

| candidate | where tested | result | outcome |
|---|---|---|---|
| LLM query decomposition (sub-queries retrieved separately, merged and deduplicated, <= 6 chunks) | Stage 0 | multi-hop recall 0.61 vs 0.58 for flat k=8: +0.03, bar was +0.15. Fixed Q11 (+1 fact) but not the within-document gaps; on Q12 it swapped one fact for another | **bar not cleared** (the other three criteria passed) |
| Flat k=8 | Stage 0 | lookup recall 0.58 -> 0.88, within-document 0.56 -> 0.83, but Q11 stayed 0.33; doubles the context (about 215k generator tokens for the benchmark, over the 200k daily free-tier limit) | not adopted: unaffordable; it served as the reference the bars were set against |
| Document-balanced k=6 | Stage 0 | multi-hop 0.47; Q11 fell to 0.00 | rejected |
| Cross-encoder rerank, pool 20 -> 4 | Stage 0b | within-document 0.67 vs target 0.765; Q08 reached 1.00, Q03 and Q09 did not move | **bar not cleared** |
| Cross-encoder rerank, pool 20 -> 6 | Stage 0b | within-document 0.75 vs target 0.765 (70% of k=8's gain, one judged fact short); about 190k generator tokens, no safety margin | **bar not cleared** (narrow miss) |

**Decision.** None of the candidates cleared its adoption bar, so none is adopted. The two failure modes are accepted as **documented limitations** instead of being optimised further:
1. **Q11-type cross-document gaps:** a single query embedding is dominated by one topic, so evidence from the second document is missed.
2. **Q03-type within-document chunk-ranking gaps:** the right paper is retrieved but not the chunk that holds a specific fact (for example the table defining Self-RAG's reflection tokens).

**Why stop here.**
- **Retrieval quality is not the variable this project measures.** It compares generation strategies on *identical* retrieval, and that holds whatever the retrieval quality is. The limitation is visible and reported (context recall per question in `summary.md`; refusals on retrieval-limited questions are read as correct behaviour for the context given).
- **The evidence cannot support more tuning.** Each question has 3 to 4 gold facts, so every difference between variants is one fact. Layering further fixes (for example reranking plus decomposition) would add variables and risk fitting five test questions.
- **The conclusions do not depend on the retrieval-limited question.** The Q11-excluded sensitivity analysis (`summary.md`) leaves every conclusion unchanged: the strategy ranking and the retrieval-versus-control gap hold, and RRR's lead over Baseline stays suggestive rather than established.
- **No change means no rerun.** The existing benchmark results stand.

**What would reopen it.** A larger quota or budget, a larger gold-fact set that can resolve small effects, or a change of goal to optimising retrieval. The one control never run is flat dense k=6 (about 7k judge tokens), without which reranking's value beyond simply retrieving more is unshown. Any future retrieval change must first confirm that the out-of-corpus question (Q13) is still refused: that check needs generator tokens and was not run for the reranked variants.

## 5. Evaluation question set (original 13; the benchmark ran the 10 in section 4.9)

Corpus: Lewis et al. (RAG), Asai et al. (Self-RAG), Singhvi et al. (DSPy Assertions), the Pinecone vector DB article, Weaviate docs, LangChain overview. Question types: **L** = single-source lookup, **M** = multi-hop (needs 2+ documents), **N** = unanswerable from the corpus (tests abstention).

**Important:** the "gold facts" below are my *draft* expectations from knowledge of these papers. Phase 4 must verify each against the source text and record page numbers in `questions.yaml` before any number is reported. Items marked (verify) I am less certain about.

| ID | Type | Source(s) | Question | Draft gold facts |
|---|---|---|---|---|
| Q01 | L | Lewis | What are the parametric and non-parametric memory components of the RAG model? | Parametric = pre-trained seq2seq transformer (BART). Non-parametric = dense vector index of Wikipedia, accessed by a pre-trained neural retriever (DPR). |
| Q02 | L | Lewis | What is the difference between RAG-Sequence and RAG-Token? | RAG-Sequence uses the same retrieved document for the whole generated sequence. RAG-Token can use a different document for each token. |
| Q03 | L | Self-RAG | Which reflection tokens does Self-RAG use, and what does each one assess? | Retrieve (whether to retrieve), IsRel (passage relevance), IsSup (whether the output is supported by the passage), IsUse (overall usefulness). |
| Q04 | L | Self-RAG | Why does Self-RAG retrieve on demand instead of always retrieving a fixed number of passages? | Indiscriminate retrieval can hurt versatility and lead to unhelpful generations. On-demand retrieval is adaptive. (verify exact wording) |
| Q05 | L | DSPy | In DSPy Assertions, what is the difference between Assert and Suggest? | Assert is a hard constraint: on failure it retries with backtracking and raises an error if still violated. Suggest is a soft constraint: it retries but continues if unmet. (verify) |
| Q06 | L | Pinecone | What is a vector database and how does it differ from a traditional database? | Stores/indexes vector embeddings for similarity search. Optimised for nearest-neighbour search on high-dimensional data rather than exact matches. (verify) |
| Q07 | L | Weaviate | What is Weaviate? | An open-source, AI-native vector database. |
| Q08 | L | LangChain | What is LangChain and what does it help developers build? | Framework for building LLM-powered applications/agents. (verify against the cleaned text) |
| Q09 | M | Lewis + Self-RAG | How do the original RAG model and Self-RAG differ in deciding when and what to retrieve? | RAG retrieves for every input (fixed top-k via DPR). Self-RAG decides adaptively whether to retrieve, using Retrieve token, and checks relevance (IsRel). |
| Q10 | M | Self-RAG + DSPy | Both Self-RAG and DSPy Assertions let a system improve an output after a first attempt. How do the mechanisms differ? | Self-RAG: trained model emits reflection tokens to critique/select outputs. DSPy Assertions: programmer-declared constraints checked at runtime, triggering backtracking/retry with feedback. |
| Q11 | M | Lewis + Pinecone | How does the vector search described in the Pinecone article relate to the retriever used in the original RAG model? | RAG's retriever (DPR) encodes queries and documents as dense vectors and finds nearest neighbours via MIPS; Pinecone describes the same similarity-search over an index as the vector DB's core function. (verify) |
| Q12 | M | Lewis + Self-RAG + DSPy | Which approaches in these documents aim to reduce factual errors in language models, and how does each work? | RAG: ground generation in retrieved documents. Self-RAG: adaptive retrieval + self-critique via reflection tokens. DSPy Assertions: enforce constraints with backtracking. |
| Q13 | N | (none) | What results did Self-RAG report on the Llama 3 model family? | **Expected: abstain.** Self-RAG (Oct 2023) predates Llama 3, so the corpus cannot answer. Gold facts: none. Score = 1 if the answer abstains, else 0 (also tracked via groundedness). |

Plan notes:
- 8 lookup, 4 multi-hop, 1 unanswerable. Q13 gets a separate "abstention accuracy" column instead of coverage.
- `questions.yaml` schema: `id, type, question, gold_facts[], sources[{doc,page}], expected_abstain`.
- The distractor paper (point-cloud survey) has no question; its value is as a retrieval distractor. Track how often it appears in top-4 (expected: ~0).
- Budget (revised, measured): see section 4.8. 13 questions x 5 conditions x 3 trials = 195 answers; about 440 generation calls and 690 judge calls, roughly 360k generator tokens and 510k judge tokens including a 25% margin. That does **not** fit one day of the free tier (200k tokens/day per model); options are in section 4.8.

---

## 6. LLM layer

### 6.1 Swappable roles
Two roles, each configured only by environment variables:

```
# .env.example
GENERATOR_BASE_URL=https://api.groq.com/openai/v1
GENERATOR_API_KEY=            # Groq key
GENERATOR_MODEL=qwen/qwen3.8-27b

JUDGE_BASE_URL=https://api.groq.com/openai/v1
JUDGE_API_KEY=                # may be the same Groq key
JUDGE_MODEL=openai/gpt-oss-20b   # reasoning model; see JUDGE_REASONING_EFFORT

# To swap the judge to GPT-4o-mini, change ONLY these three lines:
# JUDGE_BASE_URL=https://api.openai.com/v1
# JUDGE_API_KEY=sk-...
# JUDGE_MODEL=gpt-4o-mini
```
`llm/client.py` exposes `LLMClient(role, api_key=None)` with one method `chat(messages, temperature, max_tokens, json_mode=False, seed=None) -> LLMResponse`. Nothing else in the code mentions a vendor. The optional `api_key` argument **overrides the env key for that client instance only**. This is how BYOK works (section 7.4): the API builds a short-lived client from the visitor's key, and in hosted mode the server has no `GENERATOR_API_KEY` or `JUDGE_API_KEY` at all. In BYOK mode both roles use the visitor's Groq key. `pricing.yaml` has an entry per model id, and an unknown model logs a warning and records tokens with `cost_usd = null`.

Startup guard: if `JUDGE_MODEL == GENERATOR_MODEL`, log a loud warning, since the same model judging its own output biases the scores.

### 6.2 Reliability
- `tenacity` retries on 429/5xx with exponential back-off.
- A semaphore caps concurrency (default 4) to respect Groq limits.
- On-disk cache keyed on `(role, model, messages, temperature, max_tokens, trial_id)`; the trial id is part of the key so five trials are five distinct samples.
- Every result row stores model ids, settings hash and git commit, so results are traceable.

---

## 7. UI recommendation

### Option A: Streamlit
- **Pros:** fastest (1 to 2 days); Python only; built-in widgets and Plotly charts; trivial to deploy to Streamlit Cloud.
- **Cons:** recognisably "generic Streamlit"; limited layout control, animation and theming; page reruns make the live four-way streaming view clunky; hard to make visually distinctive. It signals "data-science demo" rather than "engineered product".

### Option B: FastAPI + React/Vite/TypeScript (recommended)
- **Pros:** full design control, which is what makes a portfolio piece memorable; true streaming (SSE) so the four strategy cards fill in live and race each other; the same typed API serves the CLI, the benchmark and the UI; shows full-stack skill (typed contract, API design, state management); static build can be hosted free with replay data.
- **Cons:** about 3 to 5x the build time of Streamlit; two codebases; deployment of the backend needs a host if you want live mode.

### Recommendation: Option B, built in a way that contains the risk
Reasons: you said the goal is "visually impressive, not generic", and only B can deliver that. The extra time is contained because (1) the Python core is built and tested first and is UI-independent, so Streamlit would only have saved UI time, and (2) the API contract is small (6 endpoints). **Replay mode** is the key portfolio feature and the **default view**: `ui/public/replay/` holds the frozen benchmark results, so a recruiter can explore every question, answer, metric and chart on a static host with no key and no backend. **Live mode** is opt-in and bring-your-own-key (section 7.4).

### 7.4 Replay default, "Try it live" toggle, and BYOK flow

**Modes**
| | Replay (default) | Live (opt-in) |
|---|---|---|
| Data source | Frozen `evaluation/results/` JSON served as static files | `POST /api/run`, executed by the backend with the visitor's key |
| Needs backend | No | Yes (retrieval + strategies run server-side) |
| Needs a key | No | Yes, the visitor's own Groq key |
| Questions | The 13 benchmark questions | Any of the 13, or free text |
| Metrics shown | Everything: all 3 trials, consistency, CIs, benchmark heatmap | Single trial per condition: groundedness, coverage, diversity, latency, tokens, cost. **No consistency or CIs** (they need N trials). Benchmark means are shown beside live numbers for context. |

**Flow**
1. The page loads in Replay. A persistent header badge reads **REPLAY**, and no key field is visible.
2. A **"Try it live"** switch in the header reveals a key panel. Flipping it back to Replay hides the panel and **wipes the key from memory**.
3. The panel has one password-type input (`gsk_...`), a short plain-language privacy note, a link to console.groq.com/keys, and a **"Verify key"** button. Verify calls `POST /api/live/validate`, which makes one cheap authenticated Groq call (list models) and also checks that the generator and judge model ids are available to that key. Result: valid / invalid / rate-limited / model-unavailable.
4. Once verified, the badge switches to **LIVE**, the question picker gains a free-text box, and "Run all 5 conditions" is enabled. A checkbox **"Also score groundedness (uses ~6 extra judge calls per strategy)"** lets visitors skip the judge to save their quota; it is on by default.
5. Live runs stream via SSE. Each result card is stamped **LIVE** and visually distinct from replay data. The user can switch back to Replay at any time without losing the benchmark views.
6. Errors are shown in plain language: Groq 401 (bad key), 429 (their rate limit, with retry hint), model unavailable, network failure. The key is never echoed in any message.

**Key-handling rules (a hard requirement, covered by tests)**
- **Never stored client-side:** the key lives only in React component state (memory). No `localStorage`, `sessionStorage`, cookies, IndexedDB, URL parameters, or browser autofill (`autocomplete="off"`). Closing or reloading the tab clears it.
- **Transport:** sent only over HTTPS, in a custom request header (`X-Groq-Api-Key`) on each live request, never in the URL or body. The backend refuses non-HTTPS key transport in hosted mode.
- **Server never persists it:** the key exists only as a local variable for the duration of one request, passed into a per-request `LLMClient(api_key=...)`. It is not written to disk, the cache, the database, results files or the replay data.
- **Server never logs it:** a logging filter redacts `gsk_...`-shaped strings and the key header from all logs; request headers and bodies are not logged; exception messages are scrubbed before being returned; uvicorn access logs are configured without headers. A unit test asserts a fake key never appears in captured logs or responses.
- **Cache isolation:** the on-disk LLM cache is **disabled for BYOK requests** (otherwise one visitor's result could be served to another, and keys must not influence cache keys).
- **Honest disclosure:** the key panel states that the key passes through the demo's backend to reach Groq, is used for that run only, and is never stored or logged; the source is open so this can be verified; and visitors may use a throwaway key and revoke it afterwards. Visitors can also run the project locally with their own `.env`.
- **Abuse limits (since the operator pays nothing for LLM calls but still pays for compute):** per-IP rate limit and a concurrency cap on `/api/run`, maximum question length (~500 chars), maximum one run in flight per client, request timeouts, and CORS restricted to the UI origin. A browser Content-Security-Policy restricts script and connection origins.
- **No server-side key in hosted mode:** if `GENERATOR_API_KEY` is unset and no key header is present, `/api/run` returns 401 with a message pointing to the key panel.

### UI design (what gets built)
- **Question picker:** the 13 benchmark questions (tagged lookup / multi-hop / out-of-corpus), plus a free-text box in Live mode only.
- **Retrieved context panel:** the 4 chunks with similarity scores and source/page, shared by all strategies, shown once.
- **Five answer cards side by side** (Baseline, Structured, RRR, Self-RAG, Control): rendered Markdown; for RRR and Self-RAG, an expandable stage trace (key facts / draft / critique). Live mode streams each card as it completes.
- **Claim-level groundedness view:** hover an answer to see its extracted claims coloured green (supported), amber (unsupported), red (contradicted), with the evidence passage highlighted.
- **Metric strip per card:** groundedness, coverage, FCS, latency, tokens, cost.
- **Comparison charts:** grouped bars with 95% CI whiskers; radar chart across (groundedness, coverage, consistency, diversity, speed, cost); a scatter of cost vs. FCS; a "retrieval impact" delta chart (strategy minus control); latency breakdown by stage.
- **Benchmark tab:** full 13-question heatmap (strategy x question x metric), with lookup vs. multi-hop split.
- **Method tab:** metric definitions, judge validation results, limitations.
- Visual direction: dark-first theme with a light toggle; one accent colour per strategy used consistently across all views; Inter + a monospace for numbers; subtle Framer Motion transitions; fully responsive.

### API contract
`GET /api/questions`, `GET /api/results/summary`, `GET /api/results/{question_id}` (the first three also exist as static JSON for Replay), `POST /api/live/validate` (header `X-Groq-Api-Key`), `POST /api/run` (header `X-Groq-Api-Key`; SSE stream of `retrieval`, `strategy_started`, `stage`, `strategy_done`, `metrics`, `done` events), `GET /api/health`. Response models are pydantic in `api/schemas.py`, with TypeScript types generated from the OpenAPI schema so the UI cannot drift.

---

## 8. Dependencies

**Python (3.11+)**
| Purpose | Packages |
|---|---|
| Retrieval | `chromadb`, `sentence-transformers` (pulls PyTorch; use the CPU wheel), `numpy` |
| Ingestion | `pypdf`, `pymupdf` (fallback extractor), `langchain-text-splitters` (splitter only, no full LangChain) |
| LLM access | `openai` (OpenAI-compatible SDK used for Groq and for a later GPT-4o-mini swap), `tenacity`, `diskcache` |
| Config/schemas | `pydantic`, `pydantic-settings`, `pyyaml`, `python-dotenv` |
| Metrics/stats | `pysbd` (sentence splitting), `scipy`, `pandas`, `scikit-learn` (Cohen's kappa) |
| API | `fastapi`, `uvicorn[standard]`, `sse-starlette` |
| Reporting | `matplotlib` (static figures for the README) |
| Dev | `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `pre-commit` |

**Frontend:** `react`, `react-dom`, `vite`, `typescript`, `tailwindcss`, `shadcn/ui` (Radix primitives), `recharts` (or `visx` if custom charts are needed), `framer-motion`, `@tanstack/react-query`, `react-markdown`, `lucide-react`, `openapi-typescript` (generated API types), `vitest`.

Removed from the original stack: Gradio, PyPDF2 (deprecated), faiss-cpu (replaced by Chroma), `groq` SDK, `llama-cpp-python`, `google-generativeai`, and the full LangChain package.

---

## 9. Phased build plan

Each phase ends with something runnable and an acceptance check.

| Phase | Build | Done when |
|---|---|---|
| **0. Setup and corpus audit** | New repo; `pyproject.toml`, `.gitignore`, `.env.example`; **rotate the leaked keys**; set the original coursework files aside; run extraction on every PDF and decide include/exclude; build `cleaning.py` | Every included PDF yields clean readable text (spot-checked); `pip install -e .` works; no secrets tracked |
| **1. Ingestion and retrieval** | Chunking (600/50 chars) with doc/page metadata; embedder; Chroma store (cosine); `ragbench ingest` and `ragbench retrieve "q"` | Query returns 4 chunks with similarities; chunk length stats match the ~500 to 600 target; unit tests pass; the distractor paper does not appear for RAG questions |
| **2. LLM client and strategies** | `LLMClient` with timing, usage, retry, cache; the 5 strategy modules and prompt files; `ragbench ask "q"` printing all five answers and traces | One question runs through all conditions; per-call latency/tokens logged; integration tests pass with a fake LLM |
| **3. Metrics and judge** | Lexical, consistency, cost, groundedness, coverage, stats; judge env-var swap | Metrics unit-tested on hand-made examples (e.g. known TTR/MATTR values, known cosine pairs); judge returns valid JSON on a sample; swapping `JUDGE_MODEL` needs no code change |
| **4. Question set and benchmark** | Author and **verify** gold facts in `questions.yaml`; `ragbench bench` (13 x 5 x 3 as planned; 10 x 5 x 2 as run) with caching and resume; `ragbench analyze` (CIs, paired differences); human-label 30 claims and run `validate_judge.py` | Full benchmark completes and is resumable; `summary.json` and figures generated; judge agreement reported; results reviewed for sanity (e.g. control scores lower on groundedness) |
| **5. API** | FastAPI routes, SSE streaming, replay-data endpoint, pydantic schemas, CORS, basic rate limiting, no key exposure | `curl` can run a question and stream events; OpenAPI docs render; API tests pass |
| **6. Frontend** | Vite app: question picker, context panel, five answer cards, claim highlighting, metric strips, comparison charts, benchmark heatmap, method tab; replay mode first, then live mode | Replay mode works fully offline from static files; live mode streams; looks right on desktop and mobile |
| **7. Polish and release** | README with GIF/screenshots and headline results; the app's Method page and README limitations (honest limitations); architecture diagram; CI (ruff, mypy, pytest, frontend build); static deploy of the UI | A stranger can clone, read the README, and reproduce the benchmark or browse the hosted replay in minutes |

Suggested order of effort: 0-1 about 1 day, 2-3 about 2 days, 4 about 1 to 2 days (gold-fact authoring and labelling are the slow part), 5 about 1 day, 6 about 3 to 4 days, 7 about 1 day.

---

## 10. Risks and limitations to document

1. **Judge reliability:** an LLM judge is imperfect. Mitigations: a different model family, claim-level decomposition, temperature 0, human validation, an alternate-judge agreement check. Report the agreement figures, not just the scores.
2. **Small n:** 13 questions limits statistical power. CIs are shown, and no claims of significance are made without them.
3. **Self-RAG is emulated:** do not compare numbers with the paper's.
4. **Gold-fact subjectivity:** facts were authored by the project author. Publish them in the repo.
5. **Groq rate limits and model churn:** caching and back-off for limits; pin and verify model IDs, and record them in every results row.
6. **Control's groundedness is context-relative** (D5): a correct answer from memory still scores "unsupported". This is why coverage is reported alongside.
7. **Consistency vs. brevity:** short, extractive answers are naturally more consistent. Always read consistency next to coverage.
8. **Cost figures** depend on Groq's pricing at the run date; they are stamped in `pricing.yaml`.
9. **BYOK trust:** visitors must trust the backend operator not to keep their key. We minimise exposure (memory-only, per-request, redacted logs, open source, throwaway-key advice) but cannot make it zero-trust. Documented in the UI and README.
10. **Visitor rate limits:** Groq free-tier limits belong to the visitor. A full live run is about 9 generation calls plus up to 15 judge calls, which may hit their limit, so the UI surfaces 429s clearly and offers to skip judging.
11. **Retrieval limits (accepted, section 4.10):** cross-document questions can miss half their evidence and some facts sit in chunks the dense ranker does not surface. Generation results on those questions mix retrieval and generation quality; read them with the per-question context recall.

## 11. Decisions (all resolved)
1. **Models:** generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-20b` (D1 revised). Trials reduced from 5 to 3.
2. **Distractor paper:** kept deliberately (D8).
3. **Demo mode:** Replay is the default; Live is opt-in with the visitor's own Groq key (D9, section 7.4). No server-side LLM key in hosted mode.
4. **Location:** one project folder holding the code, `.venv/` and the local index and cache (D7).

**Standing rule:** `APIKEY.txt` is off-limits. It is never read, printed, copied or referenced by any phase. It must be deleted once its keys are rotated.

### Phase 6 (frontend) - built

React 18 + Vite 5 + TypeScript strict + Tailwind 3, replay-first, BYOK live mode over fetch-based
SSE. Deviations from section 7: Tailwind v3 with CSS-variable colour tokens (no shadcn/ui);
custom SVG charts instead of recharts (CI bars, forest plots, heatmap, scatter); hash routing
instead of a router library; plain hooks instead of react-query; API types generated from the
FastAPI OpenAPI schema (openapi-typescript); replay data served statically from
`ui/public/replay` so the demo works without a backend. Smoke test of the live endpoint: first
attempt hit a 240 s timeout from a ~1 min embedding-model cold start; fixed with a background
warm-up (`live_ready`/`live_error` in `/api/health`) and a run budget that starts after
retrieval. Second attempt passed (HTTP 200, 117.7 s, ~19.7k tokens, no key text in response or
logs). Cooperative cancellation now frees the in-flight slot when a client disconnects.

Visual redesign ("lab report"): light paper theme by default; Source Serif 4 and IBM Plex Mono;
hairline borders, ledger tables and 0-2px radius in place of shadowed rounded cards; teal and brick
as the only accents; strategies distinguished by value (graphite, grey, teal, light teal, taupe)
rather than hue. CSS, tokens and markup only: no API, data, route or chart-logic changes.

### Correctness audit after the redesign

Independent recomputation of every reported figure from `runs.jsonl` matched `summary.json`,
the replay files, the real API and 1,118 values scraped from the rendered pages. Bugs found and
fixed: (1) route changes could stall after a live run (`AnimatePresence mode="wait"`), leaving
Benchmark and Method blank; (2) a live score of exactly 0 rendered as a dash; (3) the API warm-up
did not load the embedding model, so `live_ready` was true while the first live run still paid a
minute-long import. The mock API now uses a discriminating fake judge, and live results name the
models that wrote and scored them. Open measurement caveat: the judge accepted a vague Q02
no-retrieval answer as covering the "different document per token" gold fact (the coverage
prompt says vague mentions do not count), so Control FCS on Q02 (1.00) is probably overstated;
coverage is not yet in the human-labelling sample.
