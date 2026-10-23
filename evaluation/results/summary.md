# Benchmark results

## Design and limitations (read first)

- **10 questions** (5 lookup, 4 multi-hop, 1 out-of-corpus), **2 trials** per question and strategy; only trial 0 is judged (groundedness, coverage), because the free tier cannot fund the full design. The reduction from the planned 13 questions x 3 trials is documented in ARCHITECTURE_PLAN.md section 4.8.
- **Consistency rests on 2 samples per question** (one pair of answers), not 3 (three pairs), so it is a noisier estimate. Treat consistency differences as indicative only.
- Quality metrics average over the **9 answerable questions**. Confidence intervals are 95% bootstrap intervals over questions, so they are wide with this few. Differences whose intervals overlap should not be called real.
- The out-of-corpus question is reported only as abstention behaviour.
- Generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-20b` (an LLM judge; validated against a human on a small sample, see the next bullet).
- **Judge validation (human-labelled sample).** On 30 claims the judge agreed with a human label 28 times (93%; 95% CI for accuracy [0.79, 0.98]; Cohen's kappa 0.879, 3-way). All three disagreements found (two claim-level, one coverage-level) ran against the retrieval-vs-control gap: two were lenient toward the no-retrieval control and one wrongly marked an RRR claim contradicted, so the "retrieval clearly helps" finding is likely conservative, not overstated. Limits: n=30; not randomly drawn (weighted toward Control and Structured claims, plus one deliberately targeted suspect case); a single labeller who also wrote the gold facts; and coverage and FCS are validated far less than groundedness (one coverage case checked, and it went against the judge). See `claims_to_label.csv`.
- Groundedness is measured against the retrieved top-4 passages for every strategy, including the no-retrieval control, so a correct answer from memory still counts as unsupported. FCS counts a refusal on an answerable question as 0.

## Question numbering

The web app numbers the kept questions Q1-Q10; this report and the data use the original research IDs (Q01, Q04 and Q07 were dropped as redundant).

| app | research ID | type |
|---|---|---|
| Q1 | Q02 | lookup |
| Q2 | Q03 | lookup |
| Q3 | Q05 | lookup |
| Q4 | Q06 | lookup |
| Q5 | Q08 | lookup |
| Q6 | Q09 | multi-hop |
| Q7 | Q10 | multi-hop |
| Q8 | Q11 | multi-hop |
| Q9 | Q12 | multi-hop |
| Q10 | Q13 | out-of-corpus |

## Quality (answerable questions, mean [95% CI])

| strategy | groundedness G | coverage C | faithful-coverage FCS |
|---|---|---|---|
| Baseline | 1.00 [1.00, 1.00] | 0.46 [0.22, 0.70] | 0.53 [0.26, 0.78] |
| Structured | 0.91 [0.83, 0.98] | 0.62 [0.37, 0.85] | 0.65 [0.39, 0.87] |
| RRR | 0.95 [0.88, 1.00] | 0.65 [0.42, 0.85] | 0.72 [0.49, 0.90] |
| Self-RAG (prompted) | 0.94 [0.84, 1.00] | 0.55 [0.32, 0.78] | 0.63 [0.41, 0.83] |
| No retrieval (control) | 0.56 [0.33, 0.79] | 0.38 [0.17, 0.60] | 0.33 [0.14, 0.54] |

### FCS by question type

| strategy | lookup | multi_hop |
|---|---|---|
| Baseline | 0.66 [0.30, 0.96] | 0.37 [0.00, 0.73] |
| Structured | 0.86 [0.73, 0.98] | 0.39 [0.00, 0.78] |
| RRR | 0.93 [0.85, 1.00] | 0.45 [0.16, 0.71] |
| Self-RAG (prompted) | 0.83 [0.63, 1.00] | 0.37 [0.12, 0.55] |
| No retrieval (control) | 0.45 [0.17, 0.74] | 0.18 [0.00, 0.38] |

## Paired differences in FCS (strategy minus reference)

**vs Baseline** (per-question differences; Wilcoxon p is exploratory with this n)

| strategy | dG | dC | dFCS | Wilcoxon p (FCS) |
|---|---|---|---|---|
| Structured | -0.07 [-0.16, 0.00] | 0.16 [0.03, 0.33] | 0.12 [-0.02, 0.30] | 0.312 |
| RRR | -0.03 [-0.08, 0.00] | 0.19 [0.04, 0.36] | 0.19 [0.03, 0.39] | 0.125 |
| Self-RAG (prompted) | -0.04 [-0.12, 0.00] | 0.08 [-0.09, 0.28] | 0.10 [-0.10, 0.32] | 0.625 |
| No retrieval (control) | -0.33 [-0.58, -0.08] | -0.08 [-0.41, 0.25] | -0.20 [-0.51, 0.09] | 0.375 |

**vs no-retrieval control** (per-question differences; Wilcoxon p is exploratory with this n)

| strategy | dG | dC | dFCS | Wilcoxon p (FCS) |
|---|---|---|---|---|
| Baseline | 0.33 [0.08, 0.58] | 0.08 [-0.25, 0.41] | 0.20 [-0.09, 0.51] | 0.375 |
| Structured | 0.31 [0.04, 0.56] | 0.24 [-0.06, 0.54] | 0.32 [0.09, 0.55] | 0.047 |
| RRR | 0.38 [0.18, 0.58] | 0.27 [0.02, 0.53] | 0.39 [0.17, 0.62] | 0.016 |
| Self-RAG (prompted) | 0.41 [0.20, 0.60] | 0.17 [-0.07, 0.44] | 0.30 [0.08, 0.53] | 0.047 |

## Abstention

Out-of-corpus question (should abstain) and false refusals on answerable questions (should not). An answer counts as an abstention when it contains the refusal sentence.

| strategy | abstained on out-of-corpus | n | refused an answerable question | n |
|---|---|---|---|---|
| Baseline | 1.00 | 2 | 0.33 | 18 |
| Structured | 1.00 | 2 | 0.11 | 18 |
| RRR | 1.00 | 2 | 0.11 | 18 |
| Self-RAG (prompted) | 1.00 | 2 | 0.11 | 18 |
| No retrieval (control) | 0.00 | 2 | 0.00 | 18 |

## Length, diversity and consistency (answerable questions)

| strategy | words | sentences | MATTR | semantic consistency | structural consistency |
|---|---|---|---|---|---|
| Baseline | 30 [17, 47] | 1.0 [1.0, 1.0] | 0.84 [0.77, 0.91] | 0.97 [0.92, 1.00] | 0.95 [0.90, 0.99] |
| Structured | 211 [150, 258] | 11.3 [9.1, 13.2] | 0.91 [0.88, 0.93] | 0.93 [0.88, 0.97] | 0.90 [0.75, 0.99] |
| RRR | 117 [83, 146] | 5.4 [3.9, 6.9] | 0.90 [0.87, 0.92] | 0.95 [0.91, 0.97] | 0.90 [0.82, 0.96] |
| Self-RAG (prompted) | 184 [106, 261] | 9.2 [5.1, 13.4] | 0.89 [0.86, 0.91] | 0.92 [0.89, 0.95] | 0.82 [0.74, 0.89] |
| No retrieval (control) | 48 [42, 57] | 1.3 [1.0, 1.7] | 0.90 [0.88, 0.92] | 0.92 [0.89, 0.95] | 0.92 [0.87, 0.96] |

MATTR is undefined (excluded) for answers shorter than its 25-word window, which removes the original report's artefact of very short answers scoring a perfect diversity of 1.0.

## Efficiency (per answer, generation only, list prices)

| strategy | model calls | latency (s) | tokens | list-price cost (USD) |
|---|---|---|---|---|
| Baseline | 1.0 [1.0, 1.0] | 0.3 [0.2, 0.4] | 643 [605, 687] | 0.00063 |
| Structured | 1.0 [1.0, 1.0] | 0.8 [0.6, 0.9] | 939 [846, 1021] | 0.00165 |
| RRR | 3.0 [3.0, 3.0] | 1.9 [1.6, 2.2] | 2631 [2417, 2819] | 0.00369 |
| Self-RAG (prompted) | 3.0 [3.0, 3.0] | 2.1 [1.5, 2.7] | 3120 [2607, 3631] | 0.00458 |
| No retrieval (control) | 1.0 [1.0, 1.0] | 0.4 [0.3, 0.4] | 153 [144, 163] | 0.00031 |

Free-tier usage costs $0; the figures are list prices from `configs/pricing.yaml`. Latency is the sum over a strategy's sequential calls.

## Retrieval diagnostics

- Mean cosine similarity by rank: rank 1: 0.64, rank 2: 0.59, rank 3: 0.57, rank 4: 0.55
- Distractor-paper chunks in any top-4 list: **0** (over 10 questions).
- **Context recall** is the share of a question's gold facts that the retrieved top-4 actually contains, i.e. a ceiling on achievable coverage. Two measures: *judge-based* (the judge decides whether the passages contain each fact, paraphrase allowed; mean **0.74**) and *quote match* (the gold quote's words appear in one chunk; mean 0.22, a strict lower bound).
- Low coverage on a question with low context recall is a **retrieval** failure, not a generation failure. A refusal there is the correct behaviour for the context given.

| question | context recall (judge) | context recall (quote match) |
|---|---|---|
| Q02 | 1.00 | 1.00 |
| Q03 | 0.50 | 0.00 |
| Q05 | 1.00 | 0.00 |
| Q06 | 1.00 | 0.33 |
| Q08 | 0.67 | 0.67 |
| Q09 | 0.50 | 0.00 |
| Q10 | 1.00 | 0.00 |
| Q11 | 0.33 | 0.00 |
| Q12 | 0.67 | 0.00 |
| Q13 | n/a | n/a |

## Sensitivity: excluding Q11

Recomputed from the same recorded answers and judgements (no new model calls) on the 8 remaining answerable questions, shown beside the original 9. Q11 is excluded because its retrieved context held too little of the needed evidence (judge-based context recall 0.33), so its scores mostly measure retrieval failure.

**Quality (mean [95% CI]; original n=9 | excluding Q11, n=8)**

| strategy | G (all) | G (excl.) | C (all) | C (excl.) | FCS (all) | FCS (excl.) |
|---|---|---|---|---|---|---|
| Baseline | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.46 [0.22, 0.70] | 0.52 [0.27, 0.77] | 0.53 [0.26, 0.78] | 0.60 [0.33, 0.84] |
| Structured | 0.91 [0.83, 0.98] | 0.91 [0.83, 0.98] | 0.62 [0.37, 0.85] | 0.70 [0.46, 0.90] | 0.65 [0.39, 0.87] | 0.73 [0.50, 0.91] |
| RRR | 0.95 [0.88, 1.00] | 0.95 [0.88, 1.00] | 0.65 [0.42, 0.85] | 0.73 [0.55, 0.90] | 0.72 [0.49, 0.90] | 0.81 [0.65, 0.93] |
| Self-RAG (prompted) | 0.94 [0.84, 1.00] | 0.94 [0.84, 1.00] | 0.55 [0.32, 0.78] | 0.61 [0.41, 0.84] | 0.63 [0.41, 0.83] | 0.70 [0.54, 0.88] |
| No retrieval (control) | 0.56 [0.33, 0.79] | 0.56 [0.33, 0.79] | 0.38 [0.17, 0.60] | 0.43 [0.21, 0.66] | 0.33 [0.14, 0.54] | 0.37 [0.17, 0.59] |

**Retrieval vs no retrieval: paired difference (strategy minus control)**

| strategy | dG (all) | dG (excl.) | dC (all) | dC (excl.) | dFCS (all) | dFCS (excl.) |
|---|---|---|---|---|---|---|
| Baseline | 0.33 [0.08, 0.58] | 0.33 [0.08, 0.58] | 0.08 [-0.25, 0.41] | 0.09 [-0.27, 0.46] | 0.20 [-0.09, 0.51] | 0.23 [-0.10, 0.55] |
| Structured | 0.31 [0.04, 0.56] | 0.31 [0.04, 0.56] | 0.24 [-0.06, 0.54] | 0.27 [-0.06, 0.57] | 0.32 [0.09, 0.55] | 0.36 [0.12, 0.60] |
| RRR | 0.38 [0.18, 0.58] | 0.38 [0.18, 0.58] | 0.27 [0.02, 0.53] | 0.30 [0.02, 0.58] | 0.39 [0.17, 0.62] | 0.44 [0.21, 0.67] |
| Self-RAG (prompted) | 0.41 [0.20, 0.60] | 0.41 [0.20, 0.60] | 0.17 [-0.07, 0.44] | 0.19 [-0.08, 0.49] | 0.30 [0.08, 0.53] | 0.33 [0.10, 0.59] |

**Strategy vs Baseline: paired difference in FCS**

| strategy | dFCS (all) | p (all) | dFCS (excl.) | p (excl.) |
|---|---|---|---|---|
| Structured | 0.12 [-0.02, 0.30] | 0.312 | 0.14 [-0.03, 0.33] | 0.312 |
| RRR | 0.19 [0.03, 0.39] | 0.125 | 0.21 [0.04, 0.43] | 0.125 |
| Self-RAG (prompted) | 0.10 [-0.10, 0.32] | 0.625 | 0.11 [-0.12, 0.35] | 0.625 |
| No retrieval (control) | -0.20 [-0.51, 0.09] | 0.375 | -0.23 [-0.55, 0.10] | 0.375 |

**Refusals on answerable questions** (all -> excluding)

| strategy | refused an answerable question |
|---|---|
| Baseline | 0.33 -> 0.25 |
| Structured | 0.11 -> 0.00 |
| RRR | 0.11 -> 0.00 |
| Self-RAG (prompted) | 0.11 -> 0.00 |
| No retrieval (control) | 0.00 -> 0.00 |

**Reading of the sensitivity analysis**

*Written against the 2026-10-23 run (10 questions x 2 trials). If the results are regenerated from new data, re-check this text.*

**Excluding Q11 does not change any conclusion.**

- **Why the numbers move the way they do.** Q11 scored 0 for every condition, including the no-retrieval control, because its retrieved context contained only one of three gold facts and all four retrieval strategies correctly refused. Dropping it therefore lifts every condition's coverage and FCS by a similar amount (about +0.04 to +0.09), leaves groundedness exactly unchanged (no condition had a groundedness value on Q11), and leaves the rank-based Wilcoxon p-values identical (Q11 contributed zero differences between strategies). The ranking RRR > Structured > Self-RAG > Baseline > control is the same.
- **RRR's lead over Baseline is not clearer.** FCS difference +0.21 [0.04, 0.43] without Q11 versus +0.19 [0.03, 0.39] with it: the interval's lower edge barely moves, and the Wilcoxon p stays at 0.125. It remains suggestive, not established; with 8 questions it should not be called a real difference. Structured vs Baseline (+0.14 [-0.03, 0.33]) and Self-RAG vs Baseline (+0.11 [-0.12, 0.35]) stay inconclusive.
- **The retrieval-versus-no-retrieval gap holds and is slightly larger.** FCS gain over the control: Structured +0.36 [0.12, 0.60], RRR +0.44 [0.21, 0.67], Self-RAG +0.33 [0.10, 0.59], all intervals excluding zero; Baseline +0.23 [-0.10, 0.55] still does not. The groundedness gap (+0.31 to +0.41) is unchanged.
- **Refusals.** The elaborate strategies' only refusals were on Q11 (0.11 -> 0.00). Baseline still refuses 2 of the remaining 8 questions (Q03 and Q09, where context recall is 0.50) while the other strategies answered from the same context, so part of Baseline's lower coverage is a willingness-to-answer difference, not only a retrieval one.
- **Caveat.** This is a post-hoc exclusion chosen because Q11 is retrieval-limited, and the exclusion criterion (context recall 0.33) was visible in the data. It shows the conclusions are not driven by that one question; it is not a better estimate of overall performance.


## Run totals

- 100 answers (50 judged), 296 model calls (12 served from cache), 166 rate-limit retries.
- Tokens: generator 145,952, judge 85,511.
- 1 run invocation(s), 27.6 active minutes, 0 quota stop(s), 0 judge error(s).
