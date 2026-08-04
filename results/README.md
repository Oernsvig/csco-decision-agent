# Headline results

Comparison of the three policy-grounding architectures — **static** (full corpus
in context), **vector** (flat vector retrieval, ReAct), and **lexical** (document-graph
navigation, ReAct; "Lexical Graph" in the dissertation) — over the two evaluation
suites. Each scenario is run 3 times (`--runs-per-fixture 3`).

The primary correctness metric is **overall correctness (SDM-3)**: a run counts as
correct only when the fired rule id, disposition, and escalation all match the
deterministic oracle. `pass³` is the fraction of scenarios correct on all three
repeats (run-to-run reliability). Intervals are 95% scenario-cluster bootstrap.

> These figures are from the `dissertation_3rep` run and match Section 4 of the
> dissertation. Token-usage figures come from separate runs.

## Suite A — prescribed-action cases (37 scenarios, 111 runs/arm)

| arm | overall correctness | 95% CI | pass³ |
|---|---|---|---|
| static  | 94.6% (105/111) | [87.4%, 100.0%] | 34/37 (92%) |
| vector  | 73.9% (82/111)  | [63.1%, 83.8%]  | 20/37 (54%) |
| lexical | 93.7% (104/111) | [87.4%, 98.2%]  | 32/37 (86%) |

Paired differences in overall correctness:

| comparison | difference | 95% CI |
|---|---|---|
| lexical − static | −0.9 pp  | [−6.3, +3.6]  (includes 0) |
| lexical − vector | +19.8 pp | [+8.1, +31.5] |

Static and lexical were statistically indistinguishable on action cases; both
clearly exceeded vector. Vector's shortfall was primarily an **access** problem —
in 16 of its failed runs the governing policy chunk was never retrieved — whereas
lexical reached the governing rule in 110/111 runs via selective traversal.

## Suite B — no-action cases (35 scenarios, 105 runs/arm)

| arm | overall correctness | 95% CI | pass³ |
|---|---|---|---|
| static  | 67.6% (71/105) | [54.3%, 80.0%] | 18/35 (51%) |
| vector  | 51.4% (54/105) | [37.1%, 64.8%] | 12/35 (34%) |
| lexical | 93.3% (98/105) | [88.6%, 97.1%] | 28/35 (80%) |

Paired differences in overall correctness:

| comparison | difference | 95% CI |
|---|---|---|
| lexical − static | +25.7 pp | [+15.2, +37.1] |
| lexical − vector | +41.9 pp | [+30.5, +54.3] |

Lexical held nearly constant from Suite A, while static and vector dropped sharply.
Most no-action errors were **stopping-logic** failures (continuing past a prescribed
Pass-1 routing stop): all 7 lexical errors and 24 of static's 34.

## Act / no-act discrimination (combined suites)

| arm | balanced accuracy | MCC | recall | specificity | unnecessary interventions |
|---|---|---|---|---|---|
| static  | 97.2% [94.0, 99.5]  | +.945 [+.888, +.991] | 0.982 | 0.962 | 4 |
| vector  | 89.7% [83.8, 94.8]  | +.802 [+.687, +.908] | 0.955 | 0.840 | 17 |
| lexical | 99.1% [97.7, 100.0] | +.982 [+.954, +1.000] | 0.982 | 1.000 | 0 |

Paired balanced-accuracy differences: lexical − static +1.9 pp [−0.4, +4.7]
(includes 0); lexical − vector +9.4 pp [+4.7, +15.1]. Paired MCC differences:
lexical − static +.037 [−.009, +.092]; lexical − vector +.180 [+.085, +.285].

## Token usage (combined suites; separate runs)

Total and uncached tokens are the model's own reported usage; policy-context tokens
are counted with a single tokenizer across all three arms (`src/csco/tokens.py`), so
the figures are on one comparable scale.

| arm | total | uncached | policy-context |
|---|---|---|---|
| static  | 17,102 | 17,102 | 14,516 |
| vector  | 17,462 | 7,499  | 3,639  |
| lexical | 47,644 | 7,953  | 1,887  |

Lexical used ≈87% fewer policy-context tokens than static and ≈48% fewer than
vector, keeping uncached below half of static's; its higher *total* reflects
repeated transmission of cached conversation history across traversal turns.

## Cross-suite workload sensitivity

Prevalence-weighting the two suites, static and lexical break even at a no-action
share of π*≈3.4% (observed point estimate; bootstrap median 8.6%, 95% interval
2.6%–24.9%). Below that, static's small action-case edge wins; above it, lexical
leads by a widening margin. Lexical matched or exceeded vector on both suites in
99.99% of resamples (static in 98.61%).

## Reproduce

Requires an LLM API key in `.env` (and a running Neo4j for the vector and lexical
arms). Full batch reports (`BATCH_REPORT.txt`, `acc_pi.csv`, the Act/No-Act
confusion matrix and skew/prevalence tables) are written by `batch_run`:

```bash
# Suite A
python -m csco.cli.batch_run --fixtures-dir fixtures/suite_a \
  --output-dir results/suite_a --runs-per-fixture 3

# Suite B
python -m csco.cli.batch_run --fixtures-dir fixtures/suite_b \
  --output-dir results/suite_b --runs-per-fixture 3

# Both suites pooled
python -m csco.cli.batch_run --fixtures-dir fixtures/suite_a,fixtures/suite_b \
  --output-dir results/headline --runs-per-fixture 3
```
