# Fixture Sets and Benchmark Design

Practical reference for the fixture directories and the CLI commands that use them.

## Directory structure

```
fixtures/
  suite_a/       Suite A — 37 action-required scenarios (frozen)
  suite_b/       Suite B — 35 non-action scenarios
  generated/     Scratch output for ad hoc scenario_generator.py runs
```

Each scenario ships as a triplet: `<id>.yaml` (the scenario), `<id>.oracle.json`
(the deterministic ground-truth label), and `<id>.meta.json` (tags).

## Act / No-Act axis

Every scenario resolves to *Act* or *No-Act* (null). A scenario is **No-Act** when
the policy outcome is maintain-and-log: routing disposition `maintain`, or — if a
supplier pass runs — every scored supplier `maintain`, in both cases at the baseline
analyst escalation. Any monitor/replace disposition or any elevated escalation is
**Act**. The class turns on the *mandated* disposition and escalation, not on the
optional live actions a provision *permits* (a maintain outcome that merely permits,
say, intensified monitoring stays No-Act). The rule lives in `evaluation/binarize.py`
(`is_null_oracle` / `is_null_decision`) and is applied identically to the oracle and
to each agent decision.

## Suite A — action-required (`fixtures/suite_a/`)

37 scenarios, one canonical positive case per action-firing policy rule, with
deterministic spec-driven oracle labels (`oracle/deterministic.py`). Frozen
(`benchmark_meta.json`, `frozen: true`) — not regenerated once evaluation began.

## Suite B — non-action (`fixtures/suite_b/`)

35 scenarios that complement Suite A's rule-positive cases with the null-region
space, so headline accuracy is not measured only on "a rule always fires" cases
(`benchmark_meta.json`). All are neutrally named `sb_*` so the filename carries no
answer:

- **15 canonical no-action cases** (`null_mode: canonical`): one per null routing
  rule, the maintain-and-log subset — so Suite A holds only action-required cases.
- **20 null-mode fixtures**: four null modes (`no_path_on_network`,
  `all_tier1_low_risk`, `criticality_below_gate`, `dt_null_routing`) across the five
  disruption types. Each `sb_*` carries `suite`/`stratum`/`null_mode` tags read by
  `batch_run`.

## Headline run (Suite A + Suite B)

```bash
python -m csco.cli.batch_run \
  --fixtures-dir fixtures/suite_a,fixtures/suite_b \
  --output-dir results/headline --runs-per-fixture 3
```

Reports overall accuracy plus the skew/prevalence breakdown (Act/No-Act confusion
matrix, MCC, Acc(π)/cost(π) at swept null-prevalence values).

## How the suites relate to the generators

`generators/benchmark.py` builds two pools: the diagnostic pool (one positive case
per rule, `diag_*`) and the null-region/boundary pool (`sb_*`). The shipped suites
are curated from those pools — Suite A is the action subset of the diagnostic pool;
Suite B is its non-action subset plus the 20 null-mode fixtures. The shipped
fixtures are frozen; the generators are provided for transparency, not to be re-run
into the frozen directories.

```bash
# Regenerate the pools into fresh directories (not the frozen suites)
python -m csco.generators.benchmark --output-dir /tmp/diagnostic --seed 12345
python -m csco.generators.benchmark --suite-b --suite-b-output-dir /tmp/nulls --suite-b-seed 20240
```

## Ad hoc scenario generation (`fixtures/generated/`)

For targeted follow-up scenarios outside the frozen suites:

```bash
python -m csco.generators.scenario --output-dir fixtures/generated --rule CYB-HUB
python -m csco.generators.scenario --output-dir fixtures/generated --cluster C3_PRIORITY_OVERRIDE
```

Not part of headline reporting.

## Oracle validation

```bash
# Targeted predicate tests (routing NULL/DT/DT-NULL, band/tier derivation)
pytest tests/test_fairness.py -k "test_routing_rules" -v
pytest tests/test_fairness.py -k "test_null_does_not_mask" -v
```

`get_oracle(scenario_id)` (`oracle/manual.py`) resolves in order: hand-authored
`ORACLE` dict → `fixtures/suite_a/*.oracle.json` → `fixtures/suite_b/*.oracle.json`
→ `None`.
