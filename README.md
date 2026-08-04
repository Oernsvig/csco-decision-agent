# CSCO Decision Agent

A controlled comparison of **three policy-grounding architectures** for a Chief
Supply Chain Officer (CSCO) decision agent: given a supply-chain disruption
scenario, each architecture recommends a routing decision and per-supplier
actions by applying the *same* encoded disruption-response policy, and is scored
against a deterministic policy oracle.

This repository accompanies an MPhil dissertation. It contains the policy specs and
generated corpus, the three arms, the evaluation suites, the deterministic oracle
and scoring, and the headline results.

> The oracle measures **fidelity to the encoded policy**, not independent
> real-world optimality of the recommendation.

## The three architectures

All three receive the **same scenario block** and return the **same
`ScenarioDecision`**, produced by the same shared extractor (`arms/extractor.py`).
They differ only in how the policy is delivered:

| Architecture | Module | Policy delivery | LLM pattern |
|---|---|---|---|
| Static full-corpus | `arms/static.py` | All five playbooks concatenated in the system prompt (constant) | Single call |
| Flat vector retrieval | `arms/vector.py` | Neo4j vector search over playbook chunks | ReAct agent |
| Lexical document graph | `arms/lexical.py` | Neo4j document graph — `LexPlaycard` → `LexSection` → `LexProvision` with structural and reference edges | ReAct agent |

The playbooks are the shared content basis: identical rules and *Reason:* clauses
across all arms, generated deterministically from the specs with worked examples
stripped so no encoding carries an answer key.

## The Prompts

All arms are driven by a single, fixed set of prompts (`arms/prompts.py`, "The
Prompts"). Prompt blocks fall into three tiers, enforced by
`tests/test_fairness.py`:

- **Shared-literal** — byte-identical across the static and tool arms.
- **Parallel-requirement** — the same decision contract, worded per arm.
- **Arm-specific** — policy-access mechanics only, tested to contain no
  answer-bearing policy vocabulary (no rule ids, escalation values, or strategic
  priorities in the shared agent prompt; no pre-written retrieval queries).

## Evaluation

Scenarios resolve on an **Act / No-Act** axis. A scenario is *No-Act* (null) when
the policy's outcome is maintain-and-log — routing disposition `maintain`, or (if a
supplier pass runs) every supplier `maintain`, in both cases at the baseline analyst
escalation. Any monitor/replace disposition or any elevated escalation is *Act*; the
class turns on the **mandated** disposition and escalation, not on the optional live
actions a provision *permits*. This single rule (`evaluation/binarize.py`) is applied
identically to the oracle and to each agent's decision.

Two suites:

- **Suite A** — 37 action-required scenarios, one canonical positive case per
  action-firing rule (`fixtures/suite_a/`, frozen).
- **Suite B** — 35 non-action scenarios: 15 null-outcome cases plus 20 leak-free
  null-mode fixtures across four null modes and five disruption types
  (`fixtures/suite_b/`). Basis for the skew/prevalence analysis (MCC,
  Acc(π)/cost(π) swept over null prevalence).

A run is `overall_correct` (SDM-3) only when the fired rule id, disposition, and
escalation all match the oracle. See `docs/fixtures.md` for the full fixture
reference and `results/README.md` for the headline numbers.

The oracle resolves a scenario id from `fixtures/suite_a/*.oracle.json` →
`fixtures/suite_b/*.oracle.json`, so any run can be scored without knowing which
suite produced it.

## Headline results (n=3, SDM-3)

| | static | vector | lexical |
|---|---|---|---|
| Suite A (action) overall_correct | 94.6% | 73.9% | 93.7% |
| Suite B (non-action) overall_correct | 67.6% | 51.4% | 93.3% |

Full tables, `pass³` reliability, and pooling notes: [`results/README.md`](results/README.md).

## Reproducibility

The full extraction and evaluation pipeline is in `src/csco/` and is deterministic
given fixed inputs. The parts that bear on exact numerical replication:

- **Model.** The reported runs used OpenAI **gpt-4o** (as resolved on **2026-08-02**)
  with the **text-embedding-3-small** embedding model. `LLM_MODEL` defaults to the
  mutable `gpt-4o` alias — pin it to the dated snapshot from that period for exact
  replication, since the alias moves over time; `EMBEDDING_MODEL` is set explicitly.
- **Dependencies.** `pyproject.toml` pins the direct dependencies and
  `requirements-lock.txt` pins the full transitive set to the versions used
  (**langchain 0.3.x / langgraph 0.2.x**). This major line matters: the vector and
  lexical arms drive a `langgraph` ReAct loop, and on the langchain/langgraph **1.x**
  line that loop does not converge the same way — install from `requirements-lock.txt`
  (or keep the `pyproject` caps) rather than upgrading these libraries.
- **Corpus & prompts.** Regenerate the corpus from specs with
  `python -m csco.generators.playbook` before running the retrieval arms; the assembled
  prompt text is version-stamped (`PROMPT_VERSION`) and checked against golden snapshots.
- **Determinism.** Model calls use temperature 0 and each fixture is run 3× for
  run-to-run reliability. Exact hosted-model outputs can still drift across model or
  library updates, as the dissertation notes.
- **Frozen manifest.** [`MANIFEST.json`](MANIFEST.json) records the prompt/corpus
  versions and hashes, spec and fixture hashes, response/embedding models, temperature
  and retry settings, Vector `k`, Neo4j index configuration, pinned package versions,
  run dates, and the bootstrap/token scripts. Regenerate or verify with
  `python -m csco.manifest` / `python -m csco.manifest --check`.

## Setup

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # or: pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins the exact dependency versions the reported runs used.

### 2. Credentials

```bash
cp .env.example .env
# edit .env — add NEO4J_*, LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY
```

The static arm and all non-integration tests run without Neo4j. The **vector** and
**lexical** arms need a running Neo4j instance.

### 3. Generate the corpus (after editing any spec)

```bash
python -m csco.generators.playbook      # writes corpus/playbooks/*.md from specs/
```

### 4. Build and load the lexical document graph (lexical arm)

```bash
python -m csco.graph.build_lexical > corpus/corpus_lexical.cypher
python -m csco.cli.load_lexical
```

### 5. Embed (vector and lexical arms)

```bash
python -m csco.cli.embed --arm vector    # chunk and embed corpus/playbooks_embed/*.md
python -m csco.cli.embed --arm lexical   # embed LexPlaycard summaries (after load_lexical)
```

Re-run **both** embed commands after any change to the corpus (steps 3–4). Each command
first clears the prior vectors, then re-embeds from the current text, so the Neo4j indexes
never carry embeddings that no longer match the corpus they were built from — stale
embeddings silently degrade retrieval to near-random.

### 6. Check fairness invariants

```bash
pytest -m "not integration"
```

Integration tests (`-m integration`) require a live Neo4j connection.

## Running

```bash
# Single scenario
python -m csco.cli.run --arm static  --scenario fixtures/suite_a/diag_0025.yaml
python -m csco.cli.run --arm vector  --scenario fixtures/suite_a/diag_0025.yaml
python -m csco.cli.run --arm lexical --scenario fixtures/suite_a/diag_0025.yaml

# Headline: Suite A + Suite B combined, 3 runs per cell
python -m csco.cli.batch_run \
  --fixtures-dir fixtures/suite_a,fixtures/suite_b \
  --output-dir results/headline --runs-per-fixture 3
```

## Project structure

```
specs/                       Decision specs — single source of truth (5 types, 52 rules)
corpus/
  playbooks/                 Generated policy corpus (no worked examples) — shared by all arms
  corpus_lexical.cypher      Neo4j seed for the lexical graph (generated)
fixtures/
  suite_a/                   Suite A — 37 action-required scenarios (frozen)
  suite_b/                   Suite B — 35 non-action scenarios
  generated/                 Scratch output for ad hoc generators/scenario.py runs
results/                     Headline results summary (results/README.md)
docs/
  fixtures.md                Fixture-set reference and CLI usage
  troubleshooting.md         Neo4j / environment notes
tests/                       Fairness, parity, oracle, and skew/prevalence tests
src/csco/
  settings.py                Env-based config (pydantic-settings)
  models.py                  Data contracts (Scenario, ScenarioDecision, …)
  derive.py                  risk_level derivation (pure)
  parity_check.py            Verifies all 52 rules present in every encoding
  corpus_stats.py            Standalone corpus statistics CLI
  oracle/
    manual.py                Oracle data models + get_oracle() resolver
    deterministic.py         Spec-driven oracle engine (ground truth for generated fixtures)
  specs/                     Typed spec models + YAML loader
  generators/
    playbook.py              Renders corpus/playbooks/ from specs
    benchmark.py             Builds the diagnostic and null/boundary fixture pools
    scenario.py              Ad hoc single-rule / single-cluster scenario generator
  evaluation/
    binarize.py              Act/No-Act binarization
    clusters.py              Diagnostic cluster taxonomy + contrast-pair registry
    scoring.py               Per-run scoring
    run_result.py            Run identity, instrumentation, JSON persistence
    skew_analysis.py         MCC, Acc(π)/cost(π), pass^k-by-stratum
    report.py                Batch-report text generation
  arms/
    llm.py                   LLM + embeddings factory
    prompts.py               format_scenario(), the assembled system prompts, PROMPT_VERSION
    extractor.py             Shared extract_decision() — identical across arms
    runners.py               Shared invoke/extract plumbing
    static.py                Static full-corpus arm
    vector.py                Flat vector retrieval arm
    lexical.py               Lexical document graph arm
  graph/
    build_lexical.py         Generates corpus_lexical.cypher from the corpus + specs
  cli/
    run.py                   Single scenario runner
    batch_run.py             Batch runner (3-arm default; skew/prevalence + cluster reporting)
    embed.py                 Embedding CLI
    load_lexical.py          Loads corpus_lexical.cypher into Neo4j
```

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `NEO4J_URI` | vector/lexical | — | `neo4j+s://…` |
| `NEO4J_USERNAME` | vector/lexical | — | Often `neo4j` on Aura |
| `NEO4J_PASSWORD` | vector/lexical | — | |
| `NEO4J_DATABASE` | | `neo4j` | Aura sometimes uses the instance id |
| `LLM_PROVIDER` | | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | | `gpt-4o` | |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | OpenAI embedding model for the vector/lexical Neo4j indexes |
| `OPENAI_API_KEY` | ✓ | — | Also used for embeddings |
| `ANTHROPIC_API_KEY` | if anthropic | — | |

> Neo4j Aura auth errors? Try setting both `NEO4J_USERNAME` and `NEO4J_DATABASE`
> to `neo4j`. See `docs/troubleshooting.md`.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
