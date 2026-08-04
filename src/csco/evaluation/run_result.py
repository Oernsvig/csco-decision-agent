"""RunResult — container, scoring, and persistence for a single arm run.

One JSON file per run. Designed to load directly into pandas:
    import pandas as pd, json, glob
    runs = [json.load(open(f)) for f in glob.glob("results/**/*.json", recursive=True)]
    df = pd.json_normalize(runs)

The oracle measures fidelity to the encoded disruption-response policy,
not independent real-world optimality of the recommendation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from csco.models import Scenario, ScenarioDecision
from csco.oracle.manual import get_oracle
from csco.evaluation.scoring import (
    score_act_axis,
    score_carryover_match,
    score_citation_grounding,
    score_efficiency,
    score_failure_stage,
    score_lexical_references,
    score_pass_structure,
    score_reasoning_extraction,
    score_retrieval,
    score_routing,
    score_suppliers,
)


# ---------------------------------------------------------------------------
# Retrieval event (one tool call)
# ---------------------------------------------------------------------------

@dataclass
class SearchEvent:
    """One search_playbooks call in Arm 2 (flat vector)."""
    call_number: int
    query: str
    chunks: list[dict]          # [{file, heading, content_snippet}]
    governing_rule_retrieved: bool | None = None  # populated by score()


@dataclass
class OpenEvent:
    """One open() call in Arm 4 (lexical-graph arm)."""
    call_number: int
    pointer: str                        # pointer passed to open()
    node_type: str                      # "playcard" | "section" | "provision"
    pointers_returned: list[str]        # all pointers in the response
    references: list[dict] | None = None  # provision only: [{"as": "band", "pointer": "..."}]
    response_chars: int = 0             # characters of policy text in the response
    response_tokens: int = 0            # tokens of policy text (single tokenizer, shared across arms)


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

def _capture_provenance() -> dict:
    """Prompt/corpus/model versions active at run time.

    Recorded per run because result JSONs otherwise carry no version identity,
    which makes it hard to attribute a result to a prompt/corpus era without
    comparing timestamps to file mtimes. corpus_version is parsed from the rendered header
    of the embed-cut file the arms actually read (not the generator constant),
    so a stale corpus on disk is recorded as what it really is. Strictly
    additive (Design Invariant #24): legacy JSONs without this key must still
    score and report cleanly.
    """
    prov: dict = {}
    try:
        from csco.arms.prompts import PROMPT_VERSION
        prov["prompt_version"] = PROMPT_VERSION
    except Exception:
        pass
    try:
        import re
        from pathlib import Path
        embed = Path(__file__).parent.parent.parent.parent / "corpus" / "playbooks_embed" / "geopolitical.md"
        m = re.search(r"\*\*Version:\*\*\s*([0-9][0-9.]*)", embed.read_text(encoding="utf-8")[:600])
        if m:
            prov["corpus_version"] = m.group(1)
    except Exception:
        pass
    try:
        from csco.settings import get_settings
        prov["llm_model"] = get_settings().llm_model
    except Exception:
        pass
    return prov


@dataclass
class RunResult:
    """Complete record of a single arm run on a single scenario.

    The oracle measures fidelity to the encoded disruption-response policy.
    """

    # Identity
    run_id: str
    arm: str                    # "static" | "vector" | "lexical"
    scenario_id: str
    disruption_type: str
    timestamp: str              # ISO-8601

    # Performance
    elapsed_s: float = 0.0
    llm_calls_reasoning: int = 0
    llm_calls_extraction: int = 0
    tool_calls_total: int = 0
    policy_tokens: int = 0      # policy-context tokens supplied to the model (real tokenizer count)

    # Total model token spend — accumulated from every model + tool turn's
    # usage_metadata (reasoning turns, tool-call turns, and the extraction call).
    # policy_tokens above is the headline efficiency metric (policy content only);
    # these are the true billed totals across the whole run.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Provider-reported cached input tokens (prompt-cache hits). NOTE: this includes
    # *cross-run* cache and is therefore not used for the within-run analysis below —
    # kept only for reference/diagnostics.
    cache_read_tokens: int = 0

    # Per-reasoning-turn input sizes + the extractor call's input. Used to derive the
    # *within-run* uncached input analytically (independent of the provider's cache and
    # of any run-independence trick): the ReAct prefix grows monotonically, so the
    # largest reasoning turn already contains the full unique context. Fresh input =
    # that maximum (reasoning) + the extractor call (a single, always-fresh call);
    # everything else is re-sent prefix (within-run cache).
    reasoning_turn_inputs: list[int] = field(default_factory=list)
    extractor_input_tokens: int = 0

    @property
    def uncached_input_within_run(self) -> int:
        """Fresh input tokens the model processes across the run (within-run only)."""
        reasoning_fresh = max(self.reasoning_turn_inputs) if self.reasoning_turn_inputs else 0
        return reasoning_fresh + self.extractor_input_tokens

    @property
    def cached_input_within_run(self) -> int:
        """Re-sent prefix tokens (within-run cache) = total input − fresh input."""
        return max(0, self.input_tokens - self.uncached_input_within_run)

    # Decision output
    decision: dict = field(default_factory=dict)

    # Raw reasoning (before extraction)
    agent_answer_raw: str = ""

    # Retrieval trace (Arm 2 — flat vector)
    searches: list[SearchEvent] = field(default_factory=list)

    # Lexical-graph trace (Arm 4)
    opens: list[OpenEvent] = field(default_factory=list)

    # Scenario (for scoring supplier derivations)
    scenario: Scenario | None = None

    # Fixture stratification metadata (suite "A"|"B", stratum "null"|"action",
    # null_mode, boundary, target_rule_id) — carried through from the fixture
    # file/derived default so batch aggregation can post-stratify by suite/pi.
    fixture_meta: dict = field(default_factory=dict)

    # Version identity of the run (prompt_version, corpus_version, llm_model) —
    # populated by start() via _capture_provenance(); additive key.
    provenance: dict = field(default_factory=dict)

    # Scores (populated by score())
    scores: dict = field(default_factory=dict)

    # Failure classification (populated by score() or externally)
    run_status: str = "success"  # "success" | "infrastructure_failure" | "model_output_failure" | "tool_loop_failure" | "extraction_failure" | "unknown_failure"
    failure_message: str | None = None
    included_in_accuracy: bool = True  # False if infrastructure_failure
    counted_as_incorrect: bool = False  # True if model/tool/extraction failure

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        arm: str,
        scenario_id: str,
        disruption_type: str,
        scenario: Scenario | None = None,
        fixture_meta: dict | None = None,
    ) -> "RunResult":
        from datetime import datetime, timezone
        obj = cls(
            run_id=str(uuid.uuid4())[:8],
            arm=arm,
            scenario_id=scenario_id,
            disruption_type=disruption_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            scenario=scenario,
            fixture_meta=fixture_meta or {},
            provenance=_capture_provenance(),
        )
        return obj

    def __post_init__(self):
        self._start: float = time.time()

    def add_usage(self, usage_metadata: Any) -> None:
        """Accumulate one model turn's token usage onto the run totals.

        Accepts a LangChain ``usage_metadata`` mapping
        ({input_tokens, output_tokens, total_tokens}); silently ignores None or
        malformed values so instrumentation never breaks a run.
        """
        if not usage_metadata:
            return
        try:
            it = int(usage_metadata.get("input_tokens", 0) or 0)
            ot = int(usage_metadata.get("output_tokens", 0) or 0)
            tt = int(usage_metadata.get("total_tokens", 0) or 0) or (it + ot)
            details = usage_metadata.get("input_token_details") or {}
            cr = int((details.get("cache_read") if isinstance(details, dict) else 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return
        self.input_tokens += it
        self.output_tokens += ot
        self.total_tokens += tt
        self.cache_read_tokens += cr

    def finish(self, decision: ScenarioDecision) -> None:
        self.elapsed_s = round(time.time() - self._start, 1)
        self.decision = json.loads(decision.model_dump_json())
        self.score()

    # ---------------------------------------------------------------------------
    # Scoring against oracle
    # ---------------------------------------------------------------------------

    def score(self) -> None:
        """Compute correctness, retrieval, efficiency, and diagnostic metrics.

        The oracle measures fidelity to the encoded disruption-response policy,
        not independent real-world optimality of the recommendation.

        Scoring logic itself lives in csco.evaluation.scoring — this just fixes the call
        order and owns the resulting scores dict.
        """
        oracle = get_oracle(self.scenario_id)
        if oracle is None:
            self.scores = {"oracle": "not_defined"}
            return

        s: dict[str, Any] = {}

        if oracle.routing is not None:
            score_routing(self, s, oracle)
        else:
            score_suppliers(self, s, oracle)

        score_pass_structure(self, s, oracle)
        score_retrieval(self, s, oracle)
        score_lexical_references(self, s, oracle)
        score_reasoning_extraction(self, s, oracle)
        score_citation_grounding(self, s, oracle)
        score_carryover_match(self, s, oracle)
        score_efficiency(self, s)
        score_failure_stage(self, s, oracle)
        score_act_axis(self, s, oracle)

        self.scores = s

    # Thin delegates for the _score_* steps tests exercise directly
    # (test_scoring_extensions.py, test_skew_prevalence.py, test_citation_grounding.py) —
    # logic lives in csco.evaluation.scoring; kept as methods here only for that
    # direct-call surface.
    def _score_failure_stage(self, s: dict, oracle) -> None:
        score_failure_stage(self, s, oracle)

    def _score_citation_grounding(self, s: dict, oracle) -> None:
        score_citation_grounding(self, s, oracle)

    def _score_carryover_match(self, s: dict, oracle) -> None:
        score_carryover_match(self, s, oracle)

    def _score_act_axis(self, s: dict, oracle) -> None:
        score_act_axis(self, s, oracle)

    # ---------------------------------------------------------------------------
    # Serialisation
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id":               self.run_id,
            "arm":                  self.arm,
            "scenario_id":          self.scenario_id,
            "disruption_type":      self.disruption_type,
            "timestamp":            self.timestamp,
            "fixture_meta":         self.fixture_meta,
            "provenance":           self.provenance,
            "run_status":           self.run_status,
            "failure_message":      self.failure_message,
            "included_in_accuracy": self.included_in_accuracy,
            "counted_as_incorrect": self.counted_as_incorrect,
            "elapsed_s":            self.elapsed_s,
            "llm_calls_reasoning":  self.llm_calls_reasoning,
            "llm_calls_extraction": self.llm_calls_extraction,
            "llm_calls_total":      self.llm_calls_reasoning + self.llm_calls_extraction,
            "tool_calls_total":     self.tool_calls_total,
            "policy_tokens":        self.policy_tokens,
            "input_tokens":         self.input_tokens,
            "output_tokens":        self.output_tokens,
            "total_tokens":         self.total_tokens,
            "cache_read_tokens":    self.cache_read_tokens,
            "reasoning_turn_inputs": self.reasoning_turn_inputs,
            "extractor_input_tokens": self.extractor_input_tokens,
            "uncached_input_within_run": self.uncached_input_within_run,
            "cached_input_within_run":   self.cached_input_within_run,
            "agent_answer_raw":     self.agent_answer_raw,
            "decision":             self.decision,
            "searches": [
                {
                    "call_number":              s.call_number,
                    "query":                    s.query,
                    "chunks":                   s.chunks,
                    "governing_rule_retrieved": s.governing_rule_retrieved,
                }
                for s in self.searches
            ],
            "opens": [
                {
                    "call_number":     ev.call_number,
                    "pointer":         ev.pointer,
                    "node_type":       ev.node_type,
                    "pointers_returned": ev.pointers_returned,
                    "references":      ev.references,
                    "response_chars":  ev.response_chars,
                    "response_tokens": ev.response_tokens,
                }
                for ev in self.opens
            ],
            "scores": self.scores,
        }

    def save(self, output_dir: str | Path) -> Path:
        """Save to results/{scenario_id}/{arm}/{timestamp}_{run_id}.json"""
        base = Path(output_dir) / self.scenario_id / self.arm
        base.mkdir(parents=True, exist_ok=True)
        ts   = self.timestamp.replace(":", "-").replace(".", "-")[:19]
        path = base / f"{ts}_{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
