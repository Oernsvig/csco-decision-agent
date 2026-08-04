"""Tests for the scoring/oracle-loading/aggregation extensions:

  Task 1 — diagnostic oracle JSON fallback in oracle.get_oracle()
  Task 2 — first-failure attribution (scores["failure_stage"])
  Task 3/4 — pass^k cell stats and the extended batch-report section
             (light smoke coverage; the detailed spec lives in the brief)
"""

import json
from pathlib import Path

import pytest
import yaml

from csco.oracle.manual import ORACLE, ScenarioOracle, RoutingOracle, SupplierOracle, get_oracle
from csco.oracle.deterministic import build_oracle, compare_oracles
from csco.models import Scenario
from csco.specs.loader import load_spec
from csco.evaluation.run_result import RunResult

FIXTURES_DIAGNOSTIC = Path(__file__).parent.parent / "fixtures" / "suite_a"


# ---------------------------------------------------------------------------
# Task 1 — diagnostic oracle JSON fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_id", ["diag_0001", "diag_0002", "diag_0004", "diag_0007"])
def test_diagnostic_oracle_round_trip(scenario_id):
    """Oracle loaded from {scenario_id}.oracle.json must match a fresh build_oracle() call."""
    yaml_path = FIXTURES_DIAGNOSTIC / f"{scenario_id}.yaml"
    if not yaml_path.exists():
        pytest.skip(f"{yaml_path} not present")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(raw)
    spec = load_spec(scenario.disruption_type)
    fresh = build_oracle(scenario, spec)

    loaded = get_oracle(scenario_id, diagnostic_oracle_dir=FIXTURES_DIAGNOSTIC)
    assert loaded is not None, f"expected {scenario_id} to resolve via the JSON fallback"

    mismatches = compare_oracles(loaded, fresh)
    assert mismatches == [], f"{scenario_id}: {mismatches}"


def test_unknown_scenario_id_yields_not_defined_without_raising():
    """An ID present in neither source must return None, not raise."""
    oracle = get_oracle("totally_unknown_scenario_id_xyz", diagnostic_oracle_dir=FIXTURES_DIAGNOSTIC)
    assert oracle is None


def test_diagnostic_oracle_missing_file_returns_none():
    """An ID absent from both the dict and the diagnostic directory returns None cleanly."""
    oracle = get_oracle("diag_9999", diagnostic_oracle_dir=FIXTURES_DIAGNOSTIC)
    assert oracle is None


def test_score_no_longer_not_defined_for_diagnostic_fixture():
    """End-to-end proof: scoring a run against a diagnostic-only scenario_id must not
    fall back to the old {"oracle": "not_defined"} placeholder."""
    scenario_id = "diag_0004"
    yaml_path = FIXTURES_DIAGNOSTIC / f"{scenario_id}.yaml"
    if not yaml_path.exists():
        pytest.skip(f"{yaml_path} not present")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(raw)

    # Build a perfectly-correct decision straight from the fixture's own oracle so
    # the test survives fixture renumbering (Suite A = the 37 action rules).
    oracle_json = json.loads((FIXTURES_DIAGNOSTIC / f"{scenario_id}.oracle.json").read_text(encoding="utf-8"))

    rr = RunResult.start(
        arm="static",
        scenario_id=scenario_id,
        disruption_type=scenario.disruption_type,
        scenario=scenario,
    )
    if oracle_json.get("routing"):
        r = oracle_json["routing"]
        rr.decision = {
            "routing_result": {
                "fired_rule_id": r["fired_rule_id"],
                "scenario_action": r["scenario_action"],
                "disposition": r["disposition"],
                "escalation": r["escalation"],
                "permitted_actions": [],
            },
            "recommendations": [],
        }
    else:
        rr.decision = {
            "routing_result": None,
            "recommendations": [
                {
                    "supplier_id": sup["supplier_id"],
                    "disposition": sup["disposition"],
                    "escalation": sup["escalation"],
                    "fired_rule_id": sup["fired_rule_id"],
                    "permitted_actions": sup["permitted_actions"],
                    "rationale": "matches oracle",
                    "assigned_risk_level": None,
                }
                for sup in oracle_json["suppliers"]
            ],
        }
    rr.score()

    assert rr.scores.get("oracle") != "not_defined"
    assert rr.scores["overall_correct"] is True
    assert rr.scores["overall_correct_full"] is True
    assert rr.scores["failure_stage"] == "none"


# ---------------------------------------------------------------------------
# Task 2 — first-failure attribution
# ---------------------------------------------------------------------------

def _rr(arm: str = "static") -> RunResult:
    return RunResult.start(arm=arm, scenario_id="synthetic", disruption_type="cyber")


_SUPPLIER_ORACLE = ScenarioOracle(
    scenario_id="synthetic",
    disruption_type="cyber",
    routing=None,
    suppliers=[
        SupplierOracle(
            supplier_id="S1",
            disposition="monitor",
            escalation="Responsible Analyst",
            fired_rule_id="CYB-R1",
            permitted_actions=["intensify_threat_monitoring"],
        )
    ],
)

_ROUTING_ORACLE = ScenarioOracle(
    scenario_id="synthetic",
    disruption_type="cyber",
    routing=RoutingOracle(
        fired_rule_id="CYB-NULL",
        scenario_action="maintain_and_log",
        disposition="maintain",
        escalation="Responsible Analyst",
    ),
    suppliers=[],
)


def test_failure_stage_none_when_overall_correct():
    rr = _rr()
    s = {"overall_correct": True}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "none"


def test_failure_stage_extraction():
    rr = _rr()
    s = {"overall_correct": False, "extraction_valid": True, "extractor_preserved_rule": False}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "extraction"


def test_failure_stage_extraction_via_extraction_valid():
    rr = _rr()
    s = {"overall_correct": False, "extraction_valid": False}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "extraction"


def test_failure_stage_access_vector():
    rr = _rr(arm="vector")
    s = {"overall_correct": False, "retrieval_governing_provision_retrieved": False}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "access"


def test_failure_stage_access_lexical():
    rr = _rr(arm="lexical")
    s = {"overall_correct": False, "opens_to_governing_provision": None}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "access"


def test_failure_stage_static_never_fails_at_access():
    """Static always has the full corpus in context — access stage must be skipped."""
    rr = _rr(arm="static")
    # Even if a retrieval-shaped key were somehow present, static's access branch is a no-op.
    s = {"overall_correct": False, "retrieval_governing_provision_retrieved": False}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] != "access"


def test_failure_stage_classification():
    rr = _rr()
    # Classification failure = the agent's stated risk band disagrees with derive()
    # (supplier criticality was removed, so the risk band is the only classification).
    s = {"overall_correct": False, "assigned_risk_level_correct": 0.5}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "classification"


def test_failure_stage_pass_structure():
    rr = _rr()
    s = {"overall_correct": False, "pass1_pass2_correct": False}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "pass_structure"


def test_failure_stage_rule_selection_supplier_case():
    rr = _rr()
    s = {"overall_correct": False, "correctness_rule_id": 0.0, "correctness_disposition": 0.0}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "rule_selection"


def test_failure_stage_execution_supplier_case():
    rr = _rr()
    s = {
        "overall_correct": False,
        "correctness_rule_id": 1.0,
        "correctness_disposition": 0.0,
        "correctness_escalation": 1.0,
    }
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "execution"


def test_failure_stage_rule_selection_routing_case():
    rr = _rr()
    s = {"overall_correct": False, "routing_rule_correct": False}
    rr._score_failure_stage(s, _ROUTING_ORACLE)
    assert s["failure_stage"] == "rule_selection"


def test_failure_stage_execution_routing_case():
    rr = _rr()
    s = {
        "overall_correct": False,
        "routing_rule_correct": True,
        "routing_disposition_correct": False,
        "routing_escalation_correct": True,
    }
    rr._score_failure_stage(s, _ROUTING_ORACLE)
    assert s["failure_stage"] == "execution"


def test_failure_stage_earliest_broken_link_wins():
    """extraction, access, AND rule_selection all broken — extraction must win (stage 1)."""
    rr = _rr(arm="vector")
    s = {
        "overall_correct": False,
        "extraction_valid": False,
        "retrieval_governing_provision_retrieved": False,
        "correctness_rule_id": 0.0,
    }
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "extraction"


def test_failure_stage_access_beats_classification():
    rr = _rr(arm="vector")
    s = {
        "overall_correct": False,
        "retrieval_governing_provision_retrieved": False,
        "assigned_supplier_tier_correct": 0.0,
    }
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "access"


def test_failure_stage_unclassified_edge_case():
    """overall_correct is False but none of the tracked upstream fields indicate why."""
    rr = _rr()
    s = {"overall_correct": False}
    rr._score_failure_stage(s, _SUPPLIER_ORACLE)
    assert s["failure_stage"] == "unclassified"


# ---------------------------------------------------------------------------
# Task 3/4 — pass^k and extended report section smoke tests
# ---------------------------------------------------------------------------

def test_cell_stats_pass_all_k_excludes_infrastructure_failures():
    from csco.cli.batch_run import _cell_stats

    runs = [
        {"scores": {"overall_correct": True}, "run_status": "success"},
        {"scores": {"overall_correct": True}, "run_status": "success"},
        {"scores": {}, "run_status": "infrastructure_failure"},
    ]
    cs = _cell_stats(runs)
    assert cs["pass_all_k"] is True   # infra-failure rep excluded from consideration
    assert cs["n"] == 3               # existing n/consistent semantics untouched


def test_cell_stats_pass_all_k_false_on_any_scored_failure():
    from csco.cli.batch_run import _cell_stats

    runs = [
        {"scores": {"overall_correct": True}, "run_status": "success"},
        {"scores": {"overall_correct": False}, "run_status": "success"},
    ]
    cs = _cell_stats(runs)
    assert cs["pass_all_k"] is False


def test_extended_metrics_section_runs_without_raising():
    from csco.cli.batch_run import _generate_summary_report

    results_by_arm = {
        "static": [
            {
                "scenario_id": "fx1",
                "run_status": "success",
                "scores": {
                    "overall_correct": True,
                    "overall_correct_full": True,
                    "correctness_rule_id": 1.0,
                    "correctness_disposition": 1.0,
                    "correctness_escalation": 1.0,
                    "failure_stage": "none",
                },
            },
            {
                "scenario_id": "fx2",
                "run_status": "success",
                "scores": {
                    "overall_correct": False,
                    "overall_correct_full": False,
                    "correctness_rule_id": 0.0,
                    "correctness_disposition": 0.0,
                    "correctness_escalation": 0.0,
                    "failure_stage": "rule_selection",
                },
            },
        ],
    }
    report = _generate_summary_report(results_by_arm, n_runs=1)
    assert "EXTENDED SCORING METRICS" in report
    assert "SDM-3 vs SDM-full" in report
    assert "GROUNDING CROSS-TAB" in report
    assert "FAILURE-STAGE DISTRIBUTION" in report


# ---------------------------------------------------------------------------
# score_retrieval() routing-oracle blind spot fix
# ---------------------------------------------------------------------------

def test_score_retrieval_computes_for_routing_oracle_cases():
    """Routing-oracle fixtures (oracle.suppliers == []) previously made
    score_retrieval() return before setting retrieval_governing_provision_retrieved
    at all, leaving it None for every *-NULL/*-DT/*-DT-NULL Vector run — no
    visibility into whether a Vector *-NULL failure was retrieval-stage or
    reasoning-stage. Falls back to oracle.routing.fired_rule_id."""
    from csco.evaluation.run_result import SearchEvent
    from csco.evaluation.scoring import score_retrieval

    rr = _rr(arm="vector")
    rr.searches = [SearchEvent(
        call_number=1, query="q",
        chunks=[{"rule_id": "CYB-NULL", "heading": "(CYB-NULL) ...", "file": "cyber.md", "content_snippet": "..."}],
    )]
    s: dict = {}
    score_retrieval(rr, s, _ROUTING_ORACLE)
    assert s["retrieval_governing_provision_retrieved"] is True
    assert s["retrieval_rule_id_mentioned"] is True


def test_score_retrieval_routing_oracle_miss():
    from csco.evaluation.run_result import SearchEvent
    from csco.evaluation.scoring import score_retrieval

    rr = _rr(arm="vector")
    rr.searches = [SearchEvent(
        call_number=1, query="q",
        chunks=[{"rule_id": "CYB-DT", "heading": "(CYB-DT) ...", "file": "cyber.md", "content_snippet": "..."}],
    )]
    s: dict = {}
    score_retrieval(rr, s, _ROUTING_ORACLE)
    assert s["retrieval_governing_provision_retrieved"] is False


# ---------------------------------------------------------------------------
# _recs_by_supplier_id() "Supplier-" prefix normalization
# ---------------------------------------------------------------------------

def test_recs_by_supplier_id_normalizes_supplier_prefix_drift():
    """Suite B's name field is itself formatted "Supplier-{id}" — Static
    sometimes echoes that into its output supplier_id instead of the real
    id, with otherwise fully-correct content. Example: name="Supplier-S-SB-GEO-TIER-ABOVE",
    supplier_id="S-SB-GEO-TIER-ABOVE", model outputs
    supplier_id="Supplier-S-SB-GEO-TIER-ABOVE"."""
    from csco.evaluation.scoring import _recs_by_supplier_id

    recs_list = [{
        "supplier_id": "Supplier-S-SB-GEO-TIER-ABOVE",
        "disposition": "monitor",
        "escalation": "Cross-functional Management",
        "fired_rule_id": "GEO-MEDS",
    }]
    recs = _recs_by_supplier_id(recs_list, {"S-SB-GEO-TIER-ABOVE"})
    assert "S-SB-GEO-TIER-ABOVE" in recs
    assert recs["S-SB-GEO-TIER-ABOVE"]["fired_rule_id"] == "GEO-MEDS"


def test_recs_by_supplier_id_does_not_mask_genuine_wrong_supplier():
    """The normalization must be narrow: an id that doesn't reduce to a real
    expected id after stripping the prefix (or isn't prefixed at all) must
    NOT be silently matched to anything — a genuinely wrong supplier_id
    stays wrong."""
    from csco.evaluation.scoring import _recs_by_supplier_id

    recs_list = [{"supplier_id": "S-WRONG-SUPPLIER", "fired_rule_id": "GEO-MEDS"}]
    recs = _recs_by_supplier_id(recs_list, {"S-SB-GEO-TIER-ABOVE"})
    assert "S-SB-GEO-TIER-ABOVE" not in recs
    assert "S-WRONG-SUPPLIER" in recs  # left as-is, not silently dropped


def test_score_suppliers_correct_despite_supplier_prefix_drift():
    """End-to-end: score_suppliers() must score a "Supplier-"-prefixed id as
    matching its real supplier when content is otherwise fully correct."""
    from csco.evaluation.scoring import score_suppliers

    rr = _rr()
    rr.decision = {"recommendations": [{
        "supplier_id": "Supplier-S1",
        "disposition": "monitor",
        "escalation": "Responsible Analyst",
        "fired_rule_id": "CYB-R1",
        "permitted_actions": ["intensify_threat_monitoring"],
    }]}
    s: dict = {}
    score_suppliers(rr, s, _SUPPLIER_ORACLE)
    assert s["supplier_correctness"]["S1"]["disposition"] is True
    assert s["supplier_correctness"]["S1"]["escalation"] is True
    assert s["supplier_correctness"]["S1"]["rule_id"] is True
