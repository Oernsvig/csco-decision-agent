"""Tests for the skew/prevalence evaluation package:

  Task 1 — Suite B null-region + boundary-pair fixture generation
  Task 2 — Act/No-Act binarization (csco.binarize)
  Task 3 — MCC + prevalence-invariant rates (csco.skew_analysis)
  Task 4 — Acc(pi) / cost(pi) post-stratified reporting
  Task 5 — SKEW & PREVALENCE ANALYSIS report section, incl. legacy backward compat
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from csco.evaluation.binarize import is_null_decision, is_null_oracle
from csco.models import RoutingResult, ScenarioDecision, SupplierRecommendation
from csco.oracle.manual import RoutingOracle, ScenarioOracle, SupplierOracle

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Task 2 — binarization predicates
# ---------------------------------------------------------------------------

def _rec(supplier_id: str, disposition: str) -> SupplierRecommendation:
    return SupplierRecommendation(
        supplier_id=supplier_id,
        disposition=disposition,
        escalation="Responsible Analyst",
        permitted_actions=["intensify_threat_monitoring"],
        fired_rule_id="X-1",
        rationale="r",
    )


def test_is_null_oracle_routing_maintain_is_null():
    oracle = ScenarioOracle(
        scenario_id="s", disruption_type="cyber",
        routing=RoutingOracle(fired_rule_id="CYB-NULL", scenario_action="maintain_and_log",
                               disposition="maintain", escalation="Responsible Analyst"),
        suppliers=[],
    )
    assert is_null_oracle(oracle) is True


def test_is_null_oracle_routing_non_maintain_is_not_null():
    oracle = ScenarioOracle(
        scenario_id="s", disruption_type="cyber",
        routing=RoutingOracle(fired_rule_id="CYB-DT", scenario_action="structural_fallthrough",
                               disposition="monitor", escalation="Cross-functional Management"),
        suppliers=[],
    )
    assert is_null_oracle(oracle) is False


def test_is_null_oracle_all_suppliers_maintain_is_null():
    oracle = ScenarioOracle(
        scenario_id="s", disruption_type="cyber", routing=None,
        suppliers=[
            SupplierOracle(supplier_id="S1", disposition="maintain", escalation="Responsible Analyst",
                            fired_rule_id="CYB-LOW", permitted_actions=[]),
            SupplierOracle(supplier_id="S2", disposition="maintain", escalation="Responsible Analyst",
                            fired_rule_id="CYB-LOW", permitted_actions=[]),
        ],
    )
    assert is_null_oracle(oracle) is True


def test_is_null_oracle_mixed_suppliers_is_not_null():
    oracle = ScenarioOracle(
        scenario_id="s", disruption_type="cyber", routing=None,
        suppliers=[
            SupplierOracle(supplier_id="S1", disposition="maintain", escalation="Responsible Analyst",
                            fired_rule_id="CYB-LOW", permitted_actions=[]),
            SupplierOracle(supplier_id="S2", disposition="monitor", escalation="Cross-functional Management",
                            fired_rule_id="CYB-MED", permitted_actions=[]),
        ],
    )
    assert is_null_oracle(oracle) is False


def test_is_null_oracle_empty_is_conservatively_not_null():
    """No routing verdict and no suppliers scored — malformed/empty; must not
    vacuously report null via an empty all()."""
    oracle = ScenarioOracle(scenario_id="s", disruption_type="cyber", routing=None, suppliers=[])
    assert is_null_oracle(oracle) is False


def test_is_null_decision_routing_maintain():
    decision = ScenarioDecision(
        routing_result=RoutingResult(fired_rule_id="X-NULL", scenario_action="maintain_and_log",
                                      disposition="maintain", escalation="Responsible Analyst", permitted_actions=[]),
    )
    assert is_null_decision(decision) is True


def test_is_null_decision_routing_non_maintain():
    decision = ScenarioDecision(
        routing_result=RoutingResult(fired_rule_id="X-DT", scenario_action="structural_fallthrough",
                                      disposition="monitor", escalation="Cross-functional Management", permitted_actions=[]),
    )
    assert is_null_decision(decision) is False


def test_is_null_decision_empty_extraction_is_none():
    """Empty/failed extraction (no routing_result, no recommendations) must be
    excluded from the 2x2 as unbinarizable, never coerced to False."""
    decision = ScenarioDecision(routing_result=None, recommendations=[])
    assert is_null_decision(decision) is None


def test_is_null_decision_mixed_supplier_dispositions_is_not_null():
    decision = ScenarioDecision(recommendations=[_rec("S1", "maintain"), _rec("S2", "monitor")])
    assert is_null_decision(decision) is False


def test_is_null_decision_all_suppliers_maintain_is_null():
    decision = ScenarioDecision(recommendations=[_rec("S1", "maintain"), _rec("S2", "maintain")])
    assert is_null_decision(decision) is True


# ---------------------------------------------------------------------------
# RunResult.score() wiring — scores["act_axis"]
# ---------------------------------------------------------------------------

def test_run_result_act_axis_true_positive():
    from csco.evaluation.run_result import RunResult

    rr = RunResult.start(arm="static", scenario_id="synthetic_act", disruption_type="cyber")
    oracle = ScenarioOracle(
        scenario_id="synthetic_act", disruption_type="cyber", routing=None,
        suppliers=[SupplierOracle(supplier_id="S1", disposition="monitor", escalation="Cross-functional Management",
                                   fired_rule_id="CYB-R1", permitted_actions=["intensify_threat_monitoring"])],
    )
    rr.decision = {
        "routing_result": None,
        "recommendations": [{
            "supplier_id": "S1", "disposition": "monitor", "escalation": "Cross-functional Management",
            "fired_rule_id": "CYB-R1", "permitted_actions": ["intensify_threat_monitoring"],
            "rationale": "x",
        }],
    }
    s: dict = {}
    rr._score_act_axis(s, oracle)
    assert s["act_axis"] == {"oracle_null": False, "agent_null": False, "cell": "TP"}


def test_run_result_act_axis_unbinarizable_on_empty_decision():
    from csco.evaluation.run_result import RunResult

    rr = RunResult.start(arm="static", scenario_id="synthetic_empty", disruption_type="cyber")
    oracle = ScenarioOracle(
        scenario_id="synthetic_empty", disruption_type="cyber", routing=None,
        suppliers=[SupplierOracle(supplier_id="S1", disposition="monitor", escalation="Cross-functional Management",
                                   fired_rule_id="CYB-R1", permitted_actions=[])],
    )
    rr.decision = {"routing_result": None, "recommendations": []}
    s: dict = {}
    rr._score_act_axis(s, oracle)
    assert s["act_axis"]["cell"] == "unbinarizable"
    assert s["act_axis"]["agent_null"] is None


def test_run_result_act_axis_false_positive_false_alarm():
    from csco.evaluation.run_result import RunResult

    rr = RunResult.start(arm="static", scenario_id="synthetic_fp", disruption_type="cyber")
    oracle = ScenarioOracle(
        scenario_id="synthetic_fp", disruption_type="cyber",
        routing=RoutingOracle(fired_rule_id="CYB-NULL", scenario_action="maintain_and_log",
                               disposition="maintain", escalation="Responsible Analyst"),
        suppliers=[],
    )
    rr.decision = {
        "routing_result": {"fired_rule_id": "CYB-DT", "scenario_action": "structural_fallthrough",
                            "disposition": "monitor", "escalation": "Cross-functional Management", "permitted_actions": []},
        "recommendations": [],
    }
    s: dict = {}
    rr._score_act_axis(s, oracle)
    assert s["act_axis"] == {"oracle_null": True, "agent_null": False, "cell": "FP"}


# ---------------------------------------------------------------------------
# Task 3 — MCC + rates
# ---------------------------------------------------------------------------

def test_mcc_hand_computed():
    from csco.evaluation.skew_analysis import mcc_from_counts

    mcc = mcc_from_counts(tp=13, tn=80, fp=5, fn=2)
    assert isinstance(mcc, float)
    assert abs(mcc - 0.75) < 0.01


def test_mcc_single_class_guard():
    from csco.evaluation.skew_analysis import mcc_from_counts

    assert mcc_from_counts(tp=0, tn=10, fp=0, fn=0) == "n/a (single-class)"
    assert mcc_from_counts(tp=10, tn=0, fp=0, fn=0) == "n/a (single-class)"


def test_paired_mcc_difference_ci_uses_shared_scenario_clusters():
    from csco.evaluation.bootstrap import paired_mcc_difference_ci

    def _run(scenario_id: str, cell: str) -> dict:
        return {
            "scenario_id": scenario_id,
            "run_status": "success",
            "scores": {"act_axis": {"cell": cell}},
        }

    # Arm A is a perfect separator; arm B is balanced but imperfect.
    scenarios = [f"s{i}" for i in range(8)]
    runs_a = [_run(sid, "TP") for sid in scenarios[:4]] + [_run(sid, "TN") for sid in scenarios[4:]]
    runs_b = (
        [_run("s0", "TP"), _run("s1", "TP"), _run("s2", "TN"), _run("s3", "TN")]
        + [_run("s4", "FP"), _run("s5", "FP"), _run("s6", "FN"), _run("s7", "FN")]
    )

    result = paired_mcc_difference_ci(runs_a, runs_b, n_boot=500, seed=123)

    assert result["n_scenarios"] == 8
    assert result["point"] == pytest.approx(1.0)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_false_alarm_and_miss_rate():
    from csco.evaluation.skew_analysis import false_alarm_rate, miss_rate

    assert false_alarm_rate(fp=5, tn=80) == pytest.approx(5 / 85)
    assert miss_rate(fn=2, tp=13) == pytest.approx(2 / 15)
    assert false_alarm_rate(fp=0, tn=0) is None
    assert miss_rate(fn=0, tp=0) is None


def test_confusion_counts_tracks_unbinarizable_and_legacy_separately():
    from csco.evaluation.skew_analysis import confusion_counts

    runs = [
        {"scores": {"act_axis": {"cell": "TP"}}},
        {"scores": {"act_axis": {"cell": "unbinarizable"}}},
        {"scores": {}},  # legacy run predating act_axis
    ]
    counts = confusion_counts(runs)
    assert counts["TP"] == 1
    assert counts["unbinarizable"] == 1
    assert counts["no_act_axis"] == 1


# ---------------------------------------------------------------------------
# Task 4 — Acc(pi) linearity
# ---------------------------------------------------------------------------

def test_acc_pi_linearity_endpoints():
    from csco.evaluation.skew_analysis import acc_pi_rows

    runs = [
        {"scenario_id": "n1", "fixture_meta": {"stratum": "null"}, "scores": {"overall_correct": True}},
        {"scenario_id": "n2", "fixture_meta": {"stratum": "null"}, "scores": {"overall_correct": False}},
        {"scenario_id": "a1", "fixture_meta": {"stratum": "action"}, "scores": {"overall_correct": True}},
        {"scenario_id": "a2", "fixture_meta": {"stratum": "action"}, "scores": {"overall_correct": True}},
        {"scenario_id": "a3", "fixture_meta": {"stratum": "action"}, "scores": {"overall_correct": False}},
    ]
    rows = acc_pi_rows(runs, arm="static", pi_values=[0.0, 1.0])
    by_pi = {r["pi"]: r for r in rows}
    assert by_pi[0.0]["acc_pi"] == pytest.approx(by_pi[0.0]["acc_action"])
    assert by_pi[1.0]["acc_pi"] == pytest.approx(by_pi[1.0]["acc_null"])
    assert by_pi[0.0]["acc_action"] == pytest.approx(2 / 3)
    assert by_pi[1.0]["acc_null"] == pytest.approx(0.5)


def test_acc_pi_missing_stratum_is_na_not_crash():
    from csco.evaluation.skew_analysis import acc_pi_rows

    runs = [{"scenario_id": "a1", "fixture_meta": {"stratum": "action"}, "scores": {"overall_correct": True}}]
    rows = acc_pi_rows(runs, arm="static", pi_values=[0.5])
    assert rows[0]["acc_pi"] is None


def test_acc_pi_sdm_full_metric_computed_separately_from_sdm3():
    """Deck: 'SDM-full adds action levers, both weighted to null share pi' —
    Acc(pi) must be computable for overall_correct_full, not just overall_correct."""
    from csco.evaluation.skew_analysis import acc_pi_rows

    runs = [
        {"scenario_id": "n1", "fixture_meta": {"stratum": "null"},
         "scores": {"overall_correct": True, "overall_correct_full": False}},
        {"scenario_id": "a1", "fixture_meta": {"stratum": "action"},
         "scores": {"overall_correct": True, "overall_correct_full": True}},
    ]
    sdm3_rows = acc_pi_rows(runs, arm="static", pi_values=[1.0], metric="overall_correct")
    sdmfull_rows = acc_pi_rows(runs, arm="static", pi_values=[1.0], metric="overall_correct_full")
    assert sdm3_rows[0]["acc_null"] == pytest.approx(1.0)
    assert sdmfull_rows[0]["acc_null"] == pytest.approx(0.0)


def test_acc_pi_sdm_full_na_on_legacy_runs_missing_the_key():
    """Legacy runs (scored before overall_correct_full existed) must render
    n/a for SDM-full, not crash and not silently read as 0%."""
    from csco.evaluation.skew_analysis import acc_pi_rows

    runs = [
        {"scenario_id": "n1", "fixture_meta": {"stratum": "null"}, "scores": {"overall_correct": True}},
        {"scenario_id": "a1", "fixture_meta": {"stratum": "action"}, "scores": {"overall_correct": True}},
    ]
    rows = acc_pi_rows(runs, arm="static", pi_values=[0.5], metric="overall_correct_full")
    assert rows[0]["acc_pi"] is None
    assert rows[0]["n_null"] == 0
    assert rows[0]["n_action"] == 0
    assert rows[0]["n_null"] == 0


# ---------------------------------------------------------------------------
# Task 1 — Suite B generation
# ---------------------------------------------------------------------------

def test_suite_b_generation_is_deterministic(tmp_path):
    from csco.generators.benchmark import generate_suite_b

    out1 = tmp_path / "sb1"
    out2 = tmp_path / "sb2"
    meta1 = generate_suite_b(output_dir=out1, seed=777)
    meta2 = generate_suite_b(output_dir=out2, seed=777)

    assert meta1["total_fixtures"] == meta2["total_fixtures"]
    files1 = sorted(p.name for p in out1.glob("*.yaml"))
    files2 = sorted(p.name for p in out2.glob("*.yaml"))
    assert files1 == files2
    for name in files1:
        assert (out1 / name).read_text() == (out2 / name).read_text()


def test_suite_b_per_stratum_counts_match_grid(tmp_path):
    from csco.generators.benchmark import generate_suite_b, SUITE_B_NULL_MODES, SUITE_B_BOUNDARY_CONDITIONS

    meta = generate_suite_b(output_dir=tmp_path / "sb", seed=42)
    breakdown = meta["breakdown"]

    n_types = 5
    expected_null = n_types * len(SUITE_B_NULL_MODES) * 1
    expected_boundary = n_types * len(SUITE_B_BOUNDARY_CONDITIONS) * 2
    assert meta["total_fixtures"] == expected_null + expected_boundary
    # Legacy null+boundary probe set: 20 null (4 modes x 5 types) + 20 boundary
    # (2 conditions x 2 sides x 5 types). The frozen dissertation Suite B (35) is
    # produced by generate_dissertation_suites(); see test_dissertation_suites_*.
    assert meta["total_fixtures"] == 40

    for dtype, modes in breakdown["null"].items():
        assert set(modes) == set(SUITE_B_NULL_MODES)
        assert all(count == 1 for count in modes.values())

    for dtype, conds in breakdown["boundary"].items():
        assert set(conds) == set(SUITE_B_BOUNDARY_CONDITIONS)
        for cond, sides in conds.items():
            assert set(sides) == {"below", "above"}


def test_suite_b_no_path_null_mode_is_unambiguous_vs_dt_null(tmp_path):
    """Regression test: the no_path_on_network null-mode fixture must not also
    structurally satisfy DT or DT-NULL's condition (same bug/fix as
    scenario_generator._build_routing_scenario's NULL case — see
    test_fairness.py::test_generated_null_scenario_is_unambiguous_vs_dt_null).
    """
    from csco.oracle.deterministic import _routing_condition_matches, _routing_suffix
    from csco.models import Scenario
    from csco.specs.loader import load_spec
    from csco.generators.benchmark import generate_suite_b

    out = tmp_path / "sb"
    generate_suite_b(output_dir=out, seed=123)

    for path in sorted(out.glob("sb_*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("null_mode") != "no_path_on_network":
            continue
        scenario = Scenario.model_validate(raw)
        spec = load_spec(scenario.disruption_type)
        dt_rule = next(rr for rr in spec.routing_rules if _routing_suffix(rr.rule_id) == "DT")
        dt_null_rule = next(rr for rr in spec.routing_rules if _routing_suffix(rr.rule_id) == "DT-NULL")

        assert _routing_condition_matches(dt_rule, scenario, []) is False, (
            f"{path.name}: no_path_on_network fixture must NOT also satisfy DT's condition"
        )
        assert _routing_condition_matches(dt_null_rule, scenario, []) is False, (
            f"{path.name}: no_path_on_network fixture must NOT also satisfy DT-NULL's condition"
        )


def test_suite_b_fixtures_carry_stratum_metadata_and_agree_with_oracle(tmp_path):
    from csco.oracle.deterministic import build_oracle
    from csco.models import Scenario
    from csco.specs.loader import load_spec

    out = tmp_path / "sb"
    from csco.generators.benchmark import generate_suite_b
    generate_suite_b(output_dir=out, seed=99)

    checked_null = checked_boundary = 0
    for yaml_path in sorted(out.glob("sb_*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert raw["suite"] == "B"
        assert raw["stratum"] in ("null", "action")

        scenario = Scenario.model_validate(raw)
        spec = load_spec(scenario.disruption_type)
        oracle = build_oracle(scenario, spec)

        if raw["stratum"] == "null":
            assert is_null_oracle(oracle) is True
            checked_null += 1
        else:
            checked_boundary += 1
            fired = oracle.routing.fired_rule_id if oracle.routing else oracle.suppliers[0].fired_rule_id
            if raw["boundary"] == "above":
                assert fired == raw["target_rule_id"]
            elif raw["boundary"] == "below":
                assert fired != raw["target_rule_id"]

    assert checked_null == 20
    assert checked_boundary == 20


def test_suite_b_oracle_agreement_assertion_actually_fails_on_broken_fixture(monkeypatch, tmp_path):
    """Deliberately break the null-oracle predicate so a Suite B "null" fixture
    no longer resolves to null — generation must raise, not silently emit."""
    import csco.generators.benchmark as bg

    def _always_false(oracle):
        return False

    monkeypatch.setattr(bg, "is_null_oracle", _always_false)

    with pytest.raises(RuntimeError, match="generation bug"):
        bg.generate_suite_b(output_dir=tmp_path / "broken", seed=1)


# ---------------------------------------------------------------------------
# Task 5 — backward compatibility over legacy calibration_v3 results
# ---------------------------------------------------------------------------

def _load_legacy_runs(limit: int = 40) -> list[dict]:
    files = sorted((RESULTS_DIR / "legacy" / "calibration_v3").rglob("*.json"))[:limit]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def test_report_over_legacy_results_runs_clean_with_na_labels():
    from csco.cli.batch_run import _generate_summary_report

    if not (RESULTS_DIR / "legacy" / "calibration_v3").exists():
        pytest.skip("results/legacy/calibration_v3 not present in this checkout")

    runs = _load_legacy_runs()
    assert runs, "expected at least one legacy calibration_v3 run"
    assert all("act_axis" not in r.get("scores", {}) for r in runs)
    assert all("fixture_meta" not in r for r in runs)

    by_arm: dict[str, list[dict]] = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)

    report = _generate_summary_report(by_arm, n_runs=1)  # must not raise
    assert "SKEW & PREVALENCE ANALYSIS" in report
