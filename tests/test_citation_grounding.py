"""Tests for the citation-grounding instrumentation (cited_rule_was_opened,
cited_rule_was_opened_any, carryover_match). Purely additive scoring — no
participation in overall_correct.
"""

from csco.evaluation.run_result import OpenEvent, RunResult, SearchEvent
from csco.oracle.manual import RoutingOracle, ScenarioOracle, SupplierOracle


def _rr(arm: str = "lexical") -> RunResult:
    return RunResult.start(arm=arm, scenario_id="synthetic", disruption_type="cyber")


_SUPPLIER_ORACLE = ScenarioOracle(
    scenario_id="synthetic",
    disruption_type="cyber",
    routing=None,
    suppliers=[
        SupplierOracle(
            supplier_id="S1",
            disposition="monitor",
            escalation="Cross-functional Management",
            fired_rule_id="CYB-R1",
            permitted_actions=["intensify_threat_monitoring", "convene_crisis_response_center"],
        )
    ],
)

_TWO_SUPPLIER_ORACLE = ScenarioOracle(
    scenario_id="synthetic",
    disruption_type="cyber",
    routing=None,
    suppliers=[
        SupplierOracle(
            supplier_id="S1",
            disposition="monitor",
            escalation="Cross-functional Management",
            fired_rule_id="CYB-R1",
            permitted_actions=["intensify_threat_monitoring", "convene_crisis_response_center"],
        ),
        SupplierOracle(
            supplier_id="S2",
            disposition="monitor",
            escalation="Responsible Analyst",
            fired_rule_id="CYB-LOWS",
            permitted_actions=["intensify_threat_monitoring"],
        ),
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


# ---------------------------------------------------------------------------
# cited_rule_was_opened — lexical
# ---------------------------------------------------------------------------

def test_cited_rule_was_opened_true_when_opened():
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-R3", "provision", []), OpenEvent(2, "provision:CYB-R1", "provision", [])]
    rr.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": "CYB-R1"}]}
    s = {"extraction_valid": True}
    rr._score_citation_grounding(s, _SUPPLIER_ORACLE)
    assert s["cited_rule_was_opened"] is True
    assert s["cited_rule_was_opened_any"] is True


def test_cited_rule_was_opened_false_when_never_opened():
    """The CYB-HUB -> CYB-R1 skip-and-hallucinate case: cited but never open()'d."""
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-R3", "provision", []), OpenEvent(2, "provision:CYB-HUB", "provision", [])]
    rr.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": "CYB-R1"}]}
    s = {"extraction_valid": True}
    rr._score_citation_grounding(s, _SUPPLIER_ORACLE)
    assert s["cited_rule_was_opened"] is False
    assert s["cited_rule_was_opened_any"] is False


def test_cited_rule_was_opened_vacuously_true_on_null_conclusion():
    """A 'no provision matches' conclusion cites nothing -> vacuously grounded."""
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-R3", "provision", [])]
    rr.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": None}]}
    s = {"extraction_valid": True}
    rr._score_citation_grounding(s, _SUPPLIER_ORACLE)
    assert s["cited_rule_was_opened"] is True

    rr2 = _rr("lexical")
    rr2.opens = []
    rr2.decision = {"recommendations": []}
    s2 = {"extraction_valid": True}
    rr2._score_citation_grounding(s2, _SUPPLIER_ORACLE)
    assert s2["cited_rule_was_opened"] is True


def test_cited_rule_was_opened_none_on_extraction_failure():
    """An extraction failure must not masquerade as a hallucinated citation."""
    rr = _rr("lexical")
    rr.opens = []
    rr.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": "CYB-R1"}]}
    s = {"extraction_valid": False}
    rr._score_citation_grounding(s, _SUPPLIER_ORACLE)
    assert s["cited_rule_was_opened"] is None
    assert s["cited_rule_was_opened_any"] is None


def test_cited_rule_was_opened_all_vs_any_aggregation():
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-R1", "provision", [])]  # only S1's rule opened
    rr.decision = {
        "recommendations": [
            {"supplier_id": "S1", "fired_rule_id": "CYB-R1"},   # grounded
            {"supplier_id": "S2", "fired_rule_id": "CYB-LOWS"},  # not grounded
        ]
    }
    s = {"extraction_valid": True}
    rr._score_citation_grounding(s, _TWO_SUPPLIER_ORACLE)
    assert s["cited_rule_was_opened"] is False       # not all grounded
    assert s["cited_rule_was_opened_any"] is True    # but at least one is


def test_cited_rule_was_opened_routing_case():
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-NULL", "provision", [])]
    rr.decision = {"routing_result": {"fired_rule_id": "CYB-NULL"}}
    s = {"extraction_valid": True}
    rr._score_citation_grounding(s, _ROUTING_ORACLE)
    assert s["cited_rule_was_opened"] is True


def test_cited_rule_was_opened_static_not_applicable():
    """Static has the full corpus in context always — the flag is left unset."""
    rr = _rr("static")
    rr.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": "CYB-R1"}]}
    s = {"extraction_valid": True}
    rr._score_citation_grounding(s, _SUPPLIER_ORACLE)
    assert "cited_rule_was_opened" not in s


def test_cited_rule_was_opened_vector_analog():
    rr = _rr("vector")
    rr.searches = [SearchEvent(1, "q", [{"rule_id": "CYB-R1"}])]
    rr.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": "CYB-R1"}]}
    s = {"extraction_valid": True}
    rr._score_citation_grounding(s, _SUPPLIER_ORACLE)
    assert s["cited_rule_was_opened"] is True

    rr2 = _rr("vector")
    rr2.searches = [SearchEvent(1, "q", [{"rule_id": "CYB-HUB"}])]
    rr2.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": "CYB-R1"}]}
    s2 = {"extraction_valid": True}
    rr2._score_citation_grounding(s2, _SUPPLIER_ORACLE)
    assert s2["cited_rule_was_opened"] is False


# ---------------------------------------------------------------------------
# carryover_match
# ---------------------------------------------------------------------------

def test_carryover_last_opened_sibling():
    """CYB-HUB->CYB-R1 skip: cited CYB-R1, tuple is an exact copy of the last
    real provision opened (CYB-HUB), which was never followed by CYB-R1 itself."""
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-R3", "provision", []), OpenEvent(2, "provision:CYB-HUB", "provision", [])]
    rr.decision = {
        "recommendations": [{
            "supplier_id": "S1",
            "fired_rule_id": "CYB-R1",
            "disposition": "replace",
            "escalation": "Executive Leadership",
            "permitted_actions": ["activate_pre_qualified_alternate", "intensify_threat_monitoring", "convene_crisis_response_center"],
        }]
    }
    s = {
        "overall_correct": False,
        "supplier_correctness": {"S1": {"all_correct": False}},
    }
    rr._score_carryover_match(s, _SUPPLIER_ORACLE)
    assert s["carryover_match"] == "last_opened_sibling"


def test_carryover_other_sibling():
    """Tuple matches a real sibling that was NOT the last one opened."""
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-R3", "provision", [])]  # last-opened is R3, not MEDS
    rr.decision = {
        "recommendations": [{
            "supplier_id": "S1",
            "fired_rule_id": "CYB-R1",
            "disposition": "monitor",
            "escalation": "Cross-functional Management",
            "permitted_actions": ["activate_pre_qualified_alternate", "intensify_threat_monitoring"],  # == CYB-MEDS
        }]
    }
    s = {
        "overall_correct": False,
        "supplier_correctness": {"S1": {"all_correct": False}},
    }
    rr._score_carryover_match(s, _SUPPLIER_ORACLE)
    assert s["carryover_match"] == "other_sibling"


def test_carryover_no_match():
    """A tuple that matches no real sibling's consequence — genuine fabrication."""
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-R3", "provision", [])]
    rr.decision = {
        "recommendations": [{
            "supplier_id": "S1",
            "fired_rule_id": "CYB-R1",
            "disposition": "replace",
            "escalation": "Responsible Analyst",  # no real CYB rule has this disposition/escalation pairing
            "permitted_actions": ["intensify_threat_monitoring"],
        }]
    }
    s = {
        "overall_correct": False,
        "supplier_correctness": {"S1": {"all_correct": False}},
    }
    rr._score_carryover_match(s, _SUPPLIER_ORACLE)
    assert s["carryover_match"] == "no_match"


def test_carryover_none_when_correct():
    rr = _rr("lexical")
    rr.opens = [OpenEvent(1, "provision:CYB-R1", "provision", [])]
    rr.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": "CYB-R1"}]}
    s = {"overall_correct": True}
    rr._score_carryover_match(s, _SUPPLIER_ORACLE)
    assert "carryover_match" not in s


def test_carryover_none_for_non_lexical_arm():
    rr = _rr("vector")
    rr.decision = {"recommendations": [{"supplier_id": "S1", "fired_rule_id": "CYB-HUB"}]}
    s = {"overall_correct": False, "supplier_correctness": {"S1": {"all_correct": False}}}
    rr._score_carryover_match(s, _SUPPLIER_ORACLE)
    assert "carryover_match" not in s


def test_carryover_none_for_routing_case():
    rr = _rr("lexical")
    rr.decision = {"routing_result": {"fired_rule_id": "CYB-DT"}}
    s = {"overall_correct": False}
    rr._score_carryover_match(s, _ROUTING_ORACLE)
    assert "carryover_match" not in s


# ---------------------------------------------------------------------------
# Purely additive: new scorers must not perturb any pre-existing score value
# ---------------------------------------------------------------------------

def test_new_scorers_are_purely_additive_end_to_end():
    """Full .score() run against the real diag_0026 (CYB-R1) oracle: pre-existing
    keys keep their expected values with the new scorers wired into the call
    chain."""
    rr = RunResult.start(arm="lexical", scenario_id="diag_0026", disruption_type="cyber")
    rr.opens = [
        OpenEvent(1, "provision:CYB-R3", "provision", []),
        OpenEvent(2, "provision:CYB-HUB", "provision", []),
        OpenEvent(3, "provision:CYB-R1", "provision", []),
    ]
    rr.agent_answer_raw = "Provision (CYB-R1) fires."
    rr.decision = {
        "recommendations": [{
            "supplier_id": "S-GEN-CYB-01",
            "fired_rule_id": "CYB-R1",
            "disposition": "monitor",
            "escalation": "Cross-functional Management",
            "permitted_actions": ["intensify_threat_monitoring", "convene_crisis_response_center"],
            "rationale": "flexibility",
            "assigned_supplier_tier": "routine",
            "assigned_risk_level": "HIGH",
        }]
    }

    rr.score()

    assert rr.scores["overall_correct"] is True
    assert rr.scores["failure_stage"] == "none"
    assert rr.scores["cited_rule_was_opened"] is True
    assert "carryover_match" not in rr.scores  # correct run, nothing to classify
