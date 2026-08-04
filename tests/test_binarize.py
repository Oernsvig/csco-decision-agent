"""Act / No-Act construct: No-Act = mandated disposition maintain at baseline escalation.

Escalation is now part of the classifier (a maintain escalated above baseline is Act);
a *permitted optional* lever does not flip a maintain outcome to Act.
"""

from __future__ import annotations

from csco.evaluation.binarize import is_null_decision
from csco.models import RoutingResult, ScenarioDecision, SupplierRecommendation

BASELINE = "Responsible Analyst"


def _routing(disposition, escalation, actions=()):
    return ScenarioDecision(
        routing_result=RoutingResult(
            fired_rule_id="X-NULL",
            scenario_action="assess",
            disposition=disposition,
            escalation=escalation,
            permitted_actions=list(actions),
        )
    )


def _supplier(disposition, escalation, actions=()):
    return ScenarioDecision(
        recommendations=[
            SupplierRecommendation(
                supplier_id="S1",
                disposition=disposition,
                escalation=escalation,
                permitted_actions=list(actions),
                fired_rule_id="X-LOW",
                rationale="",
            )
        ]
    )


def test_maintain_at_baseline_is_no_act():
    assert is_null_decision(_routing("maintain", BASELINE)) is True
    assert is_null_decision(_supplier("maintain", BASELINE)) is True


def test_maintain_with_elevated_escalation_is_act():
    assert is_null_decision(_routing("maintain", "Executive Leadership")) is False
    assert is_null_decision(_supplier("maintain", "Cross-functional Management")) is False


def test_monitor_or_replace_is_act():
    assert is_null_decision(_routing("monitor", BASELINE)) is False
    assert is_null_decision(_supplier("replace", "Executive Leadership")) is False


def test_permitted_optional_lever_does_not_flip_maintain():
    # A maintain outcome at baseline that merely *permits* an optional lever stays No-Act.
    assert is_null_decision(_supplier("maintain", BASELINE, ("intensify_threat_monitoring",))) is True


def test_empty_decision_is_unbinarizable():
    assert is_null_decision(ScenarioDecision()) is None
