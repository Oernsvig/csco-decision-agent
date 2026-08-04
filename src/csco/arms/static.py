"""Arm 1 — Static prompt.

Full decision policy for all five disruption types is baked into the system
prompt at call time. Policy is constant across all scenarios; the agent selects
the applicable type from the scenario block. No retrieval, no tools.

Process: (1) Free-text reasoning over the provided policy, (2) Shared extractor
converts to structured ScenarioDecision (same as Arm 2 & 3).
"""

from __future__ import annotations

import logging

from csco.arms.prompts import (
    STATIC_SYSTEM_TEMPLATE,
    format_all_types_policy,
    format_scenario,
)
from csco.arms.runners import run_single_turn
from csco.derive import derive
from csco.models import Scenario, ScenarioDecision
from csco.settings import get_settings
from csco.specs.loader import load_spec
from csco.tokens import count_tokens

logger = logging.getLogger(__name__)

_CONSOLIDATED_POLICY = format_all_types_policy()
# Real token count (single tokenizer, shared with the retrieval arms) so
# policy-context tokens are comparable across all three architectures.
_POLICY_TOKEN_COUNT = count_tokens(_CONSOLIDATED_POLICY)


def run(scenario: Scenario, result=None) -> ScenarioDecision:
    """Run Arm 1: consolidated all-types policy in system prompt.

    Pass a RunResult as `result` to capture instrumentation data.
    """
    spec = load_spec(scenario.disruption_type)
    derived = [derive(s, spec) for s in scenario.suppliers]

    system_prompt = STATIC_SYSTEM_TEMPLATE.format(policy_block=_CONSOLIDATED_POLICY)
    human_prompt  = format_scenario(scenario, derived)

    logger.info(
        "Arm 1 (static) — invoking LLM for %s [policy_tokens=%d]",
        scenario.disruption_type,
        _POLICY_TOKEN_COUNT,
    )

    decision: ScenarioDecision = run_single_turn(
        system_prompt,
        human_prompt,
        result=result,
        policy_tokens=_POLICY_TOKEN_COUNT,
        on_reasoning=lambda answer: logger.debug("Arm 1 reasoning: %.200s", answer),
    )

    logger.info(
        "Arm 1 complete — %d supplier recommendations, routing=%s",
        len(decision.recommendations),
        decision.routing_result.fired_rule_id if decision.routing_result else None,
    )
    return decision
