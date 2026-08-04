"""Act / No-Act binarization — single source of truth for both oracle and agent sides.

A scenario resolves to "no-act" (null) when the policy outcome is maintain-and-log:
a routing-level stop with disposition == "maintain", or (if a supplier pass ran)
every scored supplier with disposition == "maintain" — in both cases at the baseline
analyst escalation level. Any monitor or replace disposition, or any escalation beyond
the baseline analyst level, counts as "act".

The class is determined by the *mandated* disposition and escalation, not by the
optional live actions a provision *permits*. In this corpus a maintain outcome is
always paired with the baseline escalation and carries at most optional levers
(e.g. permitted intensify-monitoring on a LOW-risk supplier), so disposition and
escalation fully determine the class; a permitted optional lever does not flip a
maintain outcome to "act". (Empirically confirmed: every maintain output — in the
oracle and across all reported runs — is at the baseline escalation, so adding the
escalation guard leaves the reported confusion matrix unchanged.)

Convention used throughout the skew/prevalence analysis: POSITIVE = action-required
(oracle_null == False).
"""

from __future__ import annotations

from csco.models import ScenarioDecision
from csco.oracle.manual import ScenarioOracle

# Lowest escalation level. A maintain outcome escalated above this signals
# action-intent and is classified as "act", not "no-act".
_BASELINE_ESCALATION = "Responsible Analyst"


def _is_maintain_and_log(disposition: str, escalation: str) -> bool:
    return disposition == "maintain" and escalation == _BASELINE_ESCALATION


def is_null_oracle(oracle: ScenarioOracle) -> bool:
    """Return True if the ground-truth oracle resolves to the null ("no-act") outcome.

    - Routing fired: null iff the routing consequence is maintain at baseline escalation.
    - No routing (supplier pass ran): null iff every scored supplier is maintain at
      baseline escalation. An oracle with no routing AND no suppliers is a malformed/
      empty case that should never occur for a real fixture; it is treated as NOT null
      (conservatively "action-required-shaped") rather than vacuously True, since an
      empty ``all()`` would otherwise silently claim nullity.
    """
    if oracle.routing is not None:
        return _is_maintain_and_log(oracle.routing.disposition, oracle.routing.escalation)
    if not oracle.suppliers:
        return False
    return all(_is_maintain_and_log(s.disposition, s.escalation) for s in oracle.suppliers)


def is_null_decision(decision: ScenarioDecision) -> bool | None:
    """Return True/False for the agent's decision on the same null/act axis, or
    None when the decision cannot be classified at all.

    None is returned for empty/failed extraction — a routing_result of None together
    with an empty recommendations list. Callers MUST treat None as "unbinarizable" and
    exclude it from the 2x2 confusion table (counted in a separate bucket), never
    silently coerce it to False.

    Deviation from a strict ``-> bool`` signature: the edge-case contract (empty/failed
    extraction must be excluded, not silently scored as a miss) requires a third state,
    so this returns ``bool | None``.
    """
    if decision.routing_result is not None:
        return _is_maintain_and_log(
            decision.routing_result.disposition, decision.routing_result.escalation
        )
    if not decision.recommendations:
        return None
    return all(
        _is_maintain_and_log(r.disposition, r.escalation) for r in decision.recommendations
    )
