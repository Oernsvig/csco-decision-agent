"""Deterministic oracle engine: derive the correct ScenarioOracle directly from spec logic.

This engine evaluates routing and supplier rules mechanically using the same
logic as the playbook encodes in prose.  It is the ground-truth source for
generated scenarios and can be used to verify hand-authored fixture oracles.

**Important:** The deterministic oracle is schema-specific and applies only to 
the five controlled policy specifications (cyber, economic, geopolitical, labour, 
natural_disaster). It performs symbolic evaluation of routing and supplier rules 
without natural language understanding. Results are valid only for scenarios 
matching the schema structure and routing rule patterns.

Public API
----------
build_oracle(scenario, spec)           -> ScenarioOracle
evaluate_routing(scenario, derived, spec) -> RoutingRule | None
evaluate_supplier(derived, scenario, spec) -> SupplierRule | None

The oracle measures **fidelity to the encoded disruption-response policy**,
not independent real-world optimality of the recommendation.

Routing evaluation
------------------
All five specs share exactly three routing rule patterns (by suffix):
  NULL     : path_on_network==False  OR  (all Tier-1 LOW AND no sub_downstream_criticality HIGH)
  DT       : tier1_on_disrupted_path==False  AND  (shared_subtier_source OR count_disrupted_paths>=2)
  DT-NULL  : tier1_on_disrupted_path==False  AND  NOT (shared_subtier_source OR count_disrupted_paths>=2)

Supplier evaluation
-------------------
Supplier rules use machine-readable ``when_all`` dicts.  Every field type
(risk_level, tier1_in_region, sub-scores by high/med/low band, path counts) is
evaluated deterministically without NLP parsing.
"""

from __future__ import annotations

import logging
from typing import Any

from csco.derive import derive
from csco.models import DerivedSupplier, Scenario
from csco.oracle.manual import ScenarioOracle, SupplierOracle, RoutingOracle
from csco.specs.schema import Spec, RoutingRule, SupplierRule

logger = logging.getLogger(__name__)

# Sub-score band thresholds — uniform across all specs
# (sub_score_bands: high >=0.66, med 0.33-0.65, low <0.33).
_SUB_HIGH_THRESHOLD = 0.66
_SUB_LOW_THRESHOLD = 0.33

_SUB_SCORE_FIELDS = frozenset({
    "sub_node_centrality",
    "sub_dependency_ratio",
    "sub_downstream_criticality",
    "sub_exposure_breadth",
    "sub_exposure_depth",
})


def _is_sub_high(value: float) -> bool:
    return value >= _SUB_HIGH_THRESHOLD


def _sub_band(value: float) -> str:
    """Classify a sub-score as high / med / low using the shared spec band thresholds."""
    if value >= _SUB_HIGH_THRESHOLD:
        return "high"
    if value < _SUB_LOW_THRESHOLD:
        return "low"
    return "med"


# ---------------------------------------------------------------------------
# Routing condition evaluators (pattern-matched by rule_id suffix)
# ---------------------------------------------------------------------------

def _routing_suffix(rule_id: str) -> str:
    """Extract routing suffix: 'NULL', 'DT', 'DT-NULL'."""
    parts = rule_id.split("-")
    # e.g. GEO-NULL -> NULL, GEO-DT -> DT, GEO-DT-NULL -> DT-NULL
    if len(parts) >= 3 and parts[-1] == "NULL" and parts[-2] == "DT":
        return "DT-NULL"
    return parts[-1]


def _routing_condition_matches(
    rule: RoutingRule,
    scenario: Scenario,
    derived: list[DerivedSupplier],
) -> bool:
    """Evaluate whether a routing rule's conditions hold for this scenario."""
    suffix = _routing_suffix(rule.rule_id)

    if suffix == "NULL":
        if not scenario.path_on_network:
            return True
        # "all Tier-1 LOW" requires at least one scored supplier.
        # When no suppliers exist (deep-tier scenarios), NULL does not fire —
        # DT or DT-NULL will handle the case instead.
        if not derived:
            return False
        all_low = all(d.risk_level == "LOW" for d in derived)
        no_high_criticality = all(
            not _is_sub_high(d.supplier.sub_downstream_criticality) for d in derived
        )
        return all_low and no_high_criticality

    if suffix == "DT":
        return (
            not scenario.tier1_on_disrupted_path
            and (
                scenario.shared_subtier_source
                or scenario.count_disrupted_paths >= 2
            )
        )

    if suffix == "DT-NULL":
        return (
            not scenario.tier1_on_disrupted_path
            and not (
                scenario.shared_subtier_source
                or scenario.count_disrupted_paths >= 2
            )
        )

    logger.warning("Unknown routing suffix %r for rule %s — treating as no-match", suffix, rule.rule_id)
    return False


def evaluate_routing(
    scenario: Scenario,
    derived: list[DerivedSupplier],
    spec: Spec,
) -> RoutingRule | None:
    """Return the first routing rule whose conditions are all satisfied, or None."""
    for rr in spec.routing_rules:
        if _routing_condition_matches(rr, scenario, derived):
            logger.debug("Routing rule %s fires", rr.rule_id)
            return rr
    return None


# ---------------------------------------------------------------------------
# Supplier condition evaluators
# ---------------------------------------------------------------------------

def _count_matches(actual: int, expr: Any) -> bool:
    if isinstance(expr, str):
        if expr.startswith(">="):
            return actual >= int(expr[2:].strip())
        if expr.startswith(">"):
            return actual > int(expr[1:].strip())
        if expr.startswith("<="):
            return actual <= int(expr[2:].strip())
        if expr.startswith("<"):
            return actual < int(expr[1:].strip())
    return actual == int(expr)


def _supplier_field_matches(
    field: str,
    expected: Any,
    ds: DerivedSupplier,
    scenario: Scenario,
) -> bool:
    """Return True if the field value for this supplier/scenario matches `expected`."""
    if field == "risk_level":
        actual = ds.risk_level
        if isinstance(expected, list):
            return actual in expected
        return actual == expected

    if field == "tier1_in_region":
        return ds.supplier.tier1_in_region == expected

    if field == "shared_subtier_source":
        return scenario.shared_subtier_source == expected

    if field == "count_disrupted_paths":
        return _count_matches(scenario.count_disrupted_paths, expected)

    if field == "tier1_on_disrupted_path":
        return scenario.tier1_on_disrupted_path == expected

    if field in _SUB_SCORE_FIELDS:
        # Sub-scores are matched by three-way band (high / med / low) so rules can
        # key on either the high band or the intermediate band (e.g. CYB-MEDS).
        return _sub_band(getattr(ds.supplier, field)) == expected

    logger.warning("Unknown when_all field %r — treating as no-match", field)
    return False


def _supplier_rule_matches(
    rule: SupplierRule,
    ds: DerivedSupplier,
    scenario: Scenario,
) -> bool:
    """Return True if ALL conditions in when_all are satisfied."""
    for field, expected in rule.when_all.items():
        if not _supplier_field_matches(field, expected, ds, scenario):
            return False
    return True


def evaluate_supplier(
    ds: DerivedSupplier,
    scenario: Scenario,
    spec: Spec,
) -> SupplierRule | None:
    """Return the first supplier rule that fires for this derived supplier, or None."""
    for sr in spec.sorted_supplier_rules:
        if _supplier_rule_matches(sr, ds, scenario):
            logger.debug("Supplier rule %s fires for %s", sr.rule_id, ds.supplier.supplier_id)
            return sr
    return None


# ---------------------------------------------------------------------------
# Full oracle construction
# ---------------------------------------------------------------------------

def build_oracle(scenario: Scenario, spec: Spec) -> ScenarioOracle:
    """Derive the correct ScenarioOracle deterministically from the spec.

    This is the ground-truth computation.  It measures fidelity to the
    encoded disruption-response policy, not independent real-world optimality.
    """
    derived = [derive(s, spec) for s in scenario.suppliers]

    # Pass 1 — routing
    fired_routing = evaluate_routing(scenario, derived, spec)
    if fired_routing is not None:
        c = fired_routing.consequence
        return ScenarioOracle(
            scenario_id="<generated>",
            disruption_type=spec.disruption_type,
            routing=RoutingOracle(
                fired_rule_id=fired_routing.rule_id,
                scenario_action=c.route,
                disposition=c.disposition,
                escalation=c.escalation,
            ),
            suppliers=[],
        )

    # Pass 2 — per supplier (only if routing exhausted and suppliers exist)
    supplier_oracles: list[SupplierOracle] = []
    strategic_priority = getattr(spec, "strategic_priority_cell", None)

    for ds in derived:
        fired_sr = evaluate_supplier(ds, scenario, spec)
        if fired_sr is None:
            logger.warning(
                "No supplier rule matched for %s (scenario=%s) — check spec completeness",
                ds.supplier.supplier_id,
                scenario.disruption_type,
            )
            continue

        c = fired_sr.consequence
        supplier_oracles.append(SupplierOracle(
            supplier_id=ds.supplier.supplier_id,
            disposition=c.disposition,
            escalation=c.escalation,
            fired_rule_id=fired_sr.rule_id,
            permitted_actions=list(c.permitted),
            strategic_priority=strategic_priority,
        ))

    return ScenarioOracle(
        scenario_id="<generated>",
        disruption_type=spec.disruption_type,
        routing=None,
        suppliers=supplier_oracles,
    )


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_oracles(
    manual: ScenarioOracle,
    deterministic: ScenarioOracle,
) -> list[str]:
    """Return a list of mismatch descriptions between manual and deterministic oracle.

    Returns an empty list if they are equivalent.
    """
    mismatches: list[str] = []

    # Routing check
    if (manual.routing is None) != (deterministic.routing is None):
        mismatches.append(
            f"routing presence mismatch: manual={'yes' if manual.routing else 'no'} "
            f"det={'yes' if deterministic.routing else 'no'}"
        )
    elif manual.routing and deterministic.routing:
        mr, dr = manual.routing, deterministic.routing
        if mr.fired_rule_id != dr.fired_rule_id:
            mismatches.append(f"routing.fired_rule_id: manual={mr.fired_rule_id!r} det={dr.fired_rule_id!r}")
        if mr.disposition != dr.disposition:
            mismatches.append(f"routing.disposition: manual={mr.disposition!r} det={dr.disposition!r}")
        if mr.escalation != dr.escalation:
            mismatches.append(f"routing.escalation: manual={mr.escalation!r} det={dr.escalation!r}")

    # Supplier checks
    manual_by_id = {s.supplier_id: s for s in manual.suppliers}
    det_by_id    = {s.supplier_id: s for s in deterministic.suppliers}

    all_ids = set(manual_by_id) | set(det_by_id)
    for sid in sorted(all_ids):
        ms = manual_by_id.get(sid)
        ds = det_by_id.get(sid)
        if ms is None:
            mismatches.append(f"supplier {sid}: missing from manual oracle")
            continue
        if ds is None:
            mismatches.append(f"supplier {sid}: missing from deterministic oracle")
            continue
        if ms.fired_rule_id != ds.fired_rule_id:
            mismatches.append(f"supplier {sid} fired_rule_id: manual={ms.fired_rule_id!r} det={ds.fired_rule_id!r}")
        if ms.disposition != ds.disposition:
            mismatches.append(f"supplier {sid} disposition: manual={ms.disposition!r} det={ds.disposition!r}")
        if ms.escalation != ds.escalation:
            mismatches.append(f"supplier {sid} escalation: manual={ms.escalation!r} det={ds.escalation!r}")
        if set(ms.permitted_actions) != set(ds.permitted_actions):
            mismatches.append(
                f"supplier {sid} permitted_actions: manual={sorted(ms.permitted_actions)} "
                f"det={sorted(ds.permitted_actions)}"
            )

    return mismatches
