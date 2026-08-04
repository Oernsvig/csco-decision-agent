"""Constrained rule-targeted scenario generator.

Generates diagnostic scenario fixtures from spec logic without random guessing.
Each generated scenario targets a specific rule (or diagnostic cluster), satisfies
all its conditions deterministically, and receives oracle labels computed by the
deterministic oracle engine.

Usage
-----
    # Generate one scenario per rule
    python -m csco.generators.scenario --output-dir fixtures/generated/

    # Target a specific rule
    python -m csco.generators.scenario --rule CYB-HUB --output-dir fixtures/generated/

    # Target a diagnostic cluster
    python -m csco.generators.scenario --cluster C3_PRIORITY_OVERRIDE --output-dir fixtures/generated/

    # Generate contrast pairs
    python -m csco.generators.scenario --pairs --output-dir fixtures/generated/

Output
------
    fixtures/generated/
      gen_0001.yaml          — scenario YAML (loadable by csco.run)
      gen_0001.meta.json     — generation metadata (target rule, clusters, contrast pair)
      gen_0001.oracle.json   — deterministic oracle output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from csco.evaluation.clusters import CONTRAST_PAIRS, ALL_CLUSTER_IDS
from csco.oracle.deterministic import build_oracle
from csco.models import Scenario
from csco.specs.loader import load_all_specs
from csco.specs.schema import Spec, SupplierRule, RoutingRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score constants
# ---------------------------------------------------------------------------

_RISK_MEDIUM_MID  = 0.50    # well inside MEDIUM for all types
_RISK_HIGH_GEO    = 0.60    # HIGH for geopolitical (>=0.55)
_RISK_HIGH_OTHER  = 0.65    # HIGH for other types (>=0.60)
_RISK_LOW_MID     = 0.35    # well inside LOW

_SUB_HIGH = 0.72    # sub-score that registers as 'high' (>=0.66)
_SUB_MED  = 0.50    # sub-score that registers as 'med'  (0.33-0.65)
_SUB_LOW  = 0.25    # sub-score that registers as 'low'  (<0.33)

# Value for a sub-score a rule does NOT gate on: use the LOW band so it never
# accidentally satisfies a rule that keys on the 'med' band (e.g. CYB-MEDS on
# sub_node_centrality: med).
_NON_TRIGGERING_SUB = _SUB_LOW

# Map a sub-score band name in a rule's when_all to a concrete value.
_SUB_BAND_VALUE = {"high": _SUB_HIGH, "med": _SUB_MED, "low": _SUB_LOW}


# ---------------------------------------------------------------------------
# Base scenario builder
# ---------------------------------------------------------------------------

def _base_scenario_fields(disruption_type: str) -> dict:
    """Minimum fields for a safe scenario that won't trigger routing rules."""
    return {
        "disruption_type": disruption_type,
        "affected_regions": ["Region-A"],
        "impacted_industries": ["Industry-X"],
        "named_companies": ["Company-Z"],
        "path_on_network": True,
        "count_disrupted_paths": 1,
        "max_disrupted_tier": 1,
        "shared_subtier_source": False,
        "tier1_on_disrupted_path": True,
        "suppliers": [],
    }


def _supplier_template(supplier_id: str, risk_score: float, tier1_in_region: bool = False) -> dict:
    """Generic supplier with low sub-scores (routine tier, configurable risk band)."""
    return {
        "supplier_id": supplier_id,
        "name": f"Supplier-{supplier_id}",
        "product_supplied": "component-X",
        "tier1_in_region": tier1_in_region,
        "risk_score": risk_score,
        "sub_exposure_breadth":       _SUB_MED,
        "sub_dependency_ratio":       _SUB_MED,
        "sub_downstream_criticality": _NON_TRIGGERING_SUB,
        "sub_node_centrality":        _NON_TRIGGERING_SUB,
        "sub_exposure_depth":         _SUB_MED,
    }


def _risk_score_for(disruption_type: str, band: str) -> float:
    if band == "HIGH":
        return _RISK_HIGH_GEO if disruption_type == "geopolitical" else _RISK_HIGH_OTHER
    if band == "LOW":
        return _RISK_LOW_MID
    return _RISK_MEDIUM_MID


# ---------------------------------------------------------------------------
# Routing scenario builders
# ---------------------------------------------------------------------------

def _build_routing_scenario(rule: RoutingRule, dtype: str) -> dict | None:
    """Return a scenario dict that satisfies the given routing rule's conditions."""
    suffix = rule.rule_id.split("-")[-1]
    parts  = rule.rule_id.split("-")

    if parts[-1] == "NULL" and len(parts) >= 3 and parts[-2] == "DT":
        suffix = "DT-NULL"

    fields = _base_scenario_fields(dtype)

    if suffix == "NULL":
        # Trigger: path_on_network == False. tier1_on_disrupted_path is set True
        # (not False) so this scenario does NOT also satisfy DT/DT-NULL's shared
        # precondition (tier1_on_disrupted_path == False) — DT and DT-NULL are
        # both structurally False here, leaving NULL as the only routing rule
        # whose condition holds. Prior to this fix, tier1_on_disrupted_path was
        # also False, which made the "canonical NULL" fixture simultaneously
        # satisfy DT-NULL's condition too, disambiguated only by spec rule-list
        # order (NULL listed first) rather than by the scenario data itself.
        fields["path_on_network"] = False
        fields["tier1_on_disrupted_path"] = True
        fields["suppliers"] = []
        return fields

    if suffix == "DT":
        fields["tier1_on_disrupted_path"] = False
        fields["shared_subtier_source"]   = True
        fields["count_disrupted_paths"]   = 2
        fields["max_disrupted_tier"]      = 3
        fields["suppliers"]               = []
        return fields

    if suffix == "DT-NULL":
        fields["tier1_on_disrupted_path"] = False
        fields["shared_subtier_source"]   = False
        fields["count_disrupted_paths"]   = 1
        fields["max_disrupted_tier"]      = 3
        fields["suppliers"]               = []
        return fields

    return None


# ---------------------------------------------------------------------------
# Supplier scenario builders
# ---------------------------------------------------------------------------

def _build_supplier_scenario(rule: SupplierRule, dtype: str, spec: Spec) -> dict | None:
    """Return a scenario dict that satisfies this supplier rule (and only this rule)."""
    wa = rule.when_all
    fields = _base_scenario_fields(dtype)

    # Determine risk score
    risk_levels = wa.get("risk_level")
    if isinstance(risk_levels, list):
        band = risk_levels[0]  # use the lower one to avoid triggering higher rules
        # Prefer MEDIUM so scenario sits in ambiguous territory
        if "MEDIUM" in risk_levels:
            band = "MEDIUM"
    elif risk_levels:
        band = risk_levels
    else:
        band = "MEDIUM"
    risk_score = _risk_score_for(dtype, band)

    # Determine tier
    tier1_in_region = bool(wa.get("tier1_in_region", False))

    # Build supplier
    supplier_id = f"S-GEN-{dtype[:3].upper()}-01"
    supplier = _supplier_template(supplier_id, risk_score, tier1_in_region)

    # A LOW-band supplier would otherwise trigger the Pass-1 NULL routing rule
    # ("all Tier-1 LOW AND sub_downstream_criticality not high") before any Pass-2
    # supplier rule is reached. Give the canonical LOW-band supplier high downstream
    # criticality so routing falls through to Pass 2. (No LOW-band supplier rule gates
    # on downstream criticality, so an explicit when_all value below still wins.)
    if band == "LOW":
        supplier["sub_downstream_criticality"] = _SUB_HIGH

    # Apply sub-score overrides from when_all (high / med / low bands)
    for _sub_field in (
        "sub_node_centrality",
        "sub_dependency_ratio",
        "sub_downstream_criticality",
        "sub_exposure_breadth",
        "sub_exposure_depth",
    ):
        band = wa.get(_sub_field)
        if band in _SUB_BAND_VALUE:
            supplier[_sub_field] = _SUB_BAND_VALUE[band]

    # Scenario-level flags
    if wa.get("shared_subtier_source"):
        fields["shared_subtier_source"] = True
    count_expr = wa.get("count_disrupted_paths")
    if count_expr is not None:
        if isinstance(count_expr, str) and count_expr.startswith(">="):
            fields["count_disrupted_paths"] = int(count_expr[2:])
        else:
            fields["count_disrupted_paths"] = int(count_expr)

    fields["suppliers"] = [supplier]
    return fields


# ---------------------------------------------------------------------------
# Contrast-pair variant builder
# ---------------------------------------------------------------------------

def _build_contrast_variant(
    base_scenario: dict,
    focal_rule: SupplierRule | RoutingRule,
    contrast_rule: SupplierRule | RoutingRule,
    vary_only: list[str],
    dtype: str,
    spec: Spec,
) -> dict | None:
    """Flip exactly the fields in vary_only to target the contrast rule instead."""
    import copy
    variant = copy.deepcopy(base_scenario)

    # We only handle supplier-rule contrast pairs here (simplest case)
    if isinstance(contrast_rule, SupplierRule) and isinstance(focal_rule, SupplierRule):
        wa_contrast = contrast_rule.when_all
        if variant.get("suppliers"):
            sup = variant["suppliers"][0]
            for field in vary_only:
                if field == "shared_subtier_source":
                    new_val = bool(wa_contrast.get("shared_subtier_source", False))
                    variant["shared_subtier_source"] = new_val
                elif field == "count_disrupted_paths":
                    expr = wa_contrast.get("count_disrupted_paths", 1)
                    if isinstance(expr, str) and expr.startswith(">="):
                        variant["count_disrupted_paths"] = max(0, int(expr[2:]) - 1)
                    else:
                        variant["count_disrupted_paths"] = int(expr) if expr else 1
                elif field == "tier1_in_region":
                    sup["tier1_in_region"] = bool(wa_contrast.get("tier1_in_region", False))
                elif field in (
                    "sub_node_centrality",
                    "sub_dependency_ratio",
                    "sub_downstream_criticality",
                    "sub_exposure_breadth",
                    "sub_exposure_depth",
                ):
                    # Set the field to the band the contrast rule keys on; if the
                    # contrast rule does not gate it, use the non-triggering (low) band.
                    expected = wa_contrast.get(field)
                    sup[field] = _SUB_BAND_VALUE.get(expected, _NON_TRIGGERING_SUB)
        return variant
    return None


# ---------------------------------------------------------------------------
# Oracle serialiser helper
# ---------------------------------------------------------------------------

def _oracle_to_dict(oracle_obj) -> dict:
    result: dict = {
        "scenario_id": oracle_obj.scenario_id,
        "disruption_type": oracle_obj.disruption_type,
        "routing": None,
        "suppliers": [],
    }
    if oracle_obj.routing:
        r = oracle_obj.routing
        result["routing"] = {
            "fired_rule_id":   r.fired_rule_id,
            "scenario_action": r.scenario_action,
            "disposition":     r.disposition,
            "escalation":      r.escalation,
        }
    for s in oracle_obj.suppliers:
        result["suppliers"].append({
            "supplier_id":      s.supplier_id,
            "fired_rule_id":    s.fired_rule_id,
            "disposition":      s.disposition,
            "escalation":       s.escalation,
            "permitted_actions":s.permitted_actions,
            "strategic_priority": s.strategic_priority,
        })
    return result


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_all(
    output_dir: Path,
    target_rule_id: str | None = None,
    target_cluster: str | None = None,
    include_pairs: bool = False,
) -> list[Path]:
    """Generate scenario fixtures from spec logic.

    Returns list of saved .yaml paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_all_specs()
    saved: list[Path] = []
    counter = 0

    for dtype, spec in specs.items():
        all_rules: list[RoutingRule | SupplierRule] = list(spec.routing_rules) + list(spec.sorted_supplier_rules)

        for rule in all_rules:
            rule_id = rule.rule_id

            # Filter by target rule
            if target_rule_id and rule_id != target_rule_id:
                continue

            # Filter by cluster
            eval_meta = getattr(rule, "evaluation", None)
            clusters = eval_meta.clusters if eval_meta else []
            if target_cluster and target_cluster not in clusters:
                continue

            # Build scenario
            if isinstance(rule, RoutingRule):
                scenario_dict = _build_routing_scenario(rule, dtype)
            else:
                scenario_dict = _build_supplier_scenario(rule, dtype, spec)

            if scenario_dict is None:
                logger.warning("Could not generate scenario for %s — skipping", rule_id)
                continue

            # Validate via deterministic oracle
            try:
                scenario_obj = Scenario.model_validate(scenario_dict)
                oracle_obj   = build_oracle(scenario_obj, spec)
            except Exception as exc:
                logger.error("Validation failed for %s: %s", rule_id, exc)
                continue

            # Verify the oracle fires the correct rule
            expected_rule = rule_id
            actual_rule   = (
                oracle_obj.routing.fired_rule_id if oracle_obj.routing
                else (oracle_obj.suppliers[0].fired_rule_id if oracle_obj.suppliers else None)
            )
            if actual_rule != expected_rule:
                logger.warning(
                    "Generated scenario for %s but oracle fires %s — adjusting or skipping",
                    expected_rule, actual_rule,
                )

            counter += 1
            scenario_id = f"gen_{counter:04d}"
            oracle_obj.scenario_id = scenario_id
            scenario_dict["_comment"] = f"Generated for rule {rule_id}"

            # Save YAML
            yaml_path = output_dir / f"{scenario_id}.yaml"
            scenario_out = {k: v for k, v in scenario_dict.items() if not k.startswith("_")}
            yaml_path.write_text(
                yaml.dump(scenario_out, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )

            # Save meta
            meta = {
                "scenario_id":       scenario_id,
                "target_rule_id":    rule_id,
                "disruption_type":   dtype,
                "target_mode":       "routing" if isinstance(rule, RoutingRule) else "supplier",
                "diagnostic_clusters": clusters,
                "oracle_source":     "deterministic_policy_evaluator",
                "contrast_pair_id":  None,
            }
            (output_dir / f"{scenario_id}.meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

            # Save oracle
            (output_dir / f"{scenario_id}.oracle.json").write_text(
                json.dumps(_oracle_to_dict(oracle_obj), indent=2), encoding="utf-8"
            )

            saved.append(yaml_path)
            logger.info("  Generated %s → %s (oracle: %s)", rule_id, scenario_id, actual_rule)

            # Generate contrast pair variant if requested
            if include_pairs and eval_meta and eval_meta.contrast_with:
                for contrast_rule_id in eval_meta.contrast_with:
                    contrast_rule_obj = spec.get_rule(contrast_rule_id)
                    if contrast_rule_obj is None:
                        continue
                    pair_key = f"{rule_id.replace('-','_')}_vs_{contrast_rule_id.replace('-','_')}"
                    pair_data = CONTRAST_PAIRS.get(pair_key, {})
                    vary_only = pair_data.get("vary_only", [])
                    if not vary_only:
                        continue

                    variant = _build_contrast_variant(
                        scenario_dict, rule, contrast_rule_obj, vary_only, dtype, spec
                    )
                    if variant is None:
                        continue

                    try:
                        variant_obj   = Scenario.model_validate(variant)
                        variant_oracle = build_oracle(variant_obj, spec)
                    except Exception as exc:
                        logger.warning("Contrast variant for %s failed: %s", contrast_rule_id, exc)
                        continue

                    counter += 1
                    vid = f"gen_{counter:04d}"
                    variant_oracle.scenario_id = vid

                    variant_yaml = {k: v for k, v in variant.items() if not k.startswith("_")}
                    (output_dir / f"{vid}.yaml").write_text(
                        yaml.dump(variant_yaml, default_flow_style=False, allow_unicode=True),
                        encoding="utf-8",
                    )
                    (output_dir / f"{vid}.meta.json").write_text(
                        json.dumps({
                            "scenario_id":       vid,
                            "target_rule_id":    contrast_rule_id,
                            "disruption_type":   dtype,
                            "target_mode":       "supplier",
                            "diagnostic_clusters": getattr(getattr(contrast_rule_obj, "evaluation", None), "clusters", []),
                            "oracle_source":     "deterministic_policy_evaluator",
                            "contrast_pair_id":  pair_key,
                            "contrast_focal_rule": rule_id,
                        }, indent=2),
                        encoding="utf-8",
                    )
                    (output_dir / f"{vid}.oracle.json").write_text(
                        json.dumps(_oracle_to_dict(variant_oracle), indent=2), encoding="utf-8"
                    )
                    saved.append(output_dir / f"{vid}.yaml")
                    logger.info(
                        "  Contrast pair %s → %s (oracle: %s)",
                        contrast_rule_id, vid,
                        variant_oracle.routing.fired_rule_id if variant_oracle.routing
                        else (variant_oracle.suppliers[0].fired_rule_id if variant_oracle.suppliers else "?"),
                    )

    logger.info("Generated %d scenarios → %s", len(saved), output_dir)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate diagnostic scenario fixtures from spec logic")
    parser.add_argument("--output-dir", default="fixtures/generated", help="Output directory")
    parser.add_argument("--rule",    default=None, help="Target a specific rule ID (e.g. CYB-HUB)")
    parser.add_argument("--cluster", default=None, help="Target a diagnostic cluster (e.g. C3_PRIORITY_OVERRIDE)")
    parser.add_argument("--pairs",   action="store_true", help="Also generate contrast-pair variants")
    args = parser.parse_args()

    if args.cluster and args.cluster not in ALL_CLUSTER_IDS:
        print(f"Unknown cluster: {args.cluster}. Available: {sorted(ALL_CLUSTER_IDS)}")
        sys.exit(1)

    saved = generate_all(
        output_dir=Path(args.output_dir),
        target_rule_id=args.rule,
        target_cluster=args.cluster,
        include_pairs=args.pairs,
    )
    print(f"\nDone. {len(saved)} scenario(s) written to {args.output_dir}/")
