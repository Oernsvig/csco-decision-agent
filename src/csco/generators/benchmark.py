"""Benchmark generator — creates the frozen Suite A diagnostic benchmark, plus Suite B.

This script generates Suite A, the 52-scenario diagnostic benchmark (one canonical
positive case per rule; the previous "hard strata" placeholders below are superseded
by ``generate_suite_b()`` — see that function's docstring for the null-region and
threshold-boundary fixture design that replaces them).

Usage:
    python -m csco.generators.benchmark \
        --output-dir fixtures/suite_a \
        --seed 12345

    python -m csco.generators.benchmark --suite-b \
        --suite-b-output-dir fixtures/suite_b \
        --suite-b-seed 20240

Output:
    fixtures/suite_a/
        benchmark_meta.json                       (frozen metadata)
        diag_0001.yaml ... diag_0052.yaml         (52 scenarios)
        diag_0001.meta.json ... diag_0052.meta.json
        diag_0001.oracle.json ... diag_0052.oracle.json

    fixtures/suite_b/
        suite_b_meta.json
        sb_0001.yaml ... sb_NNNN.yaml   (each carries suite/stratum/null_mode/
                                          boundary/target_rule_id fields)
        sb_0001.oracle.json ... , sb_0001.meta.json ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from csco.arms.prompts import PROMPT_VERSION
from csco.evaluation.binarize import is_null_oracle
from csco.derive import parse_band_expr
from csco.oracle.deterministic import build_oracle
from csco.models import Scenario
from csco.oracle.manual import ScenarioOracle
from csco.generators.scenario import generate_all as scenario_generate_all
from csco.specs.loader import load_all_specs
from csco.specs.schema import Spec

logger = logging.getLogger(__name__)

BENCHMARK_ID = "diagnostic_v1"


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def _compute_specs_hash() -> str:
    """Compute combined hash of all spec files."""
    specs_dir = Path(__file__).parent.parent.parent.parent / "specs"
    spec_files = sorted(specs_dir.glob("*.yaml"))
    
    combined = ""
    for f in spec_files:
        combined += _compute_file_hash(f)
    
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

def _generate_canonical_cases(output_dir: Path, seed: int = 12345) -> tuple[dict[str, Any], list[Path]]:
    """Generate the 52 canonical positive cases (one per rule).
    
    Returns (metadata_dict, list_of_yaml_paths).
    """
    logger.info("Generating 52 canonical positive cases (one per rule)...")
    
    # Use existing scenario generator to create one per rule
    temp_dir = output_dir.parent / "_temp_canonical"
    temp_dir.mkdir(exist_ok=True)
    
    scenario_paths = scenario_generate_all(
        output_dir=temp_dir,
        target_rule_id=None,  # All rules
        target_cluster=None,
        include_pairs=False,
    )
    
    logger.info(f"Generated {len(scenario_paths)} canonical scenarios")
    
    # Move and renumber to diagnostic naming scheme
    saved_paths = []
    counter = 1
    
    for yaml_path in sorted(scenario_paths):
        new_name = f"diag_{counter:04d}.yaml"
        new_path = output_dir / new_name
        
        # Load scenario and add metadata
        scenario_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        
        # Load oracle
        yaml_path.with_stem(yaml_path.stem).with_suffix(".oracle.json")
        
        new_path.write_text(
            yaml.dump(scenario_data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        
        saved_paths.append(new_path)
        counter += 1
    
    # Clean up temp dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    metadata = {
        "strata": "canonical_positive",
        "count": len(saved_paths),
        "description": "One scenario per rule (52 total)",
    }
    
    return metadata, saved_paths

def generate_benchmark(
    output_dir: Path,
    seed: int = 12345,
    target_size: int = 52,
) -> dict[str, Any]:
    """Generate and freeze the diagnostic benchmark.
    
    Args:
        output_dir: Directory to write benchmark artifacts
        seed: Random seed for deterministic generation
        target_size: Target scenario count (informational)
    
    Returns:
        Benchmark metadata dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating diagnostic benchmark v1 (target: {target_size} scenarios)")
    logger.info(f"Seed: {seed}")
    logger.info(f"Output: {output_dir}")
    
    # Step 1: Generate canonical cases
    canonical_meta, canonical_paths = _generate_canonical_cases(output_dir, seed=seed)
    logger.info(f"✓ Canonical cases: {canonical_meta['count']} scenarios")
    
    # Step 2: Load all generated scenarios and compute oracle
    yaml_files = sorted(output_dir.glob("diag_*.yaml"))
    logger.info(f"Processing {len(yaml_files)} scenario YAML files...")

    specs = load_all_specs()

    # Step 3: Save individual oracle and metadata JSON files
    logger.info(f"Saving oracle and metadata for {len(yaml_files)} scenarios...")
    
    for yaml_path in yaml_files:
        scenario_id = yaml_path.stem
        oracle_path = yaml_path.with_stem(scenario_id).with_suffix(".oracle.json")
        meta_path = yaml_path.with_stem(scenario_id).with_suffix(".meta.json")
        
        # Load scenario and compute oracle
        scenario_dict = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        try:
            scenario_obj = Scenario.model_validate(scenario_dict)
            dtype = scenario_obj.disruption_type
            spec = specs[dtype]
            oracle_obj = build_oracle(scenario_obj, spec)
            
            # Convert oracle to JSON-serializable dict
            def _oracle_to_dict(oracle):
                """Convert ScenarioOracle to nested dict for JSON serialization."""
                return {
                    "scenario_id": oracle.scenario_id,
                    "disruption_type": oracle.disruption_type,
                    "routing": asdict(oracle.routing) if oracle.routing else None,
                    "suppliers": [asdict(s) for s in oracle.suppliers],
                    "notes": oracle.notes,
                }
            
            oracle_dict = _oracle_to_dict(oracle_obj)
            
            # Create metadata
            scenario_meta = {
                "scenario_id": scenario_id,
                "disruption_type": dtype,
                "oracle_source": "deterministic_policy_evaluator",
            }
            
            # Write files
            oracle_path.write_text(json.dumps(oracle_dict, indent=2), encoding="utf-8")
            meta_path.write_text(json.dumps(scenario_meta, indent=2), encoding="utf-8")
            
        except Exception as e:
            logger.error(f"Failed to process {scenario_id}: {e}")
            continue
    
    logger.info("✓ Saved oracle and metadata files")

    # Step 4: Create benchmark metadata
    generated_at = datetime.utcnow().isoformat() + "Z"
    specs_hash = _compute_specs_hash()

    benchmark_meta = {
        "benchmark_id": BENCHMARK_ID,
        "target_size": target_size,
        "actual_size": len(yaml_files),
        "generation_seed": seed,
        "generated_at": generated_at,
        "specs_hash": specs_hash,
        "generator_version": "1.0",
        "oracle_version": "1.0",
        "prompt_version": PROMPT_VERSION,
        "frozen": True,
        "strata": {
            "canonical_positive": canonical_meta,
            "hard_strata": "superseded by generate_suite_b() — see fixtures/suite_b/suite_b_meta.json",
        },
        "description": (
            f"Frozen diagnostic benchmark with {len(yaml_files)} scenarios. "
            "Do not regenerate after model evaluation begins."
        ),
    }
    
    # Save benchmark metadata
    meta_path = output_dir / "benchmark_meta.json"
    meta_path.write_text(
        json.dumps(benchmark_meta, indent=2),
        encoding="utf-8",
    )
    
    logger.info(f"✓ Benchmark metadata: {meta_path}")
    logger.info(f"✓ Frozen: {benchmark_meta['frozen']}")
    logger.info(f"✓ Actual size: {benchmark_meta['actual_size']} scenarios")
    
    return benchmark_meta

# =============================================================================
# Suite B — null-region + threshold-boundary fixture generation
# =============================================================================
#
# Supersedes the HARD_STRATA_DEFINITIONS placeholders that used to live above
# (threshold_boundary/priority_overlap/cross_policy/multi_supplier/routing_edge,
# 8+6+4+3+2 "manual design required" stub). Those were never auto-generated;
# this generator produces real, oracle-verified fixtures instead.
#
# Per disruption type (5 types), generates:
#   a) CLEAR NULLS  — 1 fixture per applicable null mode (seeded jitter — the
#      value is not a hand-picked constant, just a single deterministic draw
#      from the mode's valid range). Null modes derived from the three shared
#      routing-rule patterns
#      (NULL / DT / DT-NULL, identical across all five specs):
#        1. no_path_on_network      — NULL fires via path_on_network == False
#           (unconditional; independent of any supplier data).
#        2. all_tier1_low_risk      — NULL fires via the second NULL clause
#           (all Tier-1 LOW risk_level AND no high sub_downstream_criticality).
#        3. criticality_below_gate  — the SAME second NULL clause, isolating
#           the sub_downstream_criticality < 0.66 gate as the discriminant
#           (risk_score held constant deep in LOW band; only criticality is
#           jittered). NOTE: modes 2 and 3 exercise the identical AND-clause
#           in deterministic_oracle._routing_condition_matches — the spec
#           gives no way to trigger one half of that AND independently of the
#           other, so these are two generation *strategies* (vary risk_score
#           vs. vary criticality) for the same branch, not two independently
#           triggerable conditions. Documented here rather than silently
#           merged, since the eval brief enumerates them as separate items.
#        4. dt_null_routing         — tier1_on_disrupted_path == False AND
#           NOT (shared_subtier_source OR count_disrupted_paths >= 2), i.e.
#           the DT-NULL routing rule (structural fallthrough with no material
#           downstream concern).
#      All five specs share the identical routing-rule structure, so no type
#      skips any mode.
#   b) BOUNDARY PAIRS — for every active numeric threshold in that type's
#      spec, one fixture epsilon BELOW and one epsilon ABOVE:
#        - risk_high_cut       — the risk-band HIGH cut (0.55 geopolitical,
#          0.60 others); below resolves to the generic MEDIUM rule, above
#          resolves to whichever HIGH+in-region rule the type defines.
#        - structural_tier_cut — the 0.66 sub-score high-band gate, isolated
#          on sub_node_centrality (sub_downstream_criticality held fixed
#          above-gate so crossing the centrality gate alone moves the supplier
#          into the high-centrality band); below resolves to the generic
#          MEDIUM rule, above resolves to the type's structural
#          rule (MEDS, or CYB-HUB for cyber, which keys on centrality
#          directly and fires before MEDS would even be reached).
#        - count_paths_cut     — the ">=2 disrupted paths" clause in the
#          {TYPE}-R3 supplier rule (paired with shared_subtier_source=true,
#          risk_level=MEDIUM); below (count=1) resolves to the generic
#          MEDIUM rule, above (count=2) fires R3.
#      The target rule for the ABOVE side is discovered by actually running
#      the deterministic oracle (never hardcoded), so this is robust to
#      per-type rule-priority differences (e.g. cyber's HUB rule pre-empting
#      MEDS at the same 0.66 gate).
#
# Every generated fixture is verified against the deterministic oracle before
# being written: intended-null fixtures must resolve to the null route
# (via the same is_null_oracle() predicate as the Act/No-Act binarization),
# and boundary-above fixtures must fire the targeted rule_id while the
# corresponding below fixture must NOT. A mismatch raises — nothing is ever
# silently emitted with the wrong label.
#
# Realized count: 4 null modes x 1 fixture x 5 types = 20, plus 3 boundary
# conditions x 2 fixtures (below/above) x 5 types = 30 -> 50 total, matching
# the dissertation's stated Suite B size exactly.
# =============================================================================

_SB_SUB_NEUTRAL = 0.50      # 'med' band sub-score; never registers as 'high' (>=0.66)
_SB_SUB_HIGH_ANCHOR = 0.80  # comfortably above the 0.66 gate — used as a fixed 'high' anchor
_SB_RISK_MEDIUM_MID = 0.50  # well inside every type's MEDIUM band

SUITE_B_NULL_MODES = [
    "no_path_on_network",
    "all_tier1_low_risk",
    "criticality_below_gate",
    "dt_null_routing",
]
# structural_tier_cut removed: it isolated the sub-score 'high' gate on
# node_centrality, which only discriminates for cyber now that the supplier
# criticality (structural tier) axis has been removed from the framework.
SUITE_B_BOUNDARY_CONDITIONS = ["risk_high_cut", "count_paths_cut"]


def _sb_base_fields(dtype: str) -> dict:
    return {
        "disruption_type": dtype,
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


def _sb_supplier(
    supplier_id: str,
    risk_score: float,
    *,
    tier1_in_region: bool = False,
    centrality: float = _SB_SUB_NEUTRAL,
    criticality: float = _SB_SUB_NEUTRAL,
    dep_ratio: float = _SB_SUB_NEUTRAL,
    breadth: float = _SB_SUB_NEUTRAL,
    depth: float = _SB_SUB_NEUTRAL,
) -> dict:
    return {
        "supplier_id": supplier_id,
        "name": f"Supplier-{supplier_id}",
        "product_supplied": "component-X",
        "tier1_in_region": tier1_in_region,
        "risk_score": round(risk_score, 3),
        "sub_exposure_breadth": round(breadth, 3),
        "sub_dependency_ratio": round(dep_ratio, 3),
        "sub_downstream_criticality": round(criticality, 3),
        "sub_node_centrality": round(centrality, 3),
        "sub_exposure_depth": round(depth, 3),
    }


def _sb_high_threshold(spec: Spec) -> float:
    return parse_band_expr(spec.risk_bands["HIGH"])[1]


def _sb_low_threshold(spec: Spec) -> float:
    return parse_band_expr(spec.risk_bands["LOW"])[1]


def _sb_sub_high_threshold(spec: Spec) -> float:
    return parse_band_expr(spec.sub_score_bands["high"])[1]


def _sb_effective_fired_rule(oracle: ScenarioOracle) -> str | None:
    if oracle.routing is not None:
        return oracle.routing.fired_rule_id
    if oracle.suppliers:
        return oracle.suppliers[0].fired_rule_id
    return None


# --- Null-mode builders -----------------------------------------------------
#
# Supplier IDs are deliberately NEUTRAL serials (S-SB-{TYP}-01, -02, ...).
# An earlier scheme encoded the fixture's stratum/answer class into the ID the
# model reads in the scenario block (S-SB-GEO-LOW0-0, S-SB-GEO-HIRISK-BELOW,
# S-SB-GEO-TIER-ABOVE, ...) — a literal answer-key leak: "LOW" named the
# all-LOW null branch, and "BELOW"/"ABOVE" named which side of the threshold
# the fixture tests. Discovered 2026-07-11 while diagnosing why near-identical
# all-LOW null scenarios scored 15/15 (Suite B, leaked ID) vs ~7/15 (Suite A,
# neutral ID) on the lexical arm. Stratum metadata belongs in the YAML
# side-channel keys (suite/stratum/null_mode/boundary), which the Scenario
# model ignores and the model never sees — never in model-visible fields.
# IDs only need uniqueness WITHIN a scenario (extraction and oracle matching
# are scenario-scoped), so a per-scenario serial is sufficient.

def _sb_null_no_path(dtype: str, rng: random.Random, variant: int) -> dict:
    fields = _sb_base_fields(dtype)
    fields["path_on_network"] = False
    # tier1_on_disrupted_path is True (not False) so this scenario does NOT
    # also structurally satisfy DT/DT-NULL's shared precondition
    # (tier1_on_disrupted_path == False) regardless of the randomised
    # count_disrupted_paths below — see the matching fix/comment in
    # scenario_generator._build_routing_scenario for the identical bug.
    fields["tier1_on_disrupted_path"] = True
    fields["count_disrupted_paths"] = rng.choice([0, 1, 2, 3])
    if variant == 0:
        fields["suppliers"] = []
    else:
        risk = rng.uniform(0.2, 0.9)  # irrelevant: NULL fires unconditionally on path_on_network
        fields["suppliers"] = [_sb_supplier(f"S-SB-{dtype[:3].upper()}-01", risk)]
    return fields


def _sb_null_all_low(dtype: str, spec: Spec, rng: random.Random, variant: int) -> dict:
    fields = _sb_base_fields(dtype)
    low = _sb_low_threshold(spec)
    n_suppliers = 1 if variant == 0 else 2
    suppliers = []
    for i in range(n_suppliers):
        risk = rng.uniform(max(0.02, low - 0.30), max(0.03, low - 0.01))
        suppliers.append(_sb_supplier(
            f"S-SB-{dtype[:3].upper()}-{i + 1:02d}", risk,
            criticality=rng.uniform(0.10, 0.30),
        ))
    fields["suppliers"] = suppliers
    return fields


def _sb_null_criticality_gate(dtype: str, spec: Spec, rng: random.Random, variant: int) -> dict:
    fields = _sb_base_fields(dtype)
    low = _sb_low_threshold(spec)
    gate = _sb_sub_high_threshold(spec)
    risk = max(0.02, low - 0.20)  # fixed comfortably LOW, constant across variants
    criticality = gate - rng.uniform(0.05, 0.20)  # jittered, always below gate
    fields["suppliers"] = [_sb_supplier(
        f"S-SB-{dtype[:3].upper()}-01", risk, criticality=criticality,
    )]
    return fields


def _sb_null_dt_null_routing(dtype: str, rng: random.Random, variant: int) -> dict:
    fields = _sb_base_fields(dtype)
    fields["tier1_on_disrupted_path"] = False
    fields["shared_subtier_source"] = False
    fields["count_disrupted_paths"] = rng.choice([0, 1])
    fields["max_disrupted_tier"] = rng.choice([2, 3, 4])
    fields["suppliers"] = []
    return fields


_SB_NULL_BUILDERS = {
    "no_path_on_network": _sb_null_no_path,
    "all_tier1_low_risk": _sb_null_all_low,
    "criticality_below_gate": _sb_null_criticality_gate,
    "dt_null_routing": _sb_null_dt_null_routing,
}


def _sb_build_null_scenario(mode_name: str, dtype: str, spec: Spec, rng: random.Random, variant: int) -> dict:
    builder = _SB_NULL_BUILDERS[mode_name]
    if mode_name in ("all_tier1_low_risk", "criticality_below_gate"):
        return builder(dtype, spec, rng, variant)
    return builder(dtype, rng, variant)


# --- Boundary-pair builders --------------------------------------------------

def _sb_boundary_risk_high_cut(dtype: str, spec: Spec, side: str) -> dict:
    threshold = _sb_high_threshold(spec)
    score = threshold if side == "above" else threshold - 0.01
    fields = _sb_base_fields(dtype)
    fields["suppliers"] = [_sb_supplier(
        f"S-SB-{dtype[:3].upper()}-01", score, tier1_in_region=True,
    )]
    return fields


def _sb_boundary_count_paths_cut(dtype: str, side: str) -> dict:
    fields = _sb_base_fields(dtype)
    fields["shared_subtier_source"] = True
    fields["count_disrupted_paths"] = 2 if side == "above" else 1
    fields["suppliers"] = [_sb_supplier(
        f"S-SB-{dtype[:3].upper()}-01", _SB_RISK_MEDIUM_MID, tier1_in_region=True,
    )]
    return fields


_SB_BOUNDARY_BUILDERS = {
    "risk_high_cut": _sb_boundary_risk_high_cut,
    "count_paths_cut": _sb_boundary_count_paths_cut,
}


def _sb_build_boundary_scenario(cond_name: str, dtype: str, spec: Spec, side: str) -> dict:
    builder = _SB_BOUNDARY_BUILDERS[cond_name]
    if cond_name == "count_paths_cut":
        return builder(dtype, side)
    return builder(dtype, spec, side)


# --- Fixture writer -----------------------------------------------------------

def _sb_write_fixture(
    output_dir: Path,
    scenario_id: str,
    scenario_dict: dict,
    oracle_obj: ScenarioOracle,
    *,
    suite: str,
    stratum: str,
    null_mode: str | None,
    boundary: str,
    target_rule_id: str | None,
) -> None:
    yaml_out = dict(scenario_dict)
    yaml_out["suite"] = suite
    yaml_out["stratum"] = stratum
    yaml_out["null_mode"] = null_mode
    yaml_out["boundary"] = boundary
    yaml_out["target_rule_id"] = target_rule_id

    (output_dir / f"{scenario_id}.yaml").write_text(
        yaml.dump(yaml_out, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    oracle_obj.scenario_id = scenario_id
    oracle_dict = {
        "scenario_id": oracle_obj.scenario_id,
        "disruption_type": oracle_obj.disruption_type,
        "routing": asdict(oracle_obj.routing) if oracle_obj.routing else None,
        "suppliers": [asdict(s) for s in oracle_obj.suppliers],
        "notes": oracle_obj.notes,
    }
    (output_dir / f"{scenario_id}.oracle.json").write_text(
        json.dumps(oracle_dict, indent=2), encoding="utf-8",
    )

    meta = {
        "scenario_id": scenario_id,
        "disruption_type": scenario_dict["disruption_type"],
        "suite": suite,
        "stratum": stratum,
        "null_mode": null_mode,
        "boundary": boundary,
        "target_rule_id": target_rule_id,
        "oracle_source": "deterministic_policy_evaluator",
    }
    (output_dir / f"{scenario_id}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )


def generate_suite_b(output_dir: Path, seed: int = 20240) -> dict[str, Any]:
    """Generate Suite B: seeded, deterministic null-region + boundary-pair fixtures.

    See the module-level comment block above for the full design (null modes,
    boundary conditions, and the oracle-agreement verification performed on
    every fixture before it is written). Supersedes the old
    HARD_STRATA_DEFINITIONS placeholders.

    Raises RuntimeError (does not emit a fixture) if the deterministic oracle
    disagrees with a fixture's intended stratum — that is a generation bug,
    not a warning.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_all_specs()

    breakdown: dict[str, Any] = {"null": {}, "boundary": {}}
    counter = 0
    written: list[str] = []

    for dtype, spec in specs.items():
        rng_base = f"{seed}-{dtype}"

        # --- a) CLEAR NULLS --------------------------------------------------
        for mode_name in SUITE_B_NULL_MODES:
            for variant in range(1):
                rng = random.Random(f"{rng_base}-{mode_name}-{variant}")
                scenario_dict = _sb_build_null_scenario(mode_name, dtype, spec, rng, variant)
                scenario_obj = Scenario.model_validate(scenario_dict)
                oracle_obj = build_oracle(scenario_obj, spec)

                if not is_null_oracle(oracle_obj):
                    raise RuntimeError(
                        f"Suite B generation bug: {dtype}/{mode_name}#{variant} was "
                        f"expected to resolve to the null route, but the deterministic "
                        f"oracle fired {_sb_effective_fired_rule(oracle_obj)!r} "
                        f"(not null). Fixture NOT emitted."
                    )

                counter += 1
                scenario_id = f"sb_{counter:04d}"
                _sb_write_fixture(
                    output_dir, scenario_id, scenario_dict, oracle_obj,
                    suite="B", stratum="null", null_mode=mode_name,
                    boundary="none", target_rule_id=None,
                )
                written.append(scenario_id)
                dtype_bucket = breakdown["null"].setdefault(dtype, {})
                dtype_bucket[mode_name] = dtype_bucket.get(mode_name, 0) + 1

        # --- b) BOUNDARY PAIRS -------------------------------------------------
        for cond_name in SUITE_B_BOUNDARY_CONDITIONS:
            above_dict = _sb_build_boundary_scenario(cond_name, dtype, spec, "above")
            above_obj = Scenario.model_validate(above_dict)
            above_oracle = build_oracle(above_obj, spec)
            target_rule_id = _sb_effective_fired_rule(above_oracle)
            if target_rule_id is None:
                raise RuntimeError(
                    f"Suite B generation bug: {dtype}/{cond_name} ABOVE fixture "
                    f"fired no rule at all. Fixture NOT emitted."
                )

            below_dict = _sb_build_boundary_scenario(cond_name, dtype, spec, "below")
            below_obj = Scenario.model_validate(below_dict)
            below_oracle = build_oracle(below_obj, spec)
            below_rule = _sb_effective_fired_rule(below_oracle)
            if below_rule == target_rule_id:
                raise RuntimeError(
                    f"Suite B generation bug: {dtype}/{cond_name} BELOW fixture fired "
                    f"the same rule ({below_rule}) as the ABOVE fixture — the epsilon "
                    f"perturbation did not move the classification. Fixture NOT emitted."
                )

            for side, s_dict, s_oracle in (
                ("below", below_dict, below_oracle),
                ("above", above_dict, above_oracle),
            ):
                counter += 1
                scenario_id = f"sb_{counter:04d}"
                stratum = "null" if is_null_oracle(s_oracle) else "action"
                _sb_write_fixture(
                    output_dir, scenario_id, s_dict, s_oracle,
                    suite="B", stratum=stratum, null_mode=None,
                    boundary=side, target_rule_id=target_rule_id,
                )
                written.append(scenario_id)
                cond_bucket = breakdown["boundary"].setdefault(dtype, {}).setdefault(cond_name, {})
                cond_bucket[side] = scenario_id

    meta = {
        "suite_id": "suite_b_v1",
        "generation_seed": seed,
        "total_fixtures": len(written),
        "breakdown": breakdown,
        "description": (
            f"Suite B: {len(written)} seeded, deterministic null-region and "
            "threshold-boundary fixtures. Supersedes the HARD_STRATA_DEFINITIONS "
            "placeholders. Every fixture's oracle route was verified at "
            "generation time (see generate_suite_b docstring)."
        ),
    }
    (output_dir / "suite_b_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Suite B: %d fixtures written to %s", len(written), output_dir)
    return meta


# =============================================================================
# Dissertation suites — the frozen 72-scenario evaluation set (Suite A = 37
# prescribed-action, Suite B = 35 null). Supersedes the split produced by
# generate_benchmark()/generate_suite_b() (which put all 52 canonical in Suite A
# and generated a separate 50-fixture null+boundary Suite B).
#
# Construction (matches Methodology §3.5):
#   - 52 canonical cases (one per rule) are split by is_null_oracle():
#       * 37 prescribed-action rules  -> Suite A  (diag_XXXX)
#       * 15 no-action rules          -> Suite B  (sb_XXXX, null_mode="canonical")
#   - 20 additional null scenarios (4 null modes x 5 disruption types) are added
#     to Suite B, giving 35 null scenarios and 72 in total.
# Boundary/threshold-probe pairs are NOT part of the frozen 72.
# =============================================================================

def generate_dissertation_suites(
    suite_a_dir: Path,
    suite_b_dir: Path,
    seed: int = 20240,
) -> dict[str, Any]:
    """Generate the frozen 72-scenario evaluation set (Suite A = 37, Suite B = 35).

    Every fixture is verified against the deterministic oracle before it is
    written: canonical cases must fire their target rule, and every Suite B
    fixture must resolve to the null route (is_null_oracle). A mismatch raises.
    """
    from csco.generators.scenario import _build_routing_scenario, _build_supplier_scenario
    from csco.specs.schema import RoutingRule

    suite_a_dir.mkdir(parents=True, exist_ok=True)
    suite_b_dir.mkdir(parents=True, exist_ok=True)
    specs = load_all_specs()

    a_count = 0
    b_count = 0
    a_written: list[str] = []
    b_written: list[str] = []

    # --- 1) canonical one-per-rule, split action vs null ---------------------
    for dtype, spec in specs.items():
        all_rules = list(spec.routing_rules) + list(spec.sorted_supplier_rules)
        for rule in all_rules:
            if isinstance(rule, RoutingRule):
                scenario_dict = _build_routing_scenario(rule, dtype)
            else:
                scenario_dict = _build_supplier_scenario(rule, dtype, spec)
            if scenario_dict is None:
                raise RuntimeError(f"Could not build a canonical scenario for {rule.rule_id}")

            scenario_obj = Scenario.model_validate(scenario_dict)
            oracle_obj = build_oracle(scenario_obj, spec)
            fired = _sb_effective_fired_rule(oracle_obj)
            if fired != rule.rule_id:
                raise RuntimeError(
                    f"Canonical scenario for {rule.rule_id} fired {fired!r} instead — "
                    "reachability broken, fixture NOT emitted."
                )

            if is_null_oracle(oracle_obj):
                b_count += 1
                sid = f"sb_{b_count:04d}"
                _sb_write_fixture(
                    suite_b_dir, sid, scenario_dict, oracle_obj,
                    suite="B", stratum="null", null_mode="canonical",
                    boundary="none", target_rule_id=rule.rule_id,
                )
                b_written.append(sid)
            else:
                a_count += 1
                sid = f"diag_{a_count:04d}"
                _sb_write_fixture(
                    suite_a_dir, sid, scenario_dict, oracle_obj,
                    suite="A", stratum="action", null_mode=None,
                    boundary="none", target_rule_id=rule.rule_id,
                )
                a_written.append(sid)

    # --- 2) 20 additional null scenarios (4 modes x 5 types) -----------------
    for dtype, spec in specs.items():
        for mode_name in SUITE_B_NULL_MODES:
            rng = random.Random(f"{seed}-{dtype}-{mode_name}-0")
            scenario_dict = _sb_build_null_scenario(mode_name, dtype, spec, rng, 0)
            scenario_obj = Scenario.model_validate(scenario_dict)
            oracle_obj = build_oracle(scenario_obj, spec)
            if not is_null_oracle(oracle_obj):
                raise RuntimeError(
                    f"Suite B null mode {dtype}/{mode_name} did not resolve to the null "
                    f"route (fired {_sb_effective_fired_rule(oracle_obj)!r}). Fixture NOT emitted."
                )
            b_count += 1
            sid = f"sb_{b_count:04d}"
            _sb_write_fixture(
                suite_b_dir, sid, scenario_dict, oracle_obj,
                suite="B", stratum="null", null_mode=mode_name,
                boundary="none", target_rule_id=None,
            )
            b_written.append(sid)

    specs_hash = _compute_specs_hash()
    common = {
        "generation_seed": seed,
        "specs_hash": specs_hash,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_a = {
        "suite_id": "suite_a_action_v2",
        "total_fixtures": len(a_written),
        "stratum": "prescribed_action",
        "description": "Suite A: 37 prescribed-action canonical cases (one per action rule).",
        **common,
    }
    meta_b = {
        "suite_id": "suite_b_null_v2",
        "total_fixtures": len(b_written),
        "stratum": "null",
        "description": (
            "Suite B: 15 no-action canonical cases (one per null rule) + 20 additional "
            "null scenarios (4 null modes x 5 disruption types) = 35."
        ),
        **common,
    }
    (suite_a_dir / "benchmark_meta.json").write_text(json.dumps(meta_a, indent=2), encoding="utf-8")
    (suite_b_dir / "benchmark_meta.json").write_text(json.dumps(meta_b, indent=2), encoding="utf-8")

    logger.info("Suite A: %d action fixtures -> %s", len(a_written), suite_a_dir)
    logger.info("Suite B: %d null fixtures -> %s", len(b_written), suite_b_dir)
    return {"suite_a": meta_a, "suite_b": meta_b, "total": len(a_written) + len(b_written)}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate and freeze diagnostic benchmark")
    parser.add_argument(
        "--output-dir",
        default="fixtures/suite_a",
        help="Output directory for benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for deterministic generation",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=52,
        help="Target scenario count (informational)",
    )
    parser.add_argument(
        "--suite-b",
        action="store_true",
        help="Generate Suite B (null-region + threshold-boundary fixtures) instead of the diagnostic benchmark.",
    )
    parser.add_argument(
        "--suite-b-output-dir",
        default="fixtures/suite_b",
        help="Output directory for Suite B (default: fixtures/suite_b)",
    )
    parser.add_argument(
        "--suite-b-seed",
        type=int,
        default=20240,
        help="Random seed for Suite B jitter (default: 20240)",
    )
    parser.add_argument(
        "--dissertation",
        action="store_true",
        help="Generate the frozen 72-scenario evaluation set (Suite A = 37, Suite B = 35).",
    )
    args = parser.parse_args()

    if args.dissertation:
        try:
            meta = generate_dissertation_suites(
                suite_a_dir=Path(args.output_dir),
                suite_b_dir=Path(args.suite_b_output_dir),
                seed=args.suite_b_seed,
            )
            print("\n✓ Dissertation suites generated")
            print(f"  Suite A (action): {meta['suite_a']['total_fixtures']}")
            print(f"  Suite B (null)  : {meta['suite_b']['total_fixtures']}")
            print(f"  Total           : {meta['total']}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Dissertation suite generation failed: {e}", exc_info=True)
            sys.exit(1)

    if args.suite_b:
        try:
            meta = generate_suite_b(output_dir=Path(args.suite_b_output_dir), seed=args.suite_b_seed)
            print(f"\n✓ Suite B generated: {args.suite_b_output_dir}/")
            print(f"  Fixtures: {meta['total_fixtures']}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Suite B generation failed: {e}", exc_info=True)
            sys.exit(1)

    try:
        meta = generate_benchmark(
            output_dir=Path(args.output_dir),
            seed=args.seed,
            target_size=args.target_size,
        )
        print(f"\n✓ Benchmark frozen: {args.output_dir}/")
        print(f"  Scenarios: {meta['actual_size']}")
        print(f"  Frozen: {meta['frozen']}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Benchmark generation failed: {e}", exc_info=True)
        sys.exit(1)
