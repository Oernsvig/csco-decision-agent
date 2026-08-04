"""Oracle answer keys for all scenario fixtures.

Each entry defines the ground-truth correct answer for a fixture.
Used by RunResult.score() to compute per-run correctness metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SupplierOracle:
    """Expected correct answer for one supplier."""
    supplier_id: str
    disposition: Literal["replace", "monitor", "maintain"]
    escalation: Literal["Responsible Analyst", "Cross-functional Management", "Executive Leadership"]
    fired_rule_id: str
    permitted_actions: list[str]
    strategic_priority: str | None = None        # e.g., "Continuity Protection", "Rapid Reconfiguration", "Cost Containment"
    strategic_note: str | None = None            # rationale for the priority


@dataclass
class RoutingOracle:
    """Expected correct answer when a routing rule fires."""
    fired_rule_id: str
    scenario_action: str
    disposition: Literal["replace", "monitor", "maintain"]
    escalation: Literal["Responsible Analyst", "Cross-functional Management", "Executive Leadership"]


@dataclass
class ScenarioOracle:
    """Complete expected answer for one scenario."""
    scenario_id: str
    disruption_type: str
    routing: RoutingOracle | None = None          # set if a routing rule fires
    suppliers: list[SupplierOracle] = field(default_factory=list)  # set if no routing fires
    notes: str = ""


# ---------------------------------------------------------------------------
# Hand-authored calibration answer keys are not part of this release; get_oracle()
# resolves Suite A/B scenarios from their generated *.oracle.json files.
ORACLE: dict[str, ScenarioOracle] = {}


DEFAULT_DIAGNOSTIC_ORACLE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "suite_a"

# Default location where generators.benchmark.generate_suite_b() writes
# {scenario_id}.oracle.json files for the Suite B null-region/boundary fixtures.
DEFAULT_SUITE_B_ORACLE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "suite_b"


def _load_diagnostic_oracle(scenario_id: str, oracle_dir: Path) -> ScenarioOracle | None:
    """Load and reconstruct a ScenarioOracle from {scenario_id}.oracle.json, if present.

    The file is written by benchmark_generator.py as a plain nested dict matching
    the ScenarioOracle/RoutingOracle/SupplierOracle dataclass fields (see
    deterministic_oracle.build_oracle's `_oracle_to_dict`). Returns None if the
    file does not exist.
    """
    oracle_path = oracle_dir / f"{scenario_id}.oracle.json"
    if not oracle_path.exists():
        return None

    data = json.loads(oracle_path.read_text(encoding="utf-8"))

    routing_data = data.get("routing")
    routing = RoutingOracle(**routing_data) if routing_data else None

    suppliers = [SupplierOracle(**s) for s in data.get("suppliers", [])]

    return ScenarioOracle(
        scenario_id=scenario_id,
        disruption_type=data.get("disruption_type", ""),
        routing=routing,
        suppliers=suppliers,
        notes=data.get("notes", ""),
    )


def get_oracle(
    scenario_id: str,
    diagnostic_oracle_dir: str | Path = DEFAULT_DIAGNOSTIC_ORACLE_DIR,
    suite_b_oracle_dir: str | Path = DEFAULT_SUITE_B_ORACLE_DIR,
) -> ScenarioOracle | None:
    """Return the oracle for a scenario, or None if not defined.

    Resolution order:
      1. The ORACLE dict (empty in this release; reserved for hand-authored entries).
      2. A generated {scenario_id}.oracle.json file under diagnostic_oracle_dir
         (fixtures/suite_a/ by default), as written by the benchmark generator.
      3. A generated {scenario_id}.oracle.json file under suite_b_oracle_dir
         (fixtures/suite_b/ by default), as written by generate_suite_b().
      4. None, if none of the above sources defines the scenario.
    """
    oracle = ORACLE.get(scenario_id)
    if oracle is not None:
        return oracle
    oracle = _load_diagnostic_oracle(scenario_id, Path(diagnostic_oracle_dir))
    if oracle is not None:
        return oracle
    return _load_diagnostic_oracle(scenario_id, Path(suite_b_oracle_dir))
