"""Batch runner: execute all fixtures across the three policy-grounding architectures.

Experiment framing
------------------
This runner compares three policy-grounding architectures:
  static   — full corpus in system prompt (constant context)
  vector   — flat vector retrieval over embed-cut chunks
  lexical  — lexical document graph with structural and reference edges

The oracle measures fidelity to the encoded disruption-response policy,
not independent real-world optimality of the recommendation.

Usage
-----
  # Single run per cell — quick check (20 fixtures × 3 arms = 60 runs):
  python -m csco.cli.batch_run --output-dir results/

  # N runs per cell — for repeatability measurement (recommended for experiments):
  python -m csco.cli.batch_run --output-dir results/ --runs-per-fixture 5

  # Also run generated fixtures (requires fixtures/generated/ to exist):
  python -m csco.cli.batch_run --output-dir results/ --fixtures-dir fixtures/generated/

With N > 1, each fixture×arm cell is executed N times and the report includes
per-cell accuracy rates and a repeatability section. Run-to-run consistency is
a first-class metric: the registered prediction expects the signal to appear in
repeatability, not just mean accuracy.

Note on non-determinism: even at temperature=0 the two-stage pipeline (reasoning
+ extraction) exhibits small run-to-run variance because the extraction LLM call
can parse the same free text differently across calls. This noise sits downstream
of all three architectures equally, so it does not bias inter-arm comparisons —
but it adds noise to every cell. N-run averaging reduces this noise and makes it
measurable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from csco.evaluation.binarize import is_null_oracle
from csco.models import Scenario
from csco.oracle.manual import get_oracle
from csco.evaluation.report import _generate_summary_report, _scored_non_infra_runs
from csco.evaluation.run_result import RunResult
from csco.evaluation.scoring import _cell_stats
from csco.evaluation import skew_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _classify_failure(exc: Exception, message: str) -> str:
    """Classify a run failure into a category for triage and handling.
    
    Categories:
    - infrastructure_failure: External dependencies (DB, network, etc.)
    - model_output_failure: LLM produced invalid/unparseable output
    - tool_loop_failure: Tool calling went into a loop or got stuck
    - extraction_failure: Decision extraction failed after reasoning completed
    - unknown_failure: Could not determine cause; default to conservative treatment
    
    Principle: If the failure is the architecture's fault, it counts against the architecture.
    """
    msg_lower = message.lower()
    
    # Infrastructure failures: external system issues (API quota, network, DB)
    infrastructure_keywords = [
        "connection", "timeout", "network", "database", "neo4j", "unreachable",
        "quota", "rate limit", "rate_limit", "429", "insufficient_quota",
        "billing", "overloaded", "capacity",
    ]
    if any(k in msg_lower for k in infrastructure_keywords):
        return "infrastructure_failure"
    
    # Tool loop / LLM output issues
    tool_loop_keywords = ["tool", "iteration", "max_iterations", "loop", "too many"]
    if any(k in msg_lower for k in tool_loop_keywords):
        return "tool_loop_failure"
    
    # Model output parsing / extraction issues
    model_output_keywords = ["json", "parse", "validation", "field", "schema", "model"]
    if any(k in msg_lower for k in model_output_keywords):
        if "extraction" in msg_lower or "extract" in msg_lower:
            return "extraction_failure"
        return "model_output_failure"
    
    # Extraction-specific
    if "extract" in msg_lower:
        return "extraction_failure"
    
    # Default: conservative treatment (exclude from accuracy, don't count against arm)
    return "unknown_failure"


# Fixture stratification metadata keys (Task 1). Suite B fixtures carry these
# directly in the YAML (extra keys the Scenario model safely ignores). Any
# fixture without this metadata (e.g. Suite A = fixtures/suite_a/) gets a
# derived default: suite="A", stratum computed from its oracle via the same
# is_null_oracle() predicate used for the Act/No-Act binarization (Task 2) —
# without modifying the fixture files themselves.
_FIXTURE_META_KEYS = ("suite", "stratum", "null_mode", "boundary", "target_rule_id")


def _resolve_fixture_meta(scenario_id: str, raw: dict) -> dict:
    if "suite" in raw and "stratum" in raw:
        return {k: raw.get(k) for k in _FIXTURE_META_KEYS}

    oracle = get_oracle(scenario_id)
    stratum = ("null" if is_null_oracle(oracle) else "action") if oracle is not None else None
    return {"suite": "A", "stratum": stratum, "null_mode": None, "boundary": "none", "target_rule_id": None}


def _load_fixture(path: str) -> tuple[str, Scenario, dict]:
    scenario_id = Path(path).stem
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(raw)
    fixture_meta = _resolve_fixture_meta(scenario_id, raw)
    return scenario_id, scenario, fixture_meta


def _run_all_fixtures_all_arms(
    fixtures_dir: str,
    output_dir: str,
    n_runs: int = 1,
    arms: list[str] | None = None,
    fixture_filter: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run all fixtures against the specified arms, n_runs times each.

    Default arms are the three policy-grounding architectures:
    static, vector, lexical.

    fixtures_dir: one directory, or multiple comma-separated directories
    (e.g. "fixtures/suite_a,fixtures/suite_b" to combine Suite A [the 52
    canonical one-per-rule scenarios] with Suite B) whose fixtures are merged
    into one run.

    fixture_filter: if provided, only run fixtures whose stem is in this set.

    Returns results keyed by arm. Each entry in the list is one RunResult dict.
    With n_runs > 1 there will be n_runs entries per fixture×arm cell.
    """
    if arms is None:
        arms = ["static", "vector", "lexical"]

    fixtures_dirs = [d.strip() for d in fixtures_dir.split(",") if d.strip()]
    all_fixture_files = sorted(
        f for d in fixtures_dirs for f in Path(d).glob("*.yaml")
    )
    if fixture_filter is not None:
        fixture_files = [f for f in all_fixture_files if f.stem in fixture_filter]
        missing = fixture_filter - {f.stem for f in fixture_files}
        if missing:
            logger.warning("Fixtures in --fixtures-list not found in %s: %s", fixtures_dir, sorted(missing))
    else:
        fixture_files = all_fixture_files

    if not fixture_files:
        logger.error("No fixture files found in %s (filter=%s)", fixtures_dir, fixture_filter)
        sys.exit(1)

    logger.info(
        "Found %d fixtures × %d arms × %d runs = %d total",
        len(fixture_files),
        len(arms),
        n_runs,
        len(fixture_files) * len(arms) * n_runs,
    )

    results_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)

    total_cells = len(fixture_files) * len(arms)
    cell_number = 0

    for fixture_file in fixture_files:
        scenario_id, scenario, fixture_meta = _load_fixture(str(fixture_file))

        for arm in arms:
            cell_number += 1

            for rep in range(1, n_runs + 1):
                tag = f"rep {rep}/{n_runs}" if n_runs > 1 else ""
                logger.info(
                    "[cell %d/%d%s] %s × %s (%s)",
                    cell_number,
                    total_cells,
                    f" {tag}" if tag else "",
                    scenario_id,
                    arm,
                    scenario.disruption_type,
                )

                result = RunResult.start(
                    arm=arm,
                    scenario_id=scenario_id,
                    disruption_type=scenario.disruption_type,
                    scenario=scenario,
                    fixture_meta=fixture_meta,
                )

                try:
                    if arm == "static":
                        from csco.arms.static import run
                    elif arm == "vector":
                        from csco.arms.vector import run
                    elif arm == "lexical":
                        from csco.arms.lexical import run
                    else:
                        raise ValueError(f"Unknown arm: {arm}")

                    decision = run(scenario, result=result)
                    result.finish(decision)

                    saved_path = result.save(output_dir)
                    logger.info("  → %s", saved_path)

                    run_dict = result.to_dict()
                    run_dict["rep"] = rep          # attach rep index for grouping
                    results_by_arm[arm].append(run_dict)

                except Exception as e:
                    # Classify the failure and update result accordingly
                    failure_message = str(e)
                    
                    # Attempt to classify the failure type
                    run_status = _classify_failure(e, failure_message)
                    
                    # Set failure metadata on result
                    result.run_status = run_status
                    result.failure_message = failure_message
                    
                    # Treatment rules per failure category
                    if run_status == "infrastructure_failure":
                        result.included_in_accuracy = False
                        result.counted_as_incorrect = False
                    elif run_status in ("model_output_failure", "tool_loop_failure", "extraction_failure"):
                        result.included_in_accuracy = True
                        result.counted_as_incorrect = True
                    else:  # unknown_failure
                        result.included_in_accuracy = False
                        result.counted_as_incorrect = False
                    
                    # Save the failed run for diagnostics
                    logger.error("  → FAILED [%s]: %s", run_status, failure_message, exc_info=True)
                    try:
                        saved_path = result.save(output_dir)
                        logger.info("  → Saved failed run: %s", saved_path)
                    except Exception as save_e:
                        logger.error("  → Could not save failed run: %s", save_e)
                    
                    # Still collect the failed result for batch reporting
                    run_dict = result.to_dict()
                    run_dict["rep"] = rep
                    results_by_arm[arm].append(run_dict)

    return results_by_arm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluation across three policy-grounding architectures: "
            "static-context, flat-vector-retrieval, and lexical-document-graph."
        )
    )
    parser.add_argument(
        "--fixtures-dir",
        default="fixtures/suite_a,fixtures/suite_b",
        help=(
            "Directory containing fixture YAML files. A comma-separated list of "
            "directories is also accepted; the default "
            "'fixtures/suite_a,fixtures/suite_b' runs the frozen dissertation set "
            "(Suite A = 37 prescribed-action + Suite B = 35 null = 72 scenarios)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory to save run results (default: results/)",
    )
    parser.add_argument(
        "--runs-per-fixture",
        type=int,
        default=1,
        metavar="N",
        help="Number of times to run each fixture×arm cell (default: 1). "
             "Use N≥3 to measure run-to-run consistency as a repeatability metric.",
    )
    parser.add_argument(
        "--fixtures-list",
        default=None,
        metavar="IDS",
        help=(
            "Comma-separated list of scenario IDs to run (e.g. "
            "'diag_0026,diag_0017'). Only fixtures whose stem matches "
            "an ID in this list will be executed. Useful for targeted pilots."
        ),
    )
    parser.add_argument(
        "--arms",
        default=None,
        metavar="ARMS",
        help=(
            "Comma-separated list of arms to run, overriding the default "
            "static,vector,lexical set. Valid values: static, vector, lexical."
        ),
    )
    args = parser.parse_args()

    if args.arms:
        active_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        active_arms = ["static", "vector", "lexical"]

    # Apply optional scenario filter
    fixture_filter: set[str] | None = None
    if args.fixtures_list:
        fixture_filter = {s.strip() for s in args.fixtures_list.split(",") if s.strip()}
        logger.info("Fixture filter active: %s", sorted(fixture_filter))

    logger.info(
        "Starting batch evaluation  (arms=%s, %d runs per cell)...",
        active_arms,
        args.runs_per_fixture,
    )
    results_by_arm = _run_all_fixtures_all_arms(
        args.fixtures_dir,
        args.output_dir,
        n_runs=args.runs_per_fixture,
        arms=active_arms,
        fixture_filter=fixture_filter,
    )

    report = _generate_summary_report(results_by_arm, n_runs=args.runs_per_fixture)
    print("\n" + report)

    report_path = Path(args.output_dir) / "BATCH_REPORT.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    primary_arms = [a for a in ["static", "vector", "lexical"] if results_by_arm.get(a)]
    acc_pi_all_rows = [
        row
        for arm in primary_arms
        for metric in skew_analysis.ACC_PI_METRICS
        for row in skew_analysis.acc_pi_rows(
            _scored_non_infra_runs(results_by_arm.get(arm, [])), arm, metric=metric
        )
    ]
    if acc_pi_all_rows:
        acc_pi_path = Path(args.output_dir) / "acc_pi.csv"
        skew_analysis.write_acc_pi_csv(acc_pi_path, acc_pi_all_rows)
        logger.info("Acc(pi) table saved to %s", acc_pi_path)


if __name__ == "__main__":
    main()
