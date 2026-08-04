"""CLI entrypoint: run one of the three arms against a scenario fixture.

Usage
-----
  python -m csco.cli.run --arm static  --scenario fixtures/suite_a/diag_0025.yaml
  python -m csco.cli.run --arm vector  --scenario fixtures/suite_b/sb_0001.yaml
  python -m csco.cli.run --arm lexical --scenario fixtures/suite_b/sb_0001.yaml

  # Save results for data analysis:
  python -m csco.cli.run --arm static --scenario fixtures/suite_a/diag_0025.yaml \\
                     --output-dir results/

  # Run multiple times for consistency measurement:
  for i in 1 2 3 4 5; do
      python -m csco.cli.run --arm static --scenario fixtures/suite_a/diag_0025.yaml \\
                         --output-dir results/
  done
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from csco.models import Scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _load_scenario(path: str) -> Scenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Scenario.model_validate(raw)


def _scenario_id(path: str) -> str:
    return Path(path).stem


def _print_result(decision, arm: str, scenario_path: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  ARM : {arm.upper()}")
    print(f"  SCENARIO : {Path(scenario_path).name}")
    print(f"{'═' * 60}\n")

    if decision.routing_result:
        rr = decision.routing_result
        print("ROUTING RESULT (scenario-level rule fired)")
        print(f"  rule         : {rr.fired_rule_id}")
        print(f"  action       : {rr.scenario_action}")
        print(f"  disposition  : {rr.disposition}")
        print(f"  escalation   : {rr.escalation}")
        print("  Per-supplier pass: skipped\n")
    else:
        print(f"SUPPLIER RECOMMENDATIONS  ({len(decision.recommendations)} suppliers)\n")
        for rec in decision.recommendations:
            print(f"  ── {rec.supplier_id} ──────────────────────")
            print(f"     disposition      : {rec.disposition}")
            print(f"     escalation       : {rec.escalation}")
            print(f"     permitted_actions: {rec.permitted_actions}")
            if rec.fired_rule_id:
                print(f"     rule             : {rec.fired_rule_id}")
            print(f"     rationale        : {rec.rationale}")
            print()

    print(f"{'─' * 60}")
    print("RAW JSON:")
    print(json.dumps(decision.model_dump(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CSCO Decision Agent")
    parser.add_argument(
        "--arm",
        choices=["static", "vector", "lexical"],
        required=True,
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to a scenario fixture YAML (e.g. fixtures/suite_a/diag_0025.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save run results as JSON (e.g. results/). "
             "Creates results/{scenario}/{arm}/{timestamp}_{run_id}.json",
    )
    args = parser.parse_args()

    scenario = _load_scenario(args.scenario)
    scenario_id = _scenario_id(args.scenario)
    logger.info("Loaded scenario: %s (%s)", Path(args.scenario).name, scenario.disruption_type)

    # Create RunResult for instrumentation
    result = None
    if args.output_dir:
        from csco.evaluation.run_result import RunResult
        result = RunResult.start(
            arm=args.arm,
            scenario_id=scenario_id,
            disruption_type=scenario.disruption_type,
        )

    if args.arm == "static":
        from csco.arms.static import run
        decision = run(scenario, result=result)

    elif args.arm == "vector":
        from csco.arms.vector import run
        decision = run(scenario, result=result)

    elif args.arm == "lexical":
        from csco.arms.lexical import run
        decision = run(scenario, result=result)

    else:
        logger.error("Unknown arm: %s", args.arm)
        sys.exit(1)

    _print_result(decision, args.arm, args.scenario)

    # Save results if output dir specified
    if result is not None and args.output_dir:
        result.finish(decision)
        saved_path = result.save(args.output_dir)
        logger.info("Run result saved → %s", saved_path)

        # Print scores summary
        s = result.scores
        print(f"\n{'─' * 60}")
        print("SCORES:")
        overall = s.get("overall_correct")
        print(f"  overall_correct          : {overall}")
        if "correctness_disposition" in s:
            print(f"  correctness_disposition  : {s['correctness_disposition']}")
            print(f"  correctness_escalation   : {s['correctness_escalation']}")
            print(f"  correctness_rule_id      : {s['correctness_rule_id']}")
        elif "routing_disposition_correct" in s:
            print(f"  routing_disposition_correct : {s['routing_disposition_correct']}")
            print(f"  routing_escalation_correct  : {s['routing_escalation_correct']}")
            print(f"  routing_rule_correct        : {s['routing_rule_correct']}")
        if "retrieval_governing_provision_retrieved" in s:
            print(f"  retrieval_governing_prov : {s['retrieval_governing_provision_retrieved']}")
        if "retrieval_rule_id_mentioned" in s:
            print(f"  retrieval_rule_id_loosely: {s['retrieval_rule_id_mentioned']}")
        print(f"  elapsed_s                : {result.elapsed_s}")
        print(f"  llm_calls_total          : {result.llm_calls_reasoning + result.llm_calls_extraction}")
        print(f"  tool_calls_total         : {result.tool_calls_total}")


if __name__ == "__main__":
    main()
