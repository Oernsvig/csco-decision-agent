"""Parity check: verify all 52 rules from all five specs are present in the policy block.

This module ensures that the consolidated policy block and any generated corpus
contain all decision rules without omission or modification.
"""

from __future__ import annotations

import logging
from typing import Set

from csco.specs.loader import load_all_specs
from csco.specs.schema import DisruptionType

logger = logging.getLogger(__name__)


def collect_all_rule_ids() -> dict[DisruptionType, Set[str]]:
    """Collect all rule IDs from all five specs.
    
    Returns a dict mapping disruption_type -> set of rule_ids for that type.
    """
    specs = load_all_specs()
    result: dict[DisruptionType, Set[str]] = {}
    
    for dtype, spec in specs.items():
        rule_ids: Set[str] = set()
        
        # Routing rules
        for rr in spec.routing_rules:
            rule_ids.add(rr.rule_id)
        
        # Supplier rules
        for sr in spec.supplier_rules:
            rule_ids.add(sr.rule_id)
        
        result[dtype] = rule_ids
    
    return result


def check_policy_block_completeness(policy_block: str) -> tuple[bool, list[str]]:
    """Check that all 52 rules appear in the given policy block text.
    
    Args:
        policy_block: The rendered policy block (from format_all_types_policy or similar)
    
    Returns:
        (is_complete, missing_rule_ids)
        - is_complete: True if all 52 rules are found
        - missing_rule_ids: List of rule IDs not found in the block (empty if complete)
    """
    rules_by_type = collect_all_rule_ids()
    all_expected_rule_ids: Set[str] = set()
    
    for ids in rules_by_type.values():
        all_expected_rule_ids.update(ids)
    
    missing: list[str] = []
    for rule_id in sorted(all_expected_rule_ids):
        if rule_id not in policy_block:
            missing.append(rule_id)
    
    is_complete = len(missing) == 0
    return is_complete, missing


def check_all_encodings() -> None:
    """Parity check: verify all 52 rules are present in the Arm 1 and Arm 2 encodings.

    Checks:
      1. Arm 1 (static): All rules in consolidated policy block
      2. Arm 2 (vector): All rules in hint-free corpus
    """
    from csco.arms.prompts import format_all_types_policy
    from csco.generators.playbook import generate_all_embed_cuts_text

    rules_by_type = collect_all_rule_ids()
    all_expected_rule_ids: Set[str] = set()
    for ids in rules_by_type.values():
        all_expected_rule_ids.update(ids)

    total = len(all_expected_rule_ids)

    # Arm 1: Static policy block
    policy = format_all_types_policy()
    is_complete_policy, missing_policy = check_policy_block_completeness(policy)

    # Arm 2: Vector corpus
    corpus = generate_all_embed_cuts_text()
    is_complete_corpus, missing_corpus = check_policy_block_completeness(corpus)

    # Report
    print(f"\n{'═' * 70}")
    print("PARITY CHECK — Static and Vector Encodings")
    print(f"{'═' * 70}")
    print(f"Expected total rules from specs: {total}")
    print("\nArm 1 (Static Policy Block):")
    if is_complete_policy:
        print(f"  ✓ PASS: All {total} rules present")
    else:
        print(f"  ✗ FAIL: Missing {len(missing_policy)} rules: {', '.join(missing_policy)}")

    print("\nArm 2 (Vector Corpus):")
    if is_complete_corpus:
        print(f"  ✓ PASS: All {total} rules present")
    else:
        print(f"  ✗ FAIL: Missing {len(missing_corpus)} rules: {', '.join(missing_corpus)}")

    print(f"{'═' * 70}\n")

    # Raise if any failed
    if not is_complete_policy or not is_complete_corpus:
        errors = []
        if not is_complete_policy:
            errors.append(f"Arm 1 policy: {missing_policy}")
        if not is_complete_corpus:
            errors.append(f"Arm 2 corpus: {missing_corpus}")

        raise AssertionError(
            "Parity check failed across encodings:\n" + "\n".join(errors)
        )


if __name__ == "__main__":
    check_all_encodings()
