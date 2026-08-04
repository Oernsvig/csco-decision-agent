"""Per-run scoring logic for RunResult, plus cross-run cell-aggregation helpers.

Extracted from run_result.py so that RunResult itself stays focused on
identity, instrumentation capture, and JSON persistence. Every function here
is pure with respect to its inputs — it reads from the passed-in RunResult
(or, for _cell_stats, a list of already-serialised run dicts) and writes only
into the caller-supplied scores dict. No behavior changed by this split; see
RunResult.score() for call order.

The oracle measures fidelity to the encoded disruption-response policy, not
independent real-world optimality of the recommendation.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import TYPE_CHECKING, Any

from csco.evaluation.binarize import is_null_decision, is_null_oracle
from csco.models import ScenarioDecision
from csco.oracle.manual import ScenarioOracle

if TYPE_CHECKING:
    from csco.evaluation.run_result import RunResult


def _recs_by_supplier_id(recs_list: list[dict], expected_ids: set[str] | None = None) -> dict[str, dict]:
    """Build {supplier_id: recommendation} from decision.recommendations,
    normalizing a literal "Supplier-" prefix drift.

    Some Suite B fixtures' `name` field is itself formatted "Supplier-{id}"
    (e.g. name: "Supplier-S-SB-GEO-TIER-ABOVE", supplier_id: "S-SB-GEO-TIER-
    ABOVE") — under corpus growth, Static in particular sometimes echoes
    that name-field formatting into its output supplier_id instead of the
    real id, with otherwise fully-correct content (disposition/escalation/
    fired_rule_id/permitted_actions all match). That's a measurement-
    validity bug: exact-match id lookup zeroed an otherwise-correct run.
    Narrow, mechanical fix — only strips a leading "Supplier-" when the
    remainder exactly matches a real supplier_id expected for this fixture;
    never fuzzy-matches, never masks a genuinely wrong supplier answer.
    """
    out: dict[str, dict] = {}
    for r in recs_list:
        sid = r.get("supplier_id")
        if sid is not None and expected_ids is not None and sid not in expected_ids:
            if sid.startswith("Supplier-") and sid[len("Supplier-"):] in expected_ids:
                sid = sid[len("Supplier-"):]
        out[sid] = r
    return out


# ---------------------------------------------------------------------------
# Per-run scorers (called in sequence by RunResult.score())
# ---------------------------------------------------------------------------

def score_routing(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    rd = result.decision.get("routing_result") or {}
    s["routing_rule_correct"]        = rd.get("fired_rule_id") == oracle.routing.fired_rule_id
    s["routing_disposition_correct"] = rd.get("disposition")   == oracle.routing.disposition
    s["routing_escalation_correct"]  = rd.get("escalation")    == oracle.routing.escalation
    s["overall_correct"] = all([
        s["routing_rule_correct"],
        s["routing_disposition_correct"],
        s["routing_escalation_correct"],
    ])
    # SDM-full: routing rules carry no oracle permitted_actions expectation
    # (RoutingOracle has no permitted_actions field), so it collapses to SDM-3.
    s["overall_correct_full"] = s["overall_correct"]


def score_suppliers(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    expected_ids = {e.supplier_id for e in oracle.suppliers}
    recs = _recs_by_supplier_id(result.decision.get("recommendations", []), expected_ids)

    disp_correct = esc_correct = rule_correct = actions_correct = 0
    total = len(oracle.suppliers)

    s["supplier_correctness"] = {}

    for exp in oracle.suppliers:
        got = recs.get(exp.supplier_id, {})
        det = {}

        disp_match    = got.get("disposition")   == exp.disposition
        esc_match     = got.get("escalation")    == exp.escalation
        rule_match    = got.get("fired_rule_id") == exp.fired_rule_id
        got_actions   = set(got.get("permitted_actions", []))
        exp_actions   = set(exp.permitted_actions)
        actions_match = got_actions == exp_actions

        deferred_violation = bool(got_actions - exp_actions)
        over_escalation  = _is_higher_escalation(got.get("escalation",""), exp.escalation)
        under_escalation = _is_lower_escalation(got.get("escalation",""), exp.escalation)
        over_replace  = got.get("disposition") == "replace"  and exp.disposition != "replace"
        under_replace = exp.disposition == "replace" and got.get("disposition") != "replace"

        det["disposition"]   = disp_match
        det["escalation"]    = esc_match
        det["rule_id"]       = rule_match
        det["permitted_actions"]  = actions_match
        det["deferred_action_recommended"] = deferred_violation
        det["over_escalation"]   = over_escalation
        det["under_escalation"]  = under_escalation
        det["over_replace"]      = over_replace
        det["under_replace"]     = under_replace
        det["action_set_superset_error"] = bool(got_actions - exp_actions)
        det["action_set_subset_error"]   = bool(exp_actions - got_actions)
        det["all_correct"]  = all([disp_match, esc_match, rule_match, actions_match])

        s["supplier_correctness"][exp.supplier_id] = det

        disp_correct    += int(disp_match)
        esc_correct     += int(esc_match)
        rule_correct    += int(rule_match)
        actions_correct += int(actions_match)

    denom = total if total > 0 else 1

    s["correctness_disposition"]       = disp_correct    / denom
    s["correctness_escalation"]        = esc_correct     / denom
    s["correctness_rule_id"]           = rule_correct    / denom
    s["correctness_permitted_actions"] = actions_correct / denom

    s["overall_correct"] = (
        disp_correct == total
        and esc_correct == total
        and rule_correct == total
    )
    # SDM-full: SDM-3 (overall_correct) AND permitted_actions set equality,
    # per supplier. A new, additional key — overall_correct is unchanged.
    s["overall_correct_full"] = (
        s["overall_correct"]
        and actions_correct == total
    )

    # Aggregate failure-mode flags
    s["deferred_action_recommended"] = any(
        s["supplier_correctness"][e.supplier_id]["deferred_action_recommended"]
        for e in oracle.suppliers
    )
    s["any_over_escalation"]  = any(
        s["supplier_correctness"][e.supplier_id]["over_escalation"]
        for e in oracle.suppliers
    )
    s["any_under_escalation"] = any(
        s["supplier_correctness"][e.supplier_id]["under_escalation"]
        for e in oracle.suppliers
    )
    s["any_over_replace"] = any(
        s["supplier_correctness"][e.supplier_id]["over_replace"]
        for e in oracle.suppliers
    )
    s["any_under_replace"] = any(
        s["supplier_correctness"][e.supplier_id]["under_replace"]
        for e in oracle.suppliers
    )
    s["action_set_superset_error"] = any(
        s["supplier_correctness"][e.supplier_id]["action_set_superset_error"]
        for e in oracle.suppliers
    )
    s["action_set_subset_error"] = any(
        s["supplier_correctness"][e.supplier_id]["action_set_subset_error"]
        for e in oracle.suppliers
    )

    # Direct band derivation: compare the agent's stated risk band to the
    # deterministically derived band. Policy is customised by the general risk
    # band and the disruption-type strategic priority, so only the risk band is
    # checked here.
    from csco.derive import derive
    from csco.specs.loader import load_spec

    spec = load_spec(result.disruption_type)
    derived_by_id = {
        derive(sup, spec).supplier.supplier_id: derive(sup, spec)
        for sup in result.scenario.suppliers if result.scenario
    } if result.scenario else {}

    band_correct_list = []
    for supplier_id, rec_dict in recs.items():
        if supplier_id in derived_by_id:
            derived = derived_by_id[supplier_id]
            assigned_band = rec_dict.get("assigned_risk_level")
            band_correct_list.append(assigned_band == derived.risk_level)

    if band_correct_list:
        s["assigned_risk_level_correct"] = mean(band_correct_list)

    # Strategic priority
    # Check BOTH the extracted per-supplier rationale AND the full raw answer.
    # Models often place the strategic priority in a concluding "Strategic Priority"
    # section rather than integrating it into per-supplier rationale paragraphs.
    # Checking only the extracted rationale field systematically undercounts Static,
    # which always writes a correct concluding section.
    rationale_text = " ".join(
        recs.get(e.supplier_id, {}).get("rationale", "")
        for e in oracle.suppliers
    ).lower()
    raw_text = result.agent_answer_raw.lower()
    combined_text = rationale_text + " " + raw_text

    if oracle.suppliers and oracle.suppliers[0].strategic_priority:
        priority = oracle.suppliers[0].strategic_priority.lower()
        all_priorities = {"continuity protection", "rapid reconfiguration", "cost containment"}

        # strategic_priority_mentioned: the correct priority appears anywhere in the
        # complete agent output (rationale + raw answer). Broad metric.
        s["strategic_priority_mentioned"] = priority in combined_text

        # strategic_priority_correct: alias of strategic_priority_mentioned.
        # Since we check for the correct label by construction, "mentioned" already
        # implies "correct". Kept as a separate key for report clarity.
        s["strategic_priority_correct"] = s["strategic_priority_mentioned"]

        # strategic_priority_in_rationale: stricter — priority integrated into the
        # extracted per-supplier rationale, not only in a concluding section.
        s["strategic_priority_in_rationale"] = priority in rationale_text

        # wrong_strategic_priority_used: a different disruption type's priority label
        # appears but the correct one does not.
        s["wrong_strategic_priority_used"] = (
            not s["strategic_priority_mentioned"]
            and any(
                other in combined_text
                for other in all_priorities
                if other != priority
            )
        )

    # Multi-supplier consistency
    if len(oracle.suppliers) > 1:
        all_sids = [e.supplier_id for e in oracle.suppliers]
        s["all_suppliers_correct"] = all(
            s["supplier_correctness"][sid]["all_correct"] for sid in all_sids
        )
        rule_ids_seen = {recs.get(sid, {}).get("fired_rule_id") for sid in all_sids}
        expected_rule_ids = {e.fired_rule_id for e in oracle.suppliers}
        s["supplier_cross_contamination"] = bool(rule_ids_seen - expected_rule_ids - {None})


def score_pass_structure(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    """Check whether the agent correctly applied Pass 1 before Pass 2."""
    routing_in_decision = bool(result.decision.get("routing_result"))
    routing_expected    = oracle.routing is not None
    s["pass1_pass2_correct"] = routing_in_decision == routing_expected
    s["supplier_assessed_when_routing_should_stop"] = (
        routing_expected
        and bool(result.decision.get("recommendations"))
    )


def score_retrieval(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    """Arm 2 vector retrieval metrics (strict + loose).

    Routing-oracle fixtures (oracle.suppliers == [], oracle.routing is not
    None — every *-NULL/*-DT/*-DT-NULL case) have no suppliers to key the
    governing provision off of, so retrieval metrics are computed against
    oracle.routing.fired_rule_id instead. This keeps
    retrieval_governing_provision_retrieved and friends defined for
    routing-only runs, so a *-NULL failure can be attributed to the
    retrieval stage (the governing routing chunk never surfaces) rather
    than the reasoning stage (it surfaces, gets misapplied).
    """
    if oracle.suppliers:
        governing_rule = oracle.suppliers[0].fired_rule_id
    elif oracle.routing is not None:
        governing_rule = oracle.routing.fired_rule_id
    else:
        return

    # Arm 2 — vector retrieval: STRICT and LOOSE metrics
    def _chunk_contains(chunk: dict, rule_id: str) -> bool:
        """Loose: rule_id mentioned anywhere in chunk metadata or text."""
        return (
            rule_id in chunk.get("heading", "")
            or rule_id in chunk.get("file", "")
            or rule_id in chunk.get("content_snippet", "")
        )

    def _is_governing_provision_chunk(chunk: dict, rule_id: str) -> bool:
        """Strict: chunk IS the actual provision for the rule (rule_id matches chunk metadata)."""
        return chunk.get("rule_id") == rule_id

    if result.searches and governing_rule:
        # Loose metric: rule_id mentioned anywhere
        mentioned = any(
            _chunk_contains(chunk, governing_rule)
            for se in result.searches
            for chunk in se.chunks
        )
        s["retrieval_rule_id_mentioned"] = mentioned

        # Strict metric: chunk IS the actual provision
        retrieved = any(
            _is_governing_provision_chunk(chunk, governing_rule)
            for se in result.searches
            for chunk in se.chunks
        )
        s["retrieval_governing_provision_retrieved"] = retrieved

        # Key diagnostic: retrieved governing provision but still wrong decision
        s["governing_provision_retrieved_but_wrong_decision"] = (
            retrieved and not s.get("overall_correct", False)
        )

        # Rank of first mention (diagnostic)
        for search_event in result.searches:
            for rank, chunk in enumerate(search_event.chunks, 1):
                if _chunk_contains(chunk, governing_rule):
                    s["retrieval_rank_of_rule_id_mention"] = rank
                    break
            if "retrieval_rank_of_rule_id_mention" in s:
                break

        # Definition sections retrieved (band / tier)
        all_chunks = [c for se in result.searches for c in se.chunks]
        s["retrieved_definition_band"] = any(
            c.get("section_role") == "bands" for c in all_chunks
        )
        s["retrieved_definition_tier"] = any(
            c.get("section_role") == "decision" and c.get("chunk_type") == "section"
            for c in all_chunks
        )
        s["retrieved_strategic_priority_section"] = any(
            c.get("section_role") == "posture" for c in all_chunks
        )

        # Pass-type correctness: did the agent retrieve at least one chunk from
        # the correct pass for this oracle case?
        oracle_pass = "routing" if oracle.routing else "supplier"
        s["retrieved_correct_pass"] = any(
            c.get("pass_type") == oracle_pass for c in all_chunks
        )

    # Wrong-pass rule selection: supplier rule used as routing result or vice versa.
    # Uses rule_id suffix patterns — routing rules end in NULL, DT, or DT-NULL.
    if result.searches:
        rr = result.decision.get("routing_result") or {}
        rr_id = rr.get("fired_rule_id", "")
        recs = result.decision.get("recommendations", [])

        s["supplier_rule_used_as_routing_error"] = bool(
            rr_id and not _is_routing_rule_id(rr_id)
        )
        s["routing_rule_used_as_supplier_error"] = any(
            _is_routing_rule_id(r.get("fired_rule_id", "")) for r in recs
        )
        s["wrong_pass_rule_selected"] = (
            s["supplier_rule_used_as_routing_error"]
            or s["routing_rule_used_as_supplier_error"]
        )


def score_lexical_references(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    """Arm 4 lexical-graph reference opening and usage metrics."""
    if not result.opens or not oracle.suppliers:
        return
    governing_rule    = oracle.suppliers[0].fired_rule_id
    governing_pointer = f"provision:{governing_rule}"

    governing_open = next(
        (ev for ev in result.opens if ev.pointer == governing_pointer), None
    )
    if not governing_open:
        s["opens_to_governing_provision"]       = None
        s["references_resolved"]                = False
        s["references_resolved_count"]          = 0
        s["references_expected_count"]          = 0
        s["unresolved_reference_on_failed_run"] = True
        return

    s["opens_to_governing_provision"] = governing_open.call_number

    refs = governing_open.references or []
    ref_by_type: dict[str, str] = {r["as"]: r["pointer"] for r in refs}
    s["references_expected_count"] = len(refs)

    # Check all pointers opened at any point during the run.
    # The "opened after governing provision" constraint is too strict: agents typically
    # open band/tier/escalation sections while walking earlier provisions (to evaluate
    # conditions), so the sections are consulted before or alongside the governing
    # provision rather than strictly after it.
    all_opened_ptrs = {ev.pointer for ev in result.opens}

    # Overall resolution
    ref_pointers  = set(ref_by_type.values())
    resolved_ptrs = ref_pointers & all_opened_ptrs
    s["references_resolved"]       = len(resolved_ptrs) > 0
    s["references_resolved_count"] = len(resolved_ptrs)
    # Descriptive co-occurrence metric: reference was not consulted AND the run failed.
    # NOT causal attribution — does not prove missing reference caused the wrong answer.
    s["unresolved_reference_on_failed_run"] = (
        not s["references_resolved"] and not s.get("overall_correct", False)
    )

    # Per-type reference metrics.
    recs = _recs_by_supplier_id(
        result.decision.get("recommendations", []),
        {e.supplier_id for e in oracle.suppliers},
    )
    exp  = oracle.suppliers[0] if oracle.suppliers else None

    for ref_type in ("band", "tier", "action", "escalation", "strategic_priority"):
        ptr = ref_by_type.get(ref_type)
        opened = ptr is not None and ptr in all_opened_ptrs
        s[f"reference_opened_{ref_type}"] = opened

        if ref_type == "action" and opened and exp:
            got_actions = set(recs.get(exp.supplier_id, {}).get("permitted_actions", []))
            s[f"reference_used_correctly_{ref_type}"] = got_actions == set(exp.permitted_actions)
        elif ref_type == "escalation" and opened and exp:
            got_esc = recs.get(exp.supplier_id, {}).get("escalation")
            s[f"reference_used_correctly_{ref_type}"] = got_esc == exp.escalation
        elif ref_type == "band" and opened and exp:
            # Proxy: correct rule_id means correct band interpretation
            got_rule = recs.get(exp.supplier_id, {}).get("fired_rule_id")
            s[f"reference_used_correctly_{ref_type}"] = got_rule == exp.fired_rule_id
        elif ref_type == "strategic_priority" and opened:
            sp_flag = s.get("strategic_priority_mentioned", False)
            s[f"reference_used_correctly_{ref_type}"] = sp_flag
        else:
            s[f"reference_used_correctly_{ref_type}"] = None


def score_reasoning_extraction(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    """Check whether the raw reasoning contained the correct rule and extraction preserved it."""
    if not result.agent_answer_raw:
        return

    # Collect expected rule IDs
    if oracle.routing:
        expected_rules = {oracle.routing.fired_rule_id}
    else:
        expected_rules = {exp.fired_rule_id for exp in oracle.suppliers}

    raw_lower = result.agent_answer_raw.lower()
    s["reasoning_answer_contains_correct_rule"] = any(
        rule_id.lower() in raw_lower for rule_id in expected_rules if rule_id
    )

    # Compare extracted rule IDs to what was stated in raw answer.
    # Collected from BOTH routing_result and recommendations, regardless of which
    # shape the oracle expects. extraction_valid/extractor_preserved_rule measure
    # whether the extractor faithfully pulled *some* rule_id out of the reasoning —
    # not whether the model picked the oracle-expected answer *shape* (routing vs
    # supplier). A shape mismatch is a pass1_pass2_correct / pass_structure issue
    # (scored separately by score_pass_structure), not an extraction issue.
    # Shape-restricting this used to make a should-have-routed answer that the
    # model instead scored per-supplier (or vice versa) come back as
    # extracted_rules=set() -> extraction_valid=False, which score_failure_stage
    # checks before pass_structure — silently misattributing a pass-structure
    # confusion to "extraction" in the failure-stage breakdown.
    extracted_rules: set[str] = set()
    rd = result.decision.get("routing_result") or {}
    if rd.get("fired_rule_id"):
        extracted_rules.add(rd["fired_rule_id"])
    for r in result.decision.get("recommendations", []):
        if r.get("fired_rule_id"):
            extracted_rules.add(r["fired_rule_id"])

    # Rules mentioned in raw text (pattern: TYPE-CODE, e.g. GEO-R3, CYB-HUB)
    raw_rule_pattern = re.compile(
        r"\b([A-Z]{2,5}-(?:R\d+|DT-NULL|DT-NULL|DT|NULL|HUB|DEP|SITE|MEDS|LOWS|MED|LOW))\b"
    )
    raw_mentioned = {m.group(1) for m in raw_rule_pattern.finditer(result.agent_answer_raw)}

    s["extractor_preserved_rule"] = (
        bool(extracted_rules)
        and bool(raw_mentioned & extracted_rules)
    )
    s["extraction_valid"] = bool(extracted_rules)


def score_citation_grounding(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    """Whether every rule_id the agent cited in its final decision was actually
    retrieved/opened before being cited (the skip-and-hallucinate check).

    Arm-specific definition of "opened":
      lexical: rule_id's provision pointer (provision:{rule_id}) appears in result.opens
      vector:  rule_id appears in some retrieved chunk's rule_id field
      static:  not applicable (full corpus always in context) — left unset

    Edge cases, fixed by design rather than incidental:
      - A routing_result/recommendation with no fired_rule_id (a "no provision
        matches" conclusion) cites nothing, so it is vacuously grounded — True,
        not False.
      - If extraction failed (extraction_valid is False), the field is None:
        an extraction failure must not masquerade as a hallucinated citation.
      - Multiple recommendations are scored individually and aggregated both
        ways: cited_rule_was_opened requires ALL cited rules grounded
        (primary/reported metric); cited_rule_was_opened_any requires at
        least one.
    """
    if result.arm not in ("lexical", "vector"):
        return
    if s.get("extraction_valid") is False:
        s["cited_rule_was_opened"] = None
        s["cited_rule_was_opened_any"] = None
        return

    if result.arm == "lexical":
        opened_ids = {
            ev.pointer.split("provision:", 1)[1]
            for ev in result.opens
            if ev.pointer.startswith("provision:")
        }
    else:  # vector
        opened_ids = {
            chunk.get("rule_id")
            for se in result.searches
            for chunk in se.chunks
            if chunk.get("rule_id")
        }

    cited_flags: list[bool] = []
    if oracle.routing is not None:
        rr = result.decision.get("routing_result") or {}
        rid = rr.get("fired_rule_id")
        cited_flags.append(True if not rid else rid in opened_ids)
    else:
        recs = result.decision.get("recommendations") or []
        if not recs:
            cited_flags.append(True)  # no recommendation at all cites nothing
        for r in recs:
            rid = r.get("fired_rule_id")
            cited_flags.append(True if not rid else rid in opened_ids)

    s["cited_rule_was_opened"] = all(cited_flags)
    s["cited_rule_was_opened_any"] = any(cited_flags)


def score_carryover_match(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    """Classify a failed supplier outcome's cited tuple against sibling supplier
    rules' real consequences, to distinguish relabelling (copying a real
    sibling's outcome) from genuine fabrication (no traceable source).

    Lexical only (the classification depends on the opens trace); supplier-
    oracle fixtures only (routing rules have no "sibling list" in this sense);
    only computed when overall_correct is False. Tuple equality is exact —
    (disposition, escalation, frozenset(permitted_actions)) — read from the
    spec's consequence block, never from prose parsing.

    "last_opened_sibling" means the last decision-section supplier-rule
    provision opened anywhere in the run's opens trace (trace-order, not
    "strictly before this citation" — the trace doesn't reliably expose
    per-recommendation timing, so this is a documented approximation, not
    a precise causal claim).

    carryover_match (run-level) is the most specific match_kind seen across
    any failed supplier, in priority order last_opened_sibling >
    other_sibling > no_match; correct/uncited suppliers don't contribute.
    None if the run has no failed supplier, is a routing case, or the spec
    can't be loaded.
    """
    if result.arm != "lexical" or oracle.routing is not None or not oracle.suppliers:
        return
    if s.get("overall_correct") is not False:
        return

    from csco.specs.loader import load_spec

    spec = load_spec(result.disruption_type)
    consequence_by_rule: dict[str, tuple] = {}
    known_rule_ids: set[str] = set()
    for r in spec.supplier_rules:
        c = r.consequence
        consequence_by_rule[r.rule_id] = (c.disposition, c.escalation, frozenset(c.permitted))
        known_rule_ids.add(r.rule_id)

    opened_in_order = [
        ev.pointer.split("provision:", 1)[1]
        for ev in result.opens
        if ev.pointer.startswith("provision:") and ev.pointer.split("provision:", 1)[1] in known_rule_ids
    ]
    last_opened = opened_in_order[-1] if opened_in_order else None

    recs = _recs_by_supplier_id(
        result.decision.get("recommendations", []),
        {e.supplier_id for e in oracle.suppliers},
    )
    supplier_correctness = s.get("supplier_correctness", {})

    priority = {"last_opened_sibling": 3, "other_sibling": 2, "no_match": 1}
    worst_kind: str | None = None

    for exp in oracle.suppliers:
        if supplier_correctness.get(exp.supplier_id, {}).get("all_correct"):
            continue
        rec = recs.get(exp.supplier_id)
        if not rec or not rec.get("fired_rule_id"):
            continue  # nothing cited for this supplier — no tuple to classify
        got_tuple = (
            rec.get("disposition"),
            rec.get("escalation"),
            frozenset(rec.get("permitted_actions") or []),
        )
        if last_opened and consequence_by_rule.get(last_opened) == got_tuple:
            kind = "last_opened_sibling"
        elif any(
            rid != exp.fired_rule_id and cons == got_tuple
            for rid, cons in consequence_by_rule.items()
        ):
            kind = "other_sibling"
        else:
            kind = "no_match"

        if worst_kind is None or priority[kind] > priority[worst_kind]:
            worst_kind = kind

    s["carryover_match"] = worst_kind


def score_failure_stage(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    """First-failure attribution: earliest broken link in the causal chain.

    Derived entirely from fields already populated by the sub-scorers above
    (no new measurement). Causal order, first match wins:
      1 extraction      s["extraction_valid"] is False, or
                         s["extractor_preserved_rule"] is False
                         (correct rule stated in prose, lost by the extractor)
      2 access           arm-specific — governing-rule content never reached:
                           vector:  s["retrieval_governing_provision_retrieved"] is False
                           lexical: s["opens_to_governing_provision"] is None
                                    (governing provision:<rule_id> pointer never opened)
                           static:  skipped — full corpus is always in context
      3 classification   derive() cross-check disagrees with the agent's stated
                         risk band: s["assigned_risk_level_correct"] < 1.0
      4 pass_structure   s["pass1_pass2_correct"] is False, or
                         s["supplier_assessed_when_routing_should_stop"] is True
      5 rule_selection   upstream clean, but fired_rule_id wrong:
                           routing:  s["routing_rule_correct"] is False
                           supplier: s["correctness_rule_id"] < 1.0
      6 execution        rule_id correct, but disposition/escalation wrong:
                           routing:  s["routing_disposition_correct"] or
                                     s["routing_escalation_correct"] is False
                           supplier: s["correctness_disposition"] or
                                     s["correctness_escalation"] < 1.0
    Anything left over after a confirmed failure -> "unclassified". Runs where
    overall_correct is True get "none". Never raises.
    """
    if s.get("overall_correct", False):
        s["failure_stage"] = "none"
        return

    # 1 — extraction
    if s.get("extraction_valid") is False or s.get("extractor_preserved_rule") is False:
        s["failure_stage"] = "extraction"
        return

    # 2 — access (arm-specific; static can never fail here)
    if result.arm == "vector":
        if s.get("retrieval_governing_provision_retrieved") is False:
            s["failure_stage"] = "access"
            return
    elif result.arm == "lexical":
        if "opens_to_governing_provision" in s and s.get("opens_to_governing_provision") is None:
            s["failure_stage"] = "access"
            return

    # 3 — classification
    band_ok = s.get("assigned_risk_level_correct")
    if band_ok is not None and band_ok < 1.0:
        s["failure_stage"] = "classification"
        return

    # 4 — pass structure
    if s.get("pass1_pass2_correct") is False or s.get("supplier_assessed_when_routing_should_stop") is True:
        s["failure_stage"] = "pass_structure"
        return

    # 5 / 6 — rule selection vs. execution
    if oracle.routing is not None:
        if s.get("routing_rule_correct") is False:
            s["failure_stage"] = "rule_selection"
            return
        if s.get("routing_disposition_correct") is False or s.get("routing_escalation_correct") is False:
            s["failure_stage"] = "execution"
            return
    else:
        rule_id_rate = s.get("correctness_rule_id")
        if rule_id_rate is not None and rule_id_rate < 1.0:
            s["failure_stage"] = "rule_selection"
            return
        disp_rate = s.get("correctness_disposition")
        esc_rate = s.get("correctness_escalation")
        if (disp_rate is not None and disp_rate < 1.0) or (esc_rate is not None and esc_rate < 1.0):
            s["failure_stage"] = "execution"
            return

    s["failure_stage"] = "unclassified"


def score_act_axis(result: "RunResult", s: dict, oracle: ScenarioOracle) -> None:
    """Act/No-Act binarization for the skew & prevalence analysis (see csco.evaluation.binarize).

    POSITIVE = action-required (oracle_null == False). Cells:
      TP: action required, agent acted   TN: both null
      FP: null but agent acted (false alarm)   FN: action required, agent said null (miss)
    "unbinarizable" covers empty/failed extraction — excluded from the 2x2,
    counted separately, never silently coerced into a cell.
    """
    oracle_null = is_null_oracle(oracle)
    try:
        decision_obj = ScenarioDecision.model_validate(result.decision)
        agent_null = is_null_decision(decision_obj)
    except Exception:
        agent_null = None

    if agent_null is None:
        cell = "unbinarizable"
    elif not oracle_null and not agent_null:
        cell = "TP"
    elif oracle_null and agent_null:
        cell = "TN"
    elif oracle_null and not agent_null:
        cell = "FP"
    else:  # not oracle_null and agent_null
        cell = "FN"

    s["act_axis"] = {"oracle_null": oracle_null, "agent_null": agent_null, "cell": cell}


def score_efficiency(result: "RunResult", s: dict) -> None:
    s["tool_calls_count"]  = result.tool_calls_total
    s["llm_calls_total"]   = result.llm_calls_reasoning + result.llm_calls_extraction
    s["elapsed_seconds"]   = result.elapsed_s

    # Policy-context tokens — a single, real-tokenizer scale across all arms
    # (static: whole consolidated policy; vector: retrieved chunk content;
    # lexical: opened provision/section text), so cross-arm comparisons are valid.
    if result.policy_tokens > 0:
        policy_tokens = result.policy_tokens
    elif result.searches:
        policy_tokens = sum(
            int(chunk.get("content_tokens", 0))
            for se in result.searches
            for chunk in se.chunks
        )
    elif result.opens:
        policy_tokens = sum(ev.response_tokens for ev in result.opens)
    else:
        policy_tokens = 0

    if policy_tokens > 0:
        s["policy_tokens"] = policy_tokens
    s["approx_policy_content_tokens"] = policy_tokens


# ---------------------------------------------------------------------------
# Rule-ID pass-type helper (used by score_retrieval)
# ---------------------------------------------------------------------------

def _is_routing_rule_id(rule_id: str) -> bool:
    """Return True if a rule_id is a routing rule (suffix NULL, DT, or DT-NULL).

    Routing rules end with these suffixes.  Any other suffix is a supplier rule.
    Used to detect wrong-pass errors: supplier rule used as routing result or vice versa.
    """
    if not rule_id:
        return False
    parts = rule_id.split("-")
    # CYB-DT-NULL → suffix="DT-NULL"; CYB-DT → "DT"; CYB-NULL → "NULL"
    if len(parts) >= 3 and parts[-1] == "NULL" and parts[-2] == "DT":
        return True  # DT-NULL
    return parts[-1] in ("NULL", "DT")


# ---------------------------------------------------------------------------
# Escalation comparison helpers
# ---------------------------------------------------------------------------

_ESC_RANK = {"Responsible Analyst": 0, "Cross-functional Management": 1, "Executive Leadership": 2}


def _is_higher_escalation(got: str, expected: str) -> bool:
    return _ESC_RANK.get(got, -1) > _ESC_RANK.get(expected, -1)


def _is_lower_escalation(got: str, expected: str) -> bool:
    return 0 <= _ESC_RANK.get(got, -1) < _ESC_RANK.get(expected, 99)


# ---------------------------------------------------------------------------
# Cross-run cell aggregation — used by batch_run.report and skew_analysis.
# Lives here (not in batch_run.py) so skew_analysis.py can import it directly
# instead of reaching back into batch_run.py at call time.
# ---------------------------------------------------------------------------

def _cell_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-cell statistics across N runs for one fixture×arm cell."""
    correct_flags = [r["scores"].get("overall_correct", False) for r in runs]
    n = len(correct_flags)
    k = sum(correct_flags)
    rate = k / n if n > 0 else 0.0
    consistent = (k == 0 or k == n)   # all agree

    # pass^k: every rep of this cell is overall_correct, excluding infrastructure
    # failures from the rep count (consistent with how accuracy is computed
    # elsewhere in this module). Unlike `consistent` (a determinism measure,
    # left untouched), this is a strict correctness measure.
    scored_runs = [r for r in runs if r.get("run_status") != "infrastructure_failure"]
    pass_all_k = bool(scored_runs) and all(
        r["scores"].get("overall_correct", False) for r in scored_runs
    )

    return {"n": n, "k": k, "rate": rate, "consistent": consistent, "pass_all_k": pass_all_k}
