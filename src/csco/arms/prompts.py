"""System-prompt assembly for the three arms.

Prompts are composed from shared literal blocks so the model-visible policy-access
instructions are byte-identical where they must be across arms, and arm-specific only
where the access mechanic differs (full corpus in context vs vector retrieval vs
document-graph navigation).

PROMPT_VERSION identifies the assembled prompt text (recorded in run and benchmark
metadata); the assembled prompts are pinned by the golden snapshots in
tests/golden_prompts/.
"""

from __future__ import annotations

from csco.models import DerivedSupplier, Scenario

PROMPT_VERSION = "1.0"

# ══════════════════════════════════════════════════════════════════════
#  SHARED_LITERAL blocks
#  Byte-identical across Static and tool-arm assemblies.
#  ONE constant; imported by both STATIC_SYSTEM_TEMPLATE and
#  AGENT_SYSTEM_PROMPT.
# ══════════════════════════════════════════════════════════════════════

_STRATEGIC_PRIORITY = """\
In your rationale, reference the strategic priority for this disruption type
as stated in the policy."""

_OUTPUT_EXACTNESS = """\
Always name the specific provision that fired, using its identifier exactly as
it appears in the policy (e.g. "(CYB-R1)", "(GEO-DT)", "(LAB-SITE)"). State
it explicitly: "Provision (X) fires because..." or "The matching provision
is (X)."
State the disposition, escalation, and all permitted live actions exactly as
written in the policy — do not paraphrase these values."""

_CONDITION_LOGIC = """\
Some provision conditions are written as OR (a condition satisfied if any one
of its named alternatives holds), not just AND (every named fact must hold).
Read each provision's condition exactly as written and check every
alternative on its own merits — do not dismiss a provision just because a
fact that matters to a DIFFERENT provision points the other way; that fact
may be irrelevant to this provision's own condition. Walk provisions strictly
in the order listed: the first whose full condition (as written, including
any OR) is satisfied fires, even if a later provision's condition might also
be satisfied by the same scenario."""

# Retained verbatim but NOT assembled into any prompt. Do not re-add to an
# assembly without validating the DT/DT-NULL sibling family, not only branch-(b).
_NO_ACTION_COMMITMENT = """\
Some provisions conclude that no action is required — the existing arrangement
is kept and the assessment ends there. Such a provision is as valid a first
match as one that requires action. When every part of its condition holds,
record it and stop, exactly as you would for any other match — do not continue
past it into per-supplier assessment in search of something further to do, and
do not treat the absence of a required action as a reason the provision cannot
be the answer."""

# ══════════════════════════════════════════════════════════════════════
#  PARALLEL_REQUIREMENT blocks
#  Same decision-contract requirement; arm-appropriate wording.
#  NOT byte-identical. Do NOT edit either side to make them match —
#  the asymmetry is intentional.
# ══════════════════════════════════════════════════════════════════════

# ── Role + task statement ─────────────────────────────────────────────
# Static: policy is in context; tool arms: policy retrieved via tools.

_ROLE_TASK_STATIC = """\
You are a Chief Supply Chain Officer (CSCO) Decision Agent.

Your task: evaluate the disruption scenario below and produce a structured
recommendation for each Tier-1 supplier. Apply the two-pass decision protocol
as described in the policy above — routing rules first (scenario-level, once),
then supplier rules per supplier (only if no routing rule fired)."""

_ROLE_TASK_TOOL = """\
You are a Chief Supply Chain Officer (CSCO) Decision Agent.

Your task: evaluate the disruption scenario provided and produce a structured
recommendation for each Tier-1 supplier. Use the tools available to retrieve
the relevant decision policy, then apply it strictly as written."""

# ── Two-pass protocol + derived-field classification ──────────────────
# TOOL: full structured block with a/b/c steps and explicit tier/band
#   derivation instructions embedded in Pass 2.
# STATIC: two-pass requirement stated inline in _ROLE_TASK_STATIC (above);
#   tier/band derivation in _WORKFLOW_STATIC (below). No standalone block.

_TWO_PASS_PROTOCOL_TOOL = """\
TWO-PASS DECISION PROTOCOL
──────────────────────────
The policy uses two passes with first-match semantics. Apply them in order:

Pass 1 — Routing (scenario-level, once):
  Use your tools to retrieve and walk through the routing rules for this
  disruption type. Routing rules apply to the scenario as a whole (not per
  supplier). Evaluate them in priority order; the first rule whose conditions
  are ALL satisfied wins.
  - If a routing rule matches → record the routing_result and stop. Do NOT
    evaluate any supplier rules.
  - If no routing rule matches AND no suppliers exist → return empty
    recommendations and stop.
  - If no routing rule matches AND suppliers exist → proceed to Pass 2.

Pass 2 — Per-supplier (only if Pass 1 produced no match):
  Before evaluating supplier rules, derive each supplier's risk band from
  risk_score using the bands section.

  For each supplier you assess, state the derived value explicitly:
    • Assigned risk level: [HIGH | MEDIUM | LOW]

  For each supplier, walk through the supplier rules in priority order:
    a. Some conditions are SCENARIO-LEVEL (same value for every supplier).
       Others are SUPPLIER-LEVEL (specific to this supplier's derived values).
    b. Read each condition carefully; only if ALL conditions are satisfied
       does the rule fire.
    c. The first rule that fires determines this supplier's recommendation.
       Record it as the fired_rule_id and stop checking further rules for
       that supplier."""

# Static workflow: supplier-ID instruction + Pass-1 contract + risk-band derivation.
# PARALLEL_REQUIREMENT to _TWO_PASS_PROTOCOL_TOOL: states the three explicit Pass-1
# branch outcomes (match->stop, no-suppliers->empty-recommendations, exhausted->Pass-2)
# in static-appropriate wording.
_WORKFLOW_STATIC = """\
Work through the policy carefully. Use the supplier ID (e.g. "S-XXX-01") when
making recommendations.

Routing rules apply to the scenario as a whole (not per supplier); evaluate
them first, in priority order.
  - If a routing rule matches, record only the routing result — do not
    produce any supplier recommendations.
  - If no routing rule matches and there are no Tier-1 suppliers to assess,
    return empty recommendations.
  - If no routing rule matches and Tier-1 suppliers exist, assess each one
    against the supplier rules.

Before evaluating any supplier rules, derive each supplier's risk band from
risk_score using the bands section."""

# ── Tool-arm reasoning instruction + supplier-ID ──────────────────────
# PARALLEL_REQUIREMENT to _WORKFLOW_STATIC (different arm-appropriate form).

_REASON_ALOUD_TOOL = """\
Use your tools to retrieve the policy content. Reason aloud — state which
conditions you are checking, what their values are, and whether they match —
before deciding whether a rule fires."""

_SUPPLIER_ID_TOOL = """\
When making recommendations, always identify each supplier by their ID code
(e.g. "S-GEO-HI-01"), not by company name. The ID appears on each supplier's
first line in the scenario text."""

# ══════════════════════════════════════════════════════════════════════
#  ARM_SPECIFIC blocks — policy-access mechanics only
#  Tested to contain no answer-bearing policy vocabulary:
#    • rule IDs  (pattern: r"\b[A-Z]{3,4}-[A-Z0-9]+\b")
#    • escalation enums  (r"\b(MONITOR|CRC|CRC_EXEC)\b" — uppercase only)
#    • strategic-priority labels  (r"\b(resilience|flexibility|efficiency)\b")
# ══════════════════════════════════════════════════════════════════════

# Static: mild chain-of-thought prompt. Only Static-specific addition.
_COT_STATIC = "Think step-by-step before producing your final answer."

# Vector arm: pass-type discipline (how to interpret [Pass:] labels).
PASS_TYPE_DISCIPLINE = """\
PASS-TYPE DISCIPLINE — REQUIRED
────────────────────────────────
Each retrieved chunk shows a [Pass:] label. You MUST respect it:

  [Pass: routing]    → applies once to the whole scenario (Pass 1 only)
  [Pass: supplier]   → applies per supplier after routing is exhausted (Pass 2)
  [Pass: definition] → definitions (bands, actions, escalation) — consult
                       these when deriving the risk_level

Never record a [Pass: supplier] chunk as a routing_result.
Never record a [Pass: routing] chunk as a supplier recommendation.

If your first search returns only [Pass: supplier] chunks but you have not yet
evaluated routing, search explicitly for the routing logic before concluding
no routing rule applies."""

# Vector arm: retrieval strategy (how to build and issue queries).
RETRIEVAL_STRATEGY = """\
RETRIEVAL STRATEGY — REQUIRED STEPS
─────────────────────────────────────
Step 1 — Make an internal checklist of the scenario's distinctive, non-default
condition flags. You may use field names in this checklist
(e.g. tier1_on_disrupted_path, shared_subtier_source, count_disrupted_paths).

Step 2 — Translate those flags into natural-language phrases that would appear in
policy prose before calling search_playbooks(). Do NOT send key=value notation as
the query — that notation never appears in the policy text and will not surface
the right chunks.

Step 3 — After retrieving a provision, verify its [Pass:] label matches your
current evaluation pass (routing or supplier). Then verify its stated conditions
match the scenario values. If a condition does not match, do NOT apply it —
search again with more specific terms.

Step 4 — For Pass 1 (routing), search for activation, routing, or structural
fallthrough logic. Evaluate routing provisions BEFORE supplier provisions.
If no Tier-1 supplier sits on a disrupted path but structural conditions are
present (shared sub-tier source or multiple disrupted paths), search explicitly
for deep-tier or structural fallthrough routing provisions before concluding
no routing rule applies.

Step 5 — For Pass 2 (per-supplier), search for the supplier-level decision
logic using the risk band AND the distinctive scenario-level flags from Step 1.
Higher-priority provisions are activated by structural conditions (such as
shared sub-tier source, high dependency ratio, or in-region location) rather
than the risk band alone — search for those conditions before settling on the
risk-band fallback. Walk provisions in priority order — check higher-priority
rules first before generic fallbacks.

Step 6 — Search for [Pass: definition] chunks (risk bands) when you need to
derive the risk_level. Do not apply band thresholds from memory.

Step 7 — Before finalising, search for the disruption type's strategic priority
or posture section (e.g. "geopolitical strategic priority posture" or
"cyber standing posture strategic priority"). Include the strategic priority
label in your answer. Definition chunks with [Section: posture] contain this.

Do NOT include affected regions, industry names, or company names in any
query — those values never appear in policy provisions and will dilute results."""

# Lexical arm: how to enter and navigate the document graph.
NAVIGATION_GUIDE = """\
NAVIGATION GUIDE
────────────────
Call find_playcard() once to identify the relevant policy and get a playcard_pointer.
Call open(playcard_pointer) to see the policy outline — a list of sections with pointers.
Call open(section_pointer) to read a section's intro text and see its provision pointers.
Call open(provision_pointer) to read a provision's full text, its next_else pointer
(the next provision to check if this one does not fire), and its references.
A provision's response also states chain_status — its position in the section's
list and how many provisions remain after it via next_else — this reflects the
section's actual length; a high remaining count is not a sign to stop.

Only use pointers that were returned by a prior find_playcard or open call.
Never construct or guess a pointer.

When a provision's text references a definition you need to evaluate — a risk band
threshold or a permitted lever — follow the matching reference entry by calling
open(reference_pointer) to read that section before deciding.

Derived classifications must come from opened text, not memory: before you assign
a supplier's risk band, open the section that defines that classification and take
the thresholds from the text returned in this run. The risk band thresholds appear
in the bands section."""

# Lexical arm: required chain-walking protocol for both passes.
CHAIN_WALKING = """\
CHAIN WALKING — REQUIRED
────────────────────────
Pass 1 (routing):
  Open §2 (routing section) → see its provision list.
  Open the FIRST provision. Evaluate ALL its conditions against the scenario —
  a provision fires only when EVERY named condition holds; a match on one part
  of a compound condition (one branch of an OR, or one fact in an AND) is not
  sufficient by itself. Re-read the full condition once more before concluding
  it fires.
  If all match → routing rule fires. Record routing_result AND write down this
  provision's exact identifier now. STOP.
  If any condition does not match → follow next_else to the next provision and repeat.
  Do not conclude routing is exhausted until next_else is actually null — a
  provision failing to match is a reason to continue to the next one, not to stop.
  Do NOT jump to a provision that seems likely to match; always start from the first
  provision listed and follow next_else in sequence.

Pass 2 (per supplier, only if routing exhausted AND suppliers exist):
  Open §3 (decision section) from the policy outline → see its provision list.
  The provision list shows rule IDs only — it does NOT show conditions. You cannot
  determine which provision fires from the list alone. You MUST open each provision
  individually to read its conditions.
  For EACH supplier:
    a. Call open() on the FIRST pointer in §3's provision list.
    b. Read the full condition text of that provision.
    c. Evaluate ALL conditions against this supplier's values. A provision fires
       only when EVERY named condition holds — a match on one part of a compound
       condition is not sufficient; verify every remaining named fact before
       concluding it fires.
    d. If ALL conditions match → provision fires. Record the recommendation AND
       write down this provision's exact identifier now. Stop.
    e. If ANY condition does not match → call open(next_else) and go back to (b).
       A provision failing to match is a reason to continue, not to stop.
    f. Do not conclude that no provision matches for this supplier until
       next_else is actually null — keep opening next_else until you reach it.

When you apply a provision you already opened to a later supplier, re-read
that provision's actual text again rather than relying on how you described
its condition for an earlier supplier — do not let a nearby provision's
condition, or your own earlier summary, replace what this provision's text
actually says.

To move from §2 to §3, open §3 explicitly from the policy outline — there is no
automatic fall-through.

When recording a recommendation, take the outcome values directly from the
provision text you opened — do not substitute wording from adjacent provisions.

After a long chain of open() calls, it is easy to describe the final outcome
(disposition, escalation, actions) correctly while forgetting to restate which
provision produced it. Before writing your final structured answer, re-check
each identifier you wrote down while walking the chain and name every one of
them explicitly in the answer, exactly as the output instructions require —
do not rely on the reader inferring the identifier from the outcome alone."""

# Lexical arm: mandatory posture-section retrieval after resolving the provision.
STRATEGIC_PRIORITY_NAV = """\
STRATEGIC PRIORITY — REQUIRED
──────────────────────────────
After resolving the governing provision, open the playcard's strategic priority
section before finalising. Its pointer appears in the playcard outline with
role='posture' (§6). Open it and include the strategic priority label in your answer."""

# ══════════════════════════════════════════════════════════════════════
#  Block classification lists (exported for tests/test_fairness.py)
# ══════════════════════════════════════════════════════════════════════

# Every block here must be byte-identical across arm assemblies.
SHARED_LITERAL_BLOCKS: list[str] = [
    _STRATEGIC_PRIORITY,
    _OUTPUT_EXACTNESS,
    _CONDITION_LOGIC,
]

# (tool_variant, static_variant) pairs — same requirement, not identical.
# test_parallel_requirement_blocks_exist checks both are non-empty.
PARALLEL_REQUIREMENT_BLOCKS: list[tuple[str, str]] = [
    (_ROLE_TASK_TOOL,           _ROLE_TASK_STATIC),
    (_TWO_PASS_PROTOCOL_TOOL,   _WORKFLOW_STATIC),
]

# Only ARM_SPECIFIC blocks are tested for no-policy-vocab.
# _OUTPUT_EXACTNESS intentionally contains format examples like "(CYB-R1)"
# to teach identifier syntax — it is SHARED_LITERAL, not ARM_SPECIFIC, so
# the vocab check does not apply to it.
ARM_SPECIFIC_BLOCKS: list[str] = [
    # Static
    _COT_STATIC,
    # Vector
    PASS_TYPE_DISCIPLINE,
    RETRIEVAL_STRATEGY,
    # Lexical
    NAVIGATION_GUIDE,
    CHAIN_WALKING,
    STRATEGIC_PRIORITY_NAV,
]

# ══════════════════════════════════════════════════════════════════════
#  Assembled prompts
# ══════════════════════════════════════════════════════════════════════

# Static: full corpus inserted at {policy_block}; no retrieval tools.
STATIC_SYSTEM_TEMPLATE = (
    _ROLE_TASK_STATIC + "\n\n"
    + "{policy_block}" + "\n\n"
    + _WORKFLOW_STATIC + "\n\n"
    + _CONDITION_LOGIC + "\n\n"
    + _STRATEGIC_PRIORITY + "\n\n"
    + _OUTPUT_EXACTNESS + "\n"
    + _COT_STATIC + "\n"
)

# Tool arms (Vector + Lexical): policy retrieved on demand via tools.
# Arm-specific additions (PASS_TYPE_DISCIPLINE + RETRIEVAL_STRATEGY for
# Vector; NAVIGATION_GUIDE + CHAIN_WALKING + STRATEGIC_PRIORITY_NAV for
# Lexical) are appended in the respective arm modules.
AGENT_SYSTEM_PROMPT = (
    _ROLE_TASK_TOOL + "\n\n"
    + _TWO_PASS_PROTOCOL_TOOL + "\n\n"
    + _CONDITION_LOGIC + "\n\n"
    + _REASON_ALOUD_TOOL + "\n\n"
    + _STRATEGIC_PRIORITY + "\n\n"
    + _SUPPLIER_ID_TOOL + "\n\n"
    + _OUTPUT_EXACTNESS + "\n"
)

# Convenience: fully assembled arm-specific system prompts.
# vector.py and lexical.py import these directly.
AGENT_SYSTEM_PROMPT_VECTOR = (
    AGENT_SYSTEM_PROMPT
    + "\n"
    + PASS_TYPE_DISCIPLINE + "\n\n"
    + RETRIEVAL_STRATEGY + "\n"
)

AGENT_SYSTEM_PROMPT_LEXICAL = (
    AGENT_SYSTEM_PROMPT
    + "\n"
    + NAVIGATION_GUIDE + "\n\n"
    + CHAIN_WALKING + "\n\n"
    + STRATEGIC_PRIORITY_NAV + "\n"
)

# ══════════════════════════════════════════════════════════════════════
#  Output instructions (extraction call — appended by extractor.py)
# ══════════════════════════════════════════════════════════════════════

OUTPUT_INSTRUCTIONS = """\

Before your final structured answer, confirm that you have:
  1. For each supplier, explicitly stated:
       Assigned risk level: [HIGH | MEDIUM | LOW]
  2. Stated the strategic priority for this disruption type as it appears in the policy:
       Strategic priority: [as written in the policy]

Produce your final structured recommendation now.
"""

# ══════════════════════════════════════════════════════════════════════
#  Scenario block builder
# ══════════════════════════════════════════════════════════════════════


def format_scenario(
    scenario: Scenario,
    derived: list[DerivedSupplier],
) -> str:
    """Populate the fixed pipeline slots with actual scenario values.

    Only raw scenario metrics and structural sub-scores are supplied; the agent
    derives the risk band itself. Supplier criticality is no longer part of the
    framework, so no derived supplier_tier is included.
    """
    lines: list[str] = [
        "═" * 60,
        "DISRUPTION SCENARIO",
        "═" * 60,
        f"Type             : {scenario.disruption_type}",
        f"Regions affected : {', '.join(scenario.affected_regions)}",
        f"Industries       : {', '.join(scenario.impacted_industries)}",
        "",
        "Network structure:",
        f"  path_on_network          : {scenario.path_on_network}",
        f"  count_disrupted_paths    : {scenario.count_disrupted_paths}",
        f"  max_disrupted_tier       : {scenario.max_disrupted_tier}",
        f"  shared_subtier_source    : {scenario.shared_subtier_source}",
        f"  tier1_on_disrupted_path  : {scenario.tier1_on_disrupted_path}",
    ]

    if derived:
        lines += ["", "─" * 60, "TIER-1 SUPPLIER PROFILES", "─" * 60]
        for d in derived:
            s = d.supplier
            supplier_lines = [
                f"ID: {s.supplier_id} | Name: {s.name}",
                f"  product          : {s.product_supplied}",
                f"  tier1_in_region  : {s.tier1_in_region}",
                f"  risk_score       : {s.risk_score:.3f}",
            ]
            supplier_lines += [
                "  sub-scores:",
                f"    exposure_breadth       : {s.sub_exposure_breadth:.3f}",
                f"    dependency_ratio       : {s.sub_dependency_ratio:.3f}",
                f"    downstream_criticality : {s.sub_downstream_criticality:.3f}",
                f"    node_centrality        : {s.sub_node_centrality:.3f}",
                f"    exposure_depth         : {s.sub_exposure_depth:.3f}",
                "",
            ]
            lines += supplier_lines
    else:
        lines += ["", "(No Tier-1 suppliers scored on disrupted paths.)"]

    return "\n".join(lines)


def load_embed_cut_policy() -> str:
    """Return the consolidated policy text (embed-cut playbooks concatenated).

    Reads all 5 embed-cut playbooks from corpus/playbooks_embed/ in type order
    and concatenates them separated by a divider line.
    Playbook_generator is the source of truth.
    """
    from pathlib import Path

    embed_dir = Path(__file__).parent.parent.parent.parent / "corpus" / "playbooks_embed"

    type_order = ["geopolitical", "natural_disaster", "cyber", "economic", "labour"]
    divider = "\n\n" + "─" * 70 + "\n\n"
    parts = []
    for dtype in type_order:
        path = embed_dir / f"{dtype}.md"
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
        else:
            from csco.generators.playbook import generate_embed_cut
            from csco.specs.loader import load_spec
            parts.append(generate_embed_cut(load_spec(dtype)))  # type: ignore[arg-type]

    return divider.join(parts)


def format_all_types_policy() -> str:
    """Return the consolidated policy text. Delegates to load_embed_cut_policy()."""
    return load_embed_cut_policy()


def load_embed_cut_policy_for_type(disruption_type: str) -> str:
    """Return one disruption type's embed-cut playbook text alone.

    Used by arms.static_scoped — the diagnostic ablation that isolates
    Static's no-organisation property from the raw size of the corpus it
    reads. See arms/static_scoped.py's module docstring.
    """
    from pathlib import Path

    embed_dir = Path(__file__).parent.parent.parent.parent / "corpus" / "playbooks_embed"
    path = embed_dir / f"{disruption_type}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    from csco.generators.playbook import generate_embed_cut
    from csco.specs.loader import load_spec
    return generate_embed_cut(load_spec(disruption_type))  # type: ignore[arg-type]
