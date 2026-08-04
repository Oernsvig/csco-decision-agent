"""Playbook generator: produce full playbooks and embed cuts from specs.

Two document variants per disruption type:
  - Full playbook (all 8 sections, including ## 8. Worked example)
    → corpus/playbooks/<type>.md   (human reference)
  - Embed cut (sections 1-7 only, no worked examples = no answer-key leakage)
    → corpus/playbooks_embed/<type>.md  (embedded into Arm 2, Arm 1 policy basis)

Run:  python -m csco.generators.playbook

_CORPUS_VERSION is rendered into every playbook's header ("**Version:** X.Y")
and tracks model-visible content changes to the generated corpus, the
policy-content analogue of arms/prompts.py's PROMPT_VERSION.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from csco.specs.loader import load_all_specs
from csco.specs.schema import DisruptionType, Spec

_CORPUS_VERSION = "2.7"

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_FULL_DIR = _REPO_ROOT / "corpus" / "playbooks"
_EMBED_DIR = _REPO_ROOT / "corpus" / "playbooks_embed"

# ---------------------------------------------------------------------------
# Fields that are scenario-level (the same value for every supplier in a run).
# All other when_all fields are supplier-level (assessed per supplier).
# ---------------------------------------------------------------------------

_SCENARIO_LEVEL_FIELDS: frozenset[str] = frozenset({
    "path_on_network",
    "tier1_on_disrupted_path",
    "shared_subtier_source",
    "count_disrupted_paths",
    "max_disrupted_tier",
})

# ---------------------------------------------------------------------------
# Type lookup tables
# ---------------------------------------------------------------------------

_TYPE_ORDER: list[DisruptionType] = [
    "geopolitical",
    "natural_disaster",
    "cyber",
    "economic",
    "labour",
]

_DOC_IDS: dict[DisruptionType, str] = {
    "geopolitical":     "SCR-STD-GEO-001",
    "natural_disaster": "SCR-STD-NAT-001",
    "cyber":            "SCR-STD-CYB-001",
    "economic":         "SCR-STD-ECON-001",
    "labour":           "SCR-STD-LAB-001",
}

_PRIORITY_LABELS: dict[DisruptionType, int] = {
    "geopolitical": 1,
    "natural_disaster": 2,
    "cyber": 3,
    "economic": 4,
    "labour": 5,
}

_TYPE_DISPLAY: dict[DisruptionType, str] = {
    "geopolitical":     "Geopolitical",
    "natural_disaster": "Natural-Disaster",
    "cyber":            "Cyber",
    "economic":         "Economic",
    "labour":           "Labour",
}

_TYPE_ARTICLE: dict[DisruptionType, str] = {
    "geopolitical":     "a geopolitical",
    "natural_disaster": "a natural-disaster",
    "cyber":            "a cyber",
    "economic":         "an economic",
    "labour":           "a labour",
}

_TYPE_PRIORITY_PROSE: dict[DisruptionType, str] = {
    t: "geopolitical, natural disaster, cyber, economic, labour"
    for t in _TYPE_ORDER
}

# ---------------------------------------------------------------------------
# Prose vocabulary tables
# ---------------------------------------------------------------------------

_ACTION_PROSE: dict[str, str] = {
    "activate_pre_qualified_alternate": "activate a pre-qualified alternate source",
    "intensify_threat_monitoring":      "intensify threat monitoring",
    "scenario_lever_generation":        "run scenario lever generation",
    "strategic_buffer_drawdown":        "draw down strategic buffers",
    "convene_crisis_response_center":   "convene the Crisis Response Centre",
    "reroute_logistics":                "reroute logistics",
    "expedited_emergent_design":        "commission expedited emergent design",
}

_DEFERRED_ACTION_PROSE: dict[str, str] = {
    "reshoring_nearshoring":   "reshoring or nearshoring",
    "footprint_redesign":      "footprint redesign",
    "new_in_sourcing":         "new in-sourcing",
    "ground_up_redesign":      "ground-up redesign",
    "contract_renegotiation":  "contract renegotiation",
}

_ESCALATION_PROSE: dict[str, str] = {
    "Responsible Analyst":         "**Responsible Analyst**",
    "Cross-functional Management": "**Cross-functional Management**",
    "Executive Leadership":        "**Executive Leadership**",
}

_DISPOSITION_PROSE: dict[str, str] = {
    "replace":  "**replace** the supplier",
    "monitor":  "**monitor** the supplier",
    "maintain": "**maintain** the existing relationship",
}

# Routing rule suffix → (title, descriptive body text).
# The body text is the scenario description; the consequence is appended separately.
_ROUTING_RULE_PROSE: dict[str, tuple[str, str]] = {
    "NULL": (
        "Disruption outside network boundary",
        "Where the disruption does not intersect any path in the firm's supply network, "
        "or where all Tier-1 suppliers on any affected path carry LOW risk scores with "
        "none showing high downstream criticality, no immediate action is required. "
        "The assessment is logged and the relationship is maintained at existing terms.",
    ),
    "DT": (
        "Deep-tier structural fallthrough — no Tier-1 supplier scored",
        "Where no scored Tier-1 supplier sits on a disrupted path, yet the disruption "
        "either shares a sub-tier source across paths or runs across two or more paths, "
        "the exposure is structural rather than direct. The shared or multi-path structure "
        "represents a hidden concentration even without a directly affected Tier-1, "
        "and a contingency must be staged. Do not default to maintain simply because "
        "no Tier-1 is directly scored: the structural risk is real.",
    ),
    "DT-NULL": (
        "Deep-tier with no structural concern",
        "Where no Tier-1 supplier is on a disrupted path and neither a shared sub-tier "
        "source nor two or more disrupted paths are present, the disruption reaches "
        "into lower tiers without the convergence or breadth that would justify staging "
        "a contingency.",
    ),
}

# Supplier rule suffix → human-readable title.
_SUPPLIER_RULE_TITLES: dict[str, str] = {
    "R3":   "Shared sub-tier convergence override",
    "R1":   "High-risk Tier-1 directly in the disrupted region",
    "R2":   "High-risk Tier-1 outside the disrupted region",
    "R5":   "Medium-risk in-region supplier with high downstream exposure breadth",
    "HUB":  "High-centrality hub supplier",
    "DEP":  "Medium-to-high-risk supplier with high dependency concentration",
    "SITE": "Medium-to-high-risk supplier at struck site with high dependency ratio",
    "MEDS": "Medium-risk supplier flagged by the disruption-type priority signal",
    "MED":  "Medium-risk supplier",
    "LOWS": "Low-risk supplier flagged by the disruption-type priority signal",
    "LOW":  "Low-risk routine supplier — catch-all",
}

# ---------------------------------------------------------------------------
# Prose helpers
# ---------------------------------------------------------------------------

def _article_for(dtype: DisruptionType) -> str:
    return _TYPE_ARTICLE[dtype]


def _rule_suffix(rule_id: str) -> str:
    """Extract the suffix after the type prefix, e.g. 'R3' from 'GEO-R3'."""
    for prefix in ("GEO-", "NAT-", "CYB-", "ECON-", "LAB-"):
        if rule_id.startswith(prefix):
            return rule_id[len(prefix):]
    return rule_id


def _condition_to_prose(field: str, value: Any) -> tuple[str, bool]:
    """Convert a single when_all condition to (English clause, is_scenario_level)."""
    is_scenario = field in _SCENARIO_LEVEL_FIELDS

    if field == "risk_level":
        if isinstance(value, list):
            bands = " or ".join(str(v).upper() for v in value)
            return f"the supplier's composite risk score falls in the {bands} band", False
        return f"the supplier's composite risk score falls in the {str(value).upper()} band", False

    if field == "tier1_in_region":
        if value is True:
            return "the supplier's primary site lies within the affected region", False
        return "the supplier's primary site lies outside the affected region", False

    if field == "shared_subtier_source" and value is True:
        return "two or more disrupted paths share a common lower-tier source", True

    if field == "count_disrupted_paths":
        if isinstance(value, str) and value.startswith(">="):
            n = value[2:].strip()
            return f"at least {n} supply paths are disrupted", True
        return f"the count of disrupted paths is {value}", True

    # Sub-scores are keyed by band (high / med / low).
    _SUB_SCORE_PHRASE = {
        "sub_node_centrality":        "the supplier's network centrality score",
        "sub_dependency_ratio":       "the firm's dependency ratio on this supplier",
        "sub_downstream_criticality": "the supplier's downstream criticality score",
        "sub_exposure_breadth":       "the supplier's exposure breadth score",
        "sub_exposure_depth":         "the supplier's exposure depth score",
    }
    if field in _SUB_SCORE_PHRASE and value in ("high", "med", "low"):
        band_word = {"high": "high", "med": "in the medium band", "low": "low"}[value]
        return f"{_SUB_SCORE_PHRASE[field]} is {band_word}", False

    # Fallback
    val_str = str(value).lower() if isinstance(value, bool) else str(value)
    return f"{field} is {val_str}", is_scenario


def _join_clauses(clauses: list[str]) -> str:
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _when_all_to_prose(when_all: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Convert when_all to (prose string, scenario_fields, supplier_fields).

    Returns the English condition text plus which fields are scenario-level vs supplier-level,
    so callers can add a level note when a rule mixes both.
    """
    if not when_all:
        return "no additional conditions apply (catch-all)", [], []

    clauses: list[str] = []
    scenario_fields: list[str] = []
    supplier_fields: list[str] = []

    for field, value in when_all.items():
        prose, is_scenario = _condition_to_prose(field, value)
        clauses.append(prose)
        if is_scenario:
            scenario_fields.append(field)
        else:
            supplier_fields.append(field)

    return _join_clauses(clauses), scenario_fields, supplier_fields


def _actions_to_prose(permitted: list[str]) -> str:
    parts = [_ACTION_PROSE.get(a, a) for a in permitted]
    return _join_clauses(parts) if parts else "(none)"


def _render_routing_rule_prose(rr, disruption_type: str) -> str:
    """Render a routing rule as a ### sub-section (one chunk per rule in the vector index)."""
    suffix = _rule_suffix(rr.rule_id)
    c = rr.consequence

    disp_prose = _DISPOSITION_PROSE.get(c.disposition, f"**{c.disposition}**")
    esc_prose  = _ESCALATION_PROSE.get(c.escalation,  f"**{c.escalation}**")

    title, body = _ROUTING_RULE_PROSE.get(suffix, (rr.rule_id, rr.when))

    note_sentence = ""
    if rr.note:
        note_text = rr.note.strip().replace("\n", " ").replace("  ", " ")
        note_sentence = f" *{note_text}*"

    return (
        f"### ({rr.rule_id}) {title}\n\n"
        f"{body} "
        f"The response is to {disp_prose}, escalating to {esc_prose}. "
        f"Assessment for this scenario ends here.{note_sentence}"
    )


def _render_supplier_rule_prose(sr) -> str:
    """Render a supplier rule as a ### sub-section (one chunk per rule in the vector index)."""
    suffix = _rule_suffix(sr.rule_id)

    # The "R1" suffix title mentions "directly in the disrupted region" — only accurate
    # when tier1_in_region: True is an actual condition of the rule (GEO-R1, NAT-R1, LAB-R1).
    # CYB-R1 and ECON-R1 only require risk_level=HIGH; tier1_in_region is not a condition.
    if suffix == "R1" and "tier1_in_region" not in sr.when_all:
        title = "High-risk Tier-1 supplier"
    else:
        title = _SUPPLIER_RULE_TITLES.get(suffix, sr.rule_id)

    prose, _scenario_fields, _supplier_fields = _when_all_to_prose(sr.when_all)

    c = sr.consequence
    disp_prose = _DISPOSITION_PROSE.get(c.disposition, f"**{c.disposition}**")
    esc_prose  = _ESCALATION_PROSE.get(c.escalation,  f"**{c.escalation}**")

    actions_sentence = ""
    if c.permitted:
        actions_sentence = f" Permitted live actions: {_actions_to_prose(c.permitted)}."

    rationale_sentence = ""
    note = sr.note
    grounding = sr.grounding
    if note:
        rationale_sentence = " *" + note.strip().replace("\n", " ").replace("  ", " ") + "*"
    elif grounding:
        rationale_sentence = " *" + grounding.strip().replace("\n", " ").replace("  ", " ") + "*"

    return (
        f"### ({sr.rule_id}) {title}\n\n"
        f"Where {prose}, the response is to {disp_prose}, escalating to {esc_prose}."
        f"{actions_sentence}"
        f"{rationale_sentence}"
    )


def _parse_risk_band_threshold(expr: str) -> float:
    m = re.match(r"^(>=|<=|>|<|==)\s*([0-9]+(?:\.[0-9]+)?)$", expr.strip())
    if not m:
        raise ValueError(f"Unparseable band expression: {expr!r}")
    return float(m.group(2))


def _render_risk_bands_prose(risk_bands: dict[str, str]) -> str:
    high_expr = risk_bands.get("HIGH", ">=0.55")
    low_expr  = risk_bands.get("LOW",  "<0.45")
    high_thresh = _parse_risk_band_threshold(high_expr)
    low_thresh  = _parse_risk_band_threshold(low_expr)
    medium_display = f"{low_thresh}–<{high_thresh}"
    return (
        f"a supplier is **HIGH** risk when its score is {high_expr}, "
        f"**MEDIUM** when {medium_display}, and **LOW** when {low_expr}"
    )


def _render_sub_score_bands_prose(sub_score_bands: dict[str, str]) -> str:
    high_expr = sub_score_bands.get("high", ">=0.66")
    low_expr  = sub_score_bands.get("low",  "<0.33")
    high_thresh = _parse_risk_band_threshold(high_expr)
    low_thresh  = _parse_risk_band_threshold(low_expr)
    med_display = f"{low_thresh}–<{high_thresh}"
    return f"**high** when {high_expr}, **med** when {med_display}, or **low** when {low_expr}"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_header(spec: Spec) -> str:
    dtype   = spec.disruption_type
    display = _TYPE_DISPLAY[dtype]
    doc_id  = _DOC_IDS[dtype]
    return (
        f"# {display} Disruption — Supply-Chain Disruption Response Standard\n\n"
        f"**Document ID:** {doc_id}  |  **Version:** {_CORPUS_VERSION}  |  **Status:** Controlled  |  "
        f"**Owner:** Supply Chain Resilience Office\n\n"
        f"Aligned with the firm's business-continuity management framework (ISO 22301). "
        f"Reviewed after each activation."
    )


def _build_scope(spec: Spec) -> str:
    dtype   = spec.disruption_type
    display = _TYPE_DISPLAY[dtype].lower()
    return (
        f"## Scope\n\n"
        f"This standard sets the immediate response to a **{display}** disruption for the firm's "
        f"top-ten Tier-1 suppliers. For each affected supplier it determines whether to replace, "
        f"monitor, or maintain the relationship, the level to which the situation is escalated, "
        f"and the response actions that may be taken while the disruption is live.\n\n"
        f"The standard covers the immediate response only. It does not assess commercial terms, "
        f"cost, or contracts (Procurement); capacity, lead time, inventory, or demand planning "
        f"(Supply Planning); supplier financial health (Supplier Risk); or sustainability and "
        f"emissions (Sustainability). Those considerations inform longer-term sourcing decisions "
        f"and are handled by the relevant functions, not in the immediate response set here."
    )


def _build_section_1(spec: Spec) -> str:
    risk_prose = _render_risk_bands_prose(spec.risk_bands)
    sub_prose  = _render_sub_score_bands_prose(spec.sub_score_bands)
    return (
        "## 1. Risk bands\n\n"
        f"Each affected Tier-1 supplier carries a composite risk score (`risk_score`, 0–1 scale) "
        f"reflecting how broadly and deeply the disruption reaches the supplier, how dependent "
        f"the firm is on it, and how central it is in the network. The score is banded as follows: "
        f"{risk_prose}.\n\n"
        f"The composite score is assembled from five structural sub-scores — "
        f"`sub_exposure_breadth` (breadth of downstream reach, tier-weighted), "
        f"`sub_dependency_ratio` (concentration of the firm's reliance on this supplier), "
        f"`sub_downstream_criticality` (operational importance of downstream operations fed by this node), "
        f"`sub_node_centrality` (graph centrality of the Tier-1 node; high values mean failure propagates widely), "
        f"and `sub_exposure_depth` (maximum disrupted tier, normalised) — each individually "
        f"banded as {sub_prose}."
    )


def _build_section_2(spec: Spec) -> str:
    dtype         = spec.disruption_type
    article       = _article_for(dtype)
    priority      = _PRIORITY_LABELS[dtype]
    priority_prose = _TYPE_PRIORITY_PROSE[dtype]

    routing_paras = []
    for rr in spec.routing_rules:
        routing_paras.append(_render_routing_rule_prose(rr, dtype))

    routing_block = "\n\n".join(routing_paras)

    return (
        f"## 2. Activation and routing\n\n"
        f"This standard applies when the active disruption is {article} event. "
        f"Where more than one type of disruption is active simultaneously, this type is addressed "
        f"at priority {priority} in the order {priority_prose}. "
        f"The assessment runs in two passes, each working through provisions in order; "
        f"the first provision whose conditions are fully satisfied takes effect and "
        f"ends further checking within that pass.\n\n"
        f"The routing provisions below rely on network topology flags supplied by the "
        f"Knowledge Graph query agent: `path_on_network` — whether the disruption intersects "
        f"any path in the firm's supply network; `tier1_on_disrupted_path` — whether at least "
        f"one scored Tier-1 supplier sits on a disrupted path; `shared_subtier_source` — whether "
        f"two or more disrupted paths converge on a single lower-tier (Tier ≥ 2) node, revealing "
        f"a hidden common point of failure; `count_disrupted_paths` — the number of distinct "
        f"affected paths; and `tier1_in_region` — whether a supplier's primary site lies in the "
        f"directly affected region.\n\n"
        f"**Pass 1 — Scenario-level routing.** The provisions below apply once to the scenario "
        f"as a whole, before any per-supplier assessment begins. If one fires, per-supplier "
        f"evaluation does not occur. If none fires, proceed to Section 3.\n\n"
        f"{routing_block}\n\n"
        f"**Pass 2 — Per-supplier evaluation.** If no routing provision fired, each Tier-1 "
        f"supplier on a disrupted path is assessed individually against the provisions in "
        f"Section 3."
    )


def _build_section_3(spec: Spec) -> str:
    rule_paras = []
    for sr in spec.sorted_supplier_rules:
        rule_paras.append(_render_supplier_rule_prose(sr))

    rules_block = "\n\n".join(rule_paras)

    return (
        f"## 3. Decision logic\n\n"
        f"Each Tier-1 supplier on a disrupted path is assessed against its composite risk band "
        f"(Section 1) together with the structural sub-scores this disruption type keys on. "
        f"The provisions below apply per supplier in the order listed; the first whose conditions "
        f"are fully satisfied determines that supplier's response. Some provisions mix "
        f"scenario-level facts (the same for every supplier) with supplier-level facts (assessed "
        f"individually); where this occurs it is noted explicitly.\n\n"
        f"{rules_block}"
    )


def _build_section_4(spec: Spec) -> str:
    raw_levers = getattr(spec, "levers", {})
    if isinstance(raw_levers, dict):
        permitted  = raw_levers.get("permitted_response", [])
        prohibited = raw_levers.get("prohibited_at_response", [])
    else:
        permitted  = []
        prohibited = []

    permitted_parts  = [_ACTION_PROSE.get(a, a) for a in permitted]
    prohibited_parts = [_DEFERRED_ACTION_PROSE.get(a, a) for a in prohibited]

    permitted_list  = "\n".join(f"- {p}" for p in permitted_parts)  if permitted_parts  else "- (none)"
    prohibited_list = "\n".join(f"- {p}" for p in prohibited_parts) if prohibited_parts else "- (none)"

    return (
        "## 4. Permitted and prohibited actions\n\n"
        "A response action may be taken during the live event only if it can realistically "
        "be put in place within the recovery window — that is, it relies on arrangements "
        "already in place before the disruption (pre-qualified alternates, standing buffer "
        "stock, contracted logistics flexibility). Structural measures that require building "
        "new capacity from scratch take longer than the recovery window allows and are "
        "deferred to the resilience roadmap.\n\n"
        "Actions permitted during the live response:\n\n"
        f"{permitted_list}\n\n"
        "Actions deferred to after-action (not initiated as part of the live response):\n\n"
        f"{prohibited_list}"
    )


def _build_section_5(spec: Spec) -> str:
    raw_ladder = getattr(spec, "escalation_ladder", {})
    levels = raw_ladder.get("levels", []) if isinstance(raw_ladder, dict) else []

    if levels:
        level_strs = [
            f"**{lv.get('name', '?')}** (authority: {lv.get('authority', '?')})"
            for lv in levels
        ]
        levels_prose = "; ".join(level_strs)
    else:
        levels_prose = (
            "**Responsible Analyst**; "
            "**Cross-functional Management**; "
            "**Executive Leadership**"
        )

    return (
        f"## 5. Escalation\n\n"
        f"Three escalation levels apply: {levels_prose}. "
        f"The level is set by the matching provision in Section 3. "
        f"Higher levels are reserved for directly exposed high-risk suppliers, structurally "
        f"critical suppliers, and disruptions that reach a shared source or spread across "
        f"several paths simultaneously — situations where the disruption is likely to outlast "
        f"the network's absorption capacity."
    )


def _build_section_6(spec: Spec) -> str:
    priority_cell = getattr(spec, "strategic_priority_cell", "")
    priority_note = getattr(spec, "strategic_priority_note", None)
    note_prose    = priority_note.strip() if priority_note else f"Priority: {priority_cell}."

    dtype_display = spec.disruption_type.replace("_", " ")
    return (
        f"## 6. Strategic priority\n\n"
        f"The standing posture for {dtype_display} disruptions is **{priority_cell}**. "
        f"{note_prose}"
    )


def _build_section_7(spec: Spec) -> str:
    raw_after  = getattr(spec, "after_action", {})
    if isinstance(raw_after, dict):
        record     = raw_after.get("record", ["scenario", "dispositions_taken", "outcome"])
        record_str = ", ".join(record) if isinstance(record, list) else str(record)
    else:
        record_str = "scenario, dispositions_taken, outcome"

    return (
        f"## 7. After-action\n\n"
        f"After the response, record the {record_str}. "
        f"Any structural measures identified as necessary during the response — those that "
        f"could not be executed within the recovery window — are passed to the resilience "
        f"roadmap to be put in place before the next event. "
        f"The record informs the next scheduled review of this standard."
    )


def _build_section_8(spec: Spec) -> str:
    exemplars = spec.model_extra.get("exemplars", [])
    if not exemplars:
        return "## 8. Worked example\n\n*(No exemplars defined in spec.)*"

    lines = ["## 8. Worked example", ""]
    for ex in exemplars:
        ex_id      = ex.get("exemplar_id", "?")
        rule_ref   = ex.get("rule_ref", "?")
        scenario   = ex.get("scenario", {})
        reasoning  = ex.get("reasoning", None)
        key        = ex.get("key", {})

        disposition = key.get("disposition", "?")
        escalation  = key.get("escalation", "?")

        scenario_parts = [
            f"{k}: {v}" for k, v in scenario.items() if k != "disruption_type"
        ]
        scenario_prose = "; ".join(scenario_parts) if scenario_parts else "(see spec)"

        reasoning_prose = ""
        if reasoning:
            reasoning_prose = " " + reasoning.strip().replace("\n", " ").replace("  ", " ")

        lines.append(
            f"**{ex_id}** (illustrates provision {rule_ref}). "
            f"Scenario: {scenario_prose}.{reasoning_prose} "
            f"The correct response is to **{disposition}**, escalating to **{escalation}**."
        )
        lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Full playbook and embed cut assembly
# ---------------------------------------------------------------------------

def generate_full_playbook(spec: Spec) -> str:
    """Generate the full playbook text (all 8 sections plus glossary) for a spec."""
    parts = [
        _build_header(spec),
        "",
        _build_scope(spec),
        "",
        _build_section_1(spec),
        "",
        _build_section_2(spec),
        "",
        _build_section_3(spec),
        "",
        _build_section_4(spec),
        "",
        _build_section_5(spec),
        "",
        _build_section_6(spec),
        "",
        _build_section_7(spec),
        "",
        _build_section_8(spec),
    ]
    return "\n".join(parts)


def strip_worked_examples(full_text: str) -> str:
    """Remove the ## 8. Worked example section and everything after it."""
    marker = "## 8. Worked example"
    idx = full_text.find(marker)
    if idx == -1:
        return full_text
    return full_text[:idx].rstrip()


def generate_embed_cut(spec: Spec) -> str:
    """Generate the embed cut (sections 1-7 plus glossary, no worked examples)."""
    full = generate_full_playbook(spec)
    return strip_worked_examples(full)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_all_embed_cuts_text() -> str:
    """Return concatenated embed-cut text for all 5 disruption types."""
    specs   = load_all_specs()
    divider = "\n\n" + "─" * 70 + "\n\n"
    parts   = [generate_embed_cut(specs[dtype]) for dtype in _TYPE_ORDER]
    return divider.join(parts)


def write_all_playbooks() -> None:
    """Generate and write all 5 full playbooks and 5 embed cuts to disk."""
    _FULL_DIR.mkdir(parents=True, exist_ok=True)
    _EMBED_DIR.mkdir(parents=True, exist_ok=True)

    specs = load_all_specs()
    for dtype in _TYPE_ORDER:
        spec = specs[dtype]

        full_text = generate_full_playbook(spec)
        (_FULL_DIR / f"{dtype}.md").write_text(full_text, encoding="utf-8")
        print(f"Wrote full playbook: {_FULL_DIR / f'{dtype}.md'}")

        embed_text = strip_worked_examples(full_text)
        (_EMBED_DIR / f"{dtype}.md").write_text(embed_text, encoding="utf-8")
        print(f"Wrote embed cut:     {_EMBED_DIR / f'{dtype}.md'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    write_all_playbooks()
    print("\nDone. All playbooks and embed cuts generated.")
