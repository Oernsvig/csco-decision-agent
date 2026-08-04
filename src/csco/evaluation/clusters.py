"""Evaluation cluster taxonomy and contrast-pair registry.

Clusters are diagnostic labels that tag rules (and the scenarios targeting them)
with the cognitive demand they place on the policy-grounding architecture.
One rule or scenario can belong to several clusters.

Usage
-----
    from csco.evaluation.clusters import TAXONOMY, CONTRAST_PAIRS, ALL_CLUSTER_IDS
    desc = TAXONOMY["C3_PRIORITY_OVERRIDE"]["description"]
    pair = CONTRAST_PAIRS["NAT_R5_vs_NAT_MED"]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, dict[str, str]] = {
    "C1_PASS_STRUCTURE": {
        "description": (
            "Applies scenario-level routing before supplier-level assessment. "
            "Correctly identifies when a routing rule fires and stops supplier evaluation."
        ),
    },
    "C2_DEEP_TIER": {
        "description": (
            "Distinguishes structural lower-tier exposure (shared source / multi-path) "
            "from harmless deep-tier exposure with no structural concern."
        ),
    },
    "C3_PRIORITY_OVERRIDE": {
        "description": (
            "Applies higher-priority matching provisions before lower-priority generic rules. "
            "Requires the model to walk rules in stated priority order and stop at the first match."
        ),
    },
    "C4_SCENARIO_SUPPLIER_JOIN": {
        "description": (
            "Combines scenario-level flags (shared_subtier_source, count_disrupted_paths, "
            "tier1_in_region) with supplier-level facts (risk_level, sub-scores) to evaluate "
            "a single provision."
        ),
    },
    "C5_TYPE_SPECIFICITY": {
        "description": (
            "Uses disruption-specific logic rather than generic risk response. "
            "Tests whether the correct type's playbook is applied and type-specific "
            "discriminators (NAT-R5, CYB-HUB, ECON-DEP, LAB-SITE) override generic rules."
        ),
    },
    "C6_THRESHOLD_BANDING": {
        "description": (
            "Correctly derives LOW / MEDIUM / HIGH risk band from the composite risk score, "
            "especially at threshold boundaries. Tests whether the model reads the spec's "
            "band expressions rather than applying generic intuition."
        ),
    },
    "C8_FALLBACK_DISCIPLINE": {
        "description": (
            "Applies generic MED / LOW catch-all rules without over-escalation or "
            "over-replacement when no special condition is present."
        ),
    },
    "C9_ACTION_ESCALATION_EXACTNESS": {
        "description": (
            "Copies exact disposition, escalation level, and permitted live actions as "
            "written in the policy — does not paraphrase, substitute, or add deferred actions."
        ),
    },
    "C10_STRATEGIC_POSTURE": {
        "description": (
            "Uses the correct disruption-specific strategic priority (Continuity Protection / "
            "Rapid Reconfiguration / Cost Containment) in the response rationale, as stated in "
            "Section 6 of the playbook."
        ),
    },
    "C11_MULTI_SUPPLIER_CONSISTENCY": {
        "description": (
            "Applies the policy independently and consistently to multiple suppliers in one "
            "scenario — does not let one supplier's outcome contaminate another's."
        ),
    },
    "C12_CROSS_POLICY_CONTAMINATION": {
        "description": (
            "Avoids importing rule logic, permitted actions, escalation levels, or strategic "
            "priorities from another disruption type's playbook. Critical for rules whose "
            "correct response differs from what a neighbouring type would prescribe."
        ),
    },
}

ALL_CLUSTER_IDS: frozenset[str] = frozenset(TAXONOMY.keys())

# ---------------------------------------------------------------------------
# Contrast-pair registry
# ---------------------------------------------------------------------------
# Each entry describes a minimal pair of scenarios that should differ by
# exactly one discriminating factor.  The generator uses this to produce
# positive (focal_rule fires) and negative (contrast_rule fires) variants.

CONTRAST_PAIRS: dict[str, dict] = {

    # --- shared sub-tier convergence vs generic medium -------------------
    "GEO_R3_vs_GEO_MED": {
        "focal_rule":    "GEO-R3",
        "contrast_rule": "GEO-MED",
        "vary_only":     ["shared_subtier_source", "count_disrupted_paths"],
        "purpose": (
            "Tests whether shared sub-tier convergence override is applied instead of "
            "generic medium-risk monitoring."
        ),
    },
    "NAT_R3_vs_NAT_MED": {
        "focal_rule":    "NAT-R3",
        "contrast_rule": "NAT-MED",
        "vary_only":     ["shared_subtier_source", "count_disrupted_paths"],
        "purpose":       "Natural-disaster shared sub-tier override vs generic medium.",
    },
    "CYB_R3_vs_CYB_MED": {
        "focal_rule":    "CYB-R3",
        "contrast_rule": "CYB-MED",
        "vary_only":     ["shared_subtier_source", "count_disrupted_paths"],
        "purpose":       "Cyber shared sub-tier override vs generic medium.",
    },
    "ECON_R3_vs_ECON_MED": {
        "focal_rule":    "ECON-R3",
        "contrast_rule": "ECON-MED",
        "vary_only":     ["shared_subtier_source", "count_disrupted_paths"],
        "purpose":       "Economic shared sub-tier override vs generic medium.",
    },
    "LAB_R3_vs_LAB_MED": {
        "focal_rule":    "LAB-R3",
        "contrast_rule": "LAB-MED",
        "vary_only":     ["shared_subtier_source", "count_disrupted_paths"],
        "purpose":       "Labour shared sub-tier override vs generic medium.",
    },

    # --- type-specific discriminators vs generic medium ------------------
    "NAT_R5_vs_NAT_MED": {
        "focal_rule":    "NAT-R5",
        "contrast_rule": "NAT-MED",
        "vary_only":     ["tier1_in_region", "sub_exposure_breadth"],
        "purpose": (
            "Tests whether the natural-disaster breadth override is applied instead of "
            "generic medium-risk monitoring."
        ),
    },
    "CYB_HUB_vs_CYB_MED": {
        "focal_rule":    "CYB-HUB",
        "contrast_rule": "CYB-MED",
        "vary_only":     ["sub_node_centrality"],
        "purpose":       "Tests whether cyber centrality discriminator (CYB-HUB) is applied.",
    },
    "ECON_DEP_vs_ECON_MED": {
        "focal_rule":    "ECON-DEP",
        "contrast_rule": "ECON-MED",
        "vary_only":     ["sub_dependency_ratio", "count_disrupted_paths"],
        "purpose": (
            "Tests whether broad economic concentration override is applied instead of "
            "generic medium monitoring."
        ),
    },
    "LAB_SITE_vs_LAB_MED": {
        "focal_rule":    "LAB-SITE",
        "contrast_rule": "LAB-MED",
        "vary_only":     ["tier1_in_region", "sub_dependency_ratio"],
        "purpose":       "Tests whether struck-site dependency override (LAB-SITE) is applied.",
    },

    # --- direct vs indirect high-risk -----------------------------------
    "GEO_R1_vs_GEO_R2": {
        "focal_rule":    "GEO-R1",
        "contrast_rule": "GEO-R2",
        "vary_only":     ["tier1_in_region"],
        "purpose": (
            "In-region location discriminates replace (GEO-R1) from monitor (GEO-R2) "
            "for HIGH-risk geopolitical suppliers."
        ),
    },
    "NAT_R1_vs_NAT_R2": {
        "focal_rule":    "NAT-R1",
        "contrast_rule": "NAT-R2",
        "vary_only":     ["tier1_in_region"],
        "purpose":       "Natural-disaster in-region vs out-of-region high-risk.",
    },
    "LAB_R1_vs_LAB_R2": {
        "focal_rule":    "LAB-R1",
        "contrast_rule": "LAB-R2",
        "vary_only":     ["tier1_in_region"],
        "purpose":       "Labour in-region vs out-of-region high-risk.",
    },

    # --- cross-policy contamination: cyber/economic HIGH → monitor ------
    "CYB_R1_vs_GEO_R1_analogue": {
        "focal_rule":    "CYB-R1",
        "contrast_rule": "GEO-R1",
        "vary_only":     ["disruption_type"],
        "purpose": (
            "Cyber HIGH-risk → monitor (not replace). Tests whether the model avoids "
            "importing GEO/NAT/LAB replace-on-HIGH behaviour into the cyber playbook."
        ),
    },
    "ECON_R1_vs_GEO_R1_analogue": {
        "focal_rule":    "ECON-R1",
        "contrast_rule": "GEO-R1",
        "vary_only":     ["disruption_type"],
        "purpose": (
            "Economic HIGH-risk → monitor (not replace). Tests resistance to cross-policy "
            "contamination from geopolitical policy."
        ),
    },

    # --- type-specific structural uplift (keys on the Table 1 signal) ----
    "GEO_MEDS_vs_GEO_MED": {
        "focal_rule":    "GEO-MEDS",
        "contrast_rule": "GEO-MED",
        "vary_only":     ["tier1_in_region"],
        "purpose": (
            "In-region location (struck) triggers proactive contingency (Cross-functional "
            "Management) instead of generic monitoring (Responsible Analyst) for MEDIUM-risk suppliers."
        ),
    },
    "GEO_LOWS_vs_GEO_LOW": {
        "focal_rule":    "GEO-LOWS",
        "contrast_rule": "GEO-LOW",
        "vary_only":     ["tier1_in_region"],
        "purpose": (
            "In-region location (struck) triggers monitoring (Responsible Analyst) instead of "
            "maintain for LOW-risk suppliers."
        ),
    },
    "CYB_MEDS_vs_CYB_MED": {
        "focal_rule":    "CYB-MEDS",
        "contrast_rule": "CYB-MED",
        "vary_only":     ["sub_node_centrality"],
        "purpose":       "Cyber medium node-centrality uplift (below the hub threshold).",
    },
    "ECON_MEDS_vs_ECON_MED": {
        "focal_rule":    "ECON-MEDS",
        "contrast_rule": "ECON-MED",
        "vary_only":     ["sub_dependency_ratio"],
        "purpose":       "Economic high dependency-ratio uplift for MEDIUM-risk suppliers.",
    },

    # --- deep-tier structural vs harmless --------------------------------
    "DT_vs_DT_NULL": {
        "focal_rule":    "*-DT",
        "contrast_rule": "*-DT-NULL",
        "vary_only":     ["shared_subtier_source", "count_disrupted_paths"],
        "purpose": (
            "Tests whether structural lower-tier concern (shared source / multiple paths) "
            "is distinguished from harmless deep-tier exposure with no structural concern."
        ),
    },
}
