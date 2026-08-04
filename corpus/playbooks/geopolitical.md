# Geopolitical Disruption — Supply-Chain Disruption Response Standard

**Document ID:** SCR-STD-GEO-001  |  **Version:** 2.7  |  **Status:** Controlled  |  **Owner:** Supply Chain Resilience Office

Aligned with the firm's business-continuity management framework (ISO 22301). Reviewed after each activation.

## Scope

This standard sets the immediate response to a **geopolitical** disruption for the firm's top-ten Tier-1 suppliers. For each affected supplier it determines whether to replace, monitor, or maintain the relationship, the level to which the situation is escalated, and the response actions that may be taken while the disruption is live.

The standard covers the immediate response only. It does not assess commercial terms, cost, or contracts (Procurement); capacity, lead time, inventory, or demand planning (Supply Planning); supplier financial health (Supplier Risk); or sustainability and emissions (Sustainability). Those considerations inform longer-term sourcing decisions and are handled by the relevant functions, not in the immediate response set here.

## 1. Risk bands

Each affected Tier-1 supplier carries a composite risk score (`risk_score`, 0–1 scale) reflecting how broadly and deeply the disruption reaches the supplier, how dependent the firm is on it, and how central it is in the network. The score is banded as follows: a supplier is **HIGH** risk when its score is >=0.55, **MEDIUM** when 0.45–<0.55, and **LOW** when <0.45.

The composite score is assembled from five structural sub-scores — `sub_exposure_breadth` (breadth of downstream reach, tier-weighted), `sub_dependency_ratio` (concentration of the firm's reliance on this supplier), `sub_downstream_criticality` (operational importance of downstream operations fed by this node), `sub_node_centrality` (graph centrality of the Tier-1 node; high values mean failure propagates widely), and `sub_exposure_depth` (maximum disrupted tier, normalised) — each individually banded as **high** when >=0.66, **med** when 0.33–<0.66, or **low** when <0.33.

## 2. Activation and routing

This standard applies when the active disruption is a geopolitical event. Where more than one type of disruption is active simultaneously, this type is addressed at priority 1 in the order geopolitical, natural disaster, cyber, economic, labour. The assessment runs in two passes, each working through provisions in order; the first provision whose conditions are fully satisfied takes effect and ends further checking within that pass.

The routing provisions below rely on network topology flags supplied by the Knowledge Graph query agent: `path_on_network` — whether the disruption intersects any path in the firm's supply network; `tier1_on_disrupted_path` — whether at least one scored Tier-1 supplier sits on a disrupted path; `shared_subtier_source` — whether two or more disrupted paths converge on a single lower-tier (Tier ≥ 2) node, revealing a hidden common point of failure; `count_disrupted_paths` — the number of distinct affected paths; and `tier1_in_region` — whether a supplier's primary site lies in the directly affected region.

**Pass 1 — Scenario-level routing.** The provisions below apply once to the scenario as a whole, before any per-supplier assessment begins. If one fires, per-supplier evaluation does not occur. If none fires, proceed to Section 3.

### (GEO-NULL) Disruption outside network boundary

Where the disruption does not intersect any path in the firm's supply network, or where all Tier-1 suppliers on any affected path carry LOW risk scores with none showing high downstream criticality, no immediate action is required. The assessment is logged and the relationship is maintained at existing terms. The response is to **maintain** the existing relationship, escalating to **Responsible Analyst**. Assessment for this scenario ends here.

### (GEO-DT) Deep-tier structural fallthrough — no Tier-1 supplier scored

Where no scored Tier-1 supplier sits on a disrupted path, yet the disruption either shares a sub-tier source across paths or runs across two or more paths, the exposure is structural rather than direct. The shared or multi-path structure represents a hidden concentration even without a directly affected Tier-1, and a contingency must be staged. Do not default to maintain simply because no Tier-1 is directly scored: the structural risk is real. The response is to **monitor** the supplier, escalating to **Cross-functional Management**. Assessment for this scenario ends here. *No Tier-1 scored - do NOT default to maintain. Branch on structure + material.*

### (GEO-DT-NULL) Deep-tier with no structural concern

Where no Tier-1 supplier is on a disrupted path and neither a shared sub-tier source nor two or more disrupted paths are present, the disruption reaches into lower tiers without the convergence or breadth that would justify staging a contingency. The response is to **maintain** the existing relationship, escalating to **Responsible Analyst**. Assessment for this scenario ends here.

**Pass 2 — Per-supplier evaluation.** If no routing provision fired, each Tier-1 supplier on a disrupted path is assessed individually against the provisions in Section 3.

## 3. Decision logic

Each Tier-1 supplier on a disrupted path is assessed against its composite risk band (Section 1) together with the structural sub-scores this disruption type keys on. The provisions below apply per supplier in the order listed; the first whose conditions are fully satisfied determines that supplier's response. Some provisions mix scenario-level facts (the same for every supplier) with supplier-level facts (assessed individually); where this occurs it is noted explicitly.

### (GEO-R3) Shared sub-tier convergence override

Where two or more disrupted paths share a common lower-tier source, at least 2 supply paths are disrupted, and the supplier's composite risk score falls in the MEDIUM or HIGH band, the response is to **replace** the supplier, escalating to **Executive Leadership**. Permitted live actions: activate a pre-qualified alternate source, intensify threat monitoring, and run scenario lever generation. *Per-supplier reading says 'monitor several MEDIUMs'. The shared sub-tier convergence means the apparent redundancy is illusory - one point of failure.*

### (GEO-R1) High-risk Tier-1 directly in the disrupted region

Where the supplier's composite risk score falls in the HIGH band and the supplier's primary site lies within the affected region, the response is to **replace** the supplier, escalating to **Executive Leadership**. Permitted live actions: activate a pre-qualified alternate source and intensify threat monitoring.

### (GEO-R2) High-risk Tier-1 outside the disrupted region

Where the supplier's composite risk score falls in the HIGH band and the supplier's primary site lies outside the affected region, the response is to **monitor** the supplier, escalating to **Cross-functional Management**. Permitted live actions: intensify threat monitoring and run scenario lever generation.

### (GEO-MEDS) Medium-risk supplier flagged by the disruption-type priority signal

Where the supplier's composite risk score falls in the MEDIUM band and the supplier's primary site lies within the affected region, the response is to **monitor** the supplier, escalating to **Cross-functional Management**. Permitted live actions: activate a pre-qualified alternate source and intensify threat monitoring. *High ripple potential -> proactive contingency even at MEDIUM (Tomlin 2006).*

### (GEO-MED) Medium-risk supplier

Where the supplier's composite risk score falls in the MEDIUM band, the response is to **monitor** the supplier, escalating to **Responsible Analyst**. Permitted live actions: intensify threat monitoring.

### (GEO-LOWS) Low-risk supplier flagged by the disruption-type priority signal

Where the supplier's composite risk score falls in the LOW band and the supplier's primary site lies within the affected region, the response is to **monitor** the supplier, escalating to **Responsible Analyst**. Permitted live actions: intensify threat monitoring. *Watch structurally vital nodes even at LOW own-risk (Ivanov & Dolgui 2020).*

### (GEO-LOW) Low-risk routine supplier — catch-all

Where the supplier's composite risk score falls in the LOW band, the response is to **maintain** the existing relationship, escalating to **Responsible Analyst**. Permitted live actions: intensify threat monitoring.

## 4. Permitted and prohibited actions

A response action may be taken during the live event only if it can realistically be put in place within the recovery window — that is, it relies on arrangements already in place before the disruption (pre-qualified alternates, standing buffer stock, contracted logistics flexibility). Structural measures that require building new capacity from scratch take longer than the recovery window allows and are deferred to the resilience roadmap.

Actions permitted during the live response:

- activate a pre-qualified alternate source
- intensify threat monitoring
- run scenario lever generation
- draw down strategic buffers
- convene the Crisis Response Centre
- reroute logistics
- commission expedited emergent design

Actions deferred to after-action (not initiated as part of the live response):

- reshoring or nearshoring
- footprint redesign
- new in-sourcing
- ground-up redesign
- contract renegotiation

## 5. Escalation

Three escalation levels apply: **Responsible Analyst** (authority: responsible_analyst); **Cross-functional Management** (authority: cross_functional_management); **Executive Leadership** (authority: executive_leadership). The level is set by the matching provision in Section 3. Higher levels are reserved for directly exposed high-risk suppliers, structurally critical suppliers, and disruptions that reach a shared source or spread across several paths simultaneously — situations where the disruption is likely to outlast the network's absorption capacity.

## 6. Strategic priority

The standing posture for geopolitical disruptions is **Continuity Protection**. Geopolitical disruptions persist -> favour redundancy / dual-sourcing over short-term efficiency.

## 7. After-action

After the response, record the scenario, dispositions_taken, outcome. Any structural measures identified as necessary during the response — those that could not be executed within the recovery window — are passed to the resilience roadmap to be put in place before the next event. The record informs the next scheduled review of this standard.

## 8. Worked example

**EX-GEO-03** (illustrates provision GEO-R3). Scenario: tier1_suppliers: three suppliers of a platinum-group metal, each MEDIUM (0.47-0.52); tier1_in_region: False; shared_subtier_source: True; count_disrupted_paths: 3. A per-supplier read suggests monitoring three moderate suppliers. The shared sub-tier check shows all three paths converge on a single refiner inside the disrupted region - the redundancy is illusory; one concentrated point of failure. The correct response is to **replace**, escalating to **Executive Leadership**.

**EX-GEO-DT** (illustrates provision GEO-DT). Scenario: tier1_on_disrupted_path: False; max_disrupted_tier: 3; count_disrupted_paths: 2. The correct response is to **monitor**, escalating to **Cross-functional Management**.