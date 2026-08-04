"""Pipeline data contracts (pydantic v2).

Tier1Supplier holds raw pipeline outputs only — risk_level is a DERIVED field
computed by csco.derive and is NOT present here.
"""

from typing import Literal

from pydantic import BaseModel, Field

from csco.specs.schema import DisruptionType


class Tier1Supplier(BaseModel):
    """Raw Tier-1 supplier record as emitted by the Risk-Manager pipeline."""

    supplier_id: str
    name: str
    product_supplied: str
    tier1_in_region: bool

    # Composite risk score
    risk_score: float = Field(ge=0.0, le=1.0)

    # Sub-dimension scores (each 0–1)
    sub_exposure_breadth: float = Field(ge=0.0, le=1.0)
    sub_dependency_ratio: float = Field(ge=0.0, le=1.0)
    sub_downstream_criticality: float = Field(ge=0.0, le=1.0)
    sub_node_centrality: float = Field(ge=0.0, le=1.0)
    sub_exposure_depth: float = Field(ge=0.0, le=1.0)


class Scenario(BaseModel):
    """One disruption event as emitted by the pipeline."""

    disruption_type: DisruptionType
    affected_regions: list[str]
    impacted_industries: list[str]
    named_companies: list[str]

    # Network topology flags
    path_on_network: bool
    count_disrupted_paths: int = Field(ge=0)
    max_disrupted_tier: int = Field(ge=0)
    shared_subtier_source: bool
    tier1_on_disrupted_path: bool

    suppliers: list[Tier1Supplier]


class DerivedSupplier(BaseModel):
    """Tier1Supplier enriched with derived fields, ready for evaluation."""

    supplier: Tier1Supplier
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]


class SupplierRecommendation(BaseModel):
    """Agent output for a single supplier."""

    supplier_id: str
    disposition: Literal["replace", "monitor", "maintain"]
    escalation: Literal["Responsible Analyst", "Cross-functional Management", "Executive Leadership"]
    permitted_actions: list[str]
    fired_rule_id: str | None
    rationale: str

    # Direct band derivation (newly added for validation)
    assigned_risk_level: Literal["LOW", "MEDIUM", "HIGH"] | None = None


class RoutingResult(BaseModel):
    """Scenario-level routing outcome when a routing rule fires."""

    fired_rule_id: str
    scenario_action: str
    disposition: Literal["replace", "monitor", "maintain"]
    escalation: Literal["Responsible Analyst", "Cross-functional Management", "Executive Leadership"]
    permitted_actions: list[str]


class ScenarioDecision(BaseModel):
    """Top-level agent output: routing result + per-supplier recommendations."""

    routing_result: RoutingResult | None = None
    recommendations: list[SupplierRecommendation] = Field(default_factory=list)
