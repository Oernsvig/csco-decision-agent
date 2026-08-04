"""Pydantic v2 models for a parsed *.spec.yaml file.

The spec format differs from a flat-condition design:
- risk_bands keys are uppercase: HIGH / MEDIUM / LOW
- sub_score_bands keys: high / med / low
- routing_rules use a free-form ``when:`` string (for the LLM)
- supplier_rules use ``when_all`` as a flat dict with mixed value types
- consequences use ``permitted`` (list) not ``permitted_actions``
- supplier_rules carry a ``priority`` int for ordering

Version 2.3 additions (evaluation metadata — all optional):
- Each rule carries an optional ``evaluation`` block with clusters,
  failure_modes, reference_expectations, and generation hints.
- The Spec carries an optional ``evaluation_design`` block with
  threshold probe values and contrast pairs.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

DisruptionType = Literal["geopolitical", "natural_disaster", "economic", "cyber", "labour"]


# ---------------------------------------------------------------------------
# Evaluation metadata models (optional; do not alter business logic)
# ---------------------------------------------------------------------------

class ReferenceExpectations(BaseModel):
    """Specifies which reference edges the lexical-graph arm must follow for a provision."""

    model_config = ConfigDict(extra="allow")

    band: str = "required"          # required | optional | not_applicable
    escalation: str = "required"
    action: str = "optional"
    tier: str = "optional"          # optional | not_applicable
    strategic_priority: str = "optional"


class GenerationHints(BaseModel):
    """Hints for the constrained scenario generator."""

    model_config = ConfigDict(extra="allow")

    target_mode: str = "supplier"       # "routing" | "supplier"
    create_lower_priority_overlap: bool = False
    include_threshold_edges: bool = False
    recommended_risk_scores: list[float] = []


class EvalMetadata(BaseModel):
    """Per-rule evaluation metadata — does not affect business logic."""

    model_config = ConfigDict(extra="allow")

    clusters: list[str] = []
    failure_modes: list[str] = []
    reference_expectations: ReferenceExpectations = ReferenceExpectations()
    contrast_with: list[str] = []
    generation_hints: GenerationHints = GenerationHints()


class ThresholdProbeValues(BaseModel):
    """Near-boundary risk-score values for threshold-banding tests."""

    model_config = ConfigDict(extra="allow")

    LOW_EDGE: float
    MEDIUM_LOWER_EDGE: float
    MEDIUM_UPPER_EDGE: float
    HIGH_EDGE: float


class ContrastPair(BaseModel):
    """Minimal-pair specification: focal rule vs contrast rule, one discriminator."""

    model_config = ConfigDict(extra="allow")

    pair_id: str
    focal_rule: str
    contrast_rule: str
    vary_only: list[str]
    purpose: str


class EvaluationDesign(BaseModel):
    """Spec-level benchmark configuration."""

    model_config = ConfigDict(extra="allow")

    threshold_probe_values: ThresholdProbeValues | None = None
    contrast_pairs: list[ContrastPair] = []


# ---------------------------------------------------------------------------
# Core spec models
# ---------------------------------------------------------------------------

class RoutingConsequence(BaseModel):
    model_config = ConfigDict(extra="allow")

    route: str
    disposition: Literal["replace", "monitor", "maintain"]
    escalation: Literal["Responsible Analyst", "Cross-functional Management", "Executive Leadership"]


class RoutingRule(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: str
    when: str
    consequence: RoutingConsequence
    note: str | None = None
    evaluation: EvalMetadata | None = None


class SupplierConsequence(BaseModel):
    model_config = ConfigDict(extra="allow")

    disposition: Literal["replace", "monitor", "maintain"]
    escalation: Literal["Responsible Analyst", "Cross-functional Management", "Executive Leadership"]
    permitted: list[str]


class SupplierRule(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: str
    priority: int
    # when_all is a flat dict; value types: bool, str (comparison or band/level name), list[str]
    when_all: dict[str, Any]
    consequence: SupplierConsequence
    note: str | None = None
    grounding: str | None = None
    evaluation: EvalMetadata | None = None


class Spec(BaseModel):
    """Fully parsed decision spec for one disruption type.

    ``extra='allow'`` absorbs the rich metadata sections (meta, references,
    grounding, provenance, exemplars …) without needing typed models for each.
    """

    model_config = ConfigDict(extra="allow")

    disruption_type: DisruptionType

    # Band thresholds -------------------------------------------------------
    # risk_bands keys: HIGH / MEDIUM / LOW  (uppercase)
    # sub_score_bands keys: high / med / low (lowercase)
    risk_bands: dict[str, str]
    sub_score_bands: dict[str, str]

    routing_rules: list[RoutingRule]
    supplier_rules: list[SupplierRule]

    # Evaluation design (optional — benchmark metadata only) ----------------
    evaluation_design: EvaluationDesign | None = None

    @field_validator("risk_bands")
    @classmethod
    def _risk_bands_keys(cls, v: dict[str, str]) -> dict[str, str]:
        for key in ("HIGH", "LOW"):
            if key not in v:
                raise ValueError(f"risk_bands missing required key '{key}'")
        return v

    @field_validator("sub_score_bands")
    @classmethod
    def _sub_score_bands_keys(cls, v: dict[str, str]) -> dict[str, str]:
        for key in ("high", "low"):
            if key not in v:
                raise ValueError(f"sub_score_bands missing required key '{key}'")
        return v

    @property
    def sorted_supplier_rules(self) -> list[SupplierRule]:
        """Supplier rules sorted ascending by priority (1 = highest)."""
        return sorted(self.supplier_rules, key=lambda r: r.priority)

    def get_rule(self, rule_id: str) -> RoutingRule | SupplierRule | None:
        """Return the rule with the given ID, or None."""
        for rr in self.routing_rules:
            if rr.rule_id == rule_id:
                return rr
        for sr in self.supplier_rules:
            if sr.rule_id == rule_id:
                return sr
        return None
