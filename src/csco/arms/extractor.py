"""Shared extraction function for all three arms.

All three arms reason in free text, then use this shared function to convert
the reasoning into structured `ScenarioDecision` objects. The extraction logic
is identical across arms — only the preceding reasoning differs.
"""

import logging
from csco.arms.llm import get_llm
from csco.models import ScenarioDecision

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = """\
You are extracting a structured supply-chain decision from an analyst's recommendation text.

First, classify the text as ONE of two shapes:

  ROUTING (scenario-level) — the decision applies to the scenario as a whole, not
  to any specific supplier. Signalled by phrases like "the assessment ends here",
  "no supplier evaluation needed/occurs", "no Tier-1 suppliers are scored",
  "structural fallthrough", a routing provision being named as firing, or simply
  the complete absence of any supplier being individually assessed.
    → Populate ONLY routing_result: fired_rule_id (rule 4 below), scenario_action
      (a short label for the action taken — use the exact term if the text states
      one such as "maintain_and_log" or "structural_fallthrough", otherwise a brief
      description like "maintain" or "monitor"), disposition (rule 2), escalation
      (rule 3), and permitted_actions (rule 5, if stated).
    → Leave recommendations as an EMPTY list.

  SUPPLIER (per-supplier) — the text gives a recommendation for one or more
  specific suppliers, each identified by an ID or name.
    → Populate recommendations, one entry per supplier named, using rules 1-8 below.
    → Leave routing_result as null.

Never populate both routing_result and recommendations. Never invent a placeholder,
empty-string, or literal "null" supplier_id to force a routing-level decision into
recommendations — a scenario-level decision belongs in routing_result, not as a
recommendation entry.

For the SUPPLIER shape, follow these rules exactly:

1. supplier_id — extract the supplier ID code matching the pattern S-[A-Z]+-[0-9]{2}
   (e.g. "S-GEO-SS-01", "S-GEO-HI-01", "S-DEEP-01"). If no ID pattern is found but
   a supplier name is mentioned (e.g. "AsiaPac Wafer Supply Co."), return the supplier
   name as-is. Prefer IDs when they are explicitly referenced.

2. disposition — must be one of: replace | monitor | maintain

3. escalation — must be one of: Responsible Analyst | Cross-functional Management | Executive Leadership

4. fired_rule_id — search the text for any rule ID matching the pattern TYPE-CODE.
   Examples: GEO-R3, GEO-NULL, GEO-DT, GEO-R1, GEO-MEDS, GEO-MED, GEO-LOWS, GEO-LOW,
   NAT-R3, NAT-R1, NAT-R5, NAT-DT, CYB-HUB, CYB-R3, CYB-R1, ECON-DEP, ECON-R3,
   LAB-SITE, LAB-R3, LAB-R1, and their DT/DT-NULL/MEDS/LOWS/LOW/MED variants.
   If the text says "GEO-R3 fires", "via GEO-R3", "rule GEO-R3", "matching GEO-R3",
   "condition GEO-R3", or similar — extract "GEO-R3" as the fired_rule_id.
   If no rule ID is identifiable, use null.

5. permitted_actions — use ONLY names from this controlled vocabulary (snake_case).
   Map any humanised or capitalised variant to its exact snake_case spec name:
     activate_pre_qualified_alternate   ("Activate pre-qualified alternate", "activate alternate", etc.)
     intensify_threat_monitoring        ("Intensify threat monitoring", "threat monitoring", etc.)
     scenario_lever_generation          ("Scenario lever generation", "generate levers", etc.)
     strategic_buffer_drawdown          ("Strategic buffer drawdown", "buffer drawdown", etc.)
     convene_crisis_response_center     ("Convene crisis response center", "CRC convene", etc.)
     reroute_logistics                  ("Reroute logistics", "logistics rerouting", etc.)
     expedited_emergent_design          ("Expedited emergent design", etc.)
   Only include actions explicitly listed as permitted/recommended for this supplier.
   Do NOT include prohibited or deferred actions.

6. rationale — preserve the analyst's reasoning text as written.

7. assigned_risk_level — extract the supplier's risk band as stated by the analyst.
   Must be one of: HIGH | MEDIUM | LOW
   If the analyst explicitly states "This supplier is HIGH risk" or similar,
   extract that value. If not explicitly stated, use null.
"""


def extract_decision(free_text_answer: str, result=None) -> ScenarioDecision:
    """Convert free-text agent reasoning into a structured ScenarioDecision.

    Shared across all three arms. Only the preceding reasoning differs;
    the extraction prompt and logic are identical.

    If a RunResult is passed as ``result``, the extraction call's token usage is
    accumulated onto the run's total token spend.
    """
    from csco.arms.prompts import OUTPUT_INSTRUCTIONS
    from langchain_core.messages import HumanMessage, SystemMessage

    # include_raw=True so the underlying AIMessage (and its usage_metadata) is
    # available for token accounting alongside the parsed decision.
    extraction_chain = get_llm().with_structured_output(ScenarioDecision, include_raw=True)
    raw_and_parsed = extraction_chain.invoke(
        [
            SystemMessage(_EXTRACTION_SYSTEM),
            HumanMessage(free_text_answer + OUTPUT_INSTRUCTIONS),
        ]
    )
    decision: ScenarioDecision = raw_and_parsed["parsed"]
    if result is not None:
        raw_msg = raw_and_parsed.get("raw")
        um = getattr(raw_msg, "usage_metadata", None)
        result.add_usage(um)
        # The extractor is a single call — its input is processed once (all uncached).
        try:
            result.extractor_input_tokens = int((um or {}).get("input_tokens", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            pass

    logger.debug(
        "Extracted: routing=%s, %d recommendations",
        decision.routing_result.fired_rule_id if decision.routing_result else None,
        len(decision.recommendations),
    )
    return decision
