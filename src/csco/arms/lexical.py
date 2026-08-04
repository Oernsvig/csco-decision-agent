"""Arm 4 — Lexical-Graph RAG.

The agent navigates an addressable document graph built from the embed-cut
playbook structure (headings, provision order, cross-references). It enters via
a single vector hop (find_playcard), then uses open() to traverse the document:
playcard outline → section intro → provision text → referenced sections.

Ablation target: Arm 2 (flat chunks) vs Arm 4 (chunks-plus-structure).
Both arms use the same embed-cut content; only structural edges differ.

Setup (run once before first use):
    python -m csco.graph.build_lexical > corpus/corpus_lexical.cypher
    python -m csco.cli.load_lexical
    python -m csco.cli.embed --arm lexical
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph, Neo4jVector

from csco.arms.llm import get_embeddings, get_llm
from csco.arms.prompts import AGENT_SYSTEM_PROMPT_LEXICAL, format_scenario
from csco.arms.runners import run_react_agent
from csco.derive import derive
from csco.models import Scenario, ScenarioDecision
from csco.tokens import count_tokens
from csco.settings import get_settings
from csco.specs.loader import load_spec

logger = logging.getLogger(__name__)

_LEX_PLAYCARD_INDEX = "lex_playcard_embeddings"


def _get_neo4j_graph() -> Neo4jGraph:
    s = get_settings().require_neo4j()
    return Neo4jGraph(
        url=s.neo4j_uri,
        username=s.neo4j_username,
        password=s.neo4j_password,
        database=s.neo4j_database,
        refresh_schema=False,
    )


def _get_lex_playcard_vectorstore() -> Neo4jVector:
    s = get_settings().require_neo4j()
    return Neo4jVector.from_existing_index(
        embedding=get_embeddings(),
        url=s.neo4j_uri,
        username=s.neo4j_username,
        password=s.neo4j_password,
        database=s.neo4j_database,
        index_name=_LEX_PLAYCARD_INDEX,
        node_label="LexPlaycard",
        text_node_property="summary",
        embedding_node_property="embedding",
    )


def run(
    scenario: Scenario,
    playcard_vs: Neo4jVector | None = None,
    graph: Neo4jGraph | None = None,
    result=None,
) -> ScenarioDecision:
    """Run Arm 4: lexical-graph navigation via find_playcard + open tools."""
    spec    = load_spec(scenario.disruption_type)
    derived = [derive(s, spec) for s in scenario.suppliers]

    vs = playcard_vs or _get_lex_playcard_vectorstore()
    g  = graph       or _get_neo4j_graph()

    _open_count = [0]

    # ------------------------------------------------------------------
    # Tool 1 — find_playcard
    # ------------------------------------------------------------------

    @tool
    def find_playcard(query: str) -> str:
        """Identify the relevant policy."""
        docs = vs.similarity_search(query, k=1)
        if not docs:
            return json.dumps({"error": "No matching playcard"})

        dtype   = docs[0].metadata.get("disruption_type", "?")
        pointer = f"playcard:{dtype}"

        if result is not None:
            _open_count[0] += 1
            from csco.evaluation.run_result import OpenEvent
            result.opens.append(OpenEvent(
                call_number=_open_count[0],
                pointer=pointer,
                node_type="playcard",
                pointers_returned=[pointer],
            ))
            result.tool_calls_total += 1

        return json.dumps({
            "disruption_type": dtype,
            "playcard_pointer": pointer,
        })

    # ------------------------------------------------------------------
    # Tool 2 — open (polymorphic)
    # ------------------------------------------------------------------

    @tool
    def open(pointer: str) -> str:
        """Open a part of the policy and see what it links to."""
        _open_count[0] += 1

        if pointer.startswith("playcard:"):
            payload = _open_playcard(g, pointer)
        elif pointer.startswith("section:"):
            payload = _open_section(g, pointer)
        elif pointer.startswith("provision:"):
            payload = _open_provision(g, pointer)
        else:
            payload = {"error": "not found"}

        if result is not None:
            node_type = pointer.split(":")[0]
            all_ptrs  = _collect_pointers(payload)
            refs      = payload.get("references") if node_type == "provision" else None
            # Policy content the model receives from this open: provision.text or
            # section.intro_text. Record a real token count (single tokenizer,
            # shared across arms) alongside the char length.
            policy_text = payload.get("text") or payload.get("intro_text") or ""
            from csco.evaluation.run_result import OpenEvent
            result.opens.append(OpenEvent(
                call_number=_open_count[0],
                pointer=pointer,
                node_type=node_type,
                pointers_returned=all_ptrs,
                references=refs,
                response_chars=len(policy_text),
                response_tokens=count_tokens(policy_text),
            ))
            result.tool_calls_total += 1

        return json.dumps(payload)

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    system_prompt = AGENT_SYSTEM_PROMPT_LEXICAL
    scenario_text = format_scenario(scenario, derived)

    logger.info("Arm 4 (lexical) — starting ReAct loop for %s", scenario.disruption_type)

    decision: ScenarioDecision = run_react_agent(
        get_llm(), [find_playcard, open], system_prompt, scenario_text, result=result,
    )

    logger.info(
        "Arm 4 complete — %d recommendations, routing=%s, opens=%d",
        len(decision.recommendations),
        decision.routing_result.fired_rule_id if decision.routing_result else None,
        _open_count[0],
    )
    return decision


# ------------------------------------------------------------------
# Graph query helpers
# ------------------------------------------------------------------

def _open_playcard(g: Neo4jGraph, pointer: str) -> dict:
    rows = g.query(
        "MATCH (pc:LexPlaycard {pointer: $ptr}) RETURN pc",
        {"ptr": pointer},
    )
    if not rows:
        return {"error": "not found"}

    pc = rows[0]["pc"]

    # Fetch sections in document order
    sec_rows = g.query(
        "MATCH (pc:LexPlaycard {pointer: $ptr})-[:HAS_SECTION]->(s:LexSection) "
        "RETURN s ORDER BY s.order_idx",
        {"ptr": pointer},
    )
    sections = [
        {
            "role":    r["s"]["role"],
            "heading": r["s"]["heading"],
            "pointer": r["s"]["pointer"],
        }
        for r in sec_rows
    ]

    return {
        "type":            "playcard",
        "disruption_type": pc["disruption_type"],
        "title":           pc["title"],
        "sections":        sections,
    }


def _open_section(g: Neo4jGraph, pointer: str) -> dict:
    rows = g.query(
        "MATCH (s:LexSection {pointer: $ptr}) RETURN s",
        {"ptr": pointer},
    )
    if not rows:
        return {"error": "not found"}

    sec = rows[0]["s"]

    # Provisions (in order)
    prov_rows = g.query(
        "MATCH (s:LexSection {pointer: $ptr})-[:HAS_PROVISION]->(p:LexProvision) "
        "RETURN p ORDER BY p.order_idx",
        {"ptr": pointer},
    )
    provisions = [
        {"rule_id": r["p"]["rule_id"], "pointer": r["p"]["pointer"]}
        for r in prov_rows
    ]

    # Next section
    next_rows = g.query(
        "MATCH (s:LexSection {pointer: $ptr})-[:NEXT]->(n:LexSection) RETURN n.pointer AS ptr",
        {"ptr": pointer},
    )
    next_section = next_rows[0]["ptr"] if next_rows else None

    _SECTION_PASS = {
        "routing":    "routing",
        "decision":   "supplier",
        "bands":      "definition",
        "actions":    "definition",
        "escalation": "definition",
        "posture":    "definition",
    }
    return {
        "type":         "section",
        "role":         sec["role"],
        "pass_type":    _SECTION_PASS.get(sec["role"], "other"),
        "heading":      sec["heading"],
        "intro_text":   sec["text"],
        "provisions":   provisions,
        "next_section": next_section,
    }


def _open_provision(g: Neo4jGraph, pointer: str) -> dict:
    rows = g.query(
        "MATCH (p:LexProvision {pointer: $ptr}) RETURN p",
        {"ptr": pointer},
    )
    if not rows:
        return {"error": "not found"}

    prov = rows[0]["p"]

    # next_else
    next_rows = g.query(
        "MATCH (p:LexProvision {pointer: $ptr})-[:NEXT]->(n:LexProvision) RETURN n.pointer AS ptr",
        {"ptr": pointer},
    )
    next_else = next_rows[0]["ptr"] if next_rows else None

    # REFERENCES
    ref_rows = g.query(
        "MATCH (p:LexProvision {pointer: $ptr})-[r:REFERENCES]->(s:LexSection) "
        "RETURN r.as AS as, s.pointer AS ptr",
        {"ptr": pointer},
    )
    references = [{"as": r["as"], "pointer": r["ptr"]} for r in ref_rows]

    # Chain-position ledger — pure navigation state, derived from HAS_PROVISION/
    # order_idx already on the graph (no new properties or edges needed). It
    # restates the section's actual remaining length in the most recent tool
    # message, at the exact point the stop/continue decision is made, rather
    # than relying on an instruction stated far upstream in the system prompt.
    pos_rows = g.query(
        "MATCH (s:LexSection)-[:HAS_PROVISION]->(p:LexProvision {pointer: $ptr}) "
        "RETURN p.order_idx AS idx, COUNT { (s)-[:HAS_PROVISION]->() } AS total",
        {"ptr": pointer},
    )
    if pos_rows:
        position = pos_rows[0]["idx"] + 1
        total = pos_rows[0]["total"]
        remaining = total - position
        if next_else:
            chain_status = (
                f"position {position} of {total} in this section; "
                f"{remaining} more provision(s) after this one via next_else "
                f"if it does not fire — next_else is not null."
            )
        else:
            chain_status = (
                f"position {position} of {total} in this section; "
                f"this is the last provision — next_else is null."
            )
    else:
        position = total = remaining = None
        chain_status = None

    return {
        "type":               "provision",
        "rule_id":            prov["rule_id"],
        "pass_type":          prov.get("pass", "unknown"),  # "routing" | "supplier"
        "text":               prov["text"],
        "next_else":          next_else,
        "references":         references,
        "position_in_section": position,
        "total_in_section":    total,
        "remaining_after_this": remaining,
        "chain_status":        chain_status,
    }


def _collect_pointers(payload: dict) -> list[str]:
    """Extract all pointer strings from an open() return payload."""
    ptrs: list[str] = []
    for key, val in payload.items():
        if key == "pointer" and isinstance(val, str):
            ptrs.append(val)
        elif key in ("sections", "provisions") and isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "pointer" in item:
                    ptrs.append(item["pointer"])
        elif key in ("next_section", "next_else") and isinstance(val, str):
            ptrs.append(val)
        elif key == "references" and isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "pointer" in item:
                    ptrs.append(item["pointer"])
    return ptrs


# ------------------------------------------------------------------
# Embedding setup (called by python -m csco.cli.embed --arm lexical)
# ------------------------------------------------------------------

def setup_lex_playcard_embeddings() -> Neo4jVector:
    """Embed LexPlaycard.summary nodes into lex_playcard_embeddings index."""
    s = get_settings().require_neo4j()
    logger.info("Embedding LexPlaycard.summary nodes …")
    return Neo4jVector.from_existing_graph(
        embedding=get_embeddings(),
        url=s.neo4j_uri,
        username=s.neo4j_username,
        password=s.neo4j_password,
        database=s.neo4j_database,
        index_name=_LEX_PLAYCARD_INDEX,
        node_label="LexPlaycard",
        text_node_properties=["summary"],
        embedding_node_property="embedding",
    )
