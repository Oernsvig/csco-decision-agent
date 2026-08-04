"""Arm 2 — Vector RAG (Neo4j vector index over embed-cut playbook chunks).

The agent has one tool: search_playbooks — a similarity search over
pre-embedded decision content chunks stored in a Neo4j vector index.

Setup (run once before first use):
    python -m csco.cli.embed --arm vector
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import tool
from langchain_neo4j import Neo4jVector

from csco.arms.llm import get_embeddings, get_llm
from csco.arms.prompts import AGENT_SYSTEM_PROMPT_VECTOR, format_scenario
from csco.arms.runners import run_react_agent
from csco.derive import derive
from csco.models import Scenario, ScenarioDecision
from csco.settings import get_settings
from csco.specs.loader import load_spec
from csco.tokens import count_tokens

logger = logging.getLogger(__name__)

_VECTOR_INDEX = "playbook_chunks"
_NODE_LABEL   = "Chunk"
_TEXT_PROP    = "content"
_EMB_PROP     = "embedding"

# k=5 gives a retrieval window that exposes provision-level chunks and nearby
# definition context without approximating full-context prompting.
_RETRIEVAL_K = 5

# Section-heading → role → pass_type mappings (mirrors build_lexical._HEADING_ROLE)
_H2_ROLE_MAP: dict[str, str] = {
    "1. risk bands":              "bands",
    "2. activation and routing":  "routing",
    "3. decision logic":          "decision",
    "4. permitted and prohibited": "actions",
    "5. escalation":              "escalation",
    "6. strategic priority":      "posture",
    "7. after-action":            "afteraction",
}
_ROLE_PASS_MAP: dict[str, str] = {
    "routing":    "routing",
    "decision":   "supplier",
    "bands":      "definition",
    "actions":    "definition",
    "escalation": "definition",
    "posture":    "definition",
    "afteraction": "other",
    "scope":      "other",
}


def _derive_section_role(h2: str) -> str:
    h2_lower = h2.lower().strip()
    for key, role in _H2_ROLE_MAP.items():
        if key in h2_lower:
            return role
    return "other"


def _get_vectorstore() -> Neo4jVector:
    s = get_settings().require_neo4j()
    return Neo4jVector.from_existing_index(
        embedding=get_embeddings(),
        url=s.neo4j_uri,
        username=s.neo4j_username,
        password=s.neo4j_password,
        database=s.neo4j_database,
        index_name=_VECTOR_INDEX,
        node_label=_NODE_LABEL,
        text_node_property=_TEXT_PROP,
        embedding_node_property=_EMB_PROP,
    )


def run(scenario: Scenario, vectorstore: Neo4jVector | None = None, result=None) -> ScenarioDecision:
    """Run Arm 2: ReAct agent with Neo4j vector search over embed-cut playbook chunks.

    Pass a RunResult as `result` to capture instrumentation data.
    """
    spec = load_spec(scenario.disruption_type)
    derived = [derive(s, spec) for s in scenario.suppliers]
    vs = vectorstore or _get_vectorstore()

    _search_call_count = [0]

    @tool
    def search_playbooks(query: str) -> str:
        """Identify the relevant policy. Search for decision rules, risk thresholds, and permitted actions."""
        _search_call_count[0] += 1
        call_n = _search_call_count[0]

        docs = vs.similarity_search(query, k=_RETRIEVAL_K)
        if not docs:
            logger.info("Arm 2 search #%d  query=%r  → NO RESULTS", call_n, query)
            if result is not None:
                from csco.evaluation.run_result import SearchEvent
                result.searches.append(SearchEvent(call_number=call_n, query=query, chunks=[]))
                result.tool_calls_total += 1
            return "No relevant policy found."

        logger.info("Arm 2 search #%d  query=%r", call_n, query)
        chunk_records = []
        result_parts = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            source   = meta.get("source", "?")
            fname    = os.path.basename(source) if source != "?" else "?"
            rule_id  = meta.get("rule_id")
            section_role = meta.get("section_role", "other")
            pass_type    = meta.get("pass_type", "other")
            chunk_type   = meta.get("chunk_type", "other")

            heading = next(
                (line.strip() for line in d.page_content.splitlines() if line.startswith("#")),
                d.page_content[:60].replace("\n", " "),
            )
            logger.info(
                "  chunk %d/%d  file=%-25s  pass=%-10s  rule_id=%s  heading=%s",
                i, len(docs), fname, pass_type, rule_id, heading,
            )
            chunk_records.append({
                "file":         fname,
                "heading":      heading,
                "section_role": section_role,
                "pass_type":    pass_type,
                "chunk_type":   chunk_type,
                "content_snippet": d.page_content[:2000],
                # Real token count of the full chunk the model receives (single
                # tokenizer, shared across arms) — not a chars/4 estimate.
                "content_tokens": count_tokens(d.page_content),
                "rule_id":      rule_id,
            })

            # Structured header shown to the agent so it knows what kind of chunk it received
            hdr_parts = [
                f"[Rank: {i}]",
                f"[Source: {fname}]",
                f"[Pass: {pass_type}]",
                f"[Section: {section_role}]",
                f"[Chunk: {chunk_type}]",
            ]
            if rule_id:
                hdr_parts.append(f"[Rule ID: {rule_id}]")
            result_parts.append(" ".join(hdr_parts) + "\n\n" + d.page_content)

        if result is not None:
            from csco.evaluation.run_result import SearchEvent
            result.searches.append(SearchEvent(call_number=call_n, query=query, chunks=chunk_records))
            result.tool_calls_total += 1

        return "\n\n---\n\n".join(result_parts)

    system_prompt = AGENT_SYSTEM_PROMPT_VECTOR
    scenario_text = format_scenario(scenario, derived)

    logger.info("Arm 2 (vector) — starting ReAct loop for %s", scenario.disruption_type)

    decision: ScenarioDecision = run_react_agent(
        get_llm(), [search_playbooks], system_prompt, scenario_text, result=result,
    )

    logger.info("Arm 2 total searches: %d", _search_call_count[0])

    logger.info(
        "Arm 2 complete — %d recommendations, routing=%s",
        len(decision.recommendations),
        decision.routing_result.fired_rule_id if decision.routing_result else None,
    )
    return decision


def setup_vectorstore(playbook_paths: list[str]) -> Neo4jVector:
    """Embed playbook chunks and load them into the Neo4j vector index."""
    import re
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    s = get_settings().require_neo4j()
    all_docs = []
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    for path in playbook_paths:
        raw = TextLoader(path, encoding="utf-8").load()
        chunks = splitter.split_text(raw[0].page_content)
        for chunk in chunks:
            chunk.metadata["source"] = path

            # Corpus-derived metadata — exposed to the agent via search output headers.
            h2 = chunk.metadata.get("h2", "")
            h3 = chunk.metadata.get("h3", "")

            # Section role and pass type from h2 heading
            section_role = _derive_section_role(h2)
            chunk.metadata["section_role"] = section_role
            chunk.metadata["pass_type"]    = _ROLE_PASS_MAP.get(section_role, "other")

            # chunk_type and rule_id from h3 heading
            rule_id_match = re.search(r"\(([A-Z]+-[A-Z0-9-]+)\)", h3)
            if rule_id_match:
                chunk.metadata["rule_id"]    = rule_id_match.group(1)
                chunk.metadata["chunk_type"] = "provision"
            else:
                chunk.metadata["chunk_type"] = "section" if h2 else "other"

            # MarkdownHeaderTextSplitter puts h1/h2/h3 in metadata but strips them
            # from page_content. Prepend the section path so the embedding captures
            # the section topic ("3. Decision logic", "2. Activation and routing", etc.)
            # and retrieval queries can reach the right section.
            header_parts = [
                chunk.metadata[k]
                for k in ("h1", "h2", "h3")
                if k in chunk.metadata
            ]
            if header_parts:
                section_path = " > ".join(header_parts)
                chunk.page_content = section_path + "\n\n" + chunk.page_content
        all_docs.extend(chunks)

    logger.info("Embedding %d chunks from %d playbooks", len(all_docs), len(playbook_paths))
    return Neo4jVector.from_documents(
        documents=all_docs,
        embedding=get_embeddings(),
        url=s.neo4j_uri,
        username=s.neo4j_username,
        password=s.neo4j_password,
        database=s.neo4j_database,
        index_name=_VECTOR_INDEX,
        node_label=_NODE_LABEL,
        text_node_property=_TEXT_PROP,
        embedding_node_property=_EMB_PROP,
    )
