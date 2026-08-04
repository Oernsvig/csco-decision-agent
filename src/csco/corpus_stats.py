"""Corpus statistics reporter.

Verifies and reports the measured sizes of all three policy corpora (static,
vector, lexical) to support the dissertation claim that they share identical
content. Also checks for mid-provision chunk splits that would break the
Vector-vs-Lexical parity claim.

Usage:
    python -m csco.corpus_stats

Output is printed to stdout and also written to corpus/CORPUS_STATS.txt.
"""

from __future__ import annotations

import re
from pathlib import Path


_EMBED_DIR = Path(__file__).parent.parent.parent / "corpus" / "playbooks_embed"
_STATS_PATH = Path(__file__).parent.parent.parent / "corpus" / "CORPUS_STATS.txt"

_PROVISION_HEADER = re.compile(r"^### \(([A-Z]+-[A-Z0-9-]+)\)", re.MULTILINE)


def _word_count(text: str) -> int:
    return len(text.split())


def _token_estimate(text: str) -> int:
    """Token count using the same tokenizer the arms use for policy-context tokens."""
    from csco.tokens import count_tokens

    return count_tokens(text)


def measure_static(embed_texts: dict[str, str]) -> dict:
    """Measure the static policy block (concatenated embed-cuts)."""
    divider = "\n\n" + "─" * 70 + "\n\n"
    order = ["geopolitical", "natural_disaster", "cyber", "economic", "labour"]
    parts = [embed_texts[dt] for dt in order if dt in embed_texts]
    full_text = divider.join(parts)

    provisions = _PROVISION_HEADER.findall(full_text)

    return {
        "chars":          len(full_text),
        "words":          _word_count(full_text),
        "token_estimate": _token_estimate(full_text),
        "provision_ids":  len(provisions),
        "unique_rule_ids": len(set(provisions)),
    }


def measure_vector_chunks(embed_texts: dict[str, str]) -> dict:
    """Measure vector chunk statistics (MarkdownHeaderTextSplitter output)."""
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import MarkdownHeaderTextSplitter
    except ImportError:
        return {"error": "langchain_community not installed"}

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )

    total_chunks = 0
    provision_chunks = 0
    non_provision_chunks = 0
    chunks_by_type: dict[str, int] = {}
    mid_provision_splits: list[str] = []
    chunk_chars: list[int] = []

    for dtype in ["geopolitical", "natural_disaster", "cyber", "economic", "labour"]:
        md_path = _EMBED_DIR / f"{dtype}.md"
        if not md_path.exists():
            continue
        raw = TextLoader(str(md_path), encoding="utf-8").load()
        chunks = splitter.split_text(raw[0].page_content)
        chunks_by_type[dtype] = len(chunks)
        total_chunks += len(chunks)

        for chunk in chunks:
            body = chunk.page_content
            chunk_chars.append(len(body))
            if "h3" in chunk.metadata:
                m = re.search(r"\(([A-Z]+-[A-Z0-9-]+)\)", chunk.metadata.get("h3", ""))
                if m:
                    provision_chunks += 1
                    # Check for mid-provision split: provision headers inside the body
                    # would indicate the splitter cut through a provision block
                    inner_headers = _PROVISION_HEADER.findall(body)
                    if inner_headers:
                        mid_provision_splits.append(
                            f"  {m.group(1)}: body contains {inner_headers}"
                        )
                else:
                    non_provision_chunks += 1
            else:
                non_provision_chunks += 1

    avg_chars = sum(chunk_chars) / len(chunk_chars) if chunk_chars else 0
    return {
        "total_chunks":            total_chunks,
        "provision_chunks":        provision_chunks,
        "non_provision_chunks":    non_provision_chunks,
        "chunks_by_type":          chunks_by_type,
        "avg_chunk_chars":         round(avg_chars),
        "min_chunk_chars":         min(chunk_chars) if chunk_chars else 0,
        "max_chunk_chars":         max(chunk_chars) if chunk_chars else 0,
        "mid_provision_splits":    mid_provision_splits,
        "parity_ok":               len(mid_provision_splits) == 0,
    }


def measure_lexical_graph() -> dict:
    """Query Neo4j for LexProvision and LexSection counts."""
    try:
        from csco.settings import get_settings
        from neo4j import GraphDatabase
        s = get_settings()
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
        with driver.session(database=s.neo4j_database) as session:
            prov_count = session.run("MATCH (p:LexProvision) RETURN count(p) AS n").single()["n"]
            sec_count  = session.run("MATCH (s:LexSection) RETURN count(s) AS n").single()["n"]
            pc_count   = session.run("MATCH (p:LexPlaycard) RETURN count(p) AS n").single()["n"]
            ref_count  = session.run("MATCH ()-[r:REFERENCES]->() RETURN count(r) AS n").single()["n"]
            band_refs  = session.run(
                "MATCH ()-[r:REFERENCES {as:'band'}]->() RETURN count(r) AS n"
            ).single()["n"]
            tier_refs  = session.run(
                "MATCH ()-[r:REFERENCES {as:'tier'}]->() RETURN count(r) AS n"
            ).single()["n"]
        driver.close()
        return {
            "playcard_nodes":     pc_count,
            "section_nodes":      sec_count,
            "provision_nodes":    prov_count,
            "reference_edges":    ref_count,
            "band_ref_edges":     band_refs,
            "tier_ref_edges":     tier_refs,
        }
    except Exception as exc:
        return {"error": f"Neo4j not reachable: {exc}"}


def report() -> str:
    """Generate the corpus statistics report and return it as a string."""
    lines = [
        "═" * 70,
        "CSCO CORPUS STATISTICS",
        "  (Measured from current corpus/playbooks_embed/ and Neo4j)",
        "═" * 70,
        "",
    ]

    # Load embed-cut files
    embed_texts: dict[str, str] = {}
    for dtype in ["geopolitical", "natural_disaster", "cyber", "economic", "labour"]:
        path = _EMBED_DIR / f"{dtype}.md"
        if path.exists():
            embed_texts[dtype] = path.read_text(encoding="utf-8")

    if not embed_texts:
        return "ERROR: No embed-cut files found. Run python -m csco.generators.playbook first."

    lines += [f"Embed-cut files found: {len(embed_texts)}/5"]
    for dtype, text in embed_texts.items():
        n_provisions = len(_PROVISION_HEADER.findall(text))
        lines += [f"  {dtype:20s}: {len(text):6,d} chars, {n_provisions} provisions"]

    # Static policy
    lines += ["", "─" * 70, "STATIC POLICY BLOCK", "─" * 70]
    static = measure_static(embed_texts)
    lines += [
        f"  Total chars         : {static['chars']:,}",
        f"  Words               : {static['words']:,}",
        f"  Tokens              : {static['token_estimate']:,}  (tokenizer)",
        f"  Provision rule IDs  : {static['provision_ids']}",
        f"  Unique rule IDs     : {static['unique_rule_ids']}",
    ]

    # Vector chunks
    lines += ["", "─" * 70, "VECTOR CHUNKS (MarkdownHeaderTextSplitter)", "─" * 70]
    vector = measure_vector_chunks(embed_texts)
    if "error" in vector:
        lines += [f"  ERROR: {vector['error']}"]
    else:
        lines += [
            f"  Total chunks        : {vector['total_chunks']}",
            f"  Provision chunks    : {vector['provision_chunks']}  (h3 with rule ID)",
            f"  Non-provision chunks: {vector['non_provision_chunks']}",
            "  Chunks by type:",
        ]
        for dtype, n in sorted(vector["chunks_by_type"].items()):
            lines += [f"    {dtype:20s}: {n}"]
        lines += [
            f"  Avg chunk chars     : {vector['avg_chunk_chars']}",
            f"  Min/Max chunk chars : {vector['min_chunk_chars']} / {vector['max_chunk_chars']}",
        ]
        if vector["mid_provision_splits"]:
            lines += [
                f"  ⚠ MID-PROVISION SPLITS DETECTED ({len(vector['mid_provision_splits'])}):",
            ]
            lines += vector["mid_provision_splits"]
            lines += ["  This may break the Vector-vs-Lexical parity claim."]
        else:
            lines += ["  ✓ No mid-provision splits detected."]

    # Lexical graph
    lines += ["", "─" * 70, "LEXICAL GRAPH (Neo4j)", "─" * 70]
    lexical = measure_lexical_graph()
    if "error" in lexical:
        lines += [f"  {lexical['error']}  (skipped — run after load_lexical)"]
    else:
        lines += [
            f"  LexPlaycard nodes   : {lexical['playcard_nodes']}",
            f"  LexSection nodes    : {lexical['section_nodes']}",
            f"  LexProvision nodes  : {lexical['provision_nodes']}",
            f"  REFERENCES edges    : {lexical['reference_edges']}",
            f"    as:'band'         : {lexical['band_ref_edges']}",
            f"    as:'tier'         : {lexical['tier_ref_edges']}",
        ]

    # Parity summary
    lines += ["", "─" * 70, "PARITY SUMMARY", "─" * 70]
    if "error" not in vector:
        lines += [
            f"  Static unique rule IDs  : {static['unique_rule_ids']}",
            f"  Vector provision chunks : {vector['provision_chunks']}",
            f"  Match                   : {'✓' if vector['provision_chunks'] == static['unique_rule_ids'] else '✗'}",
            f"  No mid-provision splits : {'✓' if vector['parity_ok'] else '✗ SEE ABOVE'}",
        ]
        if "error" not in lexical:
            lines += [
                f"  Lexical provisions      : {lexical['provision_nodes']}",
                f"  Static == Vector == Lexical provision count: "
                f"{'✓' if static['unique_rule_ids'] == vector['provision_chunks'] == lexical['provision_nodes'] else '✗'}",
            ]

    lines += ["", "═" * 70]
    return "\n".join(lines)


if __name__ == "__main__":
    text = report()
    print(text)
    _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATS_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"\nStats written to {_STATS_PATH}")
