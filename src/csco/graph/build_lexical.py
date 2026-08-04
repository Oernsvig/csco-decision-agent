"""Generator: emit Cypher for the Arm 4 lexical-graph from embed-cut markdowns + specs.

Node labels are LexPlaycard / LexSection / LexProvision.

Usage:
    python -m csco.graph.build_lexical > corpus/corpus_lexical.cypher

After running, load with:
    python -m csco.cli.load_lexical

Then embed LexPlaycard summaries with:
    python -m csco.cli.embed --arm lexical
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBED_DIR = Path(__file__).parent.parent.parent.parent / "corpus" / "playbooks_embed"
_SPEC_DIR  = Path(__file__).parent.parent.parent.parent / "specs"

_TYPE_ORDER = ["geopolitical", "natural_disaster", "cyber", "economic", "labour"]

_HEADING_ROLE: dict[str, str] = {
    "scope":                             "scope",
    "1. risk bands":                     "bands",
    "2. activation and routing":         "routing",
    "3. decision logic":                 "decision",
    "4. permitted and prohibited":       "actions",
    "5. escalation":                     "escalation",
    "6. strategic priority":             "posture",
    "7. after-action":                   "afteraction",
}

_PROVISION_SECTIONS = {"routing", "decision"}

_SUMMARIES = {
    "geopolitical":    "Geopolitical disruption response policy for Tier-1 suppliers.",
    "natural_disaster":"Natural-disaster disruption response policy for Tier-1 suppliers.",
    "cyber":           "Cyber disruption response policy for Tier-1 suppliers.",
    "economic":        "Economic disruption response policy for Tier-1 suppliers.",
    "labour":          "Labour disruption response policy for Tier-1 suppliers.",
}

# Regex patterns
_H1 = re.compile(r"^# (.+)$", re.MULTILINE)
_H2 = re.compile(r"^## (.+)$", re.MULTILINE)
_H3 = re.compile(r"^### \(([A-Z]+-[A-Z0-9-]+)\) (.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedSection:
    role: str
    heading: str
    order_idx: int
    intro_text: str               # prose before first ### (or all text if no ###)
    disruption_type: str

    @property
    def pointer(self) -> str:
        return f"section:{self.disruption_type}:{self.role}"


@dataclass
class ParsedProvision:
    rule_id: str
    heading: str                  # full ### heading (e.g. "(GEO-R3) Shared sub-tier...")
    pass_: str                    # "routing" | "supplier"
    order_idx: int
    text: str                     # section-path prefixed — byte-identical to Arm 2 chunk
    section_role: str
    disruption_type: str

    @property
    def pointer(self) -> str:
        return f"provision:{self.rule_id}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _heading_to_role(heading: str) -> str | None:
    h = heading.lower().strip()
    for key, role in _HEADING_ROLE.items():
        if key in h:
            return role
    return None


def parse_playbook(raw: str, disruption_type: str) -> tuple[str, list[ParsedSection], list[ParsedProvision]]:
    """Parse embed-cut markdown → (h1_title, sections, provisions)."""

    h1_match = _H1.search(raw)
    h1_title = h1_match.group(1).strip() if h1_match else disruption_type

    h2_matches = list(_H2.finditer(raw))
    sections: list[ParsedSection] = []
    provisions: list[ParsedProvision] = []

    for i, h2_match in enumerate(h2_matches):
        heading = h2_match.group(1).strip()
        role = _heading_to_role(heading)
        if role is None:
            continue

        # Section text: everything between this ## and next ##
        sec_start = h2_match.end()
        sec_end   = h2_matches[i + 1].start() if i + 1 < len(h2_matches) else len(raw)
        sec_body  = raw[sec_start:sec_end]

        h3_in_sec = list(_H3.finditer(sec_body))

        if h3_in_sec:
            intro_text = sec_body[: h3_in_sec[0].start()].strip()
        else:
            intro_text = sec_body.strip()

        sections.append(ParsedSection(
            role=role,
            heading=heading,
            order_idx=len(sections),
            intro_text=intro_text,
            disruption_type=disruption_type,
        ))

        if role in _PROVISION_SECTIONS:
            pass_name = "routing" if role == "routing" else "supplier"
            for j, h3_match in enumerate(h3_in_sec):
                rule_id     = h3_match.group(1)
                prov_title  = f"({rule_id}) {h3_match.group(2).strip()}"

                prov_start = h3_match.end()
                prov_end   = h3_in_sec[j + 1].start() if j + 1 < len(h3_in_sec) else len(sec_body)
                prov_body  = sec_body[prov_start:prov_end].strip()

                # Section-path prefix — byte-identical to the Arm 2 chunk
                section_path = f"{h1_title} > {heading} > {prov_title}"
                prov_text    = f"{section_path}\n\n{prov_body}"

                provisions.append(ParsedProvision(
                    rule_id=rule_id,
                    heading=prov_title,
                    pass_=pass_name,
                    order_idx=j,
                    text=prov_text,
                    section_role=role,
                    disruption_type=disruption_type,
                ))

    return h1_title, sections, provisions


# ---------------------------------------------------------------------------
# REFERENCES edge derivation from spec fields
# ---------------------------------------------------------------------------

def _derive_references(rule_id: str, rule: Any, disruption_type: str) -> list[dict[str, str]]:
    """Return REFERENCES edges for one rule derived purely from spec fields."""
    refs: list[dict[str, str]] = []

    # band — every rule keys on risk band
    refs.append({"as": "band", "pointer": f"section:{disruption_type}:bands"})

    # escalation — every rule has an escalation outcome
    refs.append({"as": "escalation", "pointer": f"section:{disruption_type}:escalation"})

    # action — only rules with a non-empty permitted list
    permitted = getattr(getattr(rule, "consequence", None), "permitted", [])
    if permitted:
        refs.append({"as": "action", "pointer": f"section:{disruption_type}:actions"})

    return refs


# ---------------------------------------------------------------------------
# Cypher helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Escape text for a Cypher single-quoted string literal (single-line output).

    Replaces real newlines with the Cypher escape sequence \\n so the entire
    string fits on one line in the .cypher file — this lets the semicolon-based
    statement splitter in the loader work correctly.
    """
    return (
        text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _emit_playcard(dtype: str, title: str) -> list[str]:
    summary = _esc(_SUMMARIES[dtype])
    title_esc = _esc(title)
    pointer = f"playcard:{dtype}"
    return [
        f"MERGE (pc:LexPlaycard {{pointer: '{pointer}'}})",
        f"SET pc.disruption_type = '{dtype}',",
        f"    pc.title = '{title_esc}',",
        f"    pc.summary = '{summary}';",
        "",
    ]


def _emit_section(sec: ParsedSection) -> list[str]:
    heading_esc = _esc(sec.heading)
    text_esc    = _esc(sec.intro_text)
    return [
        f"MERGE (s:LexSection {{pointer: '{sec.pointer}'}})",
        f"SET s.disruption_type = '{sec.disruption_type}',",
        f"    s.role = '{sec.role}',",
        f"    s.heading = '{heading_esc}',",
        f"    s.order_idx = {sec.order_idx},",
        f"    s.text = '{text_esc}';",
        "",
    ]


def _emit_provision(prov: ParsedProvision) -> list[str]:
    heading_esc = _esc(prov.heading)
    text_esc    = _esc(prov.text)
    return [
        f"MERGE (p:LexProvision {{pointer: '{prov.pointer}'}})",
        f"SET p.rule_id = '{prov.rule_id}',",
        f"    p.disruption_type = '{prov.disruption_type}',",
        f"    p.pass = '{prov.pass_}',",
        f"    p.order_idx = {prov.order_idx},",
        f"    p.heading = '{heading_esc}',",
        f"    p.text = '{text_esc}';",
        "",
    ]


def _emit_has_section(dtype: str, sec: ParsedSection) -> list[str]:
    return [
        f"MATCH (pc:LexPlaycard {{pointer: 'playcard:{dtype}'}})",
        f"MATCH (s:LexSection   {{pointer: '{sec.pointer}'}})",
        "MERGE (pc)-[:HAS_SECTION]->(s);",
        "",
    ]


def _emit_section_next(sec_a: ParsedSection, sec_b: ParsedSection) -> list[str]:
    return [
        f"MATCH (a:LexSection {{pointer: '{sec_a.pointer}'}})",
        f"MATCH (b:LexSection {{pointer: '{sec_b.pointer}'}})",
        "MERGE (a)-[:NEXT]->(b);",
        "",
    ]


def _emit_has_provision(sec: ParsedSection, prov: ParsedProvision) -> list[str]:
    return [
        f"MATCH (s:LexSection  {{pointer: '{sec.pointer}'}})",
        f"MATCH (p:LexProvision {{pointer: '{prov.pointer}'}})",
        "MERGE (s)-[:HAS_PROVISION]->(p);",
        "",
    ]


def _emit_provision_next(prov_a: ParsedProvision, prov_b: ParsedProvision) -> list[str]:
    return [
        f"MATCH (a:LexProvision {{pointer: '{prov_a.pointer}'}})",
        f"MATCH (b:LexProvision {{pointer: '{prov_b.pointer}'}})",
        "MERGE (a)-[:NEXT]->(b);",
        "",
    ]


def _emit_reference(prov: ParsedProvision, ref: dict[str, str]) -> list[str]:
    return [
        f"MATCH (p:LexProvision {{pointer: '{prov.pointer}'}})",
        f"MATCH (s:LexSection   {{pointer: '{ref['pointer']}'}})",
        f"MERGE (p)-[:REFERENCES {{as: '{ref['as']}'}}]->(s);",
        "",
    ]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate() -> None:
    from csco.specs.loader import load_spec

    lines: list[str] = [
        "// Arm 4 — Lexical-Graph Cypher seed",
        "// Generated by csco.graph.build_lexical — do not hand-edit.",
        "// Node labels: LexPlaycard, LexSection, LexProvision",
        "",
    ]

    for dtype in _TYPE_ORDER:
        md_path = _EMBED_DIR / f"{dtype}.md"
        if not md_path.exists():
            print(f"// WARNING: missing embed-cut for {dtype}", file=sys.stderr)
            continue

        raw = md_path.read_text(encoding="utf-8")
        spec = load_spec(dtype)

        h1_title, sections, provisions = parse_playbook(raw, dtype)

        # Build rule lookup for REFERENCES derivation
        all_rules = {r.rule_id: r for r in spec.routing_rules + spec.supplier_rules}

        lines += [f"// === {dtype.upper()} ===", ""]

        # LexPlaycard
        lines += _emit_playcard(dtype, h1_title)

        # LexSection nodes
        for sec in sections:
            lines += _emit_section(sec)

        # LexProvision nodes
        for prov in provisions:
            lines += _emit_provision(prov)

        # HAS_SECTION edges (playcard → each section)
        for sec in sections:
            lines += _emit_has_section(dtype, sec)

        # NEXT edges between sections (document order)
        for i in range(len(sections) - 1):
            lines += _emit_section_next(sections[i], sections[i + 1])

        # HAS_PROVISION edges + NEXT within each chain
        routing_provs  = [p for p in provisions if p.pass_ == "routing"]
        supplier_provs = [p for p in provisions if p.pass_ == "supplier"]

        routing_section  = next((s for s in sections if s.role == "routing"),  None)
        decision_section = next((s for s in sections if s.role == "decision"), None)

        for prov in routing_provs:
            if routing_section:
                lines += _emit_has_provision(routing_section, prov)

        for prov in supplier_provs:
            if decision_section:
                lines += _emit_has_provision(decision_section, prov)

        for i in range(len(routing_provs) - 1):
            lines += _emit_provision_next(routing_provs[i], routing_provs[i + 1])

        for i in range(len(supplier_provs) - 1):
            lines += _emit_provision_next(supplier_provs[i], supplier_provs[i + 1])

        # REFERENCES edges (from spec fields — deterministic, not NLP-derived)
        for prov in provisions:
            rule = all_rules.get(prov.rule_id)
            if rule is None:
                continue
            for ref in _derive_references(prov.rule_id, rule, dtype):
                lines += _emit_reference(prov, ref)

    print("\n".join(lines))


if __name__ == "__main__":
    generate()
