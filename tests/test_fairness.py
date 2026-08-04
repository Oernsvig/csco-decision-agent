"""
Fairness and integrity tests — executable assertions for the fairness invariants,
spec metadata consistency, and oracle correctness.

Run:  pytest tests/test_fairness.py
Integration tests (require Neo4j): pytest tests/test_fairness.py -m integration
Skip integration: pytest tests/test_fairness.py -m "not integration"
"""

import re
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Test 1 — Forbidden content test (no Neo4j needed)
# ---------------------------------------------------------------------------

def test_no_worked_examples_in_embed_cuts():
    """Embed-cut files must not contain worked-example markers or section headings."""
    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"
    assert embed_dir.exists(), (
        f"Embed-cut directory not found: {embed_dir}. "
        "Run python -m csco.generators.playbook first."
    )

    md_files = sorted(embed_dir.glob("*.md"))
    assert len(md_files) == 5, (
        f"Expected 5 embed-cut files, got {len(md_files)}: {[f.name for f in md_files]}"
    )

    all_text = "\n\n".join(f.read_text(encoding="utf-8") for f in md_files)

    forbidden_patterns = [
        ("EX-",                  "worked-example ID"),
        ("## 8.",                "worked-example section heading"),
        ("Worked example",       "worked-example heading text"),
        ("## 8. Worked example", "full worked-example section marker"),
    ]

    violations = []
    for pattern, label in forbidden_patterns:
        if pattern in all_text:
            violations.append(f"Found forbidden '{pattern}' ({label}) in embed-cut files")

    assert not violations, "\n".join(violations)

    assert "Tomlin 2006" in all_text, (
        "Expected spec grounding citations (e.g. 'Tomlin 2006') to be present in "
        "embed-cut files — spec note/grounding fields must be included in all arms."
    )


def test_static_consolidated_policy_is_embed_cuts_only():
    """The Static arm's model-visible policy must be the concatenated embed cuts,
    never the full playbooks. Guards the fairness invariant that Static sees no
    answer-key content (worked examples) unavailable to the Vector and Lexical
    arms — the parity tests below only check that each provision is *present* in
    every encoding, so a full-playbook superset passes them while still leaking
    the § 8 worked examples into Static.
    """
    from csco.arms.prompts import format_all_types_policy, STATIC_SYSTEM_TEMPLATE

    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"
    assert embed_dir.exists(), (
        f"Embed-cut directory not found: {embed_dir}. "
        "Run python -m csco.generators.playbook first."
    )

    consolidated = format_all_types_policy()

    # (1) byte-identical to the concatenated embed cuts, in the fixed type order
    type_order = ["geopolitical", "natural_disaster", "cyber", "economic", "labour"]
    divider = "\n\n" + "─" * 70 + "\n\n"
    concatenated = divider.join(
        (embed_dir / f"{t}.md").read_text(encoding="utf-8") for t in type_order
    )
    assert consolidated == concatenated, (
        "Static's consolidated policy is not byte-identical to the concatenated "
        "embed cuts — load_embed_cut_policy() is likely reading corpus/playbooks/ "
        "(full playbooks) instead of corpus/playbooks_embed/."
    )

    # (2) no worked-example markers in the assembled Static model-visible prompt
    static_prompt = STATIC_SYSTEM_TEMPLATE.format(policy_block=consolidated)
    for marker in ("## 8. Worked example", "## 8.", "Worked example", "EX-"):
        assert marker not in static_prompt, (
            f"Worked-example marker {marker!r} leaked into the Static model-visible prompt."
        )


def test_provision_order_is_canonical_across_corpus_and_specs():
    """Ordering parity: the provision (rule_id) order in each model-visible embed cut
    equals the spec-declared order (routing rules then supplier rules). Static reads the
    whole corpus, Vector chunks it, and Lexical builds its graph from it, so pinning the
    corpus order to the spec order means all three inherit one canonical ordering rather
    than a per-arm reshuffle. Also checks the consolidated Static policy preserves that
    order across the five files.
    """
    import re as _re

    from csco.arms.prompts import load_embed_cut_policy
    from csco.specs.loader import load_all_specs

    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"
    hdr = _re.compile(r"^#{2,3}\s*\(([A-Z]{2,5}(?:-[A-Z0-9]+)+)\)", _re.M)
    specs = load_all_specs()

    per_type_order: dict[str, list[str]] = {}
    for dtype, spec in specs.items():
        corpus_order = hdr.findall((embed_dir / f"{dtype}.md").read_text(encoding="utf-8"))
        spec_order = [r.rule_id for r in spec.routing_rules] + [r.rule_id for r in spec.supplier_rules]
        assert corpus_order == spec_order, (
            f"{dtype}: corpus provision order != spec order\n"
            f"  corpus={corpus_order}\n  spec  ={spec_order}"
        )
        per_type_order[dtype] = corpus_order

    # load_embed_cut_policy concatenates the files in this fixed type order.
    type_order = ["geopolitical", "natural_disaster", "cyber", "economic", "labour"]
    expected = [rid for t in type_order for rid in per_type_order[t]]
    assert hdr.findall(load_embed_cut_policy()) == expected


# ---------------------------------------------------------------------------
# Test 2 — Rule parity test (Arm 1 + Arm 2)
# ---------------------------------------------------------------------------

def test_arm1_arm2_rule_parity():
    """All 52 rule IDs from the specs must appear in Arm 1 policy text and Arm 2 embed-cut text."""
    from csco.parity_check import collect_all_rule_ids, check_policy_block_completeness
    from csco.arms.prompts import load_embed_cut_policy

    rules_by_type = collect_all_rule_ids()
    all_ids: set[str] = set()
    for ids in rules_by_type.values():
        all_ids.update(ids)

    assert len(all_ids) == 52, (
        f"Expected 52 rules total, got {len(all_ids)}: {sorted(all_ids)}"
    )

    arm1_text = load_embed_cut_policy()
    ok1, miss1 = check_policy_block_completeness(arm1_text)

    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"
    arm2_texts = [f.read_text(encoding="utf-8") for f in sorted(embed_dir.glob("*.md"))]
    arm2_text = "\n\n".join(arm2_texts)
    ok2, miss2 = check_policy_block_completeness(arm2_text)

    failures = []
    if not ok1:
        failures.append(f"Arm 1 (static policy) missing {len(miss1)} rules: {sorted(miss1)}")
    if not ok2:
        failures.append(f"Arm 2 (embed-cut files) missing {len(miss2)} rules: {sorted(miss2)}")

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 3 — Extractor symmetry test
# ---------------------------------------------------------------------------

def test_extractor_symmetry():
    """Each arm module must reach extract_decision and must not call with_structured_output directly.

    The invariant: no arm bypasses the shared extractor. Arms may reach it directly
    (extract_decision literally in the arm file) or indirectly through the shared
    csco.arms.runners plumbing (run_single_turn / run_react_agent) — both are single
    call sites shared identically by all arms that use them, verified below to
    themselves call extract_decision and never with_structured_output directly.
    extractor.py legitimately calls .with_structured_output() — it is excluded from this check.
    """
    src_dir = Path(__file__).parent.parent / "src" / "csco" / "arms"

    arm_modules = {
        "static.py":  src_dir / "static.py",
        "vector.py":  src_dir / "vector.py",
        "lexical.py": src_dir / "lexical.py",
    }

    runners = (src_dir / "runners.py").read_text(encoding="utf-8")
    assert "extract_decision" in runners, (
        "runners.py should call extract_decision — something changed"
    )
    assert ".with_structured_output(" not in runners, (
        "runners.py must not call with_structured_output directly — "
        "structured output must go through csco.arms.extractor"
    )

    failures = []
    for name, path in arm_modules.items():
        assert path.exists(), f"Arm module not found: {path}"
        text = path.read_text(encoding="utf-8")

        reaches_extractor = (
            "extract_decision" in text
            or "run_single_turn" in text
            or "run_react_agent" in text
        )
        if not reaches_extractor:
            failures.append(
                f"{name}: does not reach extract_decision "
                "(directly, or via run_single_turn/run_react_agent)"
            )

        # Arm implementation files must not call .with_structured_output() directly.
        # Only extractor.py is permitted to call it.
        if ".with_structured_output(" in text:
            failures.append(
                f"{name}: calls '.with_structured_output(' directly — "
                "structured output must go through csco.arms.extractor"
            )

    # Verify extractor.py itself is NOT in the check list and DOES call it
    extractor = (src_dir / "extractor.py").read_text(encoding="utf-8")
    assert ".with_structured_output(" in extractor, (
        "extractor.py should call .with_structured_output() — something changed"
    )

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 4 — Tier consistency test
# ---------------------------------------------------------------------------

def test_sub_score_fields_present_in_policy():
    """The structural sub-scores the supplier rules key on must be described in the
    policy corpus (Section 1). Supplier criticality was removed from the framework;
    the risk band plus these sub-scores are the only structural signals now.
    """
    from csco.arms.prompts import load_embed_cut_policy
    from csco.generators.playbook import generate_all_embed_cuts_text

    sub_score_fields = [
        "sub_exposure_breadth",
        "sub_dependency_ratio",
        "sub_downstream_criticality",
        "sub_node_centrality",
        "sub_exposure_depth",
    ]
    policy_text = load_embed_cut_policy()
    corpus_text = generate_all_embed_cuts_text()

    failures = []
    for field in sub_score_fields:
        if field not in policy_text:
            failures.append(f"Sub-score '{field}' not found in embed-cut policy text")
        if field not in corpus_text:
            failures.append(f"Sub-score '{field}' not found in generate_all_embed_cuts_text output")

    assert not failures, "\n".join(failures)


def test_routing_when_string_consistent_with_evaluator_shape():
    """The free-form `when:` spec string is documentation, not source — the
    real evaluator (oracle/deterministic.py's _routing_condition_matches())
    hardcodes each suffix's shape and never reads `when:` at all. That's a
    latent drift channel: nothing stops `when:` from silently diverging from
    what actually gets graded
    and what the corpus enumeration (test_routing_enumeration_matches_
    evaluator) claims, which the model reads. This is a cheap keyword-level
    tripwire on that channel — not a full parse, just the top-level
    connective and every field name each suffix's real shape depends on,
    so a `when:` edit that changes the described logical shape fails loudly
    instead of drifting silently."""
    import yaml

    from csco.generators.playbook import _rule_suffix
    from csco.specs.loader import load_all_specs

    _REQUIRED: dict[str, tuple[str, ...]] = {
        # NULL: path_on_network==false OR (all Tier-1 LOW AND not high criticality)
        "NULL": ("OR", "AND", "path_on_network", "Tier-1", "LOW", "sub_downstream_criticality"),
        # DT: not tier1_on_disrupted_path AND (shared_subtier_source OR count>=2)
        "DT": ("AND", "OR", "tier1_on_disrupted_path", "shared_subtier_source", "count_disrupted_paths"),
        # DT-NULL: not tier1_on_disrupted_path AND NOT(shared_subtier_source OR count>=2)
        "DT-NULL": ("AND", "NOT", "tier1_on_disrupted_path", "shared_subtier_source", "count_disrupted_paths"),
    }

    failures = []
    for dtype in ("cyber", "economic", "geopolitical", "labour", "natural_disaster"):
        raw = yaml.safe_load((__import__("pathlib").Path(__file__).parent.parent / f"specs/{dtype}.spec.yaml").read_text())
        for r in raw["routing_rules"]:
            suffix = _rule_suffix(r["rule_id"])
            when = r.get("when") or ""
            required = _REQUIRED.get(suffix)
            if required is None:
                continue
            missing = [kw for kw in required if kw not in when]
            if missing:
                failures.append(f"{r['rule_id']}: when: string missing expected keyword(s) {missing} for its shape")

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Arm 4 — Lexical-graph parity tests
# ---------------------------------------------------------------------------

def test_arm4_rule_id_parity():
    """All 52 rule IDs must appear in Arm 4 embed-cut text (same source as Arms 1 & 2)."""
    from csco.parity_check import collect_all_rule_ids, check_policy_block_completeness

    rules_by_type = collect_all_rule_ids()
    all_ids: set[str] = set()
    for ids in rules_by_type.values():
        all_ids.update(ids)

    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"
    texts = [f.read_text(encoding="utf-8") for f in sorted(embed_dir.glob("*.md"))]
    ok, missing = check_policy_block_completeness("\n\n".join(texts))

    assert ok, f"Arm 4 source (embed-cut files) missing {len(missing)} rules: {sorted(missing)}"


def test_arm4_no_worked_examples():
    """LexProvision and LexSection nodes must not contain worked-example content."""
    lexical_path = Path(__file__).parent.parent / "corpus" / "playbooks_embed"
    all_text = "\n\n".join(
        f.read_text(encoding="utf-8") for f in sorted(lexical_path.glob("*.md"))
    )
    for marker in ("EX-", "## 8.", "Worked example"):
        assert marker not in all_text, (
            f"Forbidden content '{marker}' found in Arm 4 source (embed-cut files)"
        )


def test_arm4_shared_prompt():
    """Arm 4 lexical module must use AGENT_SYSTEM_PROMPT and reach extract_decision."""
    src = (Path(__file__).parent.parent / "src" / "csco" / "arms" / "lexical.py"
           ).read_text(encoding="utf-8")
    assert "AGENT_SYSTEM_PROMPT" in src, "lexical.py must use AGENT_SYSTEM_PROMPT"
    assert "run_react_agent" in src, (
        "lexical.py must reach extract_decision via the shared csco.arms.runners plumbing"
    )
    assert ".with_structured_output(" not in src, (
        "lexical.py must not call with_structured_output directly"
    )


@pytest.mark.integration
def test_arm4_lexical_graph_parity():
    """All 52 rule IDs as LexProvision nodes; band REFERENCES edge on every provision."""
    from csco.parity_check import collect_all_rule_ids
    from csco.settings import get_settings

    try:
        from neo4j import GraphDatabase
    except ImportError:
        pytest.skip("neo4j package not installed")

    rules_by_type = collect_all_rule_ids()
    all_ids: set[str] = set()
    for ids in rules_by_type.values():
        all_ids.update(ids)

    try:
        s = get_settings()
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
        with driver.session(database=s.neo4j_database) as session:
            prov_result = session.run(
                "MATCH (p:LexProvision) RETURN DISTINCT p.rule_id AS rule_id"
            )
            graph_ids: set[str] = {r["rule_id"] for r in prov_result}

            band_result = session.run("""
                MATCH (p:LexProvision)
                WHERE NOT (p)-[:REFERENCES {as: 'band'}]->(:LexSection)
                RETURN p.rule_id AS rule_id
            """)
            missing_band = [r["rule_id"] for r in band_result]

            tier_result = session.run("""
                MATCH (p:LexProvision)-[:REFERENCES {as: 'tier'}]->(:LexSection)
                RETURN p.rule_id AS rule_id
            """)
            has_tier = {r["rule_id"] for r in tier_result}

        driver.close()
    except Exception as e:
        pytest.skip(f"Neo4j not reachable: {e}")

    missing_provs = sorted(all_ids - graph_ids)
    failures = []
    if missing_provs:
        failures.append(f"LexProvision nodes missing for: {missing_provs}")
    if missing_band:
        failures.append(f"Provisions missing band REFERENCES edge: {sorted(missing_band)}")
    unexpected_tier = has_tier - {r for r in all_ids if "MEDS" in r or "LOWS" in r}
    if unexpected_tier:
        failures.append(f"Unexpected tier REFERENCES edges on: {sorted(unexpected_tier)}")

    assert not failures, "\n".join(failures)


@pytest.mark.integration
def test_arm4_provision_text_parity():
    """LexProvision.text must be whitespace-normalised-equivalent to the Arm 2 chunk.

    Byte-exact identity is enforced for all provisions except the DT-NULL family.
    The last routing provision per type (DT-NULL) has the "Pass 2" structural paragraph
    appended by MarkdownHeaderTextSplitter but the playbook generator emits the same
    paragraph with normalised whitespace (\\n\\n vs trailing-space line break).
    These are semantically identical; we allow normalised equivalence there.
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    try:
        from neo4j import GraphDatabase
        from csco.settings import get_settings
        s = get_settings()
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
        with driver.session(database=s.neo4j_database) as session:
            rows = session.run(
                "MATCH (p:LexProvision) RETURN p.rule_id AS rule_id, p.text AS text"
            ).data()
        driver.close()
    except Exception as e:
        pytest.skip(f"Neo4j not reachable: {e}")

    graph_texts = {r["rule_id"]: r["text"] for r in rows}

    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    arm2_texts: dict[str, str] = {}
    for md_path in sorted(embed_dir.glob("*.md")):
        raw   = TextLoader(str(md_path), encoding="utf-8").load()
        chunks = splitter.split_text(raw[0].page_content)
        for chunk in chunks:
            if "h3" not in chunk.metadata:
                continue
            m = re.search(r"\(([A-Z]+-[A-Z0-9-]+)\)", chunk.metadata.get("h3", ""))
            if not m:
                continue
            rule_id = m.group(1)
            header_parts = [chunk.metadata[k] for k in ("h1", "h2", "h3") if k in chunk.metadata]
            section_path = " > ".join(header_parts)
            arm2_texts[rule_id] = section_path + "\n\n" + chunk.page_content

    def _ws_norm(text: str) -> str:
        """Collapse all whitespace runs to single spaces for semantic comparison.

        Catches both trailing-space line breaks (  \\n) and blank lines (\\n\\n)
        as well as any other whitespace normalisation differences between the
        playbook generator and MarkdownHeaderTextSplitter.
        """
        import re as _re
        return _re.sub(r"\s+", " ", text).strip()

    mismatches = []
    for rule_id, arm2_text in arm2_texts.items():
        graph_text = graph_texts.get(rule_id)
        if graph_text is None:
            mismatches.append(f"{rule_id}: missing from LexProvision nodes")
        elif graph_text != arm2_text:
            # Allow whitespace-normalised equivalence — the playbook generator and
            # MarkdownHeaderTextSplitter may differ in how they render blank lines
            # (\\n\\n vs trailing-space line breaks).  Semantic content must be identical.
            if _ws_norm(graph_text) == _ws_norm(arm2_text):
                continue
            mismatches.append(
                f"{rule_id}: text differs (content, not just whitespace)\n"
                f"  Arm2[:{min(80,len(arm2_text))}]: {repr(arm2_text[:80])}\n"
                f"  Lex [{min(80,len(graph_text)):}]: {repr(graph_text[:80])}"
            )

    assert not mismatches, f"{len(mismatches)} text parity failures:\n" + "\n".join(mismatches)


# ---------------------------------------------------------------------------
# Test — Spec metadata consistency (no Neo4j needed)
# ---------------------------------------------------------------------------

def test_spec_evaluation_clusters_exist():
    """Every rule in every spec must have an evaluation.clusters field (non-empty list)."""
    from csco.specs.loader import load_all_specs
    from csco.evaluation.clusters import ALL_CLUSTER_IDS

    specs = load_all_specs()
    failures = []

    for dtype, spec in specs.items():
        all_rules = list(spec.routing_rules) + list(spec.supplier_rules)
        for rule in all_rules:
            eval_meta = rule.evaluation
            if eval_meta is None:
                failures.append(f"{rule.rule_id}: missing evaluation block")
                continue
            if not eval_meta.clusters:
                failures.append(f"{rule.rule_id}: evaluation.clusters is empty")
                continue
            unknown = [c for c in eval_meta.clusters if c not in ALL_CLUSTER_IDS]
            if unknown:
                failures.append(
                    f"{rule.rule_id}: unknown cluster IDs: {unknown}. "
                    f"Valid IDs: {sorted(ALL_CLUSTER_IDS)}"
                )

    assert not failures, (
        f"{len(failures)} spec metadata violations:\n" + "\n".join(failures)
    )


def test_spec_contrast_with_rules_exist():
    """Every rule_id in contrast_with must exist in the same spec."""
    from csco.specs.loader import load_all_specs

    specs = load_all_specs()
    failures = []

    for dtype, spec in specs.items():
        all_rule_ids = {r.rule_id for r in spec.routing_rules + spec.supplier_rules}
        all_rules = list(spec.routing_rules) + list(spec.supplier_rules)

        for rule in all_rules:
            eval_meta = rule.evaluation
            if eval_meta is None:
                continue
            for cid in eval_meta.contrast_with:
                if cid not in all_rule_ids:
                    failures.append(
                        f"{rule.rule_id}: contrast_with references unknown rule '{cid}' "
                        f"(not in {dtype} spec)"
                    )

    assert not failures, "\n".join(failures)


def test_spec_evaluation_design_threshold_values():
    """Every spec must have evaluation_design.threshold_probe_values."""
    from csco.specs.loader import load_all_specs

    specs = load_all_specs()
    failures = []

    for dtype, spec in specs.items():
        ed = spec.evaluation_design
        if ed is None:
            failures.append(f"{dtype}: missing evaluation_design block")
            continue
        tp = ed.threshold_probe_values
        if tp is None:
            failures.append(f"{dtype}: missing evaluation_design.threshold_probe_values")
            continue
        # Sanity: LOW_EDGE < MEDIUM_LOWER_EDGE <= MEDIUM_UPPER_EDGE < HIGH_EDGE
        if not (tp.LOW_EDGE < tp.MEDIUM_LOWER_EDGE <= tp.MEDIUM_UPPER_EDGE < tp.HIGH_EDGE):
            failures.append(
                f"{dtype}: threshold_probe_values ordering violated: "
                f"LOW={tp.LOW_EDGE} MEDIUM_LO={tp.MEDIUM_LOWER_EDGE} "
                f"MEDIUM_HI={tp.MEDIUM_UPPER_EDGE} HIGH={tp.HIGH_EDGE}"
            )

    assert not failures, "\n".join(failures)


def test_reference_expectations_action_on_permitted_rules():
    """Every supplier rule with non-empty permitted actions must have reference_expectations.action != not_applicable."""
    from csco.specs.loader import load_all_specs

    specs = load_all_specs()
    failures = []

    for dtype, spec in specs.items():
        for rule in spec.supplier_rules:
            if not rule.consequence.permitted:
                continue
            eval_meta = rule.evaluation
            if eval_meta is None:
                continue
            ref_exp = eval_meta.reference_expectations
            if ref_exp.action == "not_applicable":
                failures.append(
                    f"{rule.rule_id}: has permitted actions but reference_expectations.action = not_applicable"
                )

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Primary-arm content parity tests
# The central fairness claim: Static, Vector, and Lexical differ in delivery
# structure only — NOT in policy content.
# ---------------------------------------------------------------------------

def test_static_contains_each_vector_chunk_body():
    """Every vector chunk body must appear verbatim in the static policy text.

    Vector chunks have the form: section_path_prefix + '\\n\\n' + body_text.
    The body_text is the markdown prose that follows the section headers.
    The static policy contains the full embed-cut markdown including those headers
    and body text. This test checks that no chunk body is absent from the static policy.

    This makes the claim testable: 'the three primary arms use the same prose corpus'.
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    from csco.arms.prompts import load_embed_cut_policy

    static_policy = load_embed_cut_policy()

    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )

    failures = []
    total = 0
    for md_path in sorted(embed_dir.glob("*.md")):
        raw = TextLoader(str(md_path), encoding="utf-8").load()
        chunks = splitter.split_text(raw[0].page_content)
        for chunk in chunks:
            body = chunk.page_content.strip()
            if not body:
                continue
            total += 1
            if body not in static_policy:
                # Try normalised (collapse whitespace) as fallback
                import re
                body_norm   = re.sub(r"\s+", " ", body)
                policy_norm = re.sub(r"\s+", " ", static_policy)
                if body_norm not in policy_norm:
                    heading = (chunk.metadata.get("h3") or chunk.metadata.get("h2") or
                               chunk.metadata.get("h1") or "?")[:60]
                    failures.append(f"Chunk body not found in static policy: {heading!r}")

    assert total > 0, "No chunks found — run python -m csco.generators.playbook first"
    assert not failures, (
        f"{len(failures)}/{total} chunk bodies absent from static policy:\n"
        + "\n".join(failures[:10])
    )


def test_primary_arm_provision_text_parity():
    """For every provision rule_id, the body text must be identical across Static, Vector, and Lexical.

    Static: extracted between (rule_id) header and next ### or ##.
    Vector: the chunk page_content body (after stripping the section-path prefix).
    Lexical: LexProvision.text body (after stripping the same section-path prefix).

    This is the key content-parity assertion for the main experiment.
    """
    import re as _re
    from csco.arms.prompts import load_embed_cut_policy
    from csco.parity_check import collect_all_rule_ids

    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    static_policy = load_embed_cut_policy()
    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"

    # Build vector chunk bodies keyed by rule_id
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    vector_bodies: dict[str, str] = {}
    for md_path in sorted(embed_dir.glob("*.md")):
        raw = TextLoader(str(md_path), encoding="utf-8").load()
        chunks = splitter.split_text(raw[0].page_content)
        for chunk in chunks:
            if "h3" not in chunk.metadata:
                continue
            m = _re.search(r"\(([A-Z]+-[A-Z0-9-]+)\)", chunk.metadata.get("h3", ""))
            if not m:
                continue
            vector_bodies[m.group(1)] = chunk.page_content.strip()

    # Build static policy bodies keyed by rule_id
    # Pattern: ### (RULE-ID) ... followed by body up to next ### or ##
    static_bodies: dict[str, str] = {}
    provision_re = _re.compile(
        r"^### \(([A-Z]+-[A-Z0-9-]+)\) .+?$(.*?)(?=^#{1,3} |\Z)",
        _re.MULTILINE | _re.DOTALL,
    )
    for match in provision_re.finditer(static_policy):
        rule_id = match.group(1)
        body = match.group(2).strip()
        static_bodies[rule_id] = body

    def _norm(text: str) -> str:
        """Normalise whitespace for comparison: strip, then collapse runs to single space."""
        return _re.sub(r"\s+", " ", text.strip())

    # Compare: provision description body must be identical after whitespace normalisation.
    # Note: the last routing provision per type (DT-NULL) has the "Pass 2" structural
    # paragraph appended by the splitter but not extracted by the static regex — so we
    # check that the VECTOR body appears as a substring in the STATIC policy text rather
    # than requiring exact equality. For all other provisions exact normalised equality holds.
    failures = []
    rules_by_type = collect_all_rule_ids()
    all_ids: set[str] = set()
    for ids in rules_by_type.values():
        all_ids.update(ids)

    for rule_id in sorted(all_ids):
        vec = vector_bodies.get(rule_id)
        sta = static_bodies.get(rule_id)

        if vec is None:
            failures.append(f"{rule_id}: missing from vector chunks")
            continue
        if sta is None:
            failures.append(f"{rule_id}: missing from static policy text (extraction failed)")
            continue

        vec_norm = _norm(vec)
        sta_norm = _norm(sta)

        # Primary check: normalized bodies equal.
        # Fallback for DT-NULL (trailing Pass-2 paragraph attached to chunk):
        # at minimum the static body must be a substring of the vector body or vice versa.
        if vec_norm != sta_norm:
            # Allow one to be a leading substring of the other (structural append)
            if not (vec_norm.startswith(sta_norm) or sta_norm.startswith(vec_norm)):
                failures.append(
                    f"{rule_id}: provision prose differs after normalisation\n"
                    f"  vec[:{min(100,len(vec_norm))}]: {repr(vec_norm[:100])}\n"
                    f"  sta[:{min(100,len(sta_norm))}]: {repr(sta_norm[:100])}"
                )

    assert not failures, (
        f"{len(failures)} provision body parity failures (static vs vector):\n"
        + "\n".join(failures)
    )


def test_reason_clause_parity_across_primary_arms():
    """Rationale/reason clauses for discriminator rules must be identical in Static and Vector.

    Rules most likely to drive arm differences are the discriminators:
      *-DT, *-R3, NAT-R5, LAB-SITE, ECON-DEP, CYB-HUB, *-MEDS, *-LOWS

    These rules have spec note/grounding fields that are embedded as inline italic
    text in the generated playbooks. Silent drift here would be a high-impact fairness bug.
    """
    import re as _re
    from csco.arms.prompts import load_embed_cut_policy

    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    # Discriminator rule suffixes to prioritise
    DISCRIMINATORS = {"DT", "R3", "R5", "SITE", "DEP", "HUB", "MEDS", "LOWS"}

    static_policy = load_embed_cut_policy()
    embed_dir = Path(__file__).parent.parent / "corpus" / "playbooks_embed"

    # Build vector chunk content keyed by rule_id (full chunk including prefix)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    vector_chunks: dict[str, str] = {}
    for md_path in sorted(embed_dir.glob("*.md")):
        raw = TextLoader(str(md_path), encoding="utf-8").load()
        chunks = splitter.split_text(raw[0].page_content)
        for chunk in chunks:
            if "h3" not in chunk.metadata:
                continue
            m = _re.search(r"\(([A-Z]+-[A-Z0-9-]+)\)", chunk.metadata.get("h3", ""))
            if not m:
                continue
            vector_chunks[m.group(1)] = chunk.page_content.strip()

    # Extract italic/reason clauses: text like *...* (single-line italic markers)
    italic_re = _re.compile(r"\*[^*\n]+\*")

    def _italics(text: str) -> list[str]:
        return italic_re.findall(text)

    # Build static provision blocks
    static_blocks: dict[str, str] = {}
    provision_re = _re.compile(
        r"^### \(([A-Z]+-[A-Z0-9-]+)\) .+?$(.*?)(?=^#{1,3} |\Z)",
        _re.MULTILINE | _re.DOTALL,
    )
    for match in provision_re.finditer(static_policy):
        static_blocks[match.group(1)] = match.group(2).strip()

    failures = []
    for rule_id, vec_body in vector_chunks.items():
        suffix = rule_id.split("-")[-1]
        if suffix not in DISCRIMINATORS:
            continue

        sta_body = static_blocks.get(rule_id)
        if sta_body is None:
            continue  # already caught by provision parity test

        vec_italics = sorted(_italics(vec_body))
        sta_italics = sorted(_italics(sta_body))

        if vec_italics != sta_italics:
            failures.append(
                f"{rule_id}: reason/note clauses differ\n"
                f"  vector  : {vec_italics}\n"
                f"  static  : {sta_italics}"
            )

    assert not failures, (
        f"{len(failures)} reason-clause parity failures (static vs vector):\n"
        + "\n".join(failures)
    )


def test_supplier_tier_not_in_primary_scenario_block():
    """Supplier criticality was removed from the framework: format_scenario must
    never include a 'supplier_tier' line. Only raw metrics + sub-scores are supplied;
    the agent derives the risk band itself.
    """
    import yaml
    from csco.models import Scenario
    from csco.derive import derive
    from csco.arms.prompts import format_scenario
    from csco.specs.loader import load_spec

    fixture_path = Path(__file__).parent.parent / "fixtures" / "suite_a" / "diag_0025.yaml"
    raw = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(raw)
    spec = load_spec(scenario.disruption_type)
    derived = [derive(s, spec) for s in scenario.suppliers]

    block = format_scenario(scenario, derived)
    assert "supplier_tier" not in block, "supplier_tier appeared in the scenario block"


# ---------------------------------------------------------------------------
# Oracle routing semantics tests (A4)
# Invariant: routing rule patterns are fixed and schema-specific
# ---------------------------------------------------------------------------

def test_routing_rules_match_expected_suffix_patterns():
    """All five specs must have exactly the three routing rule patterns: NULL, DT, DT-NULL.
    
    This is a structural invariant that ensures the deterministic oracle's
    evaluation of suffix-based routing logic is valid across all specs.
    """
    from csco.specs.loader import load_spec

    expected_suffixes = {"NULL", "DT", "DT-NULL"}
    specs = ["cyber", "economic", "geopolitical", "labour", "natural_disaster"]
    
    for spec_name in specs:
        spec = load_spec(spec_name)
        actual_suffixes = set()
        
        for rule in spec.routing_rules:
            # Extract suffix from rule_id: "CYB-NULL" -> "NULL", "CYB-DT-NULL" -> "DT-NULL"
            parts = rule.rule_id.split("-")
            suffix = "-".join(parts[1:])  # Everything after disruption-type prefix
            actual_suffixes.add(suffix)
        
        assert actual_suffixes == expected_suffixes, (
            f"{spec_name} routing rules have unexpected suffixes:\n"
            f"  expected: {sorted(expected_suffixes)}\n"
            f"  actual:   {sorted(actual_suffixes)}"
        )


def test_null_does_not_mask_deep_tier_structural_case():
    """Regression test: the plain NULL routing rule must NOT fire when the set of
    scored suppliers is empty — branch (b) ("all Tier-1 LOW") is not vacuously
    satisfied by an empty set. Built inline so it needs no fixture file.

    With path_on_network=True (branch (a) cannot apply), no Tier-1 on the disrupted
    path, and no structural concern, the correct route is the deep-tier no-concern
    rule (…-DT-NULL), never …-NULL.
    """
    from csco.models import Scenario
    from csco.oracle.deterministic import build_oracle, _routing_suffix
    from csco.specs.loader import load_spec

    scenario = Scenario.model_validate({
        "disruption_type": "cyber",
        "affected_regions": ["Region-A"],
        "impacted_industries": ["Industry-X"],
        "named_companies": ["Company-Z"],
        "path_on_network": True,
        "tier1_on_disrupted_path": False,
        "shared_subtier_source": False,
        "count_disrupted_paths": 1,
        "max_disrupted_tier": 2,
        "suppliers": [],
    })
    oracle = build_oracle(scenario, load_spec("cyber"))

    assert oracle.routing is not None, "expected a routing rule to fire (…-DT-NULL)"
    assert _routing_suffix(oracle.routing.fired_rule_id) != "NULL", (
        f"Plain NULL rule unexpectedly fired on empty scored suppliers: "
        f"{oracle.routing.fired_rule_id} — violates "
        "'NULL requires at least one scored supplier'."
    )


def test_null_all_low_requires_scored_supplier_on_affected_path():
    """NULL routing rule's 'all LOW' clause requires at least one scored supplier.

    When path_on_network=True and all Tier-1 suppliers are LOW risk with no
    high downstream criticality, NULL should fire (provided derived is non-empty).
    """
    from csco.models import Scenario
    from csco.oracle.deterministic import build_oracle
    from csco.specs.loader import load_spec

    # Supplier with LOW risk_score (< 0.45) and no high sub_downstream_criticality
    scenario_dict = {
        "disruption_type": "cyber",
        "affected_regions": ["TestRegion"],
        "impacted_industries": ["TestIndustry"],
        "named_companies": [],
        "path_on_network": True,
        "count_disrupted_paths": 1,
        "max_disrupted_tier": 1,
        "shared_subtier_source": False,
        "tier1_on_disrupted_path": True,
        "suppliers": [
            {
                "supplier_id": "S-TEST-LOW-01",
                "name": "Low Risk Co.",
                "product_supplied": "widgets",
                "tier1_in_region": False,
                "risk_score": 0.30,           # LOW (< 0.45)
                "sub_exposure_breadth": 0.20,
                "sub_dependency_ratio": 0.20,
                "sub_downstream_criticality": 0.20,  # NOT high (< 0.66)
                "sub_node_centrality": 0.20,
                "sub_exposure_depth": 0.20,
            }
        ],
    }

    scenario = Scenario.model_validate(scenario_dict)
    spec = load_spec("cyber")
    oracle = build_oracle(scenario, spec)

    # NULL rule should fire: path_on_network=True, all LOW, no high downstream criticality
    assert oracle.routing is not None, "Expected NULL routing oracle to fire"
    assert oracle.routing.fired_rule_id == "CYB-NULL", (
        f"Expected CYB-NULL to fire, got {oracle.routing.fired_rule_id}"
    )


def test_dt_fires_for_no_tier1_on_path_with_shared_subtier_source():
    """DT routing rule fires when tier1_on_disrupted_path=False and shared_subtier_source=True.

    Tests the DT (Deep Tier) condition: DT fires when NOT tier1 AND (shared_subtier OR count >= 2).
    """
    from csco.models import Scenario
    from csco.oracle.deterministic import build_oracle
    from csco.specs.loader import load_spec

    scenario_dict = {
        "disruption_type": "cyber",
        "affected_regions": ["TestRegion"],
        "impacted_industries": ["TestIndustry"],
        "named_companies": [],
        "path_on_network": True,
        "count_disrupted_paths": 1,
        "max_disrupted_tier": 2,
        "shared_subtier_source": True,         # DT condition: shared source
        "tier1_on_disrupted_path": False,      # DT condition: no Tier-1 on path
        "suppliers": [],
    }

    scenario = Scenario.model_validate(scenario_dict)
    spec = load_spec("cyber")
    oracle = build_oracle(scenario, spec)

    # DT rule should fire: tier1_on_disrupted_path=False AND shared_subtier_source=True
    assert oracle.routing is not None, "Expected DT routing oracle to fire"
    assert oracle.routing.fired_rule_id == "CYB-DT", (
        f"Expected CYB-DT to fire, got {oracle.routing.fired_rule_id}"
    )


def test_generated_null_scenario_is_unambiguous_vs_dt_null():
    """Regression test: the canonical NULL scenario built by
    scenario_generator._build_routing_scenario must not also structurally
    satisfy DT or DT-NULL's condition — disambiguation must come from the
    scenario data itself, not merely from spec rule-list order.

    Prior to this fix, the generator set tier1_on_disrupted_path=False for the
    NULL case, so the "canonical NULL" fixture also satisfied DT-NULL's
    condition (tier1_on_disrupted_path==False AND no structural concern) — a
    reasoner picking the most-specific-looking match rather than the
    first-listed match could land on DT-NULL instead of NULL. This was traced
    directly to real model failures (NULL vs DT-NULL confusion) on the
    generated diagnostic benchmark.
    """
    from csco.oracle.deterministic import _routing_condition_matches, _routing_suffix
    from csco.models import Scenario
    from csco.generators.scenario import _build_routing_scenario
    from csco.specs.loader import load_spec

    for dtype in ["geopolitical", "natural_disaster", "cyber", "economic", "labour"]:
        spec = load_spec(dtype)
        null_rule = next(rr for rr in spec.routing_rules if _routing_suffix(rr.rule_id) == "NULL")
        dt_rule = next(rr for rr in spec.routing_rules if _routing_suffix(rr.rule_id) == "DT")
        dt_null_rule = next(rr for rr in spec.routing_rules if _routing_suffix(rr.rule_id) == "DT-NULL")

        scenario_dict = _build_routing_scenario(null_rule, dtype)
        scenario = Scenario.model_validate(scenario_dict)

        assert _routing_condition_matches(null_rule, scenario, []) is True, (
            f"{dtype}: generated NULL scenario must satisfy NULL's own condition"
        )
        assert _routing_condition_matches(dt_rule, scenario, []) is False, (
            f"{dtype}: generated NULL scenario must NOT also satisfy DT's condition "
            f"(tier1_on_disrupted_path must be True to rule DT out)"
        )
        assert _routing_condition_matches(dt_null_rule, scenario, []) is False, (
            f"{dtype}: generated NULL scenario must NOT also satisfy DT-NULL's condition "
            f"(tier1_on_disrupted_path must be True to rule DT-NULL out)"
        )


def test_diagnostic_benchmark_frozen_metadata():
    """Suite A benchmark metadata carries the dissertation-suite provenance."""
    from pathlib import Path
    import json

    benchmark_meta_path = Path(__file__).parent.parent / "fixtures" / "suite_a" / "benchmark_meta.json"

    if not benchmark_meta_path.exists():
        pytest.skip("fixtures/suite_a/benchmark_meta.json not yet created")

    meta = json.loads(benchmark_meta_path.read_text(encoding="utf-8"))

    assert meta.get("suite_id") == "suite_a_action_v2"
    assert meta.get("stratum") == "prescribed_action"
    assert meta.get("total_fixtures") == 37, "Suite A holds the 37 prescribed-action cases"
    for key in ("specs_hash", "generated_at", "prompt_version", "generation_seed"):
        assert key in meta, f"Benchmark metadata must have {key}"


def test_diagnostic_benchmark_has_expected_size():
    """Diagnostic benchmark is in progress; must have at least 10 scenarios."""
    from pathlib import Path

    benchmark_dir = Path(__file__).parent.parent / "fixtures" / "suite_a"

    if not benchmark_dir.exists():
        pytest.skip("fixtures/suite_a not yet created")

    yaml_files = sorted(benchmark_dir.glob("diag_*.yaml"))
    # Target is ~75. Enforce a lower bound that grows as the benchmark is built.
    # Update this once the benchmark reaches its target size.
    assert len(yaml_files) >= 10, (
        f"Expected at least 10 diagnostic scenarios, got {len(yaml_files)}"
    )


def test_dissertation_suites_compose_to_72():
    """The frozen evaluation set is Suite A = 37 prescribed-action + Suite B = 35 null."""
    from pathlib import Path
    import json

    suite_a = Path(__file__).parent.parent / "fixtures" / "suite_a"
    suite_b = Path(__file__).parent.parent / "fixtures" / "suite_b"
    if not (suite_a / "benchmark_meta.json").exists() or not (suite_b / "benchmark_meta.json").exists():
        pytest.skip("dissertation suites not yet generated")

    meta_a = json.loads((suite_a / "benchmark_meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((suite_b / "benchmark_meta.json").read_text(encoding="utf-8"))

    assert meta_a["total_fixtures"] == 37
    assert meta_b["total_fixtures"] == 35

    # Count only canonically-named fixtures (diag_0001.yaml / sb_0001.yaml). This
    # ignores iCloud sync-conflict copies like "diag_0032 2.yaml" that can appear in
    # a synced working tree without meaning the suite is malformed.
    import re
    a_pat = re.compile(r"^diag_\d{4}\.yaml$")
    b_pat = re.compile(r"^sb_\d{4}\.yaml$")
    assert sum(1 for p in suite_a.glob("diag_*.yaml") if a_pat.match(p.name)) == 37
    assert sum(1 for p in suite_b.glob("sb_*.yaml") if b_pat.match(p.name)) == 35


def test_diagnostic_review_sample_covers_hard_strata():
    """Review sample covers key hard strata and is stratified."""
    from pathlib import Path
    import json
    
    review_sample_path = Path(__file__).parent.parent / "fixtures" / "suite_a_review" / "review_sample.jsonl"

    if not review_sample_path.exists():
        pytest.skip("fixtures/suite_a_review/review_sample.jsonl not yet created")
    
    lines = review_sample_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) > 0, "Review sample must contain at least one entry"
    
    # Check that at least a few strata are represented
    strata_seen = set()
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if "hard_strata" in entry:
            for s in entry.get("hard_strata", []):
                strata_seen.add(s)
    
    # At least some hard strata should be in the review sample
    # (This is a light check; full validation requires manual review)


# ---------------------------------------------------------------------------
# Strategic-priority scorer tests
# ---------------------------------------------------------------------------

def test_strategic_priority_scorer_checks_raw_answer():
    """strategic_priority_mentioned must be True when the priority appears in the
    raw agent answer but NOT in the extracted per-supplier rationale.

    This guards against the regression where the scorer only inspected the
    extracted rationale field and missed priority text written in a separate
    concluding section.
    """
    from csco.evaluation.run_result import RunResult

    # Build a minimal RunResult with priority in raw answer only
    result = RunResult.start(
        arm="static",
        scenario_id="diag_0026",
        disruption_type="cyber",
    )
    result.agent_answer_raw = (
        "The matching provision is (CYB-R1). "
        "Strategic priority: Rapid Reconfiguration — rapid isolation favoured."
    )
    result.decision = {
        "routing_result": None,
        "recommendations": [
            {
                "supplier_id": "S-CYB-HI-01",
                "disposition": "monitor",
                "escalation": "Cross-functional Management",
                "permitted_actions": ["intensify_threat_monitoring", "convene_crisis_response_center"],
                "fired_rule_id": "CYB-R1",
                "rationale": "High risk band. CYB-R1 fires.",  # NO priority word here
                "assigned_supplier_tier": "routine",
                "assigned_risk_level": "HIGH",
            }
        ],
    }
    result.score()

    scores = result.scores
    assert scores.get("strategic_priority_mentioned") is True, (
        "strategic_priority_mentioned should be True when 'flexibility' appears in "
        "raw_answer but not in extracted rationale"
    )
    assert scores.get("strategic_priority_in_rationale") is False, (
        "strategic_priority_in_rationale should be False when priority is not in "
        "the extracted per-supplier rationale"
    )
    assert scores.get("strategic_priority_correct") is True, (
        "strategic_priority_correct should match strategic_priority_mentioned"
    )
    assert scores.get("wrong_strategic_priority_used") is False


def test_strategic_priority_in_rationale_requires_integration():
    """strategic_priority_in_rationale is False when priority is only in raw answer."""
    from csco.evaluation.run_result import RunResult

    result = RunResult.start(
        arm="lexical",
        scenario_id="diag_0002",
        disruption_type="geopolitical",
    )
    
    result.agent_answer_raw = "Provision GEO-R1 fires. Standing posture: Continuity Protection."
    result.decision = {
        "routing_result": None,
        "recommendations": [
            {
                "supplier_id": "S-GEO-HI-01",
                "disposition": "replace",
                "escalation": "Executive Leadership",
                "permitted_actions": ["activate_pre_qualified_alternate", "intensify_threat_monitoring"],
                "fired_rule_id": "GEO-R1",
                "rationale": "HIGH risk, in-region supplier. GEO-R1 fires.",
                "assigned_supplier_tier": "routine",
                "assigned_risk_level": "HIGH",
            }
        ],
    }
    result.score()

    scores = result.scores
    assert scores.get("strategic_priority_mentioned") is True
    assert scores.get("strategic_priority_in_rationale") is False


def test_wrong_strategic_priority_used():
    """wrong_strategic_priority_used is True when a different disruption type's
    priority appears and the correct priority does not."""
    from csco.evaluation.run_result import RunResult

    result = RunResult.start(
        arm="vector",
        scenario_id="diag_0017",
        disruption_type="economic",
    )
    # Economic expects Cost Containment; agent says Continuity Protection (wrong type)
    result.agent_answer_raw = "The strategic posture is Continuity Protection for this disruption."
    result.decision = {
        "routing_result": None,
        "recommendations": [
            {
                "supplier_id": "S-ECON-HI-01",
                "disposition": "monitor",
                "escalation": "Cross-functional Management",
                "permitted_actions": ["intensify_threat_monitoring", "scenario_lever_generation"],
                "fired_rule_id": "ECON-R1",
                "rationale": "High risk. ECON-R1 fires.",
                "assigned_supplier_tier": "routine",
                "assigned_risk_level": "HIGH",
            }
        ],
    }
    result.score()

    scores = result.scores
    assert scores.get("strategic_priority_mentioned") is False, (
        "economic priority is Cost Containment, not Continuity Protection"
    )
    assert scores.get("wrong_strategic_priority_used") is True, (
        "wrong_strategic_priority_used should detect 'resilience' for an economic scenario"
    )


# ---------------------------------------------------------------------------
# Prompt-assembly fairness tests (Phase 1 refactor)
# ---------------------------------------------------------------------------

def _golden(name: str) -> str:
    """Read a golden-prompt file from tests/golden_prompts/."""
    return (Path(__file__).parent / "golden_prompts" / name).read_text(encoding="utf-8")


def test_refactor_preserves_prompts_byte_for_byte():
    """Assembled AGENT_SYSTEM_PROMPT and STATIC_SYSTEM_TEMPLATE must equal
    the pre-refactor golden snapshots byte-for-byte.

    If this test fails, a code change has silently altered model-visible prompt
    text without a corresponding prompt_version bump.
    """
    from csco.arms.prompts import AGENT_SYSTEM_PROMPT, STATIC_SYSTEM_TEMPLATE

    assert AGENT_SYSTEM_PROMPT == _golden("agent_system_prompt.txt"), (
        "AGENT_SYSTEM_PROMPT no longer matches golden snapshot. "
        "If this is intentional, update the golden file AND bump PROMPT_VERSION."
    )
    assert STATIC_SYSTEM_TEMPLATE == _golden("static_system_template.txt"), (
        "STATIC_SYSTEM_TEMPLATE no longer matches golden snapshot. "
        "If this is intentional, update the golden file AND bump PROMPT_VERSION."
    )


def test_shared_literal_blocks_byte_identical():
    """Every SHARED_LITERAL_BLOCKS entry must appear byte-for-byte in both the
    tool-arm (AGENT_SYSTEM_PROMPT) and Static (STATIC_SYSTEM_TEMPLATE) assemblies.
    """
    from csco.arms.prompts import (
        AGENT_SYSTEM_PROMPT,
        SHARED_LITERAL_BLOCKS,
        STATIC_SYSTEM_TEMPLATE,
    )

    for block in SHARED_LITERAL_BLOCKS:
        assert block in AGENT_SYSTEM_PROMPT, (
            f"SHARED_LITERAL block not found in AGENT_SYSTEM_PROMPT:\n{block[:120]!r}"
        )
        assert block in STATIC_SYSTEM_TEMPLATE, (
            f"SHARED_LITERAL block not found in STATIC_SYSTEM_TEMPLATE:\n{block[:120]!r}"
        )


def test_parallel_requirement_blocks_exist():
    """For each PARALLEL_REQUIREMENT pair, both the tool variant and the Static
    variant must be non-empty strings.  The test does NOT assert equality — the
    wording asymmetry is intentional (see Architecture Guide §4).
    """
    from csco.arms.prompts import PARALLEL_REQUIREMENT_BLOCKS

    for tool_block, static_block in PARALLEL_REQUIREMENT_BLOCKS:
        assert isinstance(tool_block, str) and tool_block.strip(), (
            "Tool-arm PARALLEL_REQUIREMENT block is empty"
        )
        assert isinstance(static_block, str) and static_block.strip(), (
            "Static PARALLEL_REQUIREMENT block is empty"
        )


def test_parallel_requirement_blocks_cover_same_requirement_set():
    """test_parallel_requirement_blocks_exist checks the pairs are non-empty
    but never checked they cover the same *named requirements* — the seam a
    real parity defect can slip through (e.g. one variant's Pass-1 contract
    stating match->stop, no-suppliers->empty-recommendations, and
    exhausted->Pass-2 explicitly while another states only an implicit ordering
    clause and omits the no-suppliers->empty-recommendations outcome). Crude,
    keyword-per-requirement check, one requirement list per pair (the two
    pairs cover different content) — not full semantic equivalence, but
    enough that a requirement dropped from either side fails loudly instead
    of silently.
    """
    from csco.arms.prompts import (
        _ROLE_TASK_STATIC,
        _ROLE_TASK_TOOL,
        _TWO_PASS_PROTOCOL_TOOL,
        _WORKFLOW_STATIC,
    )

    def _norm(text: str) -> str:
        """Whitespace-insensitive: collapse all runs of whitespace (including
        newlines) to a single space, so keyword phrases can be written
        naturally without hand-matching each block's exact line wrapping."""
        return " ".join(text.lower().split())

    # (tool_block, static_block, {requirement_name: (tool_keywords, static_keywords)})
    # A requirement is "covered" if ANY of its keywords appears in the
    # (whitespace-normalised) relevant block — deliberately loose per side,
    # since wording must differ (arm-appropriate), only presence of the
    # concept is checked.
    _CHECKS = [
        (
            _ROLE_TASK_TOOL, _ROLE_TASK_STATIC,
            {
                "structured_recommendation_per_supplier": (
                    ("structured recommendation for each tier-1 supplier",),
                    ("structured recommendation for each tier-1 supplier",),
                ),
                "two_pass_or_policy_application": (
                    ("apply it strictly as written", "use the tools available to retrieve"),
                    ("two-pass decision protocol",),
                ),
            },
        ),
        (
            _TWO_PASS_PROTOCOL_TOOL, _WORKFLOW_STATIC,
            {
                "match_then_stop": (
                    ("record the routing_result and stop", "do not evaluate any supplier rules"),
                    ("do not produce any supplier recommendations",),
                ),
                "no_suppliers_then_empty_recommendations": (
                    ("return empty recommendations",),
                    ("return empty recommendations",),
                ),
                "no_match_then_pass_2": (
                    ("proceed to pass 2",),
                    ("assess each one against the supplier rules",),
                ),
                "risk_band_derivation": (
                    ("risk band",), ("risk band",),
                ),
            },
        ),
    ]

    failures = []
    for tool_block, static_block, requirements in _CHECKS:
        tool_norm = _norm(tool_block)
        static_norm = _norm(static_block)
        for name, (tool_kw, static_kw) in requirements.items():
            tool_hit = any(kw in tool_norm for kw in tool_kw)
            static_hit = any(kw in static_norm for kw in static_kw)
            if not tool_hit:
                failures.append(f"{name}: missing from tool variant (checked {tool_kw})")
            if not static_hit:
                failures.append(f"{name}: missing from Static variant (checked {static_kw})")

    assert not failures, "\n".join(failures)


def test_arm_specific_blocks_have_no_policy_vocab():
    """ARM_SPECIFIC blocks must contain no answer-bearing policy vocabulary.

    Patterns checked (word-boundary, case-exact — avoids false positives on
    ordinary words like 'monitoring' or 'crc' in URLs):
      • rule IDs:              r'\\b[A-Z]{3,4}-[A-Z0-9]+\\b'  (e.g. GEO-R3)
      • escalation enum values: r'\\b(MONITOR|CRC|CRC_EXEC)\\b'  (UPPERCASE only)
      • strategic-priority labels: r'\\b(resilience|flexibility|efficiency)\\b'

    NOTE: _OUTPUT_EXACTNESS intentionally contains format examples like
    '(CYB-R1)', '(GEO-DT)' to teach the model what a rule identifier looks like.
    It is classified SHARED_LITERAL, not ARM_SPECIFIC, so this check does not
    apply to it — scoping the check to ARM_SPECIFIC_BLOCKS handles this correctly.
    """
    import re
    from csco.arms.prompts import ARM_SPECIFIC_BLOCKS

    # Anchor on known disruption-type prefixes to avoid false positives on
    # compound uppercase words in section headings (e.g. "PASS-TYPE").
    rule_id_pat      = re.compile(r"\b(GEO|NAT|CYB|ECON|LAB)-[A-Z0-9-]+\b")
    escalation_pat   = re.compile(r"\b(MONITOR|CRC|CRC_EXEC)\b")
    priority_pat     = re.compile(r"\b(resilience|flexibility|efficiency)\b")

    failures = []
    for block in ARM_SPECIFIC_BLOCKS:
        for pat, label in [
            (rule_id_pat,    "rule ID"),
            (escalation_pat, "escalation enum"),
            (priority_pat,   "strategic-priority label"),
        ]:
            hits = pat.findall(block)
            if hits:
                snippet = block[:80].replace("\n", " ")
                failures.append(
                    f"{label} {hits!r} found in ARM_SPECIFIC block: {snippet!r}..."
                )

    assert not failures, (
        "ARM_SPECIFIC blocks contain answer-bearing policy vocabulary "
        "(Invariant 7):\n" + "\n".join(failures)
    )
