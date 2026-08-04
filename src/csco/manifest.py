"""Frozen provenance manifest.

Records the exact inputs and configuration behind the reported results — prompt and
corpus versions + hashes, spec and fixture hashes, the response/embedding models,
sampling and retrieval settings, Neo4j index configuration, pinned package versions,
run dates, and the analysis scripts — so the numbers can be traced to their inputs.

Regenerate/verify:  python -m csco.manifest            # writes MANIFEST.json at repo root
                    python -m csco.manifest --check    # verify committed MANIFEST.json is current

Reads config constants from source so it needs no Neo4j/LLM dependencies.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Runs that produced the reported results (see README "Reproducibility").
RUN_DATES = ["2026-08-02"]
MANIFEST_PATH = ROOT / "MANIFEST.json"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _specs_hash(spec_files: list[Path]) -> str:
    # Same construction as generators.benchmark._compute_specs_hash (matches benchmark_meta).
    combined = "".join(_sha256_file(f) for f in spec_files)
    return _sha256_text(combined)[:16]


def _dir_hashes(d: Path) -> dict:
    files = sorted(p for p in d.rglob("*") if p.is_file())
    per_file = {str(p.relative_to(d)): _sha256_file(p) for p in files}
    aggregate = _sha256_text("".join(f"{k}:{v}" for k, v in per_file.items()))
    return {"file_count": len(per_file), "aggregate_sha256": aggregate, "files_sha256": per_file}


def _source_const(rel_path: str, name: str) -> str | int:
    src = (ROOT / rel_path).read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\s*=\s*(.+?)\s*(?:#.*)?$", src, re.M)
    if not m:
        raise KeyError(f"{name} not found in {rel_path}")
    val = m.group(1).strip()
    if val.isdigit():
        return int(val)
    return val.strip("\"'")


def build_manifest() -> dict:
    from csco.arms import prompts as P
    from csco.generators.playbook import _CORPUS_VERSION

    # Prompts (model-visible assembled system prompts)
    prompt_attrs = [
        "AGENT_SYSTEM_PROMPT",
        "AGENT_SYSTEM_PROMPT_VECTOR",
        "AGENT_SYSTEM_PROMPT_LEXICAL",
        "STATIC_SYSTEM_TEMPLATE",
    ]
    prompt_hashes = {a: _sha256_text(getattr(P, a)) for a in prompt_attrs if hasattr(P, a)}
    prompt_hashes["combined"] = _sha256_text(
        "".join(getattr(P, a) for a in prompt_attrs if hasattr(P, a))
    )

    # Corpus: the five model-visible embed-cut playbooks + lexical cypher
    embed_dir = ROOT / "corpus" / "playbooks_embed"
    types = ["cyber", "economic", "geopolitical", "labour", "natural_disaster"]
    playbook_hashes = {f"{t}.md": _sha256_file(embed_dir / f"{t}.md") for t in types}
    playbook_hashes_present = all((embed_dir / f"{t}.md").exists() for t in types)

    spec_files = sorted((ROOT / "specs").glob("*.yaml"))

    return {
        "manifest_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "python -m csco.manifest",
        "run_dates": RUN_DATES,
        "response_model": {
            "provider": "openai",
            "model_identifier": "gpt-4o",
            "model_alias_note": (
                "'gpt-4o' is a mutable alias; the runs used the snapshot it resolved to on "
                "2026-08-02. Pin LLM_MODEL to that dated snapshot for exact replication."
            ),
            "temperature": 0,
            "max_retries": "langchain_openai ChatOpenAI default (2 at the pinned version); not overridden in code",
        },
        "embedding_model": "text-embedding-3-small",
        "vector_retrieval": {"k": _source_const("src/csco/arms/vector.py", "_RETRIEVAL_K")},
        "neo4j": {
            "vector_index": {
                "index_name": _source_const("src/csco/arms/vector.py", "_VECTOR_INDEX"),
                "node_label": _source_const("src/csco/arms/vector.py", "_NODE_LABEL"),
                "text_property": _source_const("src/csco/arms/vector.py", "_TEXT_PROP"),
                "embedding_property": _source_const("src/csco/arms/vector.py", "_EMB_PROP"),
            },
            "lexical_playcard_index": {
                "index_name": _source_const("src/csco/arms/lexical.py", "_LEX_PLAYCARD_INDEX"),
                "node_label": "LexPlaycard",
            },
        },
        "prompt": {"version": P.PROMPT_VERSION, "hashes_sha256": prompt_hashes},
        "corpus": {
            "version": _CORPUS_VERSION,
            "model_visible_playbooks_present": playbook_hashes_present,
            "model_visible_playbooks_sha256": playbook_hashes,
            "combined_sha256": _sha256_text("".join(playbook_hashes[f"{t}.md"] for t in types)),
            "lexical_cypher_sha256": _sha256_file(ROOT / "corpus" / "corpus_lexical.cypher"),
            "note": "Model-visible playbooks are the embed cuts (sections 1-7, no worked examples), "
            "regenerated with `python -m csco.generators.playbook`.",
        },
        "specs": {
            "specs_hash": _specs_hash(spec_files),
            "files_sha256": {f.name: _sha256_file(f) for f in spec_files},
        },
        "fixtures": {
            d.name: _dir_hashes(d)
            for d in [ROOT / "fixtures" / s for s in ("suite_a", "suite_b")]
            if d.is_dir()
        },
        "environment": {
            "python_requires": ">=3.11",
            "manifest_generated_with_python": platform.python_version(),
            "packages": _read_lockfile(),
            "lockfile": "requirements-lock.txt",
        },
        "scripts": {
            "bootstrap": {
                "module": "src/csco/evaluation/bootstrap.py",
                "produces": "scenario-cluster bootstrap CIs and paired differences "
                "(Table 1/2 Panel B, Table 3, workload crossover)",
                "invoked_by": "python -m csco.cli.batch_run --fixtures-dir fixtures/suite_a,fixtures/suite_b "
                "--output-dir results/headline --runs-per-fixture 3  # writes BATCH_REPORT.txt",
            },
            "token_summary": {
                "modules": ["src/csco/tokens.py", "src/csco/evaluation/report.py"],
                "produces": "total / uncached / policy-context token summaries "
                "(single tokenizer via csco.tokens.count_tokens)",
                "invoked_by": "python -m csco.cli.batch_run ...  # 'Policy Content Tokens' section of BATCH_REPORT.txt",
            },
        },
    }


def _read_lockfile() -> dict:
    lock = ROOT / "requirements-lock.txt"
    pkgs: dict[str, str] = {}
    if lock.exists():
        for line in lock.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                name, ver = line.split("==", 1)
                pkgs[name] = ver
    return pkgs


def _write() -> None:
    MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")


def _check() -> int:
    if not MANIFEST_PATH.exists():
        print("MANIFEST.json missing", file=sys.stderr)
        return 1
    committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current = build_manifest()
    volatile = {"generated_at"}
    a = {k: v for k, v in committed.items() if k not in volatile}
    b = {k: v for k, v in current.items() if k not in volatile}
    if a != b:
        print("MANIFEST.json is stale — run `python -m csco.manifest`", file=sys.stderr)
        return 1
    print("MANIFEST.json is current.")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(_check())
    _write()
