"""The frozen provenance manifest must stay in sync with the repo it describes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def test_manifest_is_committed_and_current():
    manifest_path = ROOT / "MANIFEST.json"
    assert manifest_path.exists(), "MANIFEST.json must be committed at the repo root"

    if not (ROOT / "corpus" / "playbooks_embed" / "cyber.md").exists():
        pytest.skip("embed cuts not generated — run: python -m csco.generators.playbook")

    from csco.manifest import build_manifest

    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_manifest()
    volatile = {"generated_at"}
    a = {k: v for k, v in committed.items() if k not in volatile}
    b = {k: v for k, v in current.items() if k not in volatile}
    assert a == b, "MANIFEST.json is stale — regenerate with `python -m csco.manifest`"


def test_manifest_specs_hash_matches_frozen_fixture_metadata():
    m = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    meta = json.loads(
        (ROOT / "fixtures" / "suite_a" / "benchmark_meta.json").read_text(encoding="utf-8")
    )
    assert m["specs"]["specs_hash"] == meta["specs_hash"]
