"""Policy-context tokens must be measured on one common, real-tokenizer scale
across all three arms — never a word count (static) vs a chars/4 heuristic
(retrieval arms), which would make cross-arm "N% fewer tokens" claims invalid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SRC = Path(__file__).parent.parent / "src" / "csco"


def _read(rel: str) -> str:
    return (_SRC / rel).read_text(encoding="utf-8")


def test_all_arms_count_policy_tokens_via_shared_tokenizer():
    for arm in ("arms/static.py", "arms/vector.py", "arms/lexical.py"):
        src = _read(arm)
        assert "from csco.tokens import count_tokens" in src, f"{arm} must import count_tokens"
        assert "count_tokens(" in src, f"{arm} must call count_tokens"


def test_static_policy_tokens_are_not_a_word_count():
    src = _read("arms/static.py")
    assert "len(_CONSOLIDATED_POLICY.split())" not in src, (
        "Static must not measure policy tokens as a whitespace-word count"
    )
    assert "count_tokens(_CONSOLIDATED_POLICY)" in src


def test_score_efficiency_policy_path_uses_no_chars_over_four():
    src = _read("evaluation/scoring.py")
    body = src.split("def score_efficiency")[1].split("\ndef ")[0]
    assert "// 4" not in body and "//4" not in body, (
        "score_efficiency must sum real tokenizer counts, not a chars/4 estimate"
    )


def test_count_tokens_is_real_tokens_not_a_word_count():
    pytest.importorskip("tiktoken")
    from csco.tokens import count_tokens

    text = (
        "The composite risk score is banded HIGH, MEDIUM, or LOW per the policy; "
        "sub_node_centrality and sub_downstream_criticality are structural sub-scores."
    )
    n = count_tokens(text, model_name="gpt-4o")
    assert n > 0
    assert n != len(text.split()), "count_tokens must not equal the whitespace-word count"
