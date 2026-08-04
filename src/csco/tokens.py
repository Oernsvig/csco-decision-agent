"""Shared token counting.

Policy-context token counts must be on a single, common scale across all three
arms (static, vector, lexical) so cross-arm comparisons — e.g. "N% fewer
policy-context tokens" — are meaningful. This module wraps the OpenAI tokenizer
(tiktoken) so every arm measures policy text the same way, rather than mixing a
whitespace-word count (static) with a chars/4 heuristic (retrieval arms).
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=8)
def _encoding(model_name: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Non-OpenAI or unknown model: use gpt-4o's encoding as a consistent
        # cross-arm proxy. What matters is that all arms share one scale.
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Return the number of tokens in `text` under the configured model's encoding.

    All arms call this for policy-context tokens, so the counts are on one scale.
    `model_name` defaults to the configured LLM_MODEL.
    """
    if not text:
        return 0
    if model_name is None:
        try:
            from csco.settings import get_settings

            model_name = get_settings().llm_model
        except Exception:
            model_name = "gpt-4o"
    return len(_encoding(model_name).encode(text))
