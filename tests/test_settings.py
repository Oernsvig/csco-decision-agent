"""Configuration invariants."""

from __future__ import annotations


def test_embedding_model_defaults_to_text_embedding_3_small():
    """The embedding model backing the Neo4j vector indexes must default to the
    declared text-embedding-3-small, and get_embeddings() must pass it explicitly
    rather than relying on the OpenAI client's default. Inspected via the field
    default so no environment / API key is required.
    """
    from csco.settings import Settings

    field = Settings.model_fields["embedding_model"]
    assert field.default == "text-embedding-3-small"
    assert field.alias == "EMBEDDING_MODEL"


def test_get_embeddings_passes_the_configured_model():
    """get_embeddings() must instantiate OpenAIEmbeddings with model set from the
    configured embedding_model (not the library default). Read from source so the
    test does not require the langchain/openai runtime dependencies."""
    from pathlib import Path

    src = (
        Path(__file__).parent.parent / "src" / "csco" / "arms" / "llm.py"
    ).read_text(encoding="utf-8")
    assert "OpenAIEmbeddings(model=s.embedding_model" in src, (
        "get_embeddings() must pass model=s.embedding_model to OpenAIEmbeddings"
    )
