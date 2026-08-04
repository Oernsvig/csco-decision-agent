"""LLM factory — returns the configured chat model at temperature 0."""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from csco.settings import get_settings


@lru_cache(maxsize=1)
def get_llm():
    s = get_settings()
    if s.llm_provider == "openai":
        return ChatOpenAI(model=s.llm_model, temperature=0, api_key=s.openai_api_key)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=s.llm_model, temperature=0, api_key=s.anthropic_api_key)


@lru_cache(maxsize=1)
def get_embeddings():
    """OpenAI embeddings (EMBEDDING_MODEL, default text-embedding-3-small) —
    used for the Neo4j vector indexes in the vector and lexical arms."""
    s = get_settings()
    if s.openai_api_key:
        return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)
    raise RuntimeError(
        "OPENAI_API_KEY required for embeddings even when LLM_PROVIDER=anthropic. "
        "Add it to .env."
    )
