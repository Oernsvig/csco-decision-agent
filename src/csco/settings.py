"""Environment-based configuration via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j — only required by the vector and lexical arms. Optional so the Static
    # arm (and all non-integration tests) run with no Neo4j configured; the retrieval
    # arms raise a clear error via require_neo4j() if these are unset.
    neo4j_uri: str | None = Field(None, alias="NEO4J_URI")
    neo4j_username: str | None = Field(None, alias="NEO4J_USERNAME")
    neo4j_password: str | None = Field(None, alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", alias="NEO4J_DATABASE")

    def require_neo4j(self) -> "Settings":
        """Assert Neo4j credentials are present (for the vector/lexical arms)."""
        missing = [
            name
            for name, val in (
                ("NEO4J_URI", self.neo4j_uri),
                ("NEO4J_USERNAME", self.neo4j_username),
                ("NEO4J_PASSWORD", self.neo4j_password),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"{', '.join(missing)} required for the vector/lexical arms. "
                "The static arm does not need Neo4j."
            )
        return self

    # LLM
    llm_provider: Literal["openai", "anthropic"] = Field("openai", alias="LLM_PROVIDER")
    llm_model: str = Field("gpt-4o", alias="LLM_MODEL")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")

    @model_validator(mode="after")
    def _validate_api_keys(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required when LLM_PROVIDER=openai")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required when LLM_PROVIDER=anthropic")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
