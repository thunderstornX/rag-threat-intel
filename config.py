"""Typed runtime configuration.

Every component (ingest, embeddings, retrieval, generation, API)
reads from this single Settings object so we can swap a fake-Ollama
URL or an in-memory pgvector for tests without crawling helper
functions for hardcoded defaults."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                       extra="ignore")

    # ---- Ollama --------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_generation_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_alt_embedding_model: str = "all-minilm"

    # ---- Postgres ------------------------------------------------------
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "rti"
    pg_user: str = "rti"
    pg_password: str = "rti-dev-password-change-me"

    # ---- API + ingest -------------------------------------------------
    port: int = 8000
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARN|ERROR)$")

    nvd_api_base: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    nvd_api_key: str = ""
    nvd_initial_cve_count: int = 200

    @property
    def pg_dsn(self) -> str:
        return (f"postgresql://{self.pg_user}:{self.pg_password}@"
                f"{self.pg_host}:{self.pg_port}/{self.pg_db}")


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
