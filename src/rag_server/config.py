"""Runtime settings. Secrets come from the environment, never from the repo."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: str = "./data"
    embedding_backend: str = "local"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 500
    chunk_overlap: int = 80
    crawl_user_agent: str = (
        "InvestmentCaseRAG/0.1 (public research client; investment-case analysis)"
    )
    crawl_delay_seconds: float = 1.0
    crawl_timeout_seconds: float = 20.0
    crawl_max_pdf_bytes: int = 20_000_000
    cases_root: str = ""
    completed_cases_file: str = ""
    notion_token: str = ""
    notion_database_id: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def local_llm_configured(self) -> bool:
        """Local OpenAI-compatible server, for example Ollama. A key is optional."""
        return bool(self.llm_base_url.strip() and self.llm_model.strip())
