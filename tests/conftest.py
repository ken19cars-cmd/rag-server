from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_server.config import Settings
from rag_server.embeddings import FakeEmbedder
from rag_server.main import create_app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        embedding_backend="fake",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
        crawl_delay_seconds=0,
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def app(settings: Settings, embedder: FakeEmbedder):
    return create_app(settings, embedder=embedder)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
