from pathlib import Path

from rag_server.config import Settings
from rag_server.crawler import FetchResponse
from rag_server.embeddings import FakeEmbedder
from rag_server.main import create_app
from tests.pdf_bytes import SAMPLE_LINES, build_text_pdf

FIXTURES = Path(__file__).parent / "fixtures"
SEED = "https://invest.example.com/cases/tainan"
PDF_URL = "https://invest.example.com/files/brief.pdf"


class MapFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.mapping = {
            "https://invest.example.com/robots.txt": (
                200,
                "text/plain",
                (FIXTURES / "robots.txt").read_bytes(),
            ),
            SEED: (
                200,
                "text/html; charset=utf-8",
                (FIXTURES / "sample_case.html").read_bytes(),
            ),
            PDF_URL: (200, "application/pdf", (FIXTURES / "sample.pdf").read_bytes()),
        }

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        status, content_type, body = self.mapping[url]
        return FetchResponse(url=url, final_url=url, status=status, content_type=content_type, body=body)


def _assert_grounded(body: dict, chunks: dict) -> None:
    claims = body.get("summary") or body.get("claims") or []
    for claim in claims:
        assert claim["citations"]
        for citation in claim["citations"]:
            source = chunks[citation["chunk_id"]].text
            assert claim["text"] in source
            assert citation["excerpt"] in source


def test_health_uses_the_fake_embedder_and_no_llm(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_backend"] == "fake"
    assert body["analysis_mode"] == "extractive"
    assert body["local_llm_enabled"] is False


def test_pdf_and_text_ingest_then_query_and_analyze(client):
    created = client.post(
        "/api/cases",
        json={"name": "臺南安南智慧物流園區", "description": "測試案件"},
    )
    assert created.status_code == 200
    case_id = created.json()["id"]

    uploaded = client.post(
        f"/api/cases/{case_id}/documents",
        files={"file": ("brief.pdf", (FIXTURES / "sample.pdf").read_bytes(), "application/pdf")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["document"]["chunk_count"] >= 1
    assert "smart logistics" in client.app.state.service.store.list_chunks(case_id)[0].text

    text = (FIXTURES / "sample_case.txt").read_text(encoding="utf-8")
    ingested = client.post(
        f"/api/cases/{case_id}/texts",
        json={"title": "招商說明", "text": text},
    )
    assert ingested.status_code == 200

    listed = client.get("/api/cases")
    assert listed.status_code == 200
    assert listed.json()[0]["document_count"] == 2

    query = client.post(
        f"/api/cases/{case_id}/query",
        json={"question": "投資金額是多少？"},
    )
    assert query.status_code == 200
    query_body = query.json()
    assert query_body["mode"] == "extractive"
    assert "新臺幣 12 億元" in query_body["answer"]
    chunks = {chunk.id: chunk for chunk in client.app.state.service.store.list_chunks(case_id)}
    _assert_grounded(query_body, chunks)

    analysis = client.post(f"/api/cases/{case_id}/analyze", json={})
    assert analysis.status_code == 200
    body = analysis.json()
    assert body["mode"] == "extractive"
    assert body["fields"]["industry"]["value"] == "智慧物流與冷鏈倉儲"
    assert body["fields"]["location"]["status"] == "found"
    assert body["fields"]["investment_amount"]["value"] == "新臺幣 12 億元"
    assert body["fields"]["timeline"]["status"] == "found"
    assert body["fields"]["contacts"]["status"] == "found"
    assert body["fields"]["incentives"]["status"] == "unknown"
    assert any(item["field"] == "incentives" for item in body["unknowns"])
    _assert_grounded(body, chunks)

    missing = client.post("/api/cases/case_missing/query", json={"question": "金額"})
    assert missing.status_code == 404


def test_crawl_route_indexes_html_and_pdf_without_the_private_page(tmp_path):
    fetcher = MapFetcher()
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    app = create_app(settings, embedder=FakeEmbedder(), fetcher=fetcher)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        case_id = client.post("/api/cases", json={"name": "爬蟲案件"}).json()["id"]
        response = client.post(
            f"/api/cases/{case_id}/crawl",
            json={
                "seed_urls": [SEED],
                "max_pages": 10,
                "max_depth": 1,
                "delay_seconds": 0,
                "same_domain_only": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["pages_fetched"] == 1
        assert body["pdfs_fetched"] == 1
        assert body["chunks_added"] >= 2
        assert "https://invest.example.com/private/secret.html" not in fetcher.calls
        assert "https://invest.example.com/private/secret.html" in body["skipped_by_robots"]
        query = client.post(
            f"/api/cases/{case_id}/query",
            json={"question": "投資金額是多少？"},
        )
        assert "新臺幣 12 億元" in query.json()["answer"]
        pdf_query = client.post(
            f"/api/cases/{case_id}/query",
            json={"question": "smart logistics cold-chain"},
        )
        assert "smart logistics" in pdf_query.json()["answer"]


def test_local_host_seed_is_not_fetched(client):
    case_id = client.post("/api/cases", json={"name": "blocked"}).json()["id"]
    response = client.post(
        f"/api/cases/{case_id}/crawl",
        json={"seed_urls": ["http://127.0.0.1/secret"], "delay_seconds": 0, "max_pages": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pages_fetched"] == 0
    assert body["documents_indexed"] == 0
    assert any("127.0.0.1" in item for item in body["skipped"])


def test_sample_pdf_bytes_match_builder():
    assert (FIXTURES / "sample.pdf").read_bytes() == build_text_pdf(SAMPLE_LINES)
