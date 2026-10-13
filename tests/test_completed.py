"""Completed cases: offline list, Notion payload, public page enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from rag_server.completed import (
    CompletedCase,
    cases_from_notion_query,
    fetch_notion_database,
    load_case_file,
    research_case,
)
from rag_server.config import Settings
from rag_server.crawler import FetchResponse
from rag_server.embeddings import FakeEmbedder
from rag_server.main import create_app

PAGE = "https://news.example.com/cases/demo"
PRIVATE = "https://news.example.com/private/secret"
ROBOTS = "https://news.example.com/robots.txt"


class MapFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        if url == ROBOTS:
            body = b"User-agent: *\nDisallow: /private/\n"
            return FetchResponse(url=url, final_url=url, status=200, content_type="text/plain", body=body)
        if url == PAGE:
            html = """<!doctype html><html><head><title>示範完成案得標</title></head>
<body><p>最優申請人為示範營造，並提出規劃。</p>
<p><img alt="建築效果模擬圖" src="/images/rendering.png"></p></body></html>"""
            return FetchResponse(
                url=url,
                final_url=url,
                status=200,
                content_type="text/html; charset=utf-8",
                body=html.encode("utf-8"),
            )
        raise AssertionError(url)


def test_json_and_csv_lists(tmp_path: Path):
    json_path = tmp_path / "cases.json"
    json_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "示範完成案",
                        "city": "示範市",
                        "case_type": "公辦都更",
                        "notion_page_id": "page_done",
                        "winning_applicant": "示範營造",
                        "status": "招標完成",
                        "source_urls": [PAGE],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = load_case_file(json_path)
    assert loaded[0].winning_applicant == "示範營造"
    assert loaded[0].queries()[0] == "示範完成案 最優申請人"
    assert loaded[0].queries()[-1] == "示範完成案 建築效果模擬圖"

    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        "case_name,city,winning_applicant,source_urls\n示範完成案,示範市,示範營造,https://news.example.com/a\n",
        encoding="utf-8",
    )
    from_csv = load_case_file(csv_path)
    assert from_csv[0].city == "示範市"
    assert from_csv[0].source_urls == ["https://news.example.com/a"]


def test_notion_query_keeps_only_completed_pages():
    payload = {
        "results": [
            {
                "id": "page-done",
                "properties": {
                    "案件名稱": {"type": "title", "title": [{"plain_text": "示範完成案"}]},
                    "招商情形": {"type": "select", "select": {"name": "招標完成"}},
                    "縣市別": {"type": "select", "select": {"name": "示範市"}},
                    "案件類型": {"type": "select", "select": {"name": "公辦都更"}},
                    "投標廠商1(最優申請人)": {
                        "type": "rich_text",
                        "rich_text": [{"plain_text": "示範營造"}],
                    },
                },
            },
            {
                "id": "page-open",
                "properties": {
                    "案件名稱": {"type": "title", "title": [{"plain_text": "示範公告案"}]},
                    "招商情形": {"type": "select", "select": {"name": "公告中"}},
                },
            },
        ]
    }
    cases = cases_from_notion_query(payload)
    assert [case.case_name for case in cases] == ["示範完成案"]
    assert cases[0].city == "示範市"
    assert cases[0].winning_applicant == "示範營造"
    assert cases[0].notion_page_id == "page-done"


def test_fetch_notion_database_sends_completed_filter():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = fetch_notion_database("secret-token", "db123", client=client)
    assert payload == {"results": []}
    assert seen["url"] == "https://api.notion.com/v1/databases/db123/query"
    assert seen["auth"] == "Bearer secret-token"
    assert seen["body"]["filter"]["select"]["equals"] == "招標完成"


def test_research_respects_robots_and_keeps_rendering_urls():
    fetcher = MapFetcher()
    case = CompletedCase(
        case_name="示範完成案",
        source_urls=[PAGE, PRIVATE, "http://127.0.0.1/local"],
    )
    result = research_case(
        case,
        fetcher,
        user_agent="InvestmentCaseRAG/0.1",
        delay_seconds=0,
    )
    assert result.pages[0].title == "示範完成案得標"
    assert "示範營造" in result.pages[0].text
    assert result.renderings[0].url == "https://news.example.com/images/rendering.png"
    assert result.renderings[0].alt == "建築效果模擬圖"
    assert PRIVATE in result.skipped_by_robots
    assert any("127.0.0.1" in item for item in result.skipped)
    assert PRIVATE not in fetcher.calls
    assert "http://127.0.0.1/local" not in fetcher.calls


def test_completed_route_reads_the_offline_file(tmp_path: Path):
    path = tmp_path / "done.json"
    path.write_text(
        json.dumps({"cases": [{"案件名稱": "示範完成案", "縣市別": "示範市", "最優申請人": "示範營造"}]}),
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        embedding_backend="fake",
        completed_cases_file=str(path),
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    app = create_app(settings, embedder=FakeEmbedder())
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        listed = client.get("/api/completed")
        assert listed.status_code == 200
        case = listed.json()["cases"][0]
        assert case["case_name"] == "示範完成案"
        assert case["winning_applicant"] == "示範營造"
        assert "最優申請人" in case["queries"][0]

    bare = Settings(
        data_dir=str(tmp_path / "empty"),
        embedding_backend="fake",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    with TestClient(create_app(bare, embedder=FakeEmbedder())) as client:
        missing = client.get("/api/completed")
        assert missing.status_code == 400
