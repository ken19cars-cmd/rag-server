"""Status page: what works now, and what is still Phase 2."""

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_status_page_lists_ready_and_phase2(client):
    for path in ("/", "/ui"):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        page = response.text
        ready, later = page.split('id="later"', 1)
        assert "可用" in ready
        assert "尚未完成" in later
        assert "Notion 寫回" in later
        assert "表格上色" in later
        assert "期程 SVG" in later
        assert "期程 PPTX" in later
        assert "501" in later
        assert "Notion 寫回" not in ready
        assert 'href="/docs"' in page
        assert "Tailscale" in page
        assert "CASES_ROOT" in page
    assert "對上本地案件" in page
    assert "查出還沒填的頁" in page
    assert "/api/pipeline/unfilled" in page
    assert "改成 Notion 案名" not in page
    assert "local-match/rename" not in page
    assert "這一份可以略過" in page
    assert "招標完成，資料夾也有" in page
    assert "資料夾有、Notion 完全沒有" in page
    assert "進行中，已對上" in page
    assert "還沒有唯一對上的資料夾" in page
    assert "掃描沒有看到相近的資料夾" in page
    assert "掃到：" in page
    assert "/api/pipeline/local-match" in page


def test_ui_status_reports_missing_cases_root(client):
    response = client.get("/api/ui/status")
    assert response.status_code == 200
    body = response.json()
    assert body["cases_root_set"] is False
    assert body["cases_root_exists"] is False
    assert body["llm_base_url_set"] is False
    assert body["local_llm_enabled"] is False
    assert body["analysis_mode"] == "extractive"
    assert "llm_api_key" not in body
    assert "notion_token" not in body


def test_demo_analyze_and_query_do_not_need_cases_root(client):
    demo = client.get("/api/ui/demo")
    assert demo.status_code == 200
    payload = demo.json()
    assert payload["caseName"] == "示範市都更案"
    assert "示範市住宅中心" in payload["pdfText"]
    assert payload["demoRoot"].endswith("cases_root")
    assert (FIXTURES / "cases_root").is_dir()

    analyzed = client.post(
        "/api/notion/analyze",
        json={
            "caseName": "狀態頁示範",
            "pageId": payload["pageId"],
            "pdfText": payload["pdfText"],
            "templateBlocks": payload["templateBlocks"],
        },
    )
    assert analyzed.status_code == 200
    result = analyzed.json()
    assert result["caseName"] == "狀態頁示範"
    assert result["baseTable"]["bases"][0]["ownership"].endswith("100%")
    assert "中平" not in analyzed.text

    created = client.post("/api/cases", json={"name": "狀態頁提問", "description": "示範"})
    case_id = created.json()["id"]
    ingested = client.post(
        f"/api/cases/{case_id}/texts",
        json={"title": "示範問答", "text": payload["queryText"]},
    )
    assert ingested.status_code == 200
    queried = client.post(
        f"/api/cases/{case_id}/query",
        json={"question": payload["question"]},
    )
    assert queried.status_code == 200
    assert "12 億元" in queried.json()["answer"]

    missing = client.post("/api/pipeline/detect", json={})
    assert missing.status_code == 400
    assert "not set" in missing.json()["detail"]

    scanned = client.post("/api/pipeline/detect", json={"root": payload["demoRoot"]})
    assert scanned.status_code == 200
    assert any(item["case_name"] == "示範市都更案" for item in scanned.json()["cases"])
