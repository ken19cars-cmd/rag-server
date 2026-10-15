"""Completed cases: offline list, Notion payload, public page enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from rag_server.completed import (
    COMPLETED_STATUS_FILTER,
    OPEN_STATUS_FILTER,
    CompletedCase,
    admin_sync_key,
    case_match_key,
    cases_from_notion_query,
    fetch_notion_database,
    land_identity,
    load_case_file,
    loose_match_key,
    pair_notion_pages_to_folders,
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
                    "招商情形": {"type": "status", "status": {"name": "招標完成"}},
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
                    "招商情形": {"type": "status", "status": {"name": "公告中"}},
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
    assert payload["results"] == []
    assert payload["has_more"] is False
    assert seen["url"] == "https://api.notion.com/v1/databases/db123/query"
    assert seen["auth"] == "Bearer secret-token"
    assert seen["body"]["page_size"] == 100
    assert seen["body"]["filter"]["status"]["equals"] == "招標完成"


def test_fetch_notion_database_waits_out_a_rate_limit():
    waits: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"object": "error"})
        return httpx.Response(200, json={"results": [], "has_more": False})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = fetch_notion_database("secret-token", "db123", client=client, sleep=waits.append)
    assert payload["results"] == []
    assert waits == [2.0]


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


def test_fetch_notion_database_follows_cursor():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "start_cursor" not in body:
            return httpx.Response(
                200,
                json={"results": [{"id": "a"}], "has_more": True, "next_cursor": "cur-2"},
            )
        return httpx.Response(200, json={"results": [{"id": "b"}], "has_more": False})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = fetch_notion_database("secret-token", "db123", client=client)
    assert [item["id"] for item in payload["results"]] == ["a", "b"]
    assert payload["has_more"] is False
    assert calls[1]["start_cursor"] == "cur-2"
    assert calls[0]["page_size"] == 100


def test_open_notion_query_skips_completed_pages():
    payload = {
        "results": [
            {
                "id": "page-open",
                "properties": {
                    "案件名稱": {"type": "title", "title": [{"plain_text": "示範公告案"}]},
                    "招商情形": {"type": "status", "status": {"name": "公告中"}},
                },
            },
            {
                "id": "page-blank",
                "properties": {
                    "案件名稱": {"type": "title", "title": [{"plain_text": "未填情形"}]},
                    "招商情形": {"type": "status", "status": None},
                },
            },
            {
                "id": "page-done",
                "properties": {
                    "案件名稱": {"type": "title", "title": [{"plain_text": "示範完成案"}]},
                    "招商情形": {"type": "status", "status": {"name": "招標完成"}},
                },
            },
        ]
    }
    cases = cases_from_notion_query(payload, completed_only=False)
    assert [case.case_name for case in cases] == ["示範公告案", "未填情形"]
    assert cases[1].status == ""


PINGTUNG_FOLDER = "屏東高鐵屏東站北側都市更新地區(單元一)公辦都更案"
PINGTUNG_NOTION = "屏東縣高鐵屏東站北側都市更新地區（單元一）公辦都更案"


def test_admin_sync_key_fills_a_unique_county_suffix():
    assert case_match_key(PINGTUNG_FOLDER) != case_match_key(PINGTUNG_NOTION)
    assert admin_sync_key(PINGTUNG_FOLDER) == admin_sync_key(PINGTUNG_NOTION)
    assert admin_sync_key("高雄岡山區某案") == admin_sync_key("高雄市岡山區某案")
    assert admin_sync_key("嘉義市某案") != admin_sync_key("嘉義縣某案")
    assert admin_sync_key("嘉義某案") != admin_sync_key("嘉義縣某案")
    assert admin_sync_key("新竹某案") != admin_sync_key("新竹市某案")


def test_pair_renames_folder_that_omits_the_county_suffix():
    pages = cases_from_notion_query(
        {"results": [_notion_page("page-pingtung", PINGTUNG_NOTION, "公告中")]},
        completed_only=False,
    )
    paired = pair_notion_pages_to_folders(pages, [_folder(PINGTUNG_FOLDER, "屏東縣")])
    assert len(paired["matched"]) == 1
    assert paired["matched"][0]["matched_by"] == "admin"
    assert paired["matched"][0]["case_name"] == PINGTUNG_NOTION
    assert paired["renames"] == [
        {
            "from_name": PINGTUNG_FOLDER,
            "to_name": PINGTUNG_NOTION,
            "source_path": str(Path(PINGTUNG_FOLDER)),
            "notion_page_id": "page-pingtung",
            "matched_by": "admin",
        }
    ]
    assert paired["notion_without_folder"] == []


def test_case_match_key_folds_tai_space_and_dashes():
    assert case_match_key(" 台北市信義－段 ") == case_match_key("臺北市信義-段")
    assert case_match_key("ＡＢＣ") == case_match_key("abc")


def test_pair_leaves_colliding_names_unmatched():
    pages = [
        CompletedCase(case_name="重複案", status="公告中", notion_page_id="p1"),
        CompletedCase(case_name="重複案", status="進行中", notion_page_id="p2"),
    ]
    folder = type("Folder", (), {})()
    folder.case_name = "重複案"
    folder.city = "臺北市"
    folder.case_type = "都更"
    folder.folder = Path("重複案")
    folder.documents = []
    paired = pair_notion_pages_to_folders(pages, [folder])
    assert paired["matched"] == []
    assert [item["notion_page_id"] for item in paired["notion_without_folder"]] == ["p1", "p2"]
    assert paired["folder_without_notion"][0]["case_name"] == "重複案"


def test_local_match_pairs_open_pages_and_skips_completed(tmp_path: Path, monkeypatch):
    root = tmp_path / "cases"
    case_dir = root / "1.都市更新" / "00臺北市" / "台北市信義段小案"
    (case_dir / "1150921公開閱覽").mkdir(parents=True)
    (case_dir / "1140101公開閱覽").mkdir()
    (case_dir / "1150921公開閱覽" / "申請須知.txt").write_text("新", encoding="utf-8")
    (case_dir / "1140101公開閱覽" / "申請須知.txt").write_text("舊", encoding="utf-8")
    orphan = root / "1.都市更新" / "00臺北市" / "只有資料夾"
    orphan.mkdir(parents=True)
    (orphan / "需求說明書.txt").write_text("x", encoding="utf-8")
    absent = root / "1.都市更新" / "00臺北市" / "基隆市七堵區一德新村公辦都更案"
    absent.mkdir(parents=True)
    (absent / "申請須知.txt").write_text("完成案仍在磁碟", encoding="utf-8")
    nowhere = root / "1.都市更新" / "00臺北市" / "完全沒有頁"
    nowhere.mkdir(parents=True)
    (nowhere / "申請須知.txt").write_text("沒有頁", encoding="utf-8")

    seen: dict = {"filters": []}

    def fake_fetch(token, database_id, *, client=None, status_filter=None):
        seen["token"] = token
        seen["database_id"] = database_id
        seen["filters"].append(status_filter)
        return {
            "results": [
                {
                    "id": "page-open",
                    "properties": {
                        "案件名稱": {"type": "title", "title": [{"plain_text": "臺北市信義段小案"}]},
                        "招商情形": {"type": "status", "status": {"name": "公告中"}},
                    },
                },
                {
                    "id": "page-done",
                    "properties": {
                        "案件名稱": {"type": "title", "title": [{"plain_text": "只有資料夾"}]},
                        "招商情形": {"type": "status", "status": {"name": "招標完成"}},
                    },
                },
                {
                    "id": "page-keelung",
                    "properties": {
                        "案件名稱": {"type": "title", "title": [{"plain_text": "基隆市七堵區一德新村公辦都更案"}]},
                        "招商情形": {"type": "status", "status": {"name": "招標完成"}},
                        "縣市別": {"type": "select", "select": {"name": "基隆市"}},
                    },
                },
                {
                    "id": "page-done-only",
                    "properties": {
                        "案件名稱": {"type": "title", "title": [{"plain_text": "招標完成但沒有資料夾"}]},
                        "招商情形": {"type": "status", "status": {"name": "招標完成"}},
                    },
                },
                {
                    "id": "page-missing",
                    "properties": {
                        "案件名稱": {"type": "title", "title": [{"plain_text": "沒有資料夾"}]},
                        "招商情形": {"type": "status", "status": {"name": "進行中"}},
                    },
                },
            ],
            "has_more": False,
        }

    monkeypatch.setattr("rag_server.service.fetch_notion_database", fake_fetch)
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        embedding_backend="fake",
        notion_token="secret",
        notion_database_id="db123",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        listed = client.get("/api/pipeline/local-match", params={"root": str(root)})
    assert seen["token"] == "secret"
    assert seen["database_id"] == "db123"
    assert seen["filters"] == [OPEN_STATUS_FILTER, COMPLETED_STATUS_FILTER]
    assert listed.status_code == 200
    body = listed.json()
    assert len(body["matched"]) == 1
    hit = body["matched"][0]
    assert hit["case_name"] == "臺北市信義段小案"
    assert hit["folder_name"] == "台北市信義段小案"
    assert hit["notion_page_id"] == "page-open"
    assert hit["status"] == "公告中"
    assert hit["city"] == "臺北市"
    assert hit["case_type"] == "1.都市更新"
    assert [doc["filename"] for doc in hit["documents"]] == ["申請須知.txt"]
    assert "1150921" in hit["documents"][0]["source_path"]
    assert [item["case_name"] for item in body["notion_without_folder"]] == ["沒有資料夾"]
    completed_names = [item["folder_name"] for item in body["folder_completed"]]
    assert completed_names == ["只有資料夾", "基隆市七堵區一德新村公辦都更案"]
    assert body["folder_completed"][1]["status"] == "招標完成"
    assert body["folder_completed"][1]["matched_by"] == "name"
    assert [item["case_name"] for item in body["completed_without_folder"]] == ["招標完成但沒有資料夾"]
    assert [item["case_name"] for item in body["folder_without_notion"]] == ["完全沒有頁"]

    bare = Settings(
        data_dir=str(tmp_path / "empty"),
        embedding_backend="fake",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    with TestClient(create_app(bare, embedder=FakeEmbedder())) as client:
        denied = client.get("/api/pipeline/local-match", params={"root": str(root)})
    assert denied.status_code == 400


XINYI = "〈信義〉臺北市中山區長春段一小段712-1地號等4筆土地公辦都更案"
XINYI_NOTION = "臺北市中山區長春段一小段712-1地號等4筆土地公辦都更案"
YUCHENG = "臺北市南港區玉成段一小段4-2地號等19筆土地公辦都更案"
YUCHENG_NOTION = "臺北市南港區玉成段一小段4-2地號等19筆土地(產專2-32)公辦都更案"


def _folder(name: str, city: str = "臺北市"):
    folder = type("Folder", (), {})()
    folder.case_name = name
    folder.city = city
    folder.case_type = "1.都市更新"
    folder.folder = Path(name)
    folder.documents = []
    return folder


def _notion_page(page_id: str, name: str, status: str = "公告中") -> dict:
    return {
        "id": page_id,
        "properties": {
            "案件名稱": {"type": "title", "title": [{"plain_text": name}]},
            "招商情形": {"type": "status", "status": {"name": status}},
        },
    }


def test_land_identity_ignores_prefix_and_project_aside():
    xinyi = land_identity(XINYI)
    assert xinyi == land_identity(XINYI_NOTION)
    assert xinyi == ("臺北市", "中山區", "長春段1小段", "712-1", "4")
    yucheng = land_identity(YUCHENG)
    assert yucheng == land_identity(YUCHENG_NOTION)
    assert yucheng[:4] == ("臺北市", "南港區", "玉成段1小段", "4-2")
    assert land_identity("台北市中山區長春段一小段712之1地號等四筆土地")[:4] == xinyi[:4]
    assert land_identity(XINYI_NOTION.replace("712-1", "712-2")) != xinyi
    assert land_identity(XINYI_NOTION.replace("等4筆", "等5筆")) != xinyi


def test_pair_uses_land_when_titles_differ_and_plans_notion_name():
    pages = cases_from_notion_query(
        {
            "results": [
                _notion_page("page-xinyi", XINYI_NOTION),
                _notion_page("page-yucheng", YUCHENG_NOTION, "進行中"),
                _notion_page("page-done", XINYI, "招標完成"),
            ]
        },
        completed_only=False,
    )
    other_parcel = XINYI_NOTION.replace("712-1", "712-2")
    paired = pair_notion_pages_to_folders(
        pages,
        [_folder(XINYI), _folder(YUCHENG), _folder(other_parcel)],
    )
    by_page = {item["notion_page_id"]: item for item in paired["matched"]}
    assert set(by_page) == {"page-xinyi", "page-yucheng"}
    assert by_page["page-xinyi"]["matched_by"] == "land"
    assert by_page["page-xinyi"]["case_name"] == XINYI_NOTION
    assert by_page["page-xinyi"]["folder_name"] == XINYI
    planned = {item["from_name"]: item["to_name"] for item in paired["renames"]}
    assert planned[XINYI] == XINYI_NOTION
    assert planned[YUCHENG] == YUCHENG_NOTION
    assert [item["case_name"] for item in paired["folder_without_notion"]] == [other_parcel]
    assert paired["notion_without_folder"] == []


def test_pair_does_not_rename_when_two_folders_share_a_parcel():
    page = cases_from_notion_query({"results": [_notion_page("page-1", XINYI_NOTION)]}, completed_only=False)
    paired = pair_notion_pages_to_folders(page, [_folder(XINYI), _folder("〈另一標〉" + XINYI)])
    assert paired["matched"] == []
    assert paired["renames"] == []
    assert len(paired["folder_without_notion"]) == 2


def test_local_match_pairs_without_renaming_the_folder(tmp_path: Path, monkeypatch):
    root = tmp_path / "cases"
    folder = root / "1.都市更新" / "00屏東縣" / PINGTUNG_FOLDER
    folder.mkdir(parents=True)
    (folder / "申請須知.txt").write_text("屏東", encoding="utf-8")

    def fake_fetch(token, database_id, *, client=None, status_filter=None):
        return {
            "results": [
                _notion_page("page-pingtung", PINGTUNG_NOTION, "公告中"),
                _notion_page("page-done-only", "招標完成但沒有資料夾", "招標完成"),
            ],
            "has_more": False,
        }

    monkeypatch.setattr("rag_server.service.fetch_notion_database", fake_fetch)
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        embedding_backend="fake",
        notion_token="secret",
        notion_database_id="db123",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        listed = client.get("/api/pipeline/local-match", params={"root": str(root)})
        missing = client.post("/api/pipeline/local-match/rename", params={"root": str(root)})
    assert listed.status_code == 200
    body = listed.json()
    assert body["matched"][0]["folder_name"] == PINGTUNG_FOLDER
    assert body["matched"][0]["case_name"] == PINGTUNG_NOTION
    assert [item["case_name"] for item in body["completed_without_folder"]] == ["招標完成但沒有資料夾"]
    assert folder.is_dir()
    assert (folder / "申請須知.txt").read_text(encoding="utf-8") == "屏東"
    assert not (root / "1.都市更新" / "00屏東縣" / PINGTUNG_NOTION).exists()
    assert missing.status_code == 404


YUNLIN = "高速鐵路雲林車站特定區車站專用區(三)開發經營案"
TAINAN_STATION = "高速鐵路臺南車站特定區產業專用區(武東段279、286地號)開發經營案"
TAINAN_STATION_LOOSE = "高速鐵路臺南車站特定區產業專用區武東段279,286地號開發經營案"
TRADE_PARCELS = "臺中市西屯區經貿段6、8地號等兩筆土地設定地上權案"
TRADE_PARCELS_ASCII = "臺中市西屯區經貿段6,8地號等2筆土地設定地上權案"


def test_case_match_key_drops_zero_width_characters():
    assert case_match_key("案\u200b名") == case_match_key("案名")


def test_admin_sync_key_leaves_pingtung_city_alone():
    title = "屏東市舊酒廠段3、5地號國有非公用土地開發案"
    assert admin_sync_key(title) == case_match_key(title)
    assert "屏東縣市" not in admin_sync_key(title)


def test_land_identity_treats_list_punctuation_and_liang_as_the_same_parcels():
    listed = land_identity(TRADE_PARCELS)
    assert listed == land_identity(TRADE_PARCELS_ASCII)
    assert listed[3:] == ("6,8", "2")
    assert land_identity("臺北市中山區長春段一小段712-1地號等4筆土地")[3] == "712-1"
    shuijiao = "臺南市南區水交社段72、72-1、72-2地號等3筆營改土地合作開發案"
    assert land_identity(shuijiao)[3] == land_identity(shuijiao.replace("、", ","))[3]


def test_loose_match_key_ignores_parentheses_and_list_marks():
    assert case_match_key(TAINAN_STATION) != case_match_key(TAINAN_STATION_LOOSE)
    assert loose_match_key(TAINAN_STATION) == loose_match_key(TAINAN_STATION_LOOSE)
    assert loose_match_key(TRADE_PARCELS) == loose_match_key(TRADE_PARCELS_ASCII)


def test_pair_uses_loose_key_when_punctuation_is_the_only_difference():
    pages = cases_from_notion_query(
        {"results": [_notion_page("page-station", TAINAN_STATION, "前置作業")]},
        completed_only=False,
    )
    paired = pair_notion_pages_to_folders(pages, [_folder(TAINAN_STATION_LOOSE, "臺南市")])
    assert len(paired["matched"]) == 1
    assert paired["matched"][0]["matched_by"] == "loose"
    assert paired["matched"][0]["case_name"] == TAINAN_STATION
    assert paired["notion_without_folder"] == []


def test_pair_prefers_the_folder_already_named_with_the_notion_title():
    pages = cases_from_notion_query(
        {"results": [_notion_page("page-xinyi", "臺北市信義段小案")]},
        completed_only=False,
    )
    paired = pair_notion_pages_to_folders(
        pages,
        [_folder("台北市信義段小案"), _folder("臺北市信義段小案")],
    )
    assert [item["folder_name"] for item in paired["matched"]] == ["臺北市信義段小案"]
    assert [item["case_name"] for item in paired["folder_without_notion"]] == ["台北市信義段小案"]
    assert paired["renames"] == []


def test_pair_lists_near_folders_when_two_loose_names_collide():
    other = TAINAN_STATION_LOOSE.replace(",", "、")
    pages = cases_from_notion_query(
        {"results": [_notion_page("page-station", TAINAN_STATION, "前置作業")]},
        completed_only=False,
    )
    paired = pair_notion_pages_to_folders(
        pages,
        [_folder(TAINAN_STATION_LOOSE, "臺南市"), _folder(other, "臺南市")],
    )
    assert paired["matched"] == []
    near = paired["notion_without_folder"][0]["near_folders"]
    assert {item["case_name"] for item in near} == {TAINAN_STATION_LOOSE, other}


def test_local_match_pairs_a_renamed_folder_above_the_city_level(tmp_path: Path, monkeypatch):
    root = tmp_path / "cases"
    nested = root / "前置作業" / YUNLIN / "附件"
    nested.mkdir(parents=True)
    (nested / "申請須知.txt").write_text("雲林", encoding="utf-8")
    direct = root / "前置作業" / "臺南市新化區虎頭埤湖濱旅館BOT案"
    direct.mkdir(parents=True)
    (direct / "需求說明書.txt").write_text("虎", encoding="utf-8")

    def fake_fetch(token, database_id, *, client=None, status_filter=None):
        return {
            "results": [
                _notion_page("page-yunlin", YUNLIN, "前置作業"),
                _notion_page("page-bot", direct.name, "前置作業"),
                _notion_page("page-intro", "案件介紹", "公告中"),
            ],
            "has_more": False,
        }

    monkeypatch.setattr("rag_server.service.fetch_notion_database", fake_fetch)
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        embedding_backend="fake",
        notion_token="secret",
        notion_database_id="db123",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        listed = client.get("/api/pipeline/local-match", params={"root": str(root)})
    assert listed.status_code == 200
    body = listed.json()
    by_page = {item["notion_page_id"]: item for item in body["matched"]}
    assert set(by_page) == {"page-yunlin", "page-bot"}
    assert by_page["page-yunlin"]["folder_name"] == YUNLIN
    assert [doc["filename"] for doc in by_page["page-yunlin"]["documents"]] == ["申請須知.txt"]
    assert by_page["page-bot"]["folder_name"] == direct.name
    missing = body["notion_without_folder"]
    assert [item["case_name"] for item in missing] == ["案件介紹"]
    assert missing[0]["near_folders"] == []


