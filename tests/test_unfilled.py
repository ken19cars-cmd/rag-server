"""In-progress pages whose ❯計畫概述 paragraph is still empty."""

from __future__ import annotations

from pathlib import Path

import httpx

from rag_server.config import Settings
from rag_server.embeddings import FakeEmbedder
from rag_server.main import create_app
from rag_server.notion_blocks import fetch_block_children, overview_state


def _rich(text: str) -> dict:
    return {"rich_text": [{"plain_text": text}] if text else []}


def _heading(text: str, block_id: str = "h-overview") -> dict:
    return {"id": block_id, "type": "heading_2", "heading_2": _rich(text)}


def _paragraph(text: str, block_id: str = "p-overview") -> dict:
    return {"id": block_id, "type": "paragraph", "paragraph": _rich(text)}


def _bullet(text: str, block_id: str) -> dict:
    return {"id": block_id, "type": "bulleted_list_item", "bulleted_list_item": _rich(text)}


def _gray(text: str, block_id: str = "p-gray") -> dict:
    return {
        "id": block_id,
        "type": "paragraph",
        "paragraph": {"rich_text": [{"plain_text": text, "annotations": {"color": "gray"}}]},
    }


def test_overview_state_reads_the_paragraph_under_the_heading():
    assert overview_state([_heading("❯計畫概述"), _paragraph("")]) == "empty"
    assert overview_state([_heading("❯計畫概述"), _paragraph("\u200b")]) == "empty"
    assert overview_state([_heading("❯計畫概述"), _paragraph("本案位於示範區。")]) == "filled"
    assert overview_state([_heading("❯計畫概述"), _heading("❯基地基本資料", "h-base")]) == "missing_paragraph"
    assert overview_state([_heading("❯基地基本資料", "h-base"), _paragraph("只有基地")]) == "missing_heading"


def test_applied_template_is_not_case_content():
    hints = [
        _heading("❯計畫概述"),
        _paragraph(""),
        _bullet("本案是什麼案件", "b1"),
        _bullet("為什麼辦理", "b2"),
        _bullet("主要開發目的", "b3"),
        _bullet("主要開發方向", "b4"),
        _bullet("重要背景", "b5"),
    ]
    assert overview_state(hints) == "template"
    assert overview_state([_heading("❯計畫概述"), _paragraph("本案是什麼案件")]) == "template"
    assert overview_state([_heading("❯計畫概述"), _paragraph("一般資格")]) == "template"
    assert overview_state([_heading("❯計畫概述"), _paragraph("合作聯盟（3家以內）")]) == "template"
    assert overview_state([_heading("❯計畫概述"), _gray("請在此寫計畫概述")]) == "template"
    assert overview_state([_heading("❯計畫概述"), _paragraph("1.項目")]) == "template"
    assert overview_state(
        [
            _heading("❯計畫概述"),
            {
                "id": "row",
                "type": "table_row",
                "cells": ["一般資格", "單一申請人", "合作聯盟(3家以內)"],
            },
        ]
    ) == "template"
    mixed = [
        _heading("❯計畫概述"),
        _gray("請在此寫計畫概述"),
        _paragraph("本案位於示範區。", "p-real"),
    ]
    assert overview_state(mixed) == "filled"
    written_as_bullets = [
        _heading("❯計畫概述"),
        _paragraph(""),
        _bullet("本案位於示範區。", "b-real"),
    ]
    assert overview_state(written_as_bullets) == "filled"


def test_fetch_block_children_follows_the_cursor():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        if "start_cursor" not in request.url.params:
            return httpx.Response(200, json={"results": [_heading("❯計畫概述")], "has_more": True, "next_cursor": "c2"})
        return httpx.Response(200, json={"results": [_paragraph("")], "has_more": False})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    blocks = fetch_block_children("secret", "page-1", client=client)
    assert [block["id"] for block in blocks] == ["h-overview", "p-overview"]
    assert seen[0]["page_size"] == "100"
    assert seen[1]["start_cursor"] == "c2"
    client.close()


def _page(page_id: str, name: str, status: str) -> dict:
    return {
        "id": page_id,
        "properties": {
            "案件名稱": {"type": "title", "title": [{"plain_text": name}]},
            "招商情形": {"type": "status", "status": {"name": status}},
        },
    }


def test_unfilled_route_lists_blank_pages_and_skips_filled_ones(tmp_path: Path, monkeypatch):
    root = tmp_path / "cases"
    blank = root / "1.都市更新" / "00示範市" / "示範空白案"
    blank.mkdir(parents=True)
    (blank / "申請須知.txt").write_text("計畫概述：這是空白頁要對上的文字", encoding="utf-8")
    blocks = {
        "page-blank": [_heading("❯計畫概述"), _paragraph("")],
        "page-filled": [_heading("❯計畫概述"), _paragraph("已經寫過。")],
        "page-template": [
            _heading("❯計畫概述"),
            _paragraph(""),
            _bullet("本案是什麼案件", "b1"),
            _bullet("為什麼辦理", "b2"),
        ],
        "page-orphan": [_heading("❯計畫概述"), _paragraph("")],
        "page-no-heading": [_heading("❯其他", "h-other"), _paragraph("不是概述")],
        "page-broken": [_heading("❯計畫概述")],
    }
    seen_blocks: list[str] = []

    def fake_fetch(token, database_id, *, client=None, status_filter=None):
        return {
            "results": [
                _page("page-blank", "示範空白案", "前置作業"),
                _page("page-filled", "已經填過的案", "公告中"),
                _page("page-template", "只有範本提示的案", "前置作業"),
                _page("page-orphan", "沒有資料夾的空白案", "進行中"),
                _page("page-no-heading", "沒有概述標題", "前置作業"),
                _page("page-broken", "標題下沒有段落", "前置作業"),
                _page("page-done", "招標完成案", "招標完成"),
            ],
            "has_more": False,
        }

    def fake_blocks(token, block_id, *, client=None):
        seen_blocks.append(block_id)
        if block_id == "page-error":
            raise RuntimeError("notion blocks failed: 500")
        return blocks[block_id]

    monkeypatch.setattr("rag_server.service.fetch_notion_database", fake_fetch)
    monkeypatch.setattr("rag_server.service.fetch_block_children", fake_blocks)
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
        listed = client.get("/api/pipeline/unfilled", params={"root": str(root)})
    assert listed.status_code == 200
    body = listed.json()
    assert [item["case_name"] for item in body["unfilled_matched"]] == ["示範空白案"]
    assert body["unfilled_matched"][0]["overview"] == "empty"
    assert body["unfilled_matched"][0]["folder_name"] == "示範空白案"
    assert [item["case_name"] for item in body["unfilled_without_folder"]] == [
        "只有範本提示的案",
        "沒有資料夾的空白案",
        "標題下沒有段落",
    ]
    assert body["unfilled_without_folder"][0]["overview"] == "template"
    assert body["unfilled_without_folder"][2]["overview"] == "missing_paragraph"
    assert [item["case_name"] for item in body["filled"]] == ["已經填過的案"]
    assert [item["case_name"] for item in body["missing_heading"]] == ["沒有概述標題"]
    assert body["errors"] == []
    assert "page-done" not in seen_blocks
    assert blank.is_dir()

    bare = Settings(
        data_dir=str(tmp_path / "empty"),
        embedding_backend="fake",
        llm_base_url="",
        llm_api_key="",
        llm_model="",
    )
    with TestClient(create_app(bare, embedder=FakeEmbedder())) as client:
        denied = client.get("/api/pipeline/unfilled", params={"root": str(root)})
    assert denied.status_code == 400
