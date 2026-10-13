"""Local 類型 / 縣市 / 案名 / 文件 tree. No live disk and no network."""

from __future__ import annotations

import shutil
from pathlib import Path

from rag_server.corpus import classify_document, merge_document_texts, normalize_city, scan_cases_root
from rag_server.service import CaseService
from tests.pdf_bytes import build_text_pdf

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = FIXTURES / "cases_root"


def test_city_prefix_and_document_kinds():
    assert normalize_city("00基隆市") == "基隆市"
    assert classify_document("申請須知.pdf") == ("application_notice", "申請須知")
    assert classify_document("需求說明書.txt") == ("requirements", "需求說明書")
    assert classify_document("評選文件.pdf") == ("selection", "評選文件")
    assert classify_document("實施出資契約.pdf") == ("investment_contract", "實施(出資)契約")

    cases = {case.case_name: case for case in scan_cases_root(ROOT)}
    assert set(cases) == {"示範市都更案", "示範聯開案", "示範促參案", "示範標售案"}
    urban = cases["示範市都更案"]
    assert urban.case_type == "1.都市更新"
    assert urban.city == "示範市"
    assert urban.city_folder == "00示範市"
    kinds = {document.filename: document.doc_kind for document in urban.documents}
    assert kinds["申請須知.txt"] == "application_notice"
    assert kinds["評選文件.pdf"] == "selection"
    assert cases["示範聯開案"].city == "示範縣"
    assert cases["示範聯開案"].documents[0].doc_kind == "requirements"
    assert cases["示範促參案"].documents[0].doc_kind == "selection"
    assert cases["示範標售案"].documents[0].doc_kind == "investment_contract"


def test_merge_uses_prompt_separators():
    merged = merge_document_texts([("申請須知", "申請須知.txt", "計畫概述：示範")])
    assert "========== 文件 1 ==========" in merged
    assert "文件類型：申請須知" in merged
    assert "檔案名稱：申請須知.txt" in merged
    assert "========== 文件 1 結束 ==========" in merged


def test_pipeline_detect_ingest_extract_and_notion(client, tmp_path: Path):
    root = str(ROOT)
    detected = client.post("/api/pipeline/detect", json={"root": root})
    assert detected.status_code == 200
    names = {item["case_name"] for item in detected.json()["cases"]}
    assert "示範市都更案" in names
    urban = next(item for item in detected.json()["cases"] if item["case_name"] == "示範市都更案")
    assert urban["city"] == "示範市"
    assert any(document["doc_kind"] == "application_notice" for document in urban["documents"])

    missing = client.post("/api/pipeline/detect", json={})
    assert missing.status_code == 400

    ingested = client.post("/api/pipeline/ingest", json={"root": root})
    assert ingested.status_code == 200
    body = ingested.json()
    assert body["documents_indexed"] == 5
    again = client.post("/api/pipeline/ingest", json={"root": root})
    assert again.status_code == 200
    assert again.json()["documents_indexed"] == 0

    listed = client.get("/api/cases").json()
    stored = next(item for item in listed if item["name"] == "示範市都更案")
    assert stored["city"] == "示範市"
    assert stored["case_type"] == "1.都市更新"
    assert stored["stage"] == "in_progress"
    detail = client.get(f"/api/cases/{stored['id']}").json()
    notice = next(item for item in detail["documents"] if item["filename"] == "申請須知.txt")
    assert notice["doc_kind"] == "application_notice"
    assert notice["source_path"].endswith("申請須知.txt")

    folder = urban["source_path"]
    extracted = client.post("/api/pipeline/extract", json={"root": root, "caseFolder": folder})
    assert extracted.status_code == 200
    assert extracted.json()["caseName"] == "示範市都更案"
    assert "========== 文件 1 ==========" in extracted.json()["pdfText"]
    assert "示範市住宅中心" in extracted.json()["pdfText"]

    blocks = json_blocks()
    analyzed = client.post(
        "/api/notion/analyze",
        json={
            "caseName": "示範市都更案",
            "pageId": "page_demo_0001",
            "caseFolder": folder,
            "root": root,
            "templateBlocks": blocks,
        },
    )
    assert analyzed.status_code == 200
    payload = analyzed.json()
    assert set(payload) == {"pageId", "caseName", "baseTable", "landUseTable", "updates"}
    assert payload["pageId"] == "page_demo_0001"
    assert payload["baseTable"]["bases"][0]["ownership"].endswith("100%")
    assert any(item["blockId"] == "fix_schedule_p" for item in payload["updates"])
    assert "cellIndex" not in next(item for item in payload["updates"] if item["type"] == "paragraph")

    for path in (
        "/api/pipeline/notion/write",
        "/api/pipeline/schedule/svg",
        "/api/pipeline/schedule/pptx",
    ):
        blocked = client.post(path)
        assert blocked.status_code == 501
        assert blocked.json()["detail"]["phase"] == 2
        assert blocked.json()["detail"]["status"] == "not_implemented"

    copied = tmp_path / "only-text"
    shutil.copytree(ROOT / "4.招標標售", copied / "4.招標標售")
    service: CaseService = client.app.state.service
    scanned = service.scan_corpus(str(copied))
    assert scanned[0]["case_name"] == "示範標售案"
    assert build_text_pdf(["x"]).startswith(b"%PDF-")


def json_blocks() -> list[dict]:
    import json

    return json.loads((FIXTURES / "template_blocks.json").read_text(encoding="utf-8"))


def _write_nested_case(root: Path) -> Path:
    case = root / "1.都市更新" / "00示範市" / "示範市都更案"
    viewing = case / "1150921公開閱覽"
    viewing.mkdir(parents=True)
    (viewing / "更深").mkdir()
    (case / "__MACOSX" / "1150921公開閱覽").mkdir(parents=True)
    (case / ".git").mkdir()
    (case / "notes" / "__MACOSX").mkdir(parents=True)
    (case / "申請須知.txt").write_text("頂層須知", encoding="utf-8")
    (viewing / "補充.txt").write_text("公開閱覽補充", encoding="utf-8")
    (viewing / "更深" / "第二層.txt").write_text("第二層補充", encoding="utf-8")
    (viewing / "公開說明.pdf").write_bytes(
        build_text_pdf(["nested public viewing document for this case"])
    )
    (case / "__MACOSX" / "1150921公開閱覽" / "macosx-junk.pdf").write_bytes(b"%PDF-1.4 junk")
    (case / ".git" / "git-secret.txt").write_text("secret", encoding="utf-8")
    (case / "notes" / "__MACOSX" / "buried.pdf").write_bytes(b"%PDF-1.4 buried")
    return case


def test_scan_includes_nested_pdf_and_txt_but_skips_macosx_and_git(tmp_path: Path):
    case_dir = _write_nested_case(tmp_path)
    cases = scan_cases_root(tmp_path)
    assert len(cases) == 1
    case = cases[0]
    assert case.folder == case_dir
    names = {document.filename for document in case.documents}
    assert names == {"申請須知.txt", "補充.txt", "第二層.txt", "公開說明.pdf"}
    paths = {document.path.relative_to(case_dir).as_posix() for document in case.documents}
    assert "1150921公開閱覽/公開說明.pdf" in paths
    assert "1150921公開閱覽/更深/第二層.txt" in paths
    assert all("__MACOSX" not in path and not path.startswith(".git/") for path in paths)


def test_detect_ingest_extract_share_nested_documents(client, tmp_path: Path):
    case_dir = _write_nested_case(tmp_path)
    root = str(tmp_path)
    expected = {"申請須知.txt", "補充.txt", "第二層.txt", "公開說明.pdf"}

    detected = client.post("/api/pipeline/detect", json={"root": root})
    assert detected.status_code == 200
    urban = detected.json()["cases"][0]
    detected_names = {document["filename"] for document in urban["documents"]}
    assert detected_names == expected
    assert all("__MACOSX" not in document["source_path"] for document in urban["documents"])
    assert all("/.git/" not in document["source_path"] for document in urban["documents"])

    ingested = client.post("/api/pipeline/ingest", json={"root": root})
    assert ingested.status_code == 200
    assert ingested.json()["documents_indexed"] == 4
    again = client.post("/api/pipeline/ingest", json={"root": root})
    assert again.json()["documents_indexed"] == 0

    extracted = client.post(
        "/api/pipeline/extract",
        json={"root": root, "caseFolder": str(case_dir)},
    )
    assert extracted.status_code == 200
    text = extracted.json()["pdfText"]
    assert "公開閱覽補充" in text
    assert "第二層補充" in text
    assert "nested public viewing document" in text
    assert "secret" not in text
    assert "junk" not in text
    assert "buried" not in text
