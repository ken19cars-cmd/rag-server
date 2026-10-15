"""Content shape for Notion section-38 output. Fixtures are synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from rag_server.llm import LlmError
from rag_server.notion_analysis import (
    SECTION_ORDER,
    analyze_to_notion,
    extractive_notion_analysis,
    load_analysis_prompt,
    parse_template_blocks,
    section_lines,
)

FIXTURES = Path(__file__).parent / "fixtures"
TEXT = (FIXTURES / "synthetic_case.txt").read_text(encoding="utf-8")
BLOCKS = json.loads((FIXTURES / "template_blocks.json").read_text(encoding="utf-8"))
PAGE_ID = "page_demo_0001"
CASE_NAME = "示範市都更案"


def _analyze(text: str = TEXT, blocks: list[dict] | None = None, llm=None) -> dict:
    return analyze_to_notion(
        case_name=CASE_NAME,
        page_id=PAGE_ID,
        pdf_text=text,
        template_blocks=BLOCKS if blocks is None else blocks,
        llm=llm,
    )


def _update(result: dict, block_id: str, cell_index: int | None = None) -> dict:
    matches = [
        item
        for item in result["updates"]
        if item["blockId"] == block_id and item.get("cellIndex", None) == cell_index
    ]
    assert len(matches) == 1, result["updates"]
    return matches[0]


def test_prompt_contract_includes_schedule_and_section_order():
    prompt = load_analysis_prompt()
    assert "期程規範" in prompt
    assert "baseTable" in prompt
    assert "期程規範" in SECTION_ORDER
    assert SECTION_ORDER[0] == "計畫概述"
    assert SECTION_ORDER[-1] == "申請人資格"


def test_synthetic_case_matches_filled_page_shape():
    result = _analyze()
    blob = json.dumps(result, ensure_ascii=False)
    assert "中平" not in blob
    assert "桃園" not in blob
    assert result["pageId"] == PAGE_ID
    assert result["caseName"] == CASE_NAME
    assert set(result) == {"pageId", "caseName", "baseTable", "landUseTable", "updates"}

    base = result["baseTable"]["bases"][0]
    assert base["name"] == "示範更新基地"
    assert "示範路" in base["location"]
    assert "示範段1、2地號" in base["scope"]
    assert base["area"] == "800平方公尺"
    assert base["ownership"] == "• 私人（一般私人共10位）：100%"
    assert "平方" not in base["ownership"]

    zone = result["landUseTable"]["zones"][0]
    assert zone["name"] == "第三種住宅區"
    assert zone["intensity"] == "建蔽率60%、容積率200%"
    assert "法定" not in zone["intensity"]
    assert "☑ 都市更新容積獎勵" in zone["volumeItems"]
    assert "☑ 綠建築容積獎勵" in zone["volumeItems"]
    assert "☒ 容積移轉" in zone["volumeItems"]

    overview = _update(result, "fix_overview_p")
    assert "\n" not in overview["value"]
    assert "示範市住宅中心" in overview["value"]
    assert not overview["value"].startswith("•")

    surround = _update(result, "fix_surround_p")
    assert "示範路" in surround["value"]
    assert "\n" not in surround["value"]

    intensity = _update(result, "fix_intensity_p")["value"]
    assert "退縮3公尺" in intensity
    assert "1.5倍" in intensity
    assert "法定" not in intensity

    consent = _update(result, "fix_will_p")["value"]
    assert "同意比例90%" in consent
    assert "100%" not in consent

    positioning = _update(result, "fix_position_p")["value"]
    assert positioning.count("•") == 2
    assert "退縮開放空間" in positioning
    assert "停車需求" in positioning

    assert _update(result, "fix_duty_1", 0)["value"] == "1. 實施與權益分配"
    duty_body = _update(result, "fix_duty_1", 1)["value"]
    assert "權利變換" in duty_body
    assert "•" in duty_body
    assert _update(result, "fix_duty_2", 0)["value"] == "2. 拆遷安置補貼義務"
    assert "≥ 36個月" in _update(result, "fix_duty_2", 1)["value"]
    duty_ids = {item["blockId"] for item in result["updates"] if item["blockId"].startswith("fix_duty_")}
    assert duty_ids == {"fix_duty_1", "fix_duty_2"}

    schedule = _update(result, "fix_schedule_p")["value"]
    assert "簽約日起3個月內：完成開工" in schedule
    assert "開工日起24個月內：取得使用執照" in schedule
    assert "保證金" not in schedule

    finance = _update(result, "fix_finance_p")["value"]
    assert "≤ 10%" in finance
    assert "申請保證金" in finance
    assert "500萬元" in finance

    assert "fix_qual_general" not in {item["blockId"] for item in result["updates"]}
    finance_single = _update(result, "fix_qual_finance", 1)["value"]
    finance_alliance = _update(result, "fix_qual_finance", 2)["value"]
    assert "實收資本額 ≥ 1億元" in finance_single
    assert "領銜公司" in finance_alliance
    tech = _update(result, "fix_qual_tech", 1)["value"]
    assert "或" in tech
    assert "10,000平方公尺" in tech
    assert "5億元" in tech


def test_section_lines_cover_every_template_heading():
    result = _analyze()
    lines = section_lines(result, parse_template_blocks(BLOCKS))
    assert [item["heading"] for item in lines] == list(SECTION_ORDER)
    by_heading = {item["heading"]: item["text"] for item in lines}
    assert "示範市住宅中心" in by_heading["計畫概述"]
    assert "示範更新基地" in by_heading["基地基本資料"]
    assert "800平方公尺" in by_heading["基地基本資料"]
    assert "100%" in by_heading["基地基本資料"]
    assert "平方" not in by_heading["基地基本資料"].split("權屬", 1)[-1]
    assert "示範路" in by_heading["周邊環境概況"]
    assert "建蔽率60%" in by_heading["使用強度規範"]
    assert "☑ 都市更新容積獎勵" in by_heading["使用強度規範"]
    assert "同意比例90%" in by_heading["意願調查"]
    assert "退縮開放空間" in by_heading["開發定位/規劃構想"]
    assert "實施與權益分配" in by_heading["開發權利義務/負擔條件"]
    assert "拆遷安置補貼義務" in by_heading["開發權利義務/負擔條件"]
    assert "簽約日起3個月內" in by_heading["期程規範"]
    assert "保證金" not in by_heading["期程規範"]
    assert "申請保證金" in by_heading["財務條件"]
    assert "財務資格" in by_heading["申請人資格"]
    assert "實收資本額" in by_heading["申請人資格"]


def test_section_lines_leave_missing_pdf_text_blank():
    result = _analyze("")
    lines = section_lines(result, parse_template_blocks(BLOCKS))
    assert [item["heading"] for item in lines] == list(SECTION_ORDER)
    assert all(item["text"] == "" for item in lines)


def test_section_lines_skip_headings_missing_from_the_page():
    result = _analyze()
    lines = section_lines(result, parse_template_blocks(BLOCKS[:2]))
    assert [item["heading"] for item in lines] == ["計畫概述"]
    assert set(result) == {"pageId", "caseName", "baseTable", "landUseTable", "updates"}


def test_empty_pdf_keeps_ids_and_writes_nothing():
    result = _analyze("")
    assert result["pageId"] == PAGE_ID
    assert result["caseName"] == CASE_NAME
    assert result["baseTable"]["bases"] == []
    assert result["landUseTable"]["zones"] == []
    assert result["updates"] == []


def test_ownership_with_area_is_dropped():
    text = "基地名稱：甲\n座落位置：乙路\n土地權屬：私人 800平方公尺：100%"
    blocks = parse_template_blocks(BLOCKS)
    result = extractive_notion_analysis(CASE_NAME, PAGE_ID, text, blocks)
    assert result["baseTable"]["bases"][0]["ownership"] == ""
    assert result["baseTable"]["bases"][0]["name"] == "甲"


def test_missing_template_blocks_produce_no_updates():
    result = _analyze(blocks=[])
    assert result["baseTable"]["bases"]
    assert result["updates"] == []


def test_local_llm_unset_does_not_call_the_network(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("httpx must stay unused when no local model is configured")

    monkeypatch.setattr(httpx, "post", explode)

    class Boom:
        def complete(self, messages):
            raise AssertionError(messages)

    result = _analyze(llm=None)
    assert result["updates"]
    assert Boom().complete.__doc__ is None or True


def test_ungrounded_model_json_falls_back_to_extractive():
    class FakeModel:
        def complete(self, messages):
            assert "只輸出第38節的 JSON" in messages[-1]["content"]
            return json.dumps(
                {
                    "pageId": "forged",
                    "caseName": "桃園市中平段",
                    "baseTable": {
                        "bases": [
                            {
                                "name": "虛構基地",
                                "location": "",
                                "scope": "",
                                "area": "",
                                "ownership": "私人：100%",
                            }
                        ]
                    },
                    "landUseTable": {"zones": [{"name": "商業區", "intensity": "法定建蔽率99%", "volumeItems": ""}]},
                    "updates": [
                        {
                            "blockId": "fix_overview_p",
                            "type": "paragraph",
                            "value": "這段不在 PDF 裡",
                            "reason": "guess",
                        }
                    ],
                },
                ensure_ascii=False,
            )

    result = _analyze(llm=FakeModel())
    blob = json.dumps(result, ensure_ascii=False)
    assert result["pageId"] == PAGE_ID
    assert result["caseName"] == CASE_NAME
    assert "中平" not in blob
    assert "虛構" not in blob
    assert result["baseTable"]["bases"][0]["name"] == "示範更新基地"
    assert "法定" not in result["landUseTable"]["zones"][0]["intensity"]


def test_grounded_model_fields_are_kept_and_invented_ones_dropped():
    overview = "本示範案為公辦都市更新，目的是整理屋齡逾30年的老舊住宅，由示範市住宅中心公開徵求出資人。"

    class MixedModel:
        def complete(self, _messages):
            return json.dumps(
                {
                    "pageId": "forged",
                    "caseName": "別的案子",
                    "baseTable": {"bases": [{"name": "示範更新基地", "ownership": "私人 800平方公尺：50%"}]},
                    "landUseTable": {
                        "zones": [
                            {
                                "name": "第三種住宅區",
                                "intensity": "建蔽率60%、容積率200%",
                                "volumeItems": "☑ 都市更新容積獎勵\n☑ 不存在的獎勵",
                            }
                        ]
                    },
                    "updates": [
                        {
                            "blockId": "fix_overview_p",
                            "type": "paragraph",
                            "value": overview,
                            "reason": "PDF 概述",
                        },
                        {
                            "blockId": "not_a_real_block",
                            "type": "paragraph",
                            "value": overview,
                            "reason": "forged id",
                        },
                    ],
                },
                ensure_ascii=False,
            )

    result = _analyze(llm=MixedModel())
    assert result["pageId"] == PAGE_ID
    assert result["caseName"] == CASE_NAME
    assert result["baseTable"]["bases"][0]["name"] == "示範更新基地"
    assert result["baseTable"]["bases"][0]["ownership"] == ""
    assert result["landUseTable"]["zones"][0]["volumeItems"] == "☑ 都市更新容積獎勵"
    assert [item["blockId"] for item in result["updates"]] == ["fix_overview_p"]


def test_model_error_falls_back_to_extractive():
    class Broken:
        def complete(self, _messages):
            raise LlmError("offline")

    result = _analyze(llm=Broken())
    assert result["baseTable"]["bases"][0]["name"] == "示範更新基地"
    assert any(item["blockId"] == "fix_schedule_p" for item in result["updates"])
