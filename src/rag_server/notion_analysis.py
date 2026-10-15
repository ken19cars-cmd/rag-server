"""Fill Ken's Notion analysis JSON from PDF text. Never invent missing facts.

The checked-in prompt is the contract. Section order follows the blank case
template (❯計畫概述 … ❯申請人資格, including ❯期程規範). Output is only the
section-38 object: pageId, caseName, baseTable, landUseTable, updates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag_server.llm import LlmClient, LlmError, parse_json_object
from rag_server.textutil import normalize_for_match

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "notion-case-analysis.md"

SECTION_ORDER = (
    "計畫概述",
    "基地基本資料",
    "周邊環境概況",
    "使用強度規範",
    "意願調查",
    "開發定位/規劃構想",
    "開發權利義務/負擔條件",
    "期程規範",
    "財務條件",
    "申請人資格",
)

QUALIFICATION_ROWS = ("財務資格", "財務結構指標", "信用及納稅規範", "專業技術能力")

_STOP_TITLES = (
    "計畫概述",
    "基地基本資料",
    "周邊環境概況",
    "使用強度規範",
    "意願調查",
    "開發定位",
    "規劃構想",
    "開發權利義務",
    "期程規範",
    "財務條件",
    "申請人資格",
    "一般資格",
    *QUALIFICATION_ROWS,
)


@dataclass
class TemplateBlock:
    id: str
    type: str
    text: str
    cells: list[str] = field(default_factory=list)


def load_analysis_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def parse_template_blocks(raw_blocks: list[dict]) -> list[TemplateBlock]:
    parsed: list[TemplateBlock] = []
    for item in raw_blocks:
        if not isinstance(item, dict):
            continue
        block_id = str(item.get("id") or item.get("blockId") or "").strip()
        if not block_id:
            continue
        block_type = str(item.get("type") or "paragraph")
        cells = _cells_from_block(item)
        text = str(item.get("text") or item.get("plain_text") or "").strip()
        if not text and cells:
            text = cells[0]
        if not text:
            text = _rich_text(item.get(block_type))
        parsed.append(TemplateBlock(id=block_id, type=block_type, text=text, cells=cells))
    return parsed


def analyze_to_notion(
    *,
    case_name: str,
    page_id: str,
    pdf_text: str,
    template_blocks: list[dict],
    llm: LlmClient | None = None,
) -> dict:
    """Return the section-38 JSON. pageId and caseName are copied from the input."""
    blocks = parse_template_blocks(template_blocks)
    extractive = extractive_notion_analysis(case_name, page_id, pdf_text, blocks)
    if llm is None or not pdf_text.strip():
        return extractive
    try:
        raw = llm.complete(_llm_messages(case_name, page_id, pdf_text, template_blocks))
    except LlmError:
        return extractive
    accepted = _ground_model_json(raw, case_name, page_id, pdf_text, blocks)
    return accepted if accepted is not None else extractive


def extractive_notion_analysis(
    case_name: str,
    page_id: str,
    pdf_text: str,
    blocks: list[TemplateBlock],
) -> dict:
    text = pdf_text or ""
    result = _empty_result(page_id, case_name)
    result["baseTable"] = {"bases": _bases_from_text(text)}
    result["landUseTable"] = {"zones": _zones_from_text(text)}
    result["updates"] = _updates_from_text(text, blocks)
    return result


def _empty_result(page_id: str, case_name: str) -> dict:
    return {
        "pageId": page_id,
        "caseName": case_name,
        "baseTable": {"bases": []},
        "landUseTable": {"zones": []},
        "updates": [],
    }


def _bases_from_text(text: str) -> list[dict]:
    name = _labeled(text, ("基地名稱", "基地別"))
    location = _labeled(text, ("座落位置", "座落"))
    scope = _labeled(text, ("計畫範圍", "範圍"))
    area = _labeled(text, ("計畫面積", "面積"))
    ownership = _ownership(text)
    if not any((name, location, scope, area, ownership)):
        return []
    return [
        {
            "name": name,
            "location": location,
            "scope": scope,
            "area": area,
            "ownership": ownership,
        }
    ]


def _ownership(text: str) -> str:
    line = _labeled(text, ("土地權屬",))
    if not line or "%" not in line:
        return ""
    if "平方" in line:
        return ""
    bullet = line if line.startswith("•") else f"• {line}"
    return bullet


def _zones_from_text(text: str) -> list[dict]:
    name = _labeled(text, ("使用分區", "土地使用分區"))
    intensity = _intensity(text)
    volume = _volume_items(text)
    if not any((name, intensity, volume)):
        return []
    return [{"name": name, "intensity": intensity, "volumeItems": volume}]


def _intensity(text: str) -> str:
    coverage = re.search(r"(?:法定)?建蔽率(?:為)?\s*[:：]?\s*(\d+(?:\.\d+)?)\s*%", text)
    floor = re.search(r"(?:法定)?容積率(?:為)?\s*[:：]?\s*(\d+(?:\.\d+)?)\s*%", text)
    parts: list[str] = []
    if coverage:
        parts.append(f"建蔽率{coverage.group(1)}%")
    if floor:
        parts.append(f"容積率{floor.group(1)}%")
    return "、".join(parts)


def _volume_items(text: str) -> str:
    lines: list[str] = []
    for match in re.finditer(r"本案可以利用[:：]\s*([^\n。；;]{2,40})", text):
        lines.append(f"☑ {match.group(1).strip()}")
    for match in re.finditer(r"本案禁止[:：]\s*([^\n。；;]{2,40})", text):
        lines.append(f"☒ {match.group(1).strip()}")
    return "\n".join(lines)


def _updates_from_text(text: str, blocks: list[TemplateBlock]) -> list[dict]:
    updates: list[dict] = []
    _add_paragraph(updates, blocks, "計畫概述", _paragraph(_section(text, "計畫概述")))
    _add_paragraph(updates, blocks, "周邊環境概況", _paragraph(_section(text, "周邊環境概況")))
    constraints = _constraint_lines(_section(text, "使用強度規範") or text)
    _add_paragraph(updates, blocks, "使用強度規範", _bullets(constraints))
    _add_paragraph(updates, blocks, "意願調查", _consent(_section(text, "意願調查") or text))
    positioning = _positioning(text)
    _add_paragraph(updates, blocks, "開發定位/規劃構想", positioning)
    _add_duty_rows(updates, blocks, _duty_items(text))
    _add_paragraph(updates, blocks, "期程規範", _schedule(_section(text, "期程規範")))
    _add_paragraph(updates, blocks, "財務條件", _bullets(_finance_lines(_section(text, "財務條件"))))
    _add_qualification_rows(updates, blocks, text)
    return updates


def _add_paragraph(updates: list[dict], blocks: list[TemplateBlock], heading: str, value: str) -> None:
    if not value:
        return
    target = _paragraph_after(blocks, heading)
    if target is None or target.text.strip() == value.strip():
        return
    updates.append(
        {
            "blockId": target.id,
            "type": "paragraph",
            "value": value,
            "reason": f"PDF 有{heading}可引用的文字",
        }
    )


def _add_duty_rows(updates: list[dict], blocks: list[TemplateBlock], items: list[tuple[str, str]]) -> None:
    rows = _rows_after(blocks, "開發權利義務")
    for row, (title, body) in zip(rows, items):
        _put_row(updates, row, 0, title, "PDF 有此權利義務標題")
        _put_row(updates, row, 1, body, "PDF 有此權利義務內容")


def _add_qualification_rows(updates: list[dict], blocks: list[TemplateBlock], text: str) -> None:
    general = _row_by_label(blocks, "一般資格")
    if general is not None:
        single = "單一申請人" if "單一申請人" in text else ""
        alliance = _alliance_label(text)
        _put_row(updates, general, 1, single, "PDF 寫明單一申請人")
        _put_row(updates, general, 2, alliance, "PDF 寫明合作聯盟家數")
    for label in QUALIFICATION_ROWS:
        row = _row_by_label(blocks, label)
        if row is None:
            continue
        body = _section(text, label)
        single, alliance = _split_applicant_sides(body)
        _put_row(updates, row, 1, single, f"PDF 有{label}")
        _put_row(updates, row, 2, alliance, f"PDF 有{label}的合作聯盟條件")


def _put_row(
    updates: list[dict],
    row: TemplateBlock,
    cell_index: int,
    value: str,
    reason: str,
) -> None:
    if not value:
        return
    current = row.cells[cell_index] if cell_index < len(row.cells) else ""
    if current.strip() == value.strip():
        return
    updates.append(_row_update(row, cell_index, value, reason))


def _row_update(row: TemplateBlock, cell_index: int, value: str, reason: str) -> dict:
    return {
        "blockId": row.id,
        "type": "table_row",
        "cellIndex": cell_index,
        "value": value,
        "reason": reason,
    }


def _paragraph_after(blocks: list[TemplateBlock], heading: str) -> TemplateBlock | None:
    for block in _following(blocks, heading):
        if block.type == "paragraph" or block.type.startswith("paragraph"):
            return block
    return None


def _rows_after(blocks: list[TemplateBlock], heading: str) -> list[TemplateBlock]:
    return [block for block in _following(blocks, heading) if block.type == "table_row"]


def _following(blocks: list[TemplateBlock], heading: str) -> list[TemplateBlock]:
    start = _heading_index(blocks, heading)
    if start is None:
        return []
    found: list[TemplateBlock] = []
    for block in blocks[start + 1 :]:
        if block.type.startswith("heading"):
            break
        found.append(block)
    return found


def _heading_index(blocks: list[TemplateBlock], heading: str) -> int | None:
    needle = _norm_heading(heading)
    for index, block in enumerate(blocks):
        if not block.type.startswith("heading"):
            continue
        if needle and needle in _norm_heading(block.text):
            return index
    return None


def _row_by_label(blocks: list[TemplateBlock], label: str) -> TemplateBlock | None:
    for block in blocks:
        if block.type != "table_row" or not block.cells:
            continue
        if label in block.cells[0]:
            return block
    return None


def _section(text: str, title: str) -> str:
    lines = text.splitlines()
    pattern = re.compile(rf"{re.escape(title)}\s*[:：]?\s*(.*)")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if title not in stripped or stripped.find(title) > 12:
            continue
        match = pattern.search(stripped)
        if not match:
            continue
        inline = match.group(1).strip()
        if inline:
            return inline
        chunks: list[str] = []
        for follow in lines[index + 1 :]:
            if not follow.strip():
                if chunks:
                    break
                continue
            if _is_stop(follow):
                break
            chunks.append(follow.strip())
        return "\n".join(chunks).strip()
    return ""


def _is_stop(line: str) -> bool:
    stripped = line.strip().lstrip("❯#* ")
    if not stripped:
        return False
    if re.match(r"^\d+\.\s+\S", stripped):
        return True
    return any(stripped.startswith(title) for title in _STOP_TITLES)


def _labeled(text: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*[:：]\s*([^\n]+)", text)
        if match:
            return match.group(1).strip()
    return ""


def _paragraph(value: str) -> str:
    return " ".join(value.split()).strip()


def _consent(text: str) -> str:
    kept = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。；;])", text)
        if sentence.strip() and re.search(r"同意比例|同意率|不同意比例", sentence)
    ]
    return _bullets(kept)


def _positioning(text: str) -> str:
    lines = []
    for title in ("開發定位", "規劃構想"):
        body = _section(text, title)
        if body:
            lines.append(body)
    return _bullets(lines)


def _duty_items(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^(\d+)\.\s*([^：:\n]{2,40})[:：]\s*(.+)$", text))
    items: list[tuple[str, str]] = []
    for match in matches:
        title = f"{match.group(1)}. {match.group(2).strip()}"
        sentences = [part.strip() for part in re.split(r"(?<=。)", match.group(3).strip()) if part.strip()]
        items.append((title, _bullets(sentences)))
    return items


_SCHEDULE_ITEM = re.compile(
    r"^自?(?P<start>\S+?日起)(?P<deadline>\d+\S*?(?:個月|年|日))內(?P<work>.+)$"
)


def _schedule(text: str) -> str:
    kept = []
    for sentence in _sentences(text):
        if "保證金" in sentence:
            continue
        if not re.search(r"日起|個月|期限|使用執照", sentence):
            continue
        kept.append(_format_schedule_sentence(sentence))
    return _bullets(kept)


def _format_schedule_sentence(sentence: str) -> str:
    cleaned = sentence.strip().rstrip("。").strip()
    match = _SCHEDULE_ITEM.match(cleaned)
    if not match:
        return cleaned
    return f"{match.group('start')}{match.group('deadline')}內：{match.group('work').strip()}"


def _finance_lines(text: str) -> list[str]:
    kept = []
    for sentence in _sentences(text):
        if any(token in sentence for token in ("≥", "≤", "＞", "＜", "保證金", "共同負擔")):
            kept.append(sentence)
    return kept


def _sentences(text: str) -> list[str]:
    pieces: list[str] = []
    for line in text.splitlines() or [text]:
        pieces.extend(part.strip() for part in re.split(r"(?<=。)", line) if part.strip())
    return pieces


def _constraint_lines(text: str) -> list[str]:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "退縮" in stripped or "倍" in stripped:
            kept.append(stripped)
    return kept


def _split_applicant_sides(body: str) -> tuple[str, str]:
    single: list[str] = []
    alliance: list[str] = []
    pieces = [part.strip() for part in re.split(r"(?<=。)", body) if part.strip()]
    if not pieces and body.strip():
        pieces = [body.strip()]
    for piece in pieces:
        if re.search(r"領銜|全體|各成員|任一成員", piece):
            alliance.append(piece)
        else:
            single.append(piece)
    return _bullets(single), _bullets(alliance)


def _alliance_label(text: str) -> str:
    match = re.search(r"合作聯盟\s*[（(]\s*\d+\s*家以內\s*[）)]", text)
    if not match:
        return ""
    return re.sub(r"\s+", "", match.group(0)).replace("（", "(").replace("）", ")")


def _bullets(lines: list[str]) -> str:
    cleaned = [line.strip().lstrip("•").strip() for line in lines if line and line.strip()]
    if not cleaned:
        return ""
    return "\n".join(f"• {line}" for line in cleaned)


def _norm_heading(text: str) -> str:
    return re.sub(r"[\s❯*#／]", "", text.replace("/", ""))


def _cells_from_block(item: dict) -> list[str]:
    if isinstance(item.get("cells"), list):
        return [_cell_text(cell) for cell in item["cells"]]
    typed = item.get(str(item.get("type") or ""))
    if isinstance(typed, dict) and isinstance(typed.get("cells"), list):
        return [_cell_text(cell) for cell in typed["cells"]]
    return []


def _cell_text(cell: object) -> str:
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, list):
        return "".join(_rich_text(part) if isinstance(part, dict) else str(part) for part in cell).strip()
    if isinstance(cell, dict):
        return _rich_text(cell)
    return str(cell or "").strip()


def _rich_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if isinstance(value.get("rich_text"), list):
            return "".join(str(part.get("plain_text") or "") for part in value["rich_text"]).strip()
        if "plain_text" in value:
            return str(value.get("plain_text") or "").strip()
    return ""


def section_lines(result: dict, blocks: list[TemplateBlock]) -> list[dict]:
    """One line per template heading that is actually on this page.

    The section-38 JSON stays unchanged. This is only the readout for every
    heading the prompt fills: paragraph, base table, land-use table, or rows.
    """
    by_block: dict[str, list[dict]] = {}
    for item in result.get("updates") or []:
        if isinstance(item, dict):
            by_block.setdefault(str(item.get("blockId") or ""), []).append(item)
    lines: list[dict] = []
    for heading in SECTION_ORDER:
        if _heading_index(blocks, heading) is None:
            continue
        lines.append({"heading": heading, "text": _section_display(heading, result, blocks, by_block)})
    return lines


def _section_display(
    heading: str,
    result: dict,
    blocks: list[TemplateBlock],
    by_block: dict[str, list[dict]],
) -> str:
    parts: list[str] = []
    if heading == "基地基本資料":
        parts.append(_bases_text(result))
    elif heading == "使用強度規範":
        parts.append(_zones_text(result))
    paragraph = _paragraph_after(blocks, heading)
    if paragraph is not None:
        values = [str(item.get("value") or "").strip() for item in by_block.get(paragraph.id, [])]
        parts.extend(value for value in values if value)
    if heading.startswith("開發權利義務") or heading == "申請人資格":
        row_text = _row_text(_rows_after(blocks, heading), by_block)
        if row_text:
            parts.append(row_text)
    return "\n".join(part for part in parts if part).strip()


def _bases_text(result: dict) -> str:
    bases = ((result.get("baseTable") or {}).get("bases")) or []
    chunks: list[str] = []
    for base in bases:
        if not isinstance(base, dict):
            continue
        rows = [str(base.get("name") or "").strip()]
        for label, key in (("座落", "location"), ("範圍", "scope"), ("面積", "area"), ("權屬", "ownership")):
            value = str(base.get(key) or "").strip()
            if value:
                rows.append(f"{label}：{value}")
        text = "\n".join(row for row in rows if row)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def _zones_text(result: dict) -> str:
    zones = ((result.get("landUseTable") or {}).get("zones")) or []
    chunks: list[str] = []
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        rows = [str(zone.get("name") or "").strip(), str(zone.get("intensity") or "").strip()]
        volume = str(zone.get("volumeItems") or "").strip()
        if volume:
            rows.append(volume)
        text = "\n".join(row for row in rows if row)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def _row_text(rows: list[TemplateBlock], by_block: dict[str, list[dict]]) -> str:
    chunks: list[str] = []
    for row in rows:
        items = sorted(by_block.get(row.id) or [], key=lambda item: int(item.get("cellIndex") or 0))
        if not items:
            continue
        updated = {int(item.get("cellIndex") or 0) for item in items}
        values = [str(item.get("value") or "").strip() for item in items if str(item.get("value") or "").strip()]
        if row.cells and 0 not in updated and row.cells[0].strip():
            values = [row.cells[0].strip(), *values]
        line = " | ".join(values)
        if line:
            chunks.append(line)
    return "\n".join(chunks)


def _llm_messages(
    case_name: str,
    page_id: str,
    pdf_text: str,
    template_blocks: list[dict],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_analysis_prompt()},
        {
            "role": "user",
            "content": (
                f"案件名稱：\n{case_name}\n\n"
                f"Page ID：\n{page_id}\n\n"
                f"PDF：\n{pdf_text}\n\n"
                "Notion Blocks：\n"
                f"{json.dumps(template_blocks, ensure_ascii=False)}\n\n"
                "只輸出第38節的 JSON。"
            ),
        },
    ]


def _ground_model_json(
    raw: str,
    case_name: str,
    page_id: str,
    pdf_text: str,
    blocks: list[TemplateBlock],
) -> dict | None:
    payload = parse_json_object(raw)
    if not isinstance(payload, dict):
        return None
    result = _empty_result(page_id, case_name)
    bases = ((payload.get("baseTable") or {}).get("bases") if isinstance(payload.get("baseTable"), dict) else None) or []
    kept_bases = []
    if isinstance(bases, list):
        for base in bases:
            if not isinstance(base, dict):
                continue
            cleaned = {
                "name": _keep(base.get("name"), pdf_text),
                "location": _keep(base.get("location"), pdf_text),
                "scope": _keep(base.get("scope"), pdf_text),
                "area": _keep(base.get("area"), pdf_text),
                "ownership": _keep_ownership(base.get("ownership"), pdf_text),
            }
            if any(cleaned.values()):
                kept_bases.append(cleaned)
    result["baseTable"] = {"bases": kept_bases}
    zones = (
        (payload.get("landUseTable") or {}).get("zones")
        if isinstance(payload.get("landUseTable"), dict)
        else None
    ) or []
    kept_zones = []
    if isinstance(zones, list):
        for zone in zones:
            if not isinstance(zone, dict):
                continue
            cleaned_zone = {
                "name": _keep(zone.get("name"), pdf_text),
                "intensity": _keep_intensity(zone.get("intensity"), pdf_text),
                "volumeItems": _keep_volume(zone.get("volumeItems"), pdf_text),
            }
            if any(cleaned_zone.values()):
                kept_zones.append(cleaned_zone)
    result["landUseTable"] = {"zones": kept_zones}
    known = {block.id: block for block in blocks}
    updates = []
    for item in payload.get("updates") or []:
        if not isinstance(item, dict):
            continue
        block_id = str(item.get("blockId") or "")
        block = known.get(block_id)
        value = item.get("value")
        if block is None or not isinstance(value, str) or not _grounded(value, pdf_text):
            continue
        update = {
            "blockId": block_id,
            "type": "table_row" if block.type == "table_row" else "paragraph",
            "value": value.strip(),
            "reason": str(item.get("reason") or "PDF 可支持此段"),
        }
        if block.type == "table_row":
            cell_index = item.get("cellIndex")
            if not isinstance(cell_index, int):
                continue
            update["cellIndex"] = cell_index
            update["type"] = "table_row"
        updates.append(update)
    result["updates"] = updates
    if not kept_bases and not kept_zones and not updates:
        return None
    return result


def _keep(value: object, source: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip() if _grounded(value, source) else ""


def _keep_ownership(value: object, source: str) -> str:
    kept = _keep(value, source)
    if not kept or "%" not in kept or "平方" in kept:
        return ""
    return kept


def _keep_intensity(value: object, source: str) -> str:
    if not isinstance(value, str) or "法定" in value:
        return ""
    if not _grounded(value, source):
        return ""
    return value.strip()


def _keep_volume(value: object, source: str) -> str:
    if not isinstance(value, str):
        return ""
    kept = []
    for line in value.splitlines():
        if _grounded(line, source):
            kept.append(line.strip())
    return "\n".join(kept)


def _grounded(value: str, source: str) -> bool:
    parts = [part.strip() for part in re.split(r"[\n、]", value) if part.strip()]
    if not parts:
        return False
    source_norm = normalize_for_match(source)
    for part in parts:
        piece = re.sub(r"^[☑☒•\-\s]+", "", part)
        piece = piece.replace("法定", "")
        if not piece:
            return False
        if normalize_for_match(piece) not in source_norm:
            return False
    return True
