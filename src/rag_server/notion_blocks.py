"""Read Notion page blocks to see whether ❯計畫概述 has case prose."""

from __future__ import annotations

import re
import unicodedata

import httpx

from rag_server.completed import NOTION_VERSION
from rag_server.notion_analysis import TemplateBlock, _following, _heading_index, parse_template_blocks

# Lines the blank case template and the analysis prompt use as labels or hints.
# A page that only repeats these has had the template applied, not written.
_SCAFFOLD = frozenset(
    {
        "本案是什麼案件",
        "為什麼辦理",
        "主要開發目的",
        "主要開發方向",
        "重要背景",
        "計畫概述",
        "基地基本資料",
        "周邊環境概況",
        "使用強度規範",
        "意願調查",
        "開發定位規劃構想",
        "開發權利義務負擔條件",
        "期程規範",
        "財務條件",
        "申請人資格",
        "項目",
        "一般資格",
        "單一申請人",
        "合作聯盟(3家以內)",
        "財務資格",
        "財務結構指標",
        "信用及納稅規範",
        "專業技術能力",
    }
)

_CONTENT_TYPES = frozenset(
    {
        "paragraph",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
        "quote",
        "callout",
        "toggle",
        "table_row",
    }
)

_PIECE_SPLIT = re.compile(r"[\n\r]+|[•・●○▪▸►]")


def fetch_block_children(
    token: str,
    block_id: str,
    *,
    client: httpx.Client | None = None,
) -> list[dict]:
    """Return the direct children of a Notion block, following next_cursor."""
    owns = client is None
    http = client or httpx.Client(timeout=30.0)
    results: list[dict] = []
    cursor = None
    try:
        while True:
            params: dict[str, str | int] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            response = http.get(
                f"https://api.notion.com/v1/blocks/{block_id}/children",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Notion-Version": NOTION_VERSION,
                },
                params=params,
            )
            if response.status_code >= 400:
                detail = response.text[:180].replace("\n", " ")
                raise RuntimeError(f"notion blocks failed: {response.status_code} {detail}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("notion blocks did not return an object")
            results.extend(item for item in (payload.get("results") or []) if isinstance(item, dict))
            if not payload.get("has_more"):
                return results
            cursor = payload.get("next_cursor")
            if not cursor:
                return results
    except httpx.HTTPError as exc:
        raise RuntimeError(f"notion blocks failed: {exc}") from exc
    finally:
        if owns:
            http.close()


def overview_state(raw_blocks: list[dict]) -> str:
    """Whether ❯計畫概述 already describes this case.

    ``filled`` has case prose. ``template`` has only the applied template: gray
    placeholder text, or the template's own hint and label lines. ``empty`` is a
    blank paragraph. ``missing_paragraph`` has the heading and no body under it.
    ``missing_heading`` has no ❯計畫概述.
    """
    blocks = parse_template_blocks(raw_blocks)
    if _heading_index(blocks, "計畫概述") is None:
        return "missing_heading"
    body = [block for block in _following(blocks, "計畫概述") if _is_content_block(block)]
    if not body:
        return "missing_paragraph"
    raw_by_id = {
        str(item.get("id") or ""): item
        for item in raw_blocks
        if isinstance(item, dict)
    }
    verdicts = [_block_verdict(raw_by_id.get(block.id, {}), block) for block in body]
    if "content" in verdicts:
        return "filled"
    if "scaffold" in verdicts:
        return "template"
    return "empty"


def _is_content_block(block: TemplateBlock) -> bool:
    return block.type in _CONTENT_TYPES or block.type.startswith("paragraph")


def _block_verdict(raw: dict, block: TemplateBlock) -> str:
    plain: list[str] = []
    saw_placeholder = False
    for kind, text in _iter_spans(raw, block):
        if kind == "placeholder":
            if _visible(text):
                saw_placeholder = True
            continue
        plain.append(text)
    if _has_case_prose("\n".join(plain)):
        return "content"
    if _is_scaffold_text("\n".join(plain)) or saw_placeholder:
        return "scaffold"
    return "empty"


def _iter_spans(raw: dict, block: TemplateBlock):
    block_type = str(raw.get("type") or block.type)
    typed = raw.get(block_type)
    if isinstance(typed, dict) and isinstance(typed.get("rich_text"), list):
        for part in typed["rich_text"]:
            if isinstance(part, dict):
                yield _span_from_part(part)
        return
    if block.type == "table_row":
        yielded = False
        for kind, text in _table_spans(raw, typed):
            yielded = True
            yield kind, text
        if yielded:
            return
        for cell in block.cells:
            if cell.strip():
                yield "text", cell
        return
    if block.text:
        yield "text", block.text


def _table_spans(raw: dict, typed: object):
    cells = None
    if isinstance(typed, dict) and isinstance(typed.get("cells"), list):
        cells = typed["cells"]
    elif isinstance(raw.get("cells"), list):
        cells = raw["cells"]
    if not isinstance(cells, list):
        return
    for cell in cells:
        if isinstance(cell, str):
            if cell.strip():
                yield "text", cell
            continue
        if not isinstance(cell, list):
            continue
        for part in cell:
            if isinstance(part, dict):
                yield _span_from_part(part)


def _span_from_part(part: dict) -> tuple[str, str]:
    annotations = part.get("annotations")
    color = ""
    if isinstance(annotations, dict):
        color = str(annotations.get("color") or "")
    kind = "placeholder" if color.startswith("gray") else "text"
    return kind, str(part.get("plain_text") or "")


def _has_case_prose(text: str) -> bool:
    pieces = _pieces(text)
    return any(not _is_scaffold_piece(piece) for piece in pieces)


def _is_scaffold_text(text: str) -> bool:
    pieces = _pieces(text)
    return bool(pieces) and all(_is_scaffold_piece(piece) for piece in pieces)


def _is_scaffold_piece(text: str) -> bool:
    return _norm_piece(text) in _SCAFFOLD


def _pieces(text: str) -> list[str]:
    visible = _visible(text)
    if not visible:
        return []
    return [part.strip() for part in _PIECE_SPLIT.split(visible) if part.strip()]


def _visible(text: str) -> str:
    return "".join(char for char in text if unicodedata.category(char) != "Cf").strip()


def _norm_piece(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", _visible(text))
    cleaned = cleaned.strip("「」\"'“”『』")
    cleaned = re.sub(r"^[\-*＊–—－]+\s*", "", cleaned)
    cleaned = re.sub(r"^\d+\s*[.、．]\s*", "", cleaned)
    cleaned = re.sub(r"[\s❯*#／/]", "", cleaned)
    cleaned = cleaned.replace("（", "(").replace("）", ")")
    return cleaned.rstrip("。．.:：;；")
