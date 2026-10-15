"""Read Notion page blocks to see whether ❯計畫概述 has case prose."""

from __future__ import annotations

import contextvars
import re
import time
import unicodedata

import httpx

from rag_server.completed import (
    NOTION_RATE_RETRIES,
    NOTION_VERSION,
    case_match_key,
    notion_retry_delay,
)
from rag_server.notion_analysis import TemplateBlock, parse_template_blocks

_CONTAINER_TYPES = frozenset({"column_list", "column", "synced_block", "toggle", "table"})
_MAX_NESTING = 4
_TITLE_PREFIX = re.compile(r"^(?:案件名稱|案名|案件)\s*[:：]\s*")

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
_MIN_GAP_SECONDS = 0.35
_pace_var: contextvars.ContextVar["NotionPace | None"] = contextvars.ContextVar(
    "notion_block_pace",
    default=None,
)


class NotionPace:
    """Keep block reads under Notion's average of three requests a second."""

    def __init__(self, sleep=time.sleep, clock=time.monotonic, gap: float = _MIN_GAP_SECONDS):
        self._sleep = sleep
        self._clock = clock
        self._gap = gap
        self._next = 0.0

    def wait(self) -> None:
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
        self._next = self._clock() + self._gap

    def pause(self, seconds: float) -> None:
        self._sleep(seconds)
        self._next = self._clock() + self._gap


class paced_notion_reads:
    """Share one pace across every block read in a scan."""

    def __init__(self, sleep=time.sleep, clock=time.monotonic, gap: float = _MIN_GAP_SECONDS):
        self._pace = NotionPace(sleep=sleep, clock=clock, gap=gap)
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "paced_notion_reads":
        self._token = _pace_var.set(self._pace)
        return self

    def __exit__(self, *_exc) -> bool:
        if self._token is not None:
            _pace_var.reset(self._token)
        return False


def fetch_block_children(
    token: str,
    block_id: str,
    *,
    client: httpx.Client | None = None,
    sleep=time.sleep,
) -> list[dict]:
    """Return a page's blocks, including blocks nested in columns, toggles, and tables."""
    owns = client is None
    http = client or httpx.Client(timeout=30.0)
    try:
        direct = _fetch_direct_children(token, block_id, http, sleep)
        return _expand_containers(token, direct, http, sleep)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"notion blocks failed: {exc}") from exc
    finally:
        if owns:
            http.close()


def _fetch_direct_children(
    token: str,
    block_id: str,
    http: httpx.Client,
    sleep,
) -> list[dict]:
    """Return the direct children of a Notion block, following next_cursor."""
    results: list[dict] = []
    cursor = None
    while True:
        params: dict[str, str | int] = {"page_size": 100}
        if cursor:
            params["start_cursor"] = cursor
        response = _get_children(token, block_id, http, params, sleep)
        if response.status_code == 429:
            raise RuntimeError("Notion 讀取太快被限制，請稍後再按一次。")
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


def _get_children(token: str, block_id: str, http: httpx.Client, params: dict, sleep) -> httpx.Response:
    attempt = 0
    while True:
        pace = _pace_var.get()
        if pace is not None:
            pace.wait()
        response = http.get(
            f"https://api.notion.com/v1/blocks/{block_id}/children",
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
            },
            params=params,
        )
        if response.status_code != 429 or attempt >= NOTION_RATE_RETRIES:
            return response
        attempt += 1
        delay = notion_retry_delay(response, attempt)
        if pace is not None:
            pace.pause(delay)
        else:
            sleep(delay)


def _expand_containers(
    token: str,
    blocks: list[dict],
    http: httpx.Client,
    sleep,
    depth: int = 0,
) -> list[dict]:
    if depth >= _MAX_NESTING:
        return blocks
    expanded: list[dict] = []
    for block in blocks:
        expanded.append(block)
        if not _is_container(block):
            continue
        block_id = str(block.get("id") or "").strip()
        if not block_id:
            continue
        children = _fetch_direct_children(token, block_id, http, sleep)
        expanded.extend(_expand_containers(token, children, http, sleep, depth + 1))
    return expanded


def _is_container(block: dict) -> bool:
    if not block.get("has_children"):
        return False
    block_type = str(block.get("type") or "")
    if block_type in _CONTAINER_TYPES:
        return True
    if not block_type.startswith("heading"):
        return False
    typed = block.get(block_type)
    return isinstance(typed, dict) and bool(typed.get("is_toggleable"))


def overview_state(raw_blocks: list[dict], case_name: str = "") -> str:
    """Whether ❯計畫概述 already describes this case."""
    return overview_reading(raw_blocks, case_name=case_name)["state"]


def overview_reading(raw_blocks: list[dict], case_name: str = "") -> dict:
    """Classify ❯計畫概述 and keep a short quote of any case prose.

    ``filled`` has case prose. ``excerpt`` is that prose, clipped.
    ``template`` is the applied template: gray placeholder text, the template's
    own hint and label lines, or text that only repeats the page title.
    ``empty`` is a blank paragraph. ``missing_paragraph`` has the heading and no
    body under it. ``missing_heading`` has no 計畫概述; ``headings`` lists the
    headings that were on the page.
    """
    blocks = parse_template_blocks(raw_blocks)
    headings = [block.text for block in blocks if block.type.startswith("heading") and block.text][:8]
    body = _body_after_overview(blocks)
    if body is None:
        return {"state": "missing_heading", "excerpt": "", "headings": headings}
    content = [block for block in body if _is_content_block(block)]
    if not content:
        return {"state": "missing_paragraph", "excerpt": "", "headings": []}
    raw_by_id = {
        str(item.get("id") or ""): item
        for item in raw_blocks
        if isinstance(item, dict)
    }
    prose: list[str] = []
    saw_scaffold = False
    for block in content:
        verdict = _block_verdict(raw_by_id.get(block.id, {}), block)
        if verdict == "content":
            text = _prose_text(raw_by_id.get(block.id, {}), block)
            if text:
                prose.append(text)
        elif verdict == "scaffold":
            saw_scaffold = True
    if prose and not _is_only_case_name("\n".join(prose), case_name):
        return {"state": "filled", "excerpt": _clip(" ".join(prose)), "headings": []}
    if prose or saw_scaffold:
        return {"state": "template", "excerpt": "", "headings": []}
    return {"state": "empty", "excerpt": "", "headings": []}


def _body_after_overview(blocks: list[TemplateBlock]) -> list[TemplateBlock] | None:
    start = _overview_heading_index(blocks)
    if start is None:
        return None
    found: list[TemplateBlock] = []
    for block in blocks[start + 1 :]:
        if block.type.startswith("heading"):
            break
        found.append(block)
    return found


def _overview_heading_index(blocks: list[TemplateBlock]) -> int | None:
    for index, block in enumerate(blocks):
        if not block.type.startswith("heading"):
            continue
        folded = re.sub(r"[\s❯*#／]", "", block.text.replace("/", "")).replace("計劃", "計畫")
        if "計畫概述" in folded:
            return index
    return None


def _prose_text(raw: dict, block: TemplateBlock) -> str:
    parts = [_visible(text) for kind, text in _iter_spans(raw, block) if kind != "placeholder"]
    return "\n".join(part for part in parts if part)


def _is_only_case_name(text: str, case_name: str) -> bool:
    title = case_match_key(case_name)
    if not title:
        return False
    pieces = _pieces(text)
    if not pieces:
        return False
    for piece in pieces:
        if _is_scaffold_piece(piece):
            continue
        stripped = _TITLE_PREFIX.sub("", piece.strip())
        if case_match_key(stripped) != title:
            return False
    return True


def _clip(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


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
