"""Read Notion page blocks to see whether ❯計畫概述 is still empty."""

from __future__ import annotations

import unicodedata

import httpx

from rag_server.completed import NOTION_VERSION
from rag_server.notion_analysis import _heading_index, _paragraph_after, parse_template_blocks


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
    """Whether the paragraph under ❯計畫概述 already has text.

    ``filled`` has visible text. ``empty`` is a blank paragraph. ``missing_paragraph``
    has the heading and no paragraph under it. ``missing_heading`` has no ❯計畫概述.
    """
    blocks = parse_template_blocks(raw_blocks)
    if _heading_index(blocks, "計畫概述") is None:
        return "missing_heading"
    paragraph = _paragraph_after(blocks, "計畫概述")
    if paragraph is None:
        return "missing_paragraph"
    visible = "".join(char for char in paragraph.text if unicodedata.category(char) != "Cf").strip()
    return "filled" if visible else "empty"
