"""Character-window chunking that keeps paragraph boundaries when it can."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkDraft:
    text: str
    page: int | None
    start_char: int
    end_char: int
    ordinal: int = 0


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 80,
    page: int | None = None,
) -> list[ChunkDraft]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []
    size = max(40, chunk_size)
    step_overlap = overlap
    if step_overlap >= size:
        step_overlap = size // 5
    if step_overlap < 0:
        step_overlap = 0

    drafts: list[ChunkDraft] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + size, length)
        if end < length:
            window = cleaned[start:end]
            cut = max(window.rfind("\n"), window.rfind("。"), window.rfind("."))
            if cut >= int(size * 0.5):
                end = start + cut + 1
        piece = cleaned[start:end].strip()
        if piece:
            drafts.append(
                ChunkDraft(text=piece, page=page, start_char=start, end_char=end)
            )
        if end >= length:
            break
        next_start = end - step_overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return drafts


def chunk_pages(
    pages: list[tuple[int | None, str]],
    *,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    for page, text in pages:
        drafts.extend(
            chunk_text(text, chunk_size=chunk_size, overlap=overlap, page=page)
        )
    for ordinal, draft in enumerate(drafts):
        draft.ordinal = ordinal
    return drafts
