"""Text helpers shared by embedding, retrieval, and extractive analysis."""

from __future__ import annotations

import re
import unicodedata

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def normalize_for_match(text: str) -> str:
    """Fold case, width, and punctuation so Chinese and ASCII queries overlap."""
    folded = unicodedata.normalize("NFKC", text).lower()
    return _NON_WORD.sub("", folded)


def char_ngrams(text: str, size: int) -> list[str]:
    normalized = normalize_for_match(text)
    if not normalized:
        return []
    if len(normalized) < size:
        return [normalized]
    return [normalized[i : i + size] for i in range(len(normalized) - size + 1)]


def ngram_overlap(query: str, text: str, size: int = 2) -> float:
    query_grams = set(char_ngrams(query, size))
    if not query_grams:
        return 0.0
    text_grams = set(char_ngrams(text, size))
    return len(query_grams & text_grams) / len(query_grams)


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [part.strip() for part in parts if part.strip()]


def clip_text(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip()
