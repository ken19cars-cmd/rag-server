"""Swappable embedding backends.

The default backend hashes character n-grams. It runs offline for Traditional
and Simplified Chinese and does not download a model. Tests inject
``FakeEmbedder``. A multilingual sentence-transformers model is optional.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from rag_server.textutil import char_ngrams


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def hashing_embed(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    grams = char_ngrams(text, 2) + char_ngrams(text, 3)
    if not grams:
        grams = ["empty"]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return _l2_normalize(vector)


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class Embedder(Protocol):
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalMultilingualEmbedder:
    """Offline character n-gram embedder. No language-specific tokenizer."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.name = "local"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [hashing_embed(text, self.dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return hashing_embed(text, self.dimension)


class FakeEmbedder:
    """Deterministic embedder used by tests. It does not load a model."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension
        self.name = "fake"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [hashing_embed(text, self.dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return hashing_embed(text, self.dimension)


class SentenceTransformerEmbedder:
    """Optional neural embedder. The model is loaded on the first call."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.name = "sentence-transformers"
        self.dimension = 0
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install the embeddings extra or set RAG_EMBEDDING_BACKEND=local."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [[float(value) for value in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def build_embedder(backend: str, model_name: str) -> Embedder:
    name = (backend or "local").strip().lower()
    if name in {"local", "hashing", "char-ngram"}:
        return LocalMultilingualEmbedder()
    if name in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(model_name)
    if name == "fake":
        return FakeEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")
