import sys

from rag_server.embeddings import (
    FakeEmbedder,
    LocalMultilingualEmbedder,
    build_embedder,
    cosine,
)


def test_fake_embedder_is_deterministic_and_normalized():
    embedder = FakeEmbedder()
    first = embedder.embed_query("投資金額：新臺幣 12 億元")
    second = embedder.embed_texts(["投資金額：新臺幣 12 億元"])[0]
    assert first == second
    assert len(first) == embedder.dimension
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_local_embedder_ranks_related_chinese_text_higher():
    embedder = LocalMultilingualEmbedder()
    query = embedder.embed_query("這個案子的投資金額是多少")
    amount = embedder.embed_query("投資金額：新臺幣 12 億元")
    other = embedder.embed_query("今日天氣晴朗，適合散步")
    assert cosine(query, amount) > cosine(query, other)


def test_sentence_transformer_backend_is_not_loaded_until_used():
    sys.modules.pop("sentence_transformers", None)
    embedder = build_embedder(
        "sentence-transformers",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )
    assert embedder.name == "sentence-transformers"
    assert embedder._model is None
    assert "sentence_transformers" not in sys.modules
