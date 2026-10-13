from rag_server.analysis import extractive_analysis, extractive_query
from rag_server.embeddings import FakeEmbedder
from rag_server.store import ChunkRecord

FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "sample_case.txt"


def _chunk(text: str, embedder: FakeEmbedder, chunk_id: str = "chunk_test") -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        case_id="case_test",
        document_id="doc_test",
        ordinal=0,
        text=text,
        page=1,
        embedding=embedder.embed_query(text),
        filename="sample_case.txt",
        source_url=None,
        title="sample",
        source_type="text",
    )


def test_extractive_analysis_cites_fields_and_marks_unknowns():
    embedder = FakeEmbedder()
    text = FIXTURE.read_text(encoding="utf-8")
    result = extractive_analysis("case_test", [_chunk(text, embedder)], embedder)
    assert result.mode == "extractive"
    assert result.fields["industry"].value == "智慧物流與冷鏈倉儲"
    assert result.fields["location"].value == "臺南市安南區科技工業區"
    assert result.fields["investment_amount"].value == "新臺幣 12 億元"
    assert result.fields["timeline"].status == "found"
    assert "張文彬" in (result.fields["contacts"].value or "")
    assert result.fields["incentives"].status == "unknown"
    assert result.fields["incentives"].citations == []
    assert any(item.field == "incentives" for item in result.unknowns)
    for claim in result.summary:
        assert claim.citations
        assert claim.text in text
        assert claim.citations[0].excerpt in text


def test_query_returns_the_amount_sentence_with_a_citation():
    embedder = FakeEmbedder()
    text = FIXTURE.read_text(encoding="utf-8")
    result = extractive_query(
        "case_test",
        "投資金額是多少？",
        [_chunk(text, embedder)],
        embedder,
        top_k=3,
    )
    assert "新臺幣 12 億元" in result.answer
    assert "投資地點" not in result.answer
    assert result.citations
    assert result.claims[0].text in text
    assert result.claims[0].citations[0].excerpt in text


def test_simplified_chinese_label_is_extracted():
    embedder = FakeEmbedder()
    text = "产业别：冷链仓储\n投资地点：苏州市工业园区"
    result = extractive_analysis("case_test", [_chunk(text, embedder)], embedder)
    assert result.fields["industry"].value == "冷链仓储"
    assert result.fields["location"].value == "苏州市工业园区"
    assert result.fields["investment_amount"].status == "unknown"
