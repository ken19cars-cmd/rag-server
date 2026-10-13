from rag_server.chunking import chunk_pages, chunk_text


def test_chinese_text_splits_into_overlapping_windows():
    text = "甲" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert chunks[1].start_char < chunks[0].end_char
    assert chunks[0].text == "甲" * len(chunks[0].text)


def test_page_numbers_are_kept():
    chunks = chunk_pages(
        [(1, "第一頁招商說明。" * 5), (2, "第二頁投資金額：新臺幣 3 億元。")],
        chunk_size=80,
        overlap=10,
    )
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert chunks[0].ordinal == 0
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_blank_text_yields_no_chunks():
    assert chunk_text("  \n\t ") == []
