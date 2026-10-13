from pathlib import Path

from rag_server.pdf_extract import (
    UnavailableOcrEngine,
    combine_page_text,
    extract_pdf_bytes,
)
from tests.pdf_bytes import SAMPLE_LINES, build_text_pdf

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_text_pdf_fixture_extracts_industry_line():
    extracted = extract_pdf_bytes(FIXTURE.read_bytes(), ocr=UnavailableOcrEngine())
    text = extracted.full_text
    assert "smart logistics" in text
    assert "investment promotion" in text
    assert extracted.pages[0].method == "text"
    assert extracted.ocr_page_count == 0


def test_short_page_uses_ocr_when_an_engine_is_available(monkeypatch):
    class Engine:
        def available(self) -> bool:
            return True

        def ocr_image(self, image_bytes: bytes) -> str:
            assert image_bytes == b"scan"
            return "產業別：智慧物流與冷鏈倉儲"

    monkeypatch.setattr(
        "rag_server.pdf_extract.image_bytes_from_page",
        lambda _page: [b"scan"],
    )
    extracted = extract_pdf_bytes(build_text_pdf(["."]), ocr=Engine())
    assert "智慧物流" in extracted.full_text
    assert extracted.pages[0].method in {"ocr", "mixed"}


def test_scanned_page_warns_when_ocr_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "rag_server.pdf_extract.image_bytes_from_page",
        lambda _page: [b"scan"],
    )
    extracted = extract_pdf_bytes(build_text_pdf(["."]), ocr=UnavailableOcrEngine())
    assert extracted.pages[0].method == "text"
    assert any("OCR is unavailable" in warning for warning in extracted.warnings)


def test_combine_page_text_keeps_a_real_text_layer():
    text, method = combine_page_text("x" * 40, [b"image"], UnavailableOcrEngine())
    assert method == "text"
    assert text == "x" * 40


def test_sample_fixture_matches_the_builder():
    assert FIXTURE.read_bytes() == build_text_pdf(SAMPLE_LINES)
