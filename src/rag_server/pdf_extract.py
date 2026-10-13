"""PDF text extraction with an OCR path for scanned pages."""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from typing import Protocol

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MIN_TEXT_CHARS = 20


def is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


class OcrEngine(Protocol):
    def available(self) -> bool: ...

    def ocr_image(self, image_bytes: bytes) -> str: ...


@dataclass
class ExtractedPage:
    page: int
    text: str
    method: str


@dataclass
class ExtractedDocument:
    pages: list[ExtractedPage]
    warnings: list[str]

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for page in self.pages if page.method in {"ocr", "mixed"})


class TesseractOcrEngine:
    """OCR via Tesseract. Used only when the binary and Python bindings exist."""

    def __init__(self, languages: str = "chi_tra+chi_sim+eng") -> None:
        self.languages = languages

    def available(self) -> bool:
        if shutil.which("tesseract") is None:
            return False
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            return False
        return True

    def ocr_image(self, image_bytes: bytes) -> str:
        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        try:
            return pytesseract.image_to_string(image, lang=self.languages) or ""
        except pytesseract.TesseractError:
            return pytesseract.image_to_string(image, lang="eng") or ""


class UnavailableOcrEngine:
    def available(self) -> bool:
        return False

    def ocr_image(self, image_bytes: bytes) -> str:
        return ""


def image_bytes_from_page(page) -> list[bytes]:
    blobs: list[bytes] = []
    images = getattr(page, "images", None)
    if not images:
        return blobs
    for image in images:
        data = getattr(image, "data", None)
        if data:
            blobs.append(data)
    return blobs


def combine_page_text(
    text: str,
    images: list[bytes],
    ocr: OcrEngine | None,
    *,
    min_chars: int = MIN_TEXT_CHARS,
) -> tuple[str, str]:
    extracted = (text or "").strip()
    if len(extracted) >= min_chars:
        return extracted, "text"
    ocr_text = ""
    if images and ocr is not None and ocr.available():
        parts: list[str] = []
        for blob in images:
            try:
                parts.append((ocr.ocr_image(blob) or "").strip())
            except Exception:
                parts.append("")
        ocr_text = "\n".join(part for part in parts if part)
    if extracted and ocr_text:
        return f"{extracted}\n{ocr_text}".strip(), "mixed"
    if ocr_text:
        return ocr_text, "ocr"
    if extracted:
        return extracted, "text"
    return "", "empty"


def extract_pdf_bytes(
    data: bytes,
    ocr: OcrEngine | None = None,
) -> ExtractedDocument:
    if not data:
        raise PdfReadError("empty pdf")
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise PdfReadError("encrypted pdf") from exc
    warnings: list[str] = []
    pages: list[ExtractedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
            warnings.append(f"page {index}: text extraction failed")
        try:
            images = image_bytes_from_page(page)
        except Exception:
            images = []
            warnings.append(f"page {index}: image extraction failed")
        text, method = combine_page_text(raw_text, images, ocr)
        looks_scanned = len((raw_text or "").strip()) < MIN_TEXT_CHARS and bool(images)
        if looks_scanned and (ocr is None or not ocr.available()):
            warnings.append(
                f"page {index}: looks scanned, but OCR is unavailable. "
                "Install tesseract, pytesseract, and Pillow."
            )
        elif method == "empty":
            warnings.append(f"page {index}: no extractable text")
        pages.append(ExtractedPage(page=index, text=text, method=method))
    return ExtractedDocument(pages=pages, warnings=warnings)
