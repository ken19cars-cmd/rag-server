"""Build a small text-based PDF fixture without extra libraries."""

from __future__ import annotations


def build_text_pdf(lines: list[str]) -> bytes:
    content = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            content.append("0 -18 Td")
        content.append(f"({escaped}) Tj")
    content.append("ET")
    stream = ("\n".join(content) + "\n").encode("ascii")

    def obj(body: bytes) -> bytes:
        return body if body.endswith(b"\n") else body + b"\n"

    objects = [
        obj(b"<< /Type /Catalog /Pages 2 0 R >>"),
        obj(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        obj(f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream"),
        obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    position = len(header)
    numbered: list[bytes] = []
    for number, body in enumerate(objects, start=1):
        part = f"{number} 0 obj\n".encode("ascii") + body + b"endobj\n"
        numbered.append(part)
        offsets.append(position)
        position += len(part)
    xref_position = position
    xref = [b"xref\n", f"0 {len(offsets)}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode("ascii")
    return b"".join([header, *numbered, *xref, trailer])


SAMPLE_LINES = [
    "Tainan Smart Logistics Park investment promotion.",
    "Industry: smart logistics and cold-chain warehousing.",
]
