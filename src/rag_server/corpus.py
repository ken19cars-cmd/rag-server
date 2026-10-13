"""Scan a local 類型 / 縣市 / 案名 / PDF tree. Does not read Ken's D: drive."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUFFIXES = {".pdf", ".txt"}
SKIP_DIR_NAMES = frozenset({"__MACOSX", ".git"})

DOC_KINDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("application_notice", "申請須知", ("申請須知",)),
    ("requirements", "需求說明書", ("需求說明書",)),
    ("selection", "評選文件", ("評選",)),
    ("investment_contract", "實施(出資)契約", ("出資契約", "實施契約", "契約")),
)


@dataclass
class CorpusFile:
    case_type: str
    city: str
    city_folder: str
    case_name: str
    path: Path
    filename: str
    doc_kind: str
    doc_kind_label: str


@dataclass
class CorpusCase:
    case_type: str
    city: str
    city_folder: str
    case_name: str
    folder: Path
    documents: list[CorpusFile]


def classify_document(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    for kind, label, needles in DOC_KINDS:
        if any(needle in stem for needle in needles):
            return kind, label
    return "other", "其他"


def normalize_city(folder_name: str) -> str:
    """`00基隆市` → `基隆市`. The raw folder name is kept separately."""
    stripped = re.sub(r"^\d+[.．、_\-\s]*", "", folder_name).strip()
    return stripped or folder_name.strip()


def scan_cases_root(root: Path) -> list[CorpusCase]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"cases root is not a directory: {root}")
    cases: list[CorpusCase] = []
    type_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for type_dir in type_dirs:
        city_dirs = sorted(path for path in type_dir.iterdir() if path.is_dir())
        for city_dir in city_dirs:
            case_dirs = sorted(path for path in city_dir.iterdir() if path.is_dir())
            for case_dir in case_dirs:
                documents = _documents_in_case(type_dir.name, city_dir.name, case_dir)
                cases.append(
                    CorpusCase(
                        case_type=type_dir.name,
                        city=normalize_city(city_dir.name),
                        city_folder=city_dir.name,
                        case_name=case_dir.name,
                        folder=case_dir,
                        documents=documents,
                    )
                )
    return cases


def iter_case_files(case_dir: Path) -> list[Path]:
    """PDF and txt under a case folder, including nested folders.

    detect, ingest, and extract all read documents through ``scan_cases_root``,
    which uses this walk. ``__MACOSX`` and ``.git`` directories are not entered.
    """
    found: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            children = list(directory.iterdir())
        except OSError:
            return
        for child in children:
            if child.is_dir():
                if child.name in SKIP_DIR_NAMES:
                    continue
                walk(child)
                continue
            if child.is_file() and child.suffix.lower() in ALLOWED_SUFFIXES:
                found.append(child)

    walk(case_dir)
    return sorted(found, key=lambda path: path.relative_to(case_dir).as_posix())


def _documents_in_case(case_type: str, city_folder: str, case_dir: Path) -> list[CorpusFile]:
    documents: list[CorpusFile] = []
    for path in iter_case_files(case_dir):
        kind, label = classify_document(path.name)
        documents.append(
            CorpusFile(
                case_type=case_type,
                city=normalize_city(city_folder),
                city_folder=city_folder,
                case_name=case_dir.name,
                path=path,
                filename=path.name,
                doc_kind=kind,
                doc_kind_label=label,
            )
        )
    return documents


def merge_document_texts(documents: list[tuple[str, str, str]]) -> str:
    """Join texts with the separators Ken's prompt already expects.

    Each item is (doc_kind_label, filename, text).
    """
    blocks: list[str] = []
    for index, (label, filename, text) in enumerate(documents, start=1):
        body = text.strip()
        blocks.append(
            "\n".join(
                [
                    f"========== 文件 {index} ==========",
                    f"文件類型：{label}",
                    f"檔案名稱：{filename}",
                    body,
                    f"========== 文件 {index} 結束 ==========",
                ]
            )
        )
    return "\n\n".join(blocks)
