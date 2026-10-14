"""Scan a local 類型 / 縣市 / 案名 / PDF tree. Does not read Ken's D: drive."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUFFIXES = {".pdf", ".txt"}
SKIP_DIR_NAMES = frozenset({"__MACOSX", ".git"})
VERSIONED_KINDS = frozenset(
    {
        "application_notice",
        "requirements",
        "selection",
        "investment_contract",
    }
)
_MONTH = r"(1[0-2]|0?[1-9])"
_DAY = r"(3[01]|[12]\d|0?[1-9])"
_DATE_PATTERNS: tuple[tuple[str, bool], ...] = (
    (rf"(?<!\d)(20\d{{2}})[.\-/／]{_MONTH}[.\-/／]{_DAY}(?!\d)", False),
    (rf"(?<!\d)(\d{{2,3}})[.\-/／]{_MONTH}[.\-/／]{_DAY}(?!\d)", True),
    (rf"(?<!\d)(\d{{2,3}})年{_MONTH}月{_DAY}(?!\d)", True),
    (rf"(?<!\d)(20\d{{2}})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", False),
    (rf"(?<!\d)(\d{{3}})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", True),
    (rf"(?<!\d)(\d{{2}})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", True),
)

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


def _is_filing_bucket(name: str) -> bool:
    """A dated 公開閱覽 folder, or a bare date folder, is not a case name."""
    if "公開閱覽" in name:
        leftover = re.sub(r"[\d.\-/／\s年月日()（）第次]", "", name.replace("公開閱覽", ""))
        return leftover == ""
    compact = re.sub(r"[\s.\-/／]", "", name)
    return bool(re.fullmatch(r"\d{6,8}", compact) and folder_date_ordinal(name) is not None)


def case_at(root: Path, directory: Path) -> CorpusCase:
    """One case folder, including PDFs nested under it."""
    root = root.expanduser().resolve()
    directory = directory.expanduser().resolve()
    relative = directory.relative_to(root)
    case_type = relative.parts[0] if relative.parts else ""
    city_folder = relative.parts[1] if len(relative.parts) >= 3 else ""
    return CorpusCase(
        case_type=case_type,
        city=normalize_city(city_folder) if city_folder else "",
        city_folder=city_folder,
        case_name=directory.name,
        folder=directory,
        documents=_documents_in_case(case_type, city_folder or directory.parent.name, directory),
    )


def folders_for_local_match(root: Path) -> list[CorpusCase]:
    """Case folders for Notion pairing.

    The level scan is 類型 / 縣市 / 案名. A renamed folder is still a case
    when it sits directly under a type folder and the only directories inside
    it are 公開閱覽 folders, or when the PDFs sit in that folder with no
    further directories. A city folder that already holds real case names
    stays a city folder.
    """
    root = root.expanduser().resolve()
    scanned = scan_cases_root(root)
    by_parent: dict[Path, list[CorpusCase]] = {}
    for case in scanned:
        by_parent.setdefault(case.folder.resolve().parent, []).append(case)

    dropped: set[Path] = set()
    promoted: list[CorpusCase] = []
    for parent, children in by_parent.items():
        try:
            relative = parent.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            continue
        if not children or not all(_is_filing_bucket(child.case_name) for child in children):
            continue
        dropped.update(child.folder.resolve() for child in children)
        promoted.append(case_at(root, parent))

    kept = [case for case in scanned if case.folder.resolve() not in dropped]
    known = {case.folder.resolve() for case in kept}
    known.update(case.folder.resolve() for case in promoted)
    ancestors: set[Path] = set()
    for folder in known:
        ancestors.update(folder.parents)

    extras: list[CorpusCase] = []
    for type_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if type_dir.name in SKIP_DIR_NAMES or type_dir.name.startswith("."):
            continue
        for child in sorted(path for path in type_dir.iterdir() if path.is_dir()):
            if child.name in SKIP_DIR_NAMES or child.name.startswith("."):
                continue
            resolved = child.resolve()
            if resolved in known or resolved in ancestors:
                continue
            nested = [
                path
                for path in child.iterdir()
                if path.is_dir() and path.name not in SKIP_DIR_NAMES and not path.name.startswith(".")
            ]
            if nested:
                continue
            has_file = any(path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES for path in child.iterdir())
            if not has_file:
                continue
            extras.append(case_at(root, resolved))
    return kept + promoted + extras


def retarget_folders_named_like(root: Path, cases: list[CorpusCase], hits) -> list[CorpusCase]:
    """Use a directory whose name is a page title when the level scan missed it.

    ``hits(name)`` is true when that name is a Notion title. A parent replaces
    its scanned children only when the parent hits and none of the children do.
    A scanned case that misses is replaced by a nested directory that hits.
    """
    root = root.expanduser().resolve()
    by_parent: dict[Path, list[CorpusCase]] = {}
    for case in cases:
        by_parent.setdefault(case.folder.resolve().parent, []).append(case)

    consumed: set[Path] = set()
    result: list[CorpusCase] = []
    seen: set[Path] = set()
    for parent, children in by_parent.items():
        try:
            relative = parent.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) < 2 or not hits(parent.name):
            continue
        if any(hits(child.case_name) for child in children):
            continue
        resolved = parent.resolve()
        if resolved in seen:
            continue
        result.append(case_at(root, parent))
        seen.add(resolved)
        consumed.update(child.folder.resolve() for child in children)

    for case in cases:
        resolved = case.folder.resolve()
        if resolved in consumed:
            continue
        if hits(case.case_name):
            if resolved not in seen:
                result.append(case)
                seen.add(resolved)
            continue
        found: list[Path] = []
        for _current, dirnames, _filenames in os.walk(resolved):
            current = Path(_current)
            kept: list[str] = []
            for name in dirnames:
                if name in SKIP_DIR_NAMES or name.startswith(".") or _is_filing_bucket(name):
                    continue
                kept.append(name)
                directory = (current / name).resolve()
                if hits(name):
                    found.append(directory)
            dirnames[:] = kept
        replacements = [
            directory
            for directory in found
            if not any(other != directory and other in directory.parents for other in found)
        ]
        if not replacements:
            if resolved not in seen:
                result.append(case)
                seen.add(resolved)
            continue
        for directory in replacements:
            if directory in seen:
                continue
            result.append(case_at(root, directory))
            seen.add(directory)
    return result


def folder_date_ordinal(name: str) -> int | None:
    """Parse a ROC or Gregorian date embedded in a folder name.

    ``1150921`` and ``115.09.21`` are ROC dates. ``20250921`` is Gregorian.
    The result is comparable across both calendars.
    """
    for pattern, roc in _DATE_PATTERNS:
        match = re.search(pattern, name)
        if match is None:
            continue
        year, month, day = (int(match.group(index)) for index in range(1, 4))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        gregorian_year = year + 1911 if roc else year
        return gregorian_year * 10000 + month * 100 + day
    return None


def viewing_date_ordinal(case_dir: Path, path: Path) -> int | None:
    """Date of the nearest 公開閱覽 folder that contains this file."""
    case_dir = case_dir.resolve()
    current = path.resolve().parent
    while current != case_dir:
        if "公開閱覽" in current.name:
            found = folder_date_ordinal(current.name)
            if found is not None:
                return found
            ancestor = current.parent
            while ancestor != case_dir:
                if "公開閱覽" in ancestor.name:
                    break
                found = folder_date_ordinal(ancestor.name)
                if found is not None:
                    return found
                ancestor = ancestor.parent
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def select_latest_documents(case_dir: Path, paths: list[Path]) -> list[Path]:
    """Keep the newest dated 公開閱覽 copy of each investment document.

    申請須知、需求說明書、評選文件、契約 may be repeated in older viewing
    folders. Other files, and a kind that has no dated viewing copy, stay.
    detect, ingest, and extract all use this through ``scan_cases_root``.
    """
    rows: list[tuple[Path, str, int | None]] = []
    latest: dict[str, int] = {}
    for path in paths:
        kind, _label = classify_document(path.name)
        ordinal = viewing_date_ordinal(case_dir, path)
        rows.append((path, kind, ordinal))
        if kind in VERSIONED_KINDS and ordinal is not None:
            latest[kind] = max(latest.get(kind, ordinal), ordinal)
    chosen: list[Path] = []
    for path, kind, ordinal in rows:
        if kind in latest and ordinal != latest[kind]:
            continue
        chosen.append(path)
    return chosen


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
    for path in select_latest_documents(case_dir, iter_case_files(case_dir)):
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
