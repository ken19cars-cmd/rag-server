"""Completed-case list and public-page enrichment.

In-progress cases come from the local PDF tree. Completed cases come from a
JSON/CSV export, or from Notion when a token is configured. The research job
builds public queries and fetches only allowed pages. It stores article text
and rendering image URLs. It does not log in or bypass CAPTCHA.
"""

from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import httpx

from rag_server.crawler import (
    RateLimiter,
    RobotsPolicy,
    decode_html,
    html_to_text_and_links,
    rejection_reason,
)

QUERY_SUFFIXES = ("最優申請人", "得標", "規劃", "建築效果模擬圖")
NOTION_VERSION = "2022-06-28"
NOTION_RATE_RETRIES = 5
RENDERING_HINTS = ("效果圖", "模擬圖", "示意圖", "建築效果", "外觀透視", "rendering", "perspective")


@dataclass
class CompletedCase:
    case_name: str
    city: str = ""
    case_type: str = ""
    notion_page_id: str = ""
    winning_applicant: str = ""
    status: str = "招標完成"
    source_urls: list[str] = field(default_factory=list)

    def queries(self) -> list[str]:
        return [f"{self.case_name} {suffix}" for suffix in QUERY_SUFFIXES]


@dataclass
class PublicPage:
    url: str
    title: str
    text: str


@dataclass
class Rendering:
    url: str
    alt: str
    source_page: str


@dataclass
class ResearchResult:
    case_name: str
    queries: list[str]
    pages: list[PublicPage] = field(default_factory=list)
    renderings: list[Rendering] = field(default_factory=list)
    skipped_by_robots: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SearchBackend:
    def search(self, query: str) -> list[str]:
        raise NotImplementedError


def load_case_file(path: Path) -> list[CompletedCase]:
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(raw.splitlines()))
        return [_case_from_mapping(row) for row in rows if (row.get("case_name") or row.get("案件名稱"))]
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("cases") or []
    return [_case_from_mapping(item) for item in payload if isinstance(item, dict)]


COMPLETED_STATUS_FILTER = {
    "property": "招商情形",
    "status": {"equals": "招標完成"},
}
OPEN_STATUS_FILTER = {
    "property": "招商情形",
    "status": {"does_not_equal": "招標完成"},
}


def notion_retry_delay(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait after a Notion 429. Honor Retry-After, otherwise back off."""
    header = response.headers.get("retry-after", "").strip()
    try:
        seconds = float(header)
    except ValueError:
        seconds = 0.0
    backoff = min(8.0, 0.5 * (2 ** (attempt - 1)))
    return max(seconds, backoff)


def _notion_response(request, *, sleep=time.sleep) -> httpx.Response:
    attempt = 0
    while True:
        response = request()
        if response.status_code != 429 or attempt >= NOTION_RATE_RETRIES:
            return response
        attempt += 1
        sleep(notion_retry_delay(response, attempt))


def fetch_notion_database(
    token: str,
    database_id: str,
    *,
    client: httpx.Client | None = None,
    status_filter: dict | None = None,
    sleep=time.sleep,
) -> dict:
    """Query a Notion database. The default filter is 招商情形 = 招標完成.

    Tests pass a client; the default opens one. Follows next_cursor until the
    database is exhausted.
    """
    chosen = status_filter or COMPLETED_STATUS_FILTER
    owns = client is None
    http = client or httpx.Client(timeout=20.0)
    results: list = []
    cursor = None
    try:
        while True:
            body: dict = {"filter": chosen, "page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            response = _notion_response(
                lambda: http.post(
                    f"https://api.notion.com/v1/databases/{database_id}/query",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Notion-Version": NOTION_VERSION,
                        "Content-Type": "application/json",
                    },
                    json=body,
                ),
                sleep=sleep,
            )
            if response.status_code == 429:
                raise RuntimeError("Notion 讀取太快被限制，請稍後再按一次。")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("notion query did not return an object")
            results.extend(payload.get("results") or [])
            if not payload.get("has_more"):
                return {"results": results, "has_more": False}
            cursor = payload.get("next_cursor")
            if not cursor:
                return {"results": results, "has_more": False}
    except httpx.HTTPError as exc:
        raise RuntimeError(f"notion query failed: {exc}") from exc
    finally:
        if owns:
            http.close()


def cases_from_notion_query(payload: dict, *, completed_only: bool = True) -> list[CompletedCase]:
    """Map a Notion database query body. Property names follow the existing site.

    ``completed_only`` keeps 招標完成 pages. Local-folder matching passes
    ``completed_only=False`` and skips those pages.
    """
    cases: list[CompletedCase] = []
    for page in payload.get("results") or []:
        props = page.get("properties") or {}
        name = _notion_text(props, ("案件名稱", "名稱", "Name")) or _notion_title(props)
        if not name:
            continue
        status = _notion_text(props, ("招商情形", "狀態", "Status"))
        if completed_only and status != "招標完成":
            continue
        if not completed_only and status == "招標完成":
            continue
        cases.append(
            CompletedCase(
                case_name=name,
                city=_notion_text(props, ("縣市別", "縣市", "City")),
                case_type=_notion_text(props, ("案件類型", "類型", "Type")),
                notion_page_id=str(page.get("id") or ""),
                winning_applicant=_notion_text(
                    props,
                    ("投標廠商1(最優申請人)", "投標廠商1（最優申請人）", "投標廠商1", "最優申請人"),
                ),
                status=status or ("招標完成" if completed_only else ""),
            )
        )
    return cases


_DASHES = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\ufe58": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)


def admin_sync_key(name: str) -> str:
    """Fold a missing 縣 or 市 onto the one official name that stem belongs to.

    ``屏東`` becomes ``屏東縣``. ``嘉義`` is left alone because it could be
    ``嘉義市`` or ``嘉義縣``. ``新竹`` is left alone for the same reason.
    """
    text = case_match_key(name)
    stems: dict[str, list[str]] = {}
    for city in _CITIES:
        stems.setdefault(city[:-1], []).append(city)
    unique = {stem: names[0] for stem, names in stems.items() if len(names) == 1 and stem}
    for stem, city in sorted(unique.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(re.escape(stem) + "(?![縣市])", city, text)
    return text


def case_match_key(name: str) -> str:
    """Fold a Notion title and a case-folder name onto one comparison key.

    NFKC folds full-width letters. 台 and 臺 count as the same character.
    Spaces drop out, and dash variants become ASCII hyphens.
    """
    folded = unicodedata.normalize("NFKC", name).translate(_DASHES).replace("台", "臺")
    folded = "".join(char for char in folded if unicodedata.category(char) != "Cf")
    return re.sub(r"\s+", "", folded).casefold()


def loose_match_key(name: str) -> str:
    """Match key that also ignores list punctuation and count wording.

    ``6、8`` and ``6,8`` are the same list. ``等兩筆`` and ``等2筆`` are the
    same count. Parentheses stay out of the comparison, but the text inside
    them stays.
    """
    text = case_match_key(name)
    text = re.sub(r"[、，,､()]", "", text)
    return re.sub(
        r"等([0-9一二三四五六七八九十兩]+)筆",
        lambda match: "等" + _cn_to_int(match.group(1)) + "筆",
        text,
    )


_CITIES = (
    "臺北市",
    "新北市",
    "桃園市",
    "臺中市",
    "臺南市",
    "高雄市",
    "基隆市",
    "新竹市",
    "嘉義市",
    "新竹縣",
    "苗栗縣",
    "彰化縣",
    "南投縣",
    "雲林縣",
    "嘉義縣",
    "屏東縣",
    "宜蘭縣",
    "花蓮縣",
    "臺東縣",
    "澎湖縣",
    "金門縣",
    "連江縣",
)
_CITY_RE = re.compile("|".join(_CITIES))
_DISTRICT_RE = re.compile(r"(\D{1,8}?[區鄉鎮市])")
_SECTION_RE = re.compile(r"(\D+?段(?:[0-9一二三四五六七八九十]+小段)?)")
_PARCEL_RE = re.compile(
    r"((?:\d+(?:[-之]\d+)?[、,，]*)+)地號(?:等([0-9一二三四五六七八九十兩]+)筆)?"
)
_LEADING_TAG = re.compile(r"^[〈【\[（(][^〉】\]）)]{0,24}[〉】\]）)]")
_PAREN = re.compile(r"[（(]([^（）()]*)[）)]")
_FORBIDDEN_FOLDER_CHARS = set('<>:"/\\|?*')
_RESERVED_FOLDER_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
    f"LPT{i}" for i in range(1, 10)
}
_CN_DIGIT = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(text: str) -> str:
    if text.isdigit():
        return str(int(text))
    if text in ("十", "兩"):
        return "10" if text == "十" else "2"
    if len(text) == 1 and text in _CN_DIGIT:
        return str(_CN_DIGIT[text])
    if len(text) == 2 and text[0] == "十" and text[1] in _CN_DIGIT:
        return str(10 + _CN_DIGIT[text[1]])
    if len(text) == 2 and text[1] == "十" and text[0] in _CN_DIGIT:
        return str(_CN_DIGIT[text[0]] * 10)
    if len(text) == 3 and text[1] == "十" and text[0] in _CN_DIGIT and text[2] in _CN_DIGIT:
        return str(_CN_DIGIT[text[0]] * 10 + _CN_DIGIT[text[2]])
    return text


def _prepare_land_text(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).translate(_DASHES).replace("台", "臺")
    text = re.sub(r"\s+", "", text)
    while True:
        stripped = _LEADING_TAG.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped

    def _drop_aside(match: re.Match[str]) -> str:
        inner = match.group(1)
        if "段" in inner or "地號" in inner:
            return match.group(0)
        return ""

    return _PAREN.sub(_drop_aside, text)


def land_identity(name: str, city: str = "") -> tuple[str, str, str, str, str] | None:
    """縣市、行政區、地段、地號, plus 等N筆 when the name states a count.

    A leading tag such as ``〈信義〉`` and a parenthetical aside such as
    ``(產專2-32)`` are ignored. Names without a cadastral parcel return None.
    """
    text = _prepare_land_text(name)
    city_match = _CITY_RE.search(text)
    if city_match:
        found_city = city_match.group(0)
        rest = text[city_match.end() :]
    else:
        fallback = _CITY_RE.search(_prepare_land_text(city))
        if fallback is None:
            return None
        found_city = fallback.group(0)
        rest = text
    district_match = _DISTRICT_RE.match(rest)
    if district_match is None:
        return None
    rest = rest[district_match.end() :]
    section_match = _SECTION_RE.match(rest)
    if section_match is None:
        return None
    section = re.sub(
        r"([0-9一二三四五六七八九十]+)小段",
        lambda match: _cn_to_int(match.group(1)) + "小段",
        section_match.group(1),
    )
    parcel_match = _PARCEL_RE.search(text)
    if parcel_match is None:
        return None
    parcel = _normalize_parcels(parcel_match.group(1))
    count = _cn_to_int(parcel_match.group(2)) if parcel_match.group(2) else ""
    return (found_city, district_match.group(1), section, parcel, count)


def _normalize_parcels(raw: str) -> str:
    """``6、8`` and ``6,8`` become the same parcel list. ``712-1`` stays one parcel."""
    parts = [part.replace("之", "-") for part in re.findall(r"\d+(?:[-之]\d+)?", raw)]

    def sort_key(part: str) -> tuple[int, ...]:
        return tuple(int(piece) if piece.isdigit() else 0 for piece in part.split("-"))

    return ",".join(sorted(parts, key=sort_key))


def _same_land(
    left: tuple[str, str, str, str, str] | None,
    right: tuple[str, str, str, str, str] | None,
) -> bool:
    if left is None or right is None:
        return False
    if left[:4] != right[:4]:
        return False
    return not left[4] or not right[4] or left[4] == right[4]


def windows_folder_name(name: str) -> str | None:
    """Return a Windows directory name, or None when the Notion title cannot be one."""
    cleaned = name.strip()
    if not cleaned or cleaned.endswith(".") or cleaned.endswith(" "):
        return None
    if any(ord(char) < 32 or char in _FORBIDDEN_FOLDER_CHARS for char in cleaned):
        return None
    if cleaned.split(".", 1)[0].upper() in _RESERVED_FOLDER_NAMES:
        return None
    return cleaned


def _pick_folder(page: CompletedCase, page_group: list[CompletedCase], folder_group: list):
    """Return the one folder for this page, or None when the pair is not unique.

    Several folders may fold to the same key. The folder whose name is already
    the Notion title is that page's folder. Any other collision stays unpaired.
    """
    if len(page_group) != 1 or not folder_group:
        return None
    if len(folder_group) == 1:
        return folder_group[0]
    target = windows_folder_name(page.case_name) or page.case_name
    exact = [
        folder
        for folder in folder_group
        if folder.case_name == page.case_name or folder.case_name == target
    ]
    if len(exact) == 1:
        return exact[0]
    return None


def pair_notion_pages_to_folders(pages: list[CompletedCase], folders: list) -> dict:
    """Pair open Notion pages to local case folders.

    A normalized title pairs first, and only when exactly one page and one
    folder share it. When several folders fold together, the folder already
    named with the Notion title wins. Pages and folders still left over pair
    when 縣市、行政區、地段、地號 are the same, then when a missing 縣 or 市 is
    the only gap and that stem belongs to one official name, then when the
    only difference is list punctuation, parentheses, or 等兩筆 versus 等2筆.
    Each pass keeps a pair only when it is unique. The Notion title is the
    name a folder should have; ``renames`` lists pairs whose folder name still
    differs. Unmatched pages carry ``near_folders`` for names the scan did see.
    """
    folders_by_key: dict[str, list] = {}
    for folder in folders:
        folders_by_key.setdefault(case_match_key(folder.case_name), []).append(folder)

    pages_by_key: dict[str, list[CompletedCase]] = {}
    for page in pages:
        pages_by_key.setdefault(case_match_key(page.case_name), []).append(page)

    matched: list[dict] = []
    used_folder_ids: set[int] = set()
    used_page_ids: set[int] = set()
    for page in pages:
        key = case_match_key(page.case_name)
        folder = _pick_folder(page, pages_by_key[key], folders_by_key.get(key) or [])
        if folder is None or id(folder) in used_folder_ids or id(page) in used_page_ids:
            continue
        matched.append(_matched_row(page, folder, "name"))
        used_folder_ids.add(id(folder))
        used_page_ids.add(id(page))

    remaining_pages = [page for page in pages if id(page) not in used_page_ids]
    remaining_folders = [folder for folder in folders if id(folder) not in used_folder_ids]
    page_land = {id(page): land_identity(page.case_name, page.city) for page in remaining_pages}
    folder_land = {
        id(folder): land_identity(folder.case_name, folder.city) for folder in remaining_folders
    }
    for page in remaining_pages:
        compatible = [
            folder
            for folder in remaining_folders
            if id(folder) not in used_folder_ids
            and _same_land(page_land[id(page)], folder_land[id(folder)])
        ]
        if len(compatible) > 1:
            target = windows_folder_name(page.case_name) or page.case_name
            exact = [
                folder
                for folder in compatible
                if folder.case_name == page.case_name or folder.case_name == target
            ]
            compatible = exact if len(exact) == 1 else []
        if len(compatible) != 1:
            continue
        folder = compatible[0]
        reverse = [
            other
            for other in remaining_pages
            if id(other) not in used_page_ids and _same_land(page_land[id(other)], folder_land[id(folder)])
        ]
        if len(reverse) != 1:
            continue
        matched.append(_matched_row(page, folder, "land"))
        used_folder_ids.add(id(folder))
        used_page_ids.add(id(page))

    remaining_pages = [page for page in pages if id(page) not in used_page_ids]
    remaining_folders = [folder for folder in folders if id(folder) not in used_folder_ids]
    pages_by_admin: dict[str, list[CompletedCase]] = {}
    folders_by_admin: dict[str, list] = {}
    for page in remaining_pages:
        pages_by_admin.setdefault(admin_sync_key(page.case_name), []).append(page)
    for folder in remaining_folders:
        folders_by_admin.setdefault(admin_sync_key(folder.case_name), []).append(folder)
    for page in remaining_pages:
        key = admin_sync_key(page.case_name)
        if not key:
            continue
        page_group = pages_by_admin[key]
        folder_group = folders_by_admin.get(key) or []
        folder = _pick_folder(page, page_group, folder_group)
        if folder is None or id(folder) in used_folder_ids or id(page) in used_page_ids:
            continue
        matched.append(_matched_row(page, folder, "admin"))
        used_folder_ids.add(id(folder))
        used_page_ids.add(id(page))

    remaining_pages = [page for page in pages if id(page) not in used_page_ids]
    remaining_folders = [folder for folder in folders if id(folder) not in used_folder_ids]
    pages_by_loose: dict[str, list[CompletedCase]] = {}
    folders_by_loose: dict[str, list] = {}
    for page in remaining_pages:
        pages_by_loose.setdefault(loose_match_key(page.case_name), []).append(page)
    for folder in remaining_folders:
        folders_by_loose.setdefault(loose_match_key(folder.case_name), []).append(folder)
    for page in remaining_pages:
        key = loose_match_key(page.case_name)
        if not key:
            continue
        folder = _pick_folder(page, pages_by_loose[key], folders_by_loose.get(key) or [])
        if folder is None or id(folder) in used_folder_ids or id(page) in used_page_ids:
            continue
        matched.append(_matched_row(page, folder, "loose"))
        used_folder_ids.add(id(folder))
        used_page_ids.add(id(page))

    renames = []
    for item in matched:
        target = windows_folder_name(item["case_name"]) or item["case_name"]
        if item["folder_name"] == target:
            continue
        renames.append(
            {
                "from_name": item["folder_name"],
                "to_name": target,
                "source_path": item["source_path"],
                "notion_page_id": item["notion_page_id"],
                "matched_by": item["matched_by"],
            }
        )
    return {
        "matched": matched,
        "notion_without_folder": [
            {
                **_notion_only_row(page),
                "near_folders": _near_folders(page, folders, used_folder_ids),
            }
            for page in pages
            if id(page) not in used_page_ids
        ],
        "folder_without_notion": [
            _folder_only_row(folder) for folder in folders if id(folder) not in used_folder_ids
        ],
        "renames": renames,
    }


def describe_completed_folders(remaining_folders: list, completed_pages: list[CompletedCase]) -> dict:
    """Say which leftover folders already exist as 招標完成 pages.

    Unique name or land pairs become ``folder_completed``. A folder that hits
    more than one completed page stays out of the automatic pair and is listed
    under ``folder_completed_ambiguous``. Folders with no completed page stay
    in ``folder_without_notion``.
    """
    paired = pair_notion_pages_to_folders(completed_pages, remaining_folders)
    ambiguous: list[dict] = []
    absent: list[dict] = []
    claimed_ids: set[str] = set()
    for folder in paired["folder_without_notion"]:
        hits = _completed_hits(folder, completed_pages)
        if not hits:
            absent.append(folder)
            continue
        claimed_ids.update(hit.notion_page_id for hit in hits)
        ambiguous.append(
            {
                **folder,
                "notion_pages": [
                    {
                        "case_name": hit.case_name,
                        "notion_page_id": hit.notion_page_id,
                        "status": hit.status,
                    }
                    for hit in hits
                ],
            }
        )
    completed_without = [
        page
        for page in paired["notion_without_folder"]
        if page["notion_page_id"] not in claimed_ids
    ]
    return {
        "folder_completed": paired["matched"],
        "folder_completed_ambiguous": ambiguous,
        "folder_without_notion": absent,
        "completed_without_folder": completed_without,
        "renames": paired["renames"],
    }


def _completed_hits(folder: dict, completed_pages: list[CompletedCase]) -> list[CompletedCase]:
    key = case_match_key(folder["case_name"])
    admin = admin_sync_key(folder["case_name"])
    loose = loose_match_key(folder["case_name"])
    land = land_identity(folder["case_name"], folder.get("city") or "")
    return [
        page
        for page in completed_pages
        if case_match_key(page.case_name) == key
        or (admin and admin_sync_key(page.case_name) == admin)
        or (loose and loose_match_key(page.case_name) == loose)
        or _same_land(land, land_identity(page.case_name, page.city))
    ]


def _near_folders(page: CompletedCase, folders: list, used_folder_ids: set[int]) -> list[dict]:
    """Folders the scan saw whose names are close to this unmatched page."""
    page_loose = loose_match_key(page.case_name)
    page_admin = admin_sync_key(page.case_name)
    page_land = land_identity(page.case_name, page.city)
    ranked: list[tuple[int, str, object]] = []
    for folder in folders:
        if id(folder) in used_folder_ids:
            continue
        other = loose_match_key(folder.case_name)
        score = 0
        if page_loose and other == page_loose:
            score = 100
        elif page_admin and admin_sync_key(folder.case_name) == page_admin:
            score = 90
        else:
            land = land_identity(folder.case_name, getattr(folder, "city", "") or "")
            if page_land and land and page_land[:3] == land[:3]:
                score = 80
            elif (
                page_loose
                and other
                and len(page_loose) >= 8
                and len(other) >= 8
                and (page_loose in other or other in page_loose)
            ):
                score = 70
        if score:
            ranked.append((score, folder.case_name, folder))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [_folder_only_row(folder) for _score, _name, folder in ranked[:3]]


def _document_rows(documents: list) -> list[dict]:
    return [
        {
            "filename": document.filename,
            "source_path": str(document.path),
            "doc_kind": document.doc_kind,
            "doc_kind_label": document.doc_kind_label,
        }
        for document in documents
    ]


def _matched_row(page: CompletedCase, folder, matched_by: str) -> dict:
    return {
        "case_name": page.case_name,
        "folder_name": folder.case_name,
        "notion_page_id": page.notion_page_id,
        "status": page.status,
        "city": folder.city,
        "case_type": folder.case_type,
        "source_path": str(folder.folder),
        "documents": _document_rows(folder.documents),
        "matched_by": matched_by,
    }


def _notion_only_row(page: CompletedCase) -> dict:
    return {
        "case_name": page.case_name,
        "notion_page_id": page.notion_page_id,
        "status": page.status,
        "city": page.city,
        "case_type": page.case_type,
    }


def _folder_only_row(folder) -> dict:
    return {
        "case_name": folder.case_name,
        "city": folder.city,
        "case_type": folder.case_type,
        "source_path": str(folder.folder),
        "documents": _document_rows(folder.documents),
    }


def research_case(
    case: CompletedCase,
    fetcher,
    *,
    user_agent: str,
    delay_seconds: float,
    search: SearchBackend | None = None,
    max_pages: int = 8,
) -> ResearchResult:
    result = ResearchResult(case_name=case.case_name, queries=case.queries())
    urls: list[str] = []
    seen: set[str] = set()
    for url in case.source_urls:
        if url not in seen:
            seen.add(url)
            urls.append(url)
    if search is not None:
        for query in result.queries:
            try:
                found = search.search(query)
            except Exception as exc:
                result.errors.append(f"{query}: {exc}")
                continue
            for url in found:
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
    limiter = RateLimiter(delay_seconds)
    policy = RobotsPolicy(fetcher, user_agent, limiter)
    fetched = 0
    for url in urls:
        if fetched >= max_pages:
            break
        reason = rejection_reason(url)
        if reason:
            result.skipped.append(f"{url}: {reason}")
            continue
        if not policy.allowed(url):
            result.skipped_by_robots.append(url)
            continue
        limiter.wait()
        try:
            response = fetcher.fetch(url)
        except Exception as exc:
            result.errors.append(f"{url}: {exc}")
            continue
        if response.status >= 400:
            result.errors.append(f"{url}: HTTP {response.status}")
            continue
        content_type = response.content_type.lower()
        if "html" not in content_type and not response.body.lstrip().lower().startswith(b"<!doctype html"):
            result.skipped.append(f"{url}: not an html page")
            continue
        html = decode_html(response.body, response.content_type)
        title, text, _links = html_to_text_and_links(html, response.final_url or url)
        result.pages.append(PublicPage(url=response.final_url or url, title=title, text=text))
        result.renderings.extend(_rendering_images(html, response.final_url or url))
        fetched += 1
    result.errors.extend(policy.errors)
    return result


def _rendering_images(html: str, page_url: str) -> list[Rendering]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    found: list[Rendering] = []
    for image in soup.find_all("img"):
        alt = str(image.get("alt") or "")
        src = str(image.get("src") or "").strip()
        nearby = alt + " " + str(image.get("title") or "")
        parent_text = image.parent.get_text(" ", strip=True) if image.parent else ""
        blob = f"{nearby} {parent_text}"
        if not src or not any(hint.lower() in blob.lower() for hint in RENDERING_HINTS):
            continue
        found.append(Rendering(url=urljoin(page_url, src), alt=alt, source_page=page_url))
    return found


def _case_from_mapping(item: dict) -> CompletedCase:
    urls = item.get("source_urls") or item.get("urls") or ""
    if isinstance(urls, str):
        urls = [part.strip() for part in urls.split() if part.strip()]
    return CompletedCase(
        case_name=str(item.get("case_name") or item.get("案件名稱") or "").strip(),
        city=str(item.get("city") or item.get("縣市") or item.get("縣市別") or ""),
        case_type=str(item.get("case_type") or item.get("案件類型") or ""),
        notion_page_id=str(item.get("notion_page_id") or item.get("pageId") or ""),
        winning_applicant=str(item.get("winning_applicant") or item.get("最優申請人") or ""),
        status=str(item.get("status") or item.get("招商情形") or "招標完成"),
        source_urls=[str(url) for url in urls],
    )


def _notion_title(properties: dict) -> str:
    for prop in properties.values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return _plain(prop.get("title"))
    return ""


def _notion_text(properties: dict, names: tuple[str, ...]) -> str:
    for name in names:
        prop = properties.get(name)
        if isinstance(prop, dict):
            return _plain_prop(prop)
    return ""


def _plain_prop(prop: dict) -> str:
    kind = prop.get("type")
    if kind == "title":
        return _plain(prop.get("title"))
    if kind == "rich_text":
        return _plain(prop.get("rich_text"))
    if kind == "select":
        return str((prop.get("select") or {}).get("name") or "")
    if kind == "status":
        return str((prop.get("status") or {}).get("name") or "")
    if kind == "url":
        return str(prop.get("url") or "")
    return ""


def _plain(items: object) -> str:
    if not isinstance(items, list):
        return ""
    return "".join(str(item.get("plain_text") or "") for item in items if isinstance(item, dict)).strip()
