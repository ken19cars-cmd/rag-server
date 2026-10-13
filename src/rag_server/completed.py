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


def fetch_notion_database(
    token: str,
    database_id: str,
    *,
    client: httpx.Client | None = None,
    status_filter: dict | None = None,
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
            response = http.post(
                f"https://api.notion.com/v1/databases/{database_id}/query",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Notion-Version": NOTION_VERSION,
                    "Content-Type": "application/json",
                },
                json=body,
            )
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


def case_match_key(name: str) -> str:
    """Fold a Notion title and a case-folder name onto one comparison key.

    NFKC folds full-width letters. 台 and 臺 count as the same character.
    Spaces drop out, and dash variants become ASCII hyphens.
    """
    folded = unicodedata.normalize("NFKC", name).translate(_DASHES).replace("台", "臺")
    return re.sub(r"\s+", "", folded).casefold()


def pair_notion_pages_to_folders(pages: list[CompletedCase], folders: list) -> dict:
    """Pair open Notion pages to local case folders by normalized name.

    A pair is kept only when exactly one page and one folder share the key.
    Collisions stay in the unmatched lists so a later write cannot pick the
    wrong page. ``folders`` are corpus cases from ``scan_cases_root``.
    """
    folders_by_key: dict[str, list] = {}
    for folder in folders:
        folders_by_key.setdefault(case_match_key(folder.case_name), []).append(folder)

    pages_by_key: dict[str, list[CompletedCase]] = {}
    for page in pages:
        pages_by_key.setdefault(case_match_key(page.case_name), []).append(page)

    matched: list[dict] = []
    notion_without_folder: list[dict] = []
    used_folder_ids: set[int] = set()
    for page in pages:
        key = case_match_key(page.case_name)
        page_group = pages_by_key[key]
        folder_group = folders_by_key.get(key) or []
        if len(page_group) == 1 and len(folder_group) == 1:
            folder = folder_group[0]
            matched.append(_matched_row(page, folder))
            used_folder_ids.add(id(folder))
        else:
            notion_without_folder.append(_notion_only_row(page))

    folder_without_notion = [
        _folder_only_row(folder) for folder in folders if id(folder) not in used_folder_ids
    ]
    return {
        "matched": matched,
        "notion_without_folder": notion_without_folder,
        "folder_without_notion": folder_without_notion,
    }


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


def _matched_row(page: CompletedCase, folder) -> dict:
    return {
        "case_name": page.case_name,
        "folder_name": folder.case_name,
        "notion_page_id": page.notion_page_id,
        "status": page.status,
        "city": folder.city,
        "case_type": folder.case_type,
        "source_path": str(folder.folder),
        "documents": _document_rows(folder.documents),
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
