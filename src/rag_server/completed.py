"""Completed-case list and public-page enrichment.

In-progress cases come from the local PDF tree. Completed cases come from a
JSON/CSV export, or from Notion when a token is configured. The research job
builds public queries and fetches only allowed pages. It stores article text
and rendering image URLs. It does not log in or bypass CAPTCHA.
"""

from __future__ import annotations

import csv
import json
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


def fetch_notion_database(
    token: str,
    database_id: str,
    *,
    client: httpx.Client | None = None,
) -> dict:
    """Query Notion for 招標完成 pages. Tests pass a client; the default opens one."""
    owns = client is None
    http = client or httpx.Client(timeout=20.0)
    try:
        response = http.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
            json={
                "filter": {
                    "property": "招商情形",
                    "status": {"equals": "招標完成"},
                }
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("notion query did not return an object")
        return payload
    except httpx.HTTPError as exc:
        raise RuntimeError(f"notion query failed: {exc}") from exc
    finally:
        if owns:
            http.close()


def cases_from_notion_query(payload: dict) -> list[CompletedCase]:
    """Map a Notion database query body. Property names follow the existing site."""
    cases: list[CompletedCase] = []
    for page in payload.get("results") or []:
        props = page.get("properties") or {}
        name = _notion_text(props, ("案件名稱", "名稱", "Name")) or _notion_title(props)
        if not name:
            continue
        status = _notion_text(props, ("招商情形", "狀態", "Status"))
        if status and status != "招標完成":
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
                status=status or "招標完成",
            )
        )
    return cases


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
