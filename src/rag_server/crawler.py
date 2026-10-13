"""Polite public-web crawler for investment-case HTML pages and linked PDFs.

The crawler identifies itself, waits between requests, and obeys robots.txt.
It does not log in, hide its client, or solve CAPTCHAs.
"""

from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from rag_server.pdf_extract import is_pdf_bytes

USER_AGENT_TOKEN = "InvestmentCaseRAG"


@dataclass
class FetchResponse:
    url: str
    final_url: str
    status: int
    content_type: str
    body: bytes


class Fetcher:
    def fetch(self, url: str) -> FetchResponse:
        raise NotImplementedError


class HttpFetcher(Fetcher):
    def __init__(
        self,
        user_agent: str,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.user_agent = user_agent
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/pdf,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def fetch(self, url: str) -> FetchResponse:
        response = self._client.get(url)
        return FetchResponse(
            url=url,
            final_url=str(response.url),
            status=response.status_code,
            content_type=response.headers.get("content-type", ""),
            body=response.content,
        )

    def close(self) -> None:
        self._client.close()


class RateLimiter:
    def __init__(
        self,
        delay_seconds: float,
        *,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self._clock = clock
        self._sleep = sleep
        self._last = 0.0
        self._started = False

    def wait(self) -> None:
        now = self._clock()
        if self._started and self.delay_seconds > 0:
            remaining = self.delay_seconds - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._started = True
        self._last = now


@dataclass
class CrawledDocument:
    url: str
    kind: str
    title: str
    text: str
    body: bytes
    content_type: str


@dataclass
class CrawlResult:
    documents: list[CrawledDocument] = field(default_factory=list)
    skipped_by_robots: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pages_fetched: int = 0
    pdfs_fetched: int = 0


def rejection_reason(url: str) -> str | None:
    """Return why a URL must not be fetched, or None when it is a public http(s) URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "only http and https URLs are fetched"
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return "missing host"
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        return "local hosts are not fetched"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        return "non-public IP addresses are not fetched"
    return None


def origin_of(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def decode_html(body: bytes, content_type: str) -> str:
    charset = ""
    lowered = content_type.lower()
    if "charset=" in lowered:
        charset = lowered.split("charset=", 1)[1].split(";", 1)[0].strip(" \"'")
    if charset:
        try:
            return body.decode(charset, errors="replace")
        except LookupError:
            pass
    head = body[:4096]
    import re

    match = re.search(br"charset=[\"']?([\w-]+)", head, re.IGNORECASE)
    if match:
        sniffed = match.group(1).decode("ascii", errors="ignore")
        try:
            return body.decode(sniffed, errors="replace")
        except LookupError:
            pass
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("big5", errors="replace")


def html_to_text_and_links(html: str, base_url: str) -> tuple[str, str, list[str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.get_text(strip=True)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href:
            continue
        absolute = urldefrag(urljoin(base_url, href))[0]
        if absolute:
            links.append(absolute)
    return title, text, links


def looks_like_pdf(url: str, content_type: str, body: bytes) -> bool:
    if "application/pdf" in content_type.lower():
        return True
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return True
    return is_pdf_bytes(body)


class RobotsPolicy:
    def __init__(self, fetcher: Fetcher, user_agent: str, limiter: RateLimiter) -> None:
        self.fetcher = fetcher
        self.user_agent = user_agent
        self.limiter = limiter
        self._cache: dict[str, RobotFileParser | None] = {}
        self.errors: list[str] = []

    def allowed(self, url: str) -> bool:
        parser = self._parser_for(origin_of(url))
        if parser is None:
            return False
        return parser.can_fetch(self.user_agent, url)

    def delay_for(self, url: str) -> float | None:
        parser = self._cache.get(origin_of(url))
        if parser is None:
            return None
        delay = parser.crawl_delay(self.user_agent)
        if delay is None:
            return None
        return float(delay)

    def _parser_for(self, origin: str) -> RobotFileParser | None:
        if origin in self._cache:
            return self._cache[origin]
        robots_url = origin + "/robots.txt"
        self.limiter.wait()
        try:
            response = self.fetcher.fetch(robots_url)
        except Exception as exc:
            self.errors.append(f"{robots_url}: {exc}")
            self._cache[origin] = None
            return None
        parser = RobotFileParser()
        parser.set_url(robots_url)
        if response.status == 404:
            parser.parse([])
            self._cache[origin] = parser
            return parser
        if response.status >= 400:
            self.errors.append(f"{robots_url}: HTTP {response.status}")
            self._cache[origin] = None
            return None
        text = decode_html(response.body, response.content_type)
        parser.parse(text.splitlines())
        self._cache[origin] = parser
        return parser


def same_host(left: str, right: str) -> bool:
    return (urlparse(left).hostname or "").lower() == (urlparse(right).hostname or "").lower()


class Crawler:
    def __init__(
        self,
        fetcher: Fetcher,
        *,
        user_agent: str,
        delay_seconds: float,
        max_pdf_bytes: int = 20_000_000,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.fetcher = fetcher
        self.user_agent = user_agent
        self.max_pdf_bytes = max_pdf_bytes
        self.limiter = RateLimiter(delay_seconds, clock=clock, sleep=sleep)

    def crawl(
        self,
        seed_urls: list[str],
        *,
        max_pages: int,
        max_depth: int,
        same_domain_only: bool,
    ) -> CrawlResult:
        result = CrawlResult()
        policy = RobotsPolicy(self.fetcher, self.user_agent, self.limiter)
        queue: list[tuple[str, int, str | None]] = []
        seen: set[str] = set()
        for seed in seed_urls:
            normalized = urldefrag(seed.strip())[0]
            reason = rejection_reason(normalized)
            if reason:
                result.skipped.append(f"{normalized}: {reason}")
                continue
            queue.append((normalized, 0, None))

        while queue and (result.pages_fetched + result.pdfs_fetched) < max_pages:
            url, depth, seed_host = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            reason = rejection_reason(url)
            if reason:
                result.skipped.append(f"{url}: {reason}")
                continue
            if seed_host and same_domain_only and not same_host(url, f"https://{seed_host}"):
                result.skipped.append(f"{url}: outside the seed domain")
                continue
            if not policy.allowed(url):
                result.skipped_by_robots.append(url)
                continue
            robots_delay = policy.delay_for(url)
            if robots_delay is not None and robots_delay > self.limiter.delay_seconds:
                self.limiter.delay_seconds = robots_delay
            self.limiter.wait()
            try:
                response = self.fetcher.fetch(url)
            except Exception as exc:
                result.errors.append(f"{url}: {exc}")
                continue
            final_url = urldefrag(response.final_url)[0] or url
            if final_url != url:
                final_reason = rejection_reason(final_url)
                if final_reason:
                    result.skipped.append(f"{final_url}: {final_reason}")
                    continue
                if not policy.allowed(final_url):
                    result.skipped_by_robots.append(final_url)
                    continue
            if response.status >= 400:
                result.errors.append(f"{url}: HTTP {response.status}")
                continue
            host = urlparse(final_url).hostname or urlparse(url).hostname or ""
            content_type = response.content_type
            if looks_like_pdf(final_url, content_type, response.body):
                if len(response.body) > self.max_pdf_bytes:
                    result.skipped.append(f"{final_url}: pdf exceeds size limit")
                    continue
                result.documents.append(
                    CrawledDocument(
                        url=final_url,
                        kind="pdf",
                        title=final_url.rsplit("/", 1)[-1] or "document.pdf",
                        text="",
                        body=response.body,
                        content_type=content_type,
                    )
                )
                result.pdfs_fetched += 1
                continue
            if not _looks_like_html(final_url, content_type, response.body):
                result.skipped.append(f"{final_url}: unsupported content type")
                continue
            html = decode_html(response.body, content_type)
            title, text, links = html_to_text_and_links(html, final_url)
            result.documents.append(
                CrawledDocument(
                    url=final_url,
                    kind="html",
                    title=title or final_url,
                    text=text,
                    body=b"",
                    content_type=content_type,
                )
            )
            result.pages_fetched += 1
            if depth >= max_depth:
                continue
            for link in links:
                if link in seen:
                    continue
                link_reason = rejection_reason(link)
                if link_reason:
                    continue
                if same_domain_only and not same_host(link, final_url):
                    result.skipped.append(f"{link}: outside the seed domain")
                    continue
                queue.append((link, depth + 1, host))
        result.errors.extend(policy.errors)
        result.skipped = _unique(result.skipped)
        result.skipped_by_robots = _unique(result.skipped_by_robots)
        result.errors = _unique(result.errors)
        return result


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _looks_like_html(url: str, content_type: str, body: bytes) -> bool:
    lowered = content_type.lower()
    if "html" in lowered or "text/plain" in lowered or lowered.startswith("text/"):
        return True
    path = urlparse(url).path.lower()
    if path.endswith((".html", ".htm", "/")) or "." not in path.rsplit("/", 1)[-1]:
        return True
    head = body[:200].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head
