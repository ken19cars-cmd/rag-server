from pathlib import Path

import httpx

from rag_server.crawler import (
    Crawler,
    FetchResponse,
    HttpFetcher,
    RateLimiter,
    rejection_reason,
)
from tests.pdf_bytes import SAMPLE_LINES, build_text_pdf

FIXTURES = Path(__file__).parent / "fixtures"
SEED = "https://invest.example.com/cases/tainan"
PDF_URL = "https://invest.example.com/files/brief.pdf"
PRIVATE_URL = "https://invest.example.com/private/secret.html"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class MapFetcher:
    def __init__(self, mapping: dict[str, tuple[int, str, bytes]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        status, content_type, body = self.mapping[url]
        return FetchResponse(
            url=url,
            final_url=url,
            status=status,
            content_type=content_type,
            body=body,
        )


def _mapping() -> dict[str, tuple[int, str, bytes]]:
    return {
        "https://invest.example.com/robots.txt": (
            200,
            "text/plain",
            (FIXTURES / "robots.txt").read_bytes(),
        ),
        SEED: (
            200,
            "text/html; charset=utf-8",
            (FIXTURES / "sample_case.html").read_bytes(),
        ),
        PDF_URL: (200, "application/pdf", build_text_pdf(SAMPLE_LINES)),
    }


def test_rate_limiter_waits_between_requests():
    clock = Clock()
    limiter = RateLimiter(1.0, clock=clock.monotonic, sleep=clock.sleep)
    limiter.wait()
    limiter.wait()
    limiter.wait()
    assert clock.slept == [1.0, 1.0]


def test_private_and_non_http_urls_are_rejected():
    assert rejection_reason("http://127.0.0.1/admin") is not None
    assert rejection_reason("http://localhost/cases") is not None
    assert rejection_reason("file:///etc/passwd") is not None
    assert rejection_reason("https://invest.example.com/cases") is None


def test_crawl_respects_robots_and_stays_on_the_seed_domain():
    fetcher = MapFetcher(_mapping())
    crawler = Crawler(
        fetcher,
        user_agent="InvestmentCaseRAG/0.1 (public research client)",
        delay_seconds=0,
    )
    result = crawler.crawl(
        [SEED],
        max_pages=10,
        max_depth=1,
        same_domain_only=True,
    )
    assert result.pages_fetched == 1
    assert result.pdfs_fetched == 1
    assert PRIVATE_URL not in fetcher.calls
    assert PRIVATE_URL in result.skipped_by_robots
    assert any(item.startswith("https://other.example.org/news") for item in result.skipped)
    assert "mailto:invest@example.com.tw" not in fetcher.calls
    html = result.documents[0]
    assert html.kind == "html"
    assert "新臺幣 12 億元" in html.text
    assert result.documents[1].kind == "pdf"
    assert fetcher.calls[0].endswith("/robots.txt")


def test_disallowed_seed_is_not_fetched():
    fetcher = MapFetcher(
        {
            "https://invest.example.com/robots.txt": (
                200,
                "text/plain",
                b"User-agent: *\nDisallow: /cases/\n",
            )
        }
    )
    crawler = Crawler(fetcher, user_agent="InvestmentCaseRAG/0.1", delay_seconds=0)
    result = crawler.crawl([SEED], max_pages=5, max_depth=1, same_domain_only=True)
    assert fetcher.calls == ["https://invest.example.com/robots.txt"]
    assert result.pages_fetched == 0
    assert SEED in result.skipped_by_robots


def test_http_fetcher_uses_an_identifying_user_agent():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, text="ok")

    fetcher = HttpFetcher(
        "InvestmentCaseRAG/0.1 (public research client)",
        timeout=5,
        transport=httpx.MockTransport(handler),
    )
    try:
        response = fetcher.fetch("https://example.com/invest")
    finally:
        fetcher.close()
    assert response.status == 200
    assert seen["ua"] == "InvestmentCaseRAG/0.1 (public research client)"
