"""FastAPI application for investment-case RAG."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import json

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from rag_server.analysis import IndexMismatch
from rag_server.config import Settings
from rag_server.crawler import Fetcher
from rag_server.embeddings import Embedder, build_embedder
from rag_server.completed import CompletedCase
from rag_server.llm import LlmClient, OpenAICompatibleClient
from rag_server.pdf_extract import OcrEngine
from rag_server.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    CaseDetail,
    CaseSummary,
    CrawlRequest,
    CrawlResponse,
    CorpusExtractRequest,
    CorpusScanRequest,
    CreateCaseRequest,
    HealthResponse,
    IngestResponse,
    NotionAnalysisResult,
    NotionAnalyzeRequest,
    Phase2Stub,
    QueryRequest,
    QueryResponse,
    TextIngestRequest,
)
from rag_server.service import CaseNotFound, CaseService, IngestError

MAX_UPLOAD_BYTES = 20_000_000
UI_PAGE = Path(__file__).resolve().parent / "static" / "index.html"
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
DEMO_QUERY_TEXT = "投資金額：新臺幣 12 億元。\n產業別：示範住宅更新。\n投資地點：示範市示範區。"


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    fetcher: Fetcher | None = None,
    ocr: OcrEngine | None = None,
    llm: LlmClient | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    resolved_embedder = embedder or build_embedder(
        resolved.embedding_backend,
        resolved.embedding_model,
    )
    resolved_llm = llm
    if resolved_llm is None and resolved.local_llm_configured:
        resolved_llm = OpenAICompatibleClient(
            resolved.llm_base_url,
            resolved.llm_api_key or "ollama",
            resolved.llm_model,
        )
    service = CaseService(
        resolved,
        resolved_embedder,
        fetcher=fetcher,
        ocr=ocr,
        llm=resolved_llm,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service.open()
        yield
        app.state.service.close()

    app = FastAPI(
        title="Investment Case RAG",
        summary="Analyze investment-promotion cases from PDFs and public web pages.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.settings = resolved

    @app.exception_handler(CaseNotFound)
    def _missing_case(_request: Request, _exc: CaseNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "case not found"})

    @app.exception_handler(IngestError)
    def _bad_document(_request: Request, exc: IngestError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(IndexMismatch)
    def _bad_index(_request: Request, exc: IndexMismatch) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    def ui_page() -> FileResponse:
        if not UI_PAGE.is_file():
            raise HTTPException(status_code=404, detail="status page is missing")
        return FileResponse(UI_PAGE, media_type="text/html; charset=utf-8")

    @app.get("/api/ui/status")
    def ui_status() -> dict:
        current: CaseService = app.state.service
        settings: Settings = app.state.settings
        root = settings.cases_root.strip()
        root_path = Path(root).expanduser() if root else None
        backend = getattr(current.embedder, "name", current.embedder.__class__.__name__)
        return {
            "status": "ok",
            "embedding_backend": backend,
            "analysis_mode": "extractive",
            "local_llm_enabled": current.llm is not None,
            "llm_base_url_set": bool(settings.llm_base_url.strip()),
            "llm_model_set": bool(settings.llm_model.strip()),
            "llm_base_url": settings.llm_base_url.strip(),
            "ocr_available": current.ocr.available(),
            "cases_root_set": bool(root),
            "cases_root_exists": bool(root_path and root_path.is_dir()),
            "cases_root": root,
            "completed_cases_file_set": bool(settings.completed_cases_file.strip()),
            "notion_configured": bool(settings.notion_token.strip() and settings.notion_database_id.strip()),
            "computer_a": "案件 PDF 與 RAG 服務",
            "computer_b": "Ollama，只經 Tailscale",
        }

    @app.get("/api/ui/demo")
    def ui_demo() -> dict:
        text_path = FIXTURES / "synthetic_case.txt"
        blocks_path = FIXTURES / "template_blocks.json"
        cases_root = FIXTURES / "cases_root"
        if not text_path.is_file() or not blocks_path.is_file():
            raise HTTPException(status_code=404, detail="demo fixture is missing")
        return {
            "caseName": "示範市都更案",
            "pageId": "page_demo_0001",
            "pdfText": text_path.read_text(encoding="utf-8"),
            "templateBlocks": json.loads(blocks_path.read_text(encoding="utf-8")),
            "queryText": DEMO_QUERY_TEXT,
            "question": "投資金額是多少？",
            "demoRoot": str(cases_root) if cases_root.is_dir() else "",
        }

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current: CaseService = app.state.service
        backend = getattr(current.embedder, "name", current.embedder.__class__.__name__)
        return HealthResponse(
            status="ok",
            embedding_backend=backend,
            local_llm_enabled=current.llm is not None,
            ocr_available=current.ocr.available(),
        )

    @app.post("/api/cases", response_model=CaseDetail)
    def create_case(payload: CreateCaseRequest) -> CaseDetail:
        return app.state.service.create_case(payload.name, payload.description)

    @app.get("/api/cases", response_model=list[CaseSummary])
    def list_cases() -> list[CaseSummary]:
        return app.state.service.list_cases()

    @app.get("/api/cases/{case_id}", response_model=CaseDetail)
    def get_case(case_id: str) -> CaseDetail:
        return app.state.service.get_case(case_id)

    @app.post("/api/cases/{case_id}/documents", response_model=IngestResponse)
    def upload_pdf(case_id: str, file: UploadFile = File(...)) -> IngestResponse:
        filename = Path(file.filename or "upload.pdf").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="upload a .pdf file")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="PDF exceeds the 20 MB upload limit")
        return app.state.service.ingest_pdf(case_id, filename=filename, data=data)

    @app.post("/api/cases/{case_id}/texts", response_model=IngestResponse)
    def ingest_text(case_id: str, payload: TextIngestRequest) -> IngestResponse:
        return app.state.service.ingest_text(
            case_id,
            title=payload.title,
            text=payload.text,
            source_url=payload.source_url,
        )

    @app.post("/api/cases/{case_id}/crawl", response_model=CrawlResponse)
    def crawl(case_id: str, payload: CrawlRequest) -> CrawlResponse:
        return app.state.service.crawl(case_id, payload)

    @app.post("/api/cases/{case_id}/query", response_model=QueryResponse)
    def query_case(case_id: str, payload: QueryRequest) -> QueryResponse:
        return app.state.service.query(case_id, payload.question, payload.top_k)

    @app.post("/api/cases/{case_id}/analyze", response_model=AnalysisResponse)
    def analyze_case_route(case_id: str, payload: AnalyzeRequest | None = None) -> AnalysisResponse:
        del payload
        return app.state.service.analyze(case_id)

    @app.post("/api/pipeline/detect")
    def detect_cases(payload: CorpusScanRequest | None = None) -> dict:
        root = payload.root if payload else ""
        return {"cases": app.state.service.scan_corpus(root)}

    @app.post("/api/pipeline/extract")
    def extract_case(payload: CorpusExtractRequest) -> dict:
        name, text = app.state.service.merged_case_text(payload.caseFolder, payload.root)
        return {"caseName": name, "pdfText": text}

    @app.post("/api/pipeline/ingest")
    def ingest_tree(payload: CorpusScanRequest | None = None) -> dict:
        root = payload.root if payload else ""
        return app.state.service.ingest_corpus(root)

    @app.get("/api/pipeline/local-match")
    def local_match(root: str = "") -> dict:
        return app.state.service.match_open_notion_cases(root)

    @app.get("/api/pipeline/unfilled")
    def unfilled_pages(root: str = "") -> dict:
        return app.state.service.list_unfilled_notion_cases(root)

    @app.post(
        "/api/notion/analyze",
        response_model=NotionAnalysisResult,
        response_model_exclude_none=True,
    )
    def analyze_notion(payload: NotionAnalyzeRequest) -> NotionAnalysisResult:
        result = app.state.service.analyze_notion(
            case_name=payload.caseName,
            page_id=payload.pageId,
            pdf_text=payload.pdfText,
            case_id=payload.caseId,
            case_folder=payload.caseFolder,
            root=payload.root,
            template_blocks=payload.templateBlocks,
        )
        return NotionAnalysisResult.model_validate(result)

    @app.get("/api/completed")
    def list_completed() -> dict:
        cases = app.state.service.list_completed()
        return {
            "cases": [
                {
                    "case_name": case.case_name,
                    "city": case.city,
                    "case_type": case.case_type,
                    "notion_page_id": case.notion_page_id,
                    "winning_applicant": case.winning_applicant,
                    "status": case.status,
                    "source_urls": case.source_urls,
                    "queries": case.queries(),
                }
                for case in cases
            ]
        }

    @app.post("/api/completed/research")
    def research_completed(payload: dict) -> dict:
        case = CompletedCase(
            case_name=str(payload.get("case_name") or payload.get("caseName") or ""),
            city=str(payload.get("city") or ""),
            case_type=str(payload.get("case_type") or ""),
            notion_page_id=str(payload.get("notion_page_id") or payload.get("pageId") or ""),
            winning_applicant=str(payload.get("winning_applicant") or ""),
            status=str(payload.get("status") or "招標完成"),
            source_urls=list(payload.get("source_urls") or []),
        )
        if not case.case_name:
            raise HTTPException(status_code=400, detail="case_name is required")
        found = app.state.service.research_completed(
            case,
            delay_seconds=float(payload.get("delay_seconds") or 0),
        )
        return {
            "case_name": found.case_name,
            "queries": found.queries,
            "pages": [{"url": page.url, "title": page.title, "text": page.text} for page in found.pages],
            "renderings": [
                {"url": image.url, "alt": image.alt, "source_page": image.source_page}
                for image in found.renderings
            ],
            "skipped_by_robots": found.skipped_by_robots,
            "skipped": found.skipped,
            "errors": found.errors,
        }

    def _phase2(branch: str, note: str) -> Phase2Stub:
        raise HTTPException(
            status_code=501,
            detail=Phase2Stub(branch=branch, note=note).model_dump(),
        )

    @app.post("/api/pipeline/notion/write")
    def notion_write() -> None:
        _phase2(
            "更新Notion內容 / 寫入Notion資料表 / 表格上色",
            "Phase 2 writes the section-38 JSON back to Notion and colors table rows.",
        )

    @app.post("/api/pipeline/schedule/svg")
    def schedule_svg() -> None:
        _phase2(
            "產生垂直簡約時程圖 / 上傳時程圖 / 期程圖導入Notion",
            "Phase 2 builds the schedule SVG from ❯期程規範 and uploads it.",
        )

    @app.post("/api/pipeline/schedule/pptx")
    def schedule_pptx() -> None:
        _phase2("產生簡約期程PPTX", "Phase 2 builds the schedule PPTX. It is not required for the local demo.")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "rag_server.main:app",
        host=settings.host,
        port=settings.port,
        factory=False,
    )


if __name__ == "__main__":
    main()
