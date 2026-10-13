"""API request and response models. Field names stay in English."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateCaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class TextIngestRequest(BaseModel):
    title: str = Field(default="untitled", max_length=200)
    text: str = Field(min_length=1)
    source_url: str | None = None


class CrawlRequest(BaseModel):
    seed_urls: list[str] = Field(min_length=1, max_length=50)
    max_pages: int = Field(default=10, ge=1, le=100)
    max_depth: int = Field(default=1, ge=0, le=3)
    delay_seconds: float = Field(default=1.0, ge=0, le=60)
    same_domain_only: bool = True


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class AnalyzeRequest(BaseModel):
    """Structured analysis reads the whole case index. The body may be empty."""


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    excerpt: str
    page: int | None = None
    source_url: str | None = None
    filename: str | None = None
    title: str | None = None
    score: float | None = None


class Claim(BaseModel):
    text: str
    citations: list[Citation]


class FieldValue(BaseModel):
    value: str | None
    status: Literal["found", "unknown"]
    label: str
    citations: list[Citation]


class UnknownField(BaseModel):
    field: str
    label: str


class CaseSummary(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    document_count: int
    chunk_count: int
    stage: str = "in_progress"
    case_type: str = ""
    city: str = ""
    source_path: str = ""
    notion_page_id: str = ""


class DocumentSummary(BaseModel):
    id: str
    source_type: str
    title: str
    filename: str | None
    source_url: str | None
    page_count: int
    chunk_count: int
    ocr_page_count: int
    warnings: list[str]
    doc_kind: str = ""
    source_path: str = ""


class CaseDetail(CaseSummary):
    documents: list[DocumentSummary]


class IngestResponse(BaseModel):
    case_id: str
    document: DocumentSummary


class QueryResponse(BaseModel):
    case_id: str
    question: str
    mode: Literal["extractive"]
    answer: str
    claims: list[Claim]
    citations: list[Citation]
    warnings: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    case_id: str
    mode: Literal["extractive"]
    summary: list[Claim]
    fields: dict[str, FieldValue]
    unknowns: list[UnknownField]
    warnings: list[str] = Field(default_factory=list)


class CrawlResponse(BaseModel):
    case_id: str
    pages_fetched: int
    pdfs_fetched: int
    documents_indexed: int
    chunks_added: int
    skipped_by_robots: list[str]
    skipped: list[str]
    errors: list[str]


class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    analysis_mode: Literal["extractive"] = "extractive"
    local_llm_enabled: bool
    ocr_available: bool


class NotionAnalyzeRequest(BaseModel):
    caseName: str = Field(min_length=1, max_length=300)
    pageId: str = Field(min_length=1, max_length=80)
    pdfText: str = ""
    caseId: str = ""
    caseFolder: str = ""
    root: str = ""
    templateBlocks: list[dict] = Field(default_factory=list)


class NotionBase(BaseModel):
    name: str = ""
    location: str = ""
    scope: str = ""
    area: str = ""
    ownership: str = ""


class NotionBaseTable(BaseModel):
    bases: list[NotionBase] = Field(default_factory=list)


class NotionZone(BaseModel):
    name: str = ""
    intensity: str = ""
    volumeItems: str = ""


class NotionLandUseTable(BaseModel):
    zones: list[NotionZone] = Field(default_factory=list)


class NotionBlockUpdate(BaseModel):
    blockId: str
    type: str
    value: str
    reason: str
    cellIndex: int | None = None


class NotionAnalysisResult(BaseModel):
    """Section 38 of Ken's analysis prompt. No extra wrapper fields."""

    model_config = ConfigDict(extra="ignore")

    pageId: str
    caseName: str
    baseTable: NotionBaseTable
    landUseTable: NotionLandUseTable
    updates: list[NotionBlockUpdate]


class CorpusScanRequest(BaseModel):
    root: str = ""


class CorpusExtractRequest(BaseModel):
    root: str = ""
    caseFolder: str = Field(min_length=1)


class Phase2Stub(BaseModel):
    phase: Literal[2] = 2
    status: Literal["not_implemented"] = "not_implemented"
    branch: str
    note: str
