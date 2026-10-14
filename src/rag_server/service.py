"""Case index: ingest PDFs and crawled pages, then query them."""

from __future__ import annotations

from pathlib import Path

from pypdf.errors import PdfReadError

from rag_server.analysis import analyze_case, answer_question
from rag_server.chunking import chunk_pages
from rag_server.completed import (
    OPEN_STATUS_FILTER,
    CompletedCase,
    SearchBackend,
    cases_from_notion_query,
    fetch_notion_database,
    load_case_file,
    pair_notion_pages_to_folders,
    research_case,
    windows_folder_name,
)
from rag_server.config import Settings
from rag_server.corpus import CorpusCase, merge_document_texts, scan_cases_root
from rag_server.crawler import CrawledDocument, Crawler, Fetcher, HttpFetcher
from rag_server.embeddings import Embedder
from rag_server.llm import LlmClient
from rag_server.notion_analysis import analyze_to_notion
from rag_server.pdf_extract import ExtractedPage, OcrEngine, TesseractOcrEngine, extract_pdf_bytes
from rag_server.schemas import (
    AnalysisResponse,
    CaseDetail,
    CaseSummary,
    CrawlRequest,
    CrawlResponse,
    DocumentSummary,
    IngestResponse,
    QueryResponse,
)
from rag_server.store import CaseStore, new_id, warnings_from_row


class CaseNotFound(Exception):
    pass


class IngestError(Exception):
    pass


def _corpus_case_dict(case: CorpusCase) -> dict:
    return {
        "case_type": case.case_type,
        "city": case.city,
        "city_folder": case.city_folder,
        "case_name": case.case_name,
        "source_path": str(case.folder),
        "documents": [
            {
                "filename": document.filename,
                "source_path": str(document.path),
                "doc_kind": document.doc_kind,
                "doc_kind_label": document.doc_kind_label,
            }
            for document in case.documents
        ],
    }


class CaseService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        *,
        store: CaseStore | None = None,
        fetcher: Fetcher | None = None,
        ocr: OcrEngine | None = None,
        llm: LlmClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store or CaseStore(Path(settings.data_dir) / "index.sqlite")
        self.fetcher = fetcher
        self.ocr = ocr if ocr is not None else TesseractOcrEngine()
        self.llm = llm

    def open(self) -> None:
        self.store.open()

    def close(self) -> None:
        self.store.close()

    def create_case(self, name: str, description: str = "", **fields) -> CaseDetail:
        case_id = self.store.create_case(name.strip(), description.strip(), **fields)
        return self.get_case(case_id)

    def list_cases(self) -> list[CaseSummary]:
        return [self._case_summary(row) for row in self.store.list_cases()]

    def get_case(self, case_id: str) -> CaseDetail:
        row = self._require_case(case_id)
        documents = [self._document_summary(item) for item in self.store.list_documents(case_id)]
        summary = self._case_summary(row)
        return CaseDetail(**summary.model_dump(), documents=documents)

    def ingest_text(
        self,
        case_id: str,
        *,
        title: str,
        text: str,
        source_url: str | None = None,
        source_type: str = "text",
        filename: str | None = None,
        doc_kind: str = "",
        source_path: str = "",
    ) -> IngestResponse:
        self._require_case(case_id)
        cleaned = text.strip()
        if not cleaned:
            raise IngestError("document has no text")
        document = self._index_pages(
            case_id,
            pages=[ExtractedPage(page=1, text=cleaned, method="text")],
            source_type=source_type,
            title=title.strip() or "untitled",
            filename=filename,
            source_url=source_url,
            warnings=[],
            doc_kind=doc_kind,
            source_path=source_path,
        )
        return IngestResponse(case_id=case_id, document=document)

    def ingest_pdf(
        self,
        case_id: str,
        *,
        filename: str,
        data: bytes,
        source_url: str | None = None,
        doc_kind: str = "",
        source_path: str = "",
    ) -> IngestResponse:
        self._require_case(case_id)
        if not data.startswith(b"%PDF-"):
            raise IngestError("file is not a PDF")
        try:
            extracted = extract_pdf_bytes(data, ocr=self.ocr)
        except PdfReadError as exc:
            raise IngestError(f"could not read PDF: {exc}") from exc
        except Exception as exc:
            raise IngestError(f"could not read PDF: {exc}") from exc
        pages = extracted.pages or [ExtractedPage(page=1, text="", method="empty")]
        if not extracted.full_text.strip():
            warnings = list(extracted.warnings)
            warnings.append("no text could be extracted from this PDF")
            document = self._index_pages(
                case_id,
                pages=[],
                source_type="pdf",
                title=filename,
                filename=filename,
                source_url=source_url,
                warnings=warnings,
                page_count=len(pages),
                ocr_page_count=extracted.ocr_page_count,
                doc_kind=doc_kind,
                source_path=source_path,
            )
            return IngestResponse(case_id=case_id, document=document)
        document = self._index_pages(
            case_id,
            pages=pages,
            source_type="pdf",
            title=filename,
            filename=filename,
            source_url=source_url,
            warnings=extracted.warnings,
            page_count=len(pages),
            ocr_page_count=extracted.ocr_page_count,
            doc_kind=doc_kind,
            source_path=source_path,
        )
        return IngestResponse(case_id=case_id, document=document)

    def crawl(self, case_id: str, request: CrawlRequest) -> CrawlResponse:
        self._require_case(case_id)
        fetcher = self.fetcher
        owns_fetcher = False
        if fetcher is None:
            fetcher = HttpFetcher(
                self.settings.crawl_user_agent,
                self.settings.crawl_timeout_seconds,
            )
            owns_fetcher = True
        try:
            crawler = Crawler(
                fetcher,
                user_agent=self.settings.crawl_user_agent,
                delay_seconds=request.delay_seconds,
                max_pdf_bytes=self.settings.crawl_max_pdf_bytes,
            )
            crawled = crawler.crawl(
                request.seed_urls,
                max_pages=request.max_pages,
                max_depth=request.max_depth,
                same_domain_only=request.same_domain_only,
            )
            chunks_added = 0
            documents_indexed = 0
            for document in crawled.documents:
                try:
                    ingested = self._ingest_crawled(case_id, document)
                except IngestError as exc:
                    crawled.errors.append(f"{document.url}: {exc}")
                    continue
                documents_indexed += 1
                chunks_added += ingested.document.chunk_count
            return CrawlResponse(
                case_id=case_id,
                pages_fetched=crawled.pages_fetched,
                pdfs_fetched=crawled.pdfs_fetched,
                documents_indexed=documents_indexed,
                chunks_added=chunks_added,
                skipped_by_robots=crawled.skipped_by_robots,
                skipped=crawled.skipped,
                errors=crawled.errors,
            )
        finally:
            if owns_fetcher:
                fetcher.close()

    def query(self, case_id: str, question: str, top_k: int) -> QueryResponse:
        self._require_case(case_id)
        chunks = self.store.list_chunks(case_id)
        return answer_question(case_id, question, chunks, self.embedder, top_k)

    def analyze(self, case_id: str) -> AnalysisResponse:
        self._require_case(case_id)
        chunks = self.store.list_chunks(case_id)
        return analyze_case(case_id, chunks, self.embedder)

    def _ingest_crawled(self, case_id: str, document: CrawledDocument) -> IngestResponse:
        if document.kind == "pdf":
            return self.ingest_pdf(
                case_id,
                filename=document.title or "crawled.pdf",
                data=document.body,
                source_url=document.url,
            )
        return self.ingest_text(
            case_id,
            title=document.title or document.url,
            text=document.text,
            source_url=document.url,
            source_type="html",
        )

    def _index_pages(
        self,
        case_id: str,
        *,
        pages: list[ExtractedPage],
        source_type: str,
        title: str,
        filename: str | None,
        source_url: str | None,
        warnings: list[str],
        page_count: int | None = None,
        ocr_page_count: int | None = None,
        doc_kind: str = "",
        source_path: str = "",
    ) -> DocumentSummary:
        usable = [(page.page, page.text) for page in pages if page.text.strip()]
        drafts = chunk_pages(
            usable,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        vectors = self.embedder.embed_texts([draft.text for draft in drafts]) if drafts else []
        chunk_rows = []
        for draft, vector in zip(drafts, vectors):
            chunk_rows.append(
                {
                    "id": new_id("chunk"),
                    "ordinal": draft.ordinal,
                    "text": draft.text,
                    "page": draft.page,
                    "start_char": draft.start_char,
                    "end_char": draft.end_char,
                    "embedding": vector,
                }
            )
        counted_pages = page_count if page_count is not None else len(pages)
        counted_ocr = ocr_page_count
        if counted_ocr is None:
            counted_ocr = sum(1 for page in pages if page.method in {"ocr", "mixed"})
        document_id = self.store.add_document(
            case_id=case_id,
            source_type=source_type,
            title=title,
            filename=filename,
            source_url=source_url,
            page_count=counted_pages,
            ocr_page_count=counted_ocr,
            warnings=warnings,
            chunks=chunk_rows,
            doc_kind=doc_kind,
            source_path=source_path,
        )
        stored = next(row for row in self.store.list_documents(case_id) if row["id"] == document_id)
        summary = self._document_summary(stored)
        summary.chunk_count = len(chunk_rows)
        return summary

    def _require_case(self, case_id: str):
        row = self.store.get_case(case_id)
        if row is None:
            raise CaseNotFound(case_id)
        return row

    def _case_summary(self, row) -> CaseSummary:
        document_count, chunk_count = self.store.counts_for_case(row["id"])
        return CaseSummary(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            document_count=document_count,
            chunk_count=chunk_count,
            stage=row["stage"] or "in_progress",
            case_type=row["case_type"] or "",
            city=row["city"] or "",
            source_path=row["source_path"] or "",
            notion_page_id=row["notion_page_id"] or "",
        )

    def _document_summary(self, row) -> DocumentSummary:
        chunk_count = self.store.count_chunks_for_document(row["id"])
        return DocumentSummary(
            id=row["id"],
            source_type=row["source_type"],
            title=row["title"],
            filename=row["filename"],
            source_url=row["source_url"],
            page_count=row["page_count"],
            chunk_count=chunk_count,
            ocr_page_count=row["ocr_page_count"],
            warnings=warnings_from_row(row),
            doc_kind=row["doc_kind"] or "",
            source_path=row["source_path"] or "",
        )

    def scan_corpus(self, root: str = "") -> list[dict]:
        cases = scan_cases_root(self._cases_root(root))
        return [_corpus_case_dict(case) for case in cases]

    def ingest_corpus(self, root: str = "") -> dict:
        cases = scan_cases_root(self._cases_root(root))
        documents_indexed = 0
        case_ids: list[str] = []
        for case in cases:
            case_id = self._upsert_corpus_case(case)
            case_ids.append(case_id)
            for document in case.documents:
                if self.store.find_document_by_source_path(str(document.path)):
                    continue
                if document.path.suffix.lower() == ".pdf":
                    self.ingest_pdf(
                        case_id,
                        filename=document.filename,
                        data=document.path.read_bytes(),
                        doc_kind=document.doc_kind,
                        source_path=str(document.path),
                    )
                else:
                    self.ingest_text(
                        case_id,
                        title=document.filename,
                        text=document.path.read_text(encoding="utf-8"),
                        filename=document.filename,
                        doc_kind=document.doc_kind,
                        source_path=str(document.path),
                    )
                documents_indexed += 1
        return {
            "cases": [_corpus_case_dict(case) for case in cases],
            "case_ids": case_ids,
            "documents_indexed": documents_indexed,
        }

    def merged_case_text(self, case_folder: str, root: str = "") -> tuple[str, str]:
        folder = Path(case_folder)
        if not folder.is_absolute():
            folder = self._cases_root(root) / folder
        folder = folder.resolve()
        matches = [case for case in scan_cases_root(self._cases_root(root)) if case.folder.resolve() == folder]
        if not matches and folder.is_dir():
            parent_root = folder.parent.parent.parent
            matches = [case for case in scan_cases_root(parent_root) if case.folder.resolve() == folder]
        if not matches:
            raise IngestError(f"case folder not found: {case_folder}")
        case = matches[0]
        parts: list[tuple[str, str, str]] = []
        for document in case.documents:
            if document.path.suffix.lower() == ".pdf":
                extracted = extract_pdf_bytes(document.path.read_bytes(), ocr=self.ocr)
                text = extracted.full_text
            else:
                text = document.path.read_text(encoding="utf-8")
            parts.append((document.doc_kind_label, document.filename, text))
        return case.case_name, merge_document_texts(parts)

    def analyze_notion(
        self,
        *,
        case_name: str,
        page_id: str,
        pdf_text: str = "",
        case_id: str = "",
        case_folder: str = "",
        root: str = "",
        template_blocks: list[dict] | None = None,
    ) -> dict:
        text = pdf_text
        resolved_name = case_name
        if not text and case_id:
            self._require_case(case_id)
            text = "\n\n".join(chunk.text for chunk in self.store.list_chunks(case_id))
        if not text and case_folder:
            resolved_name, text = self.merged_case_text(case_folder, root)
            if case_name:
                resolved_name = case_name
        return analyze_to_notion(
            case_name=resolved_name,
            page_id=page_id,
            pdf_text=text,
            template_blocks=template_blocks or [],
            llm=self.llm,
        )

    def match_open_notion_cases(self, root: str = "") -> dict:
        """Pair Notion pages that are not 招標完成 with local case folders."""
        pages, folders, _root = self._open_pages_and_folders(root)
        return pair_notion_pages_to_folders(pages, folders)

    def rename_open_case_folders(self, root: str = "") -> dict:
        """Rename uniquely matched case folders to the Notion page title.

        招標完成 pages are not in this set. A folder is renamed only when the
        pairing is one page to one folder, and the new name stays inside
        CASES_ROOT. This does not write Notion.
        """
        pages, folders, root_path = self._open_pages_and_folders(root)
        paired = pair_notion_pages_to_folders(pages, folders)
        renamed, skipped = self._rename_paired_folders(paired["renames"], root_path)
        result = pair_notion_pages_to_folders(pages, scan_cases_root(root_path))
        result["renamed"] = renamed
        result["skipped"] = skipped
        return result

    def _open_pages_and_folders(self, root: str):
        token = self.settings.notion_token.strip()
        database_id = self.settings.notion_database_id.strip()
        if not token or not database_id:
            raise IngestError("set RAG_NOTION_TOKEN and RAG_NOTION_DATABASE_ID")
        try:
            payload = fetch_notion_database(
                token,
                database_id,
                status_filter=OPEN_STATUS_FILTER,
            )
        except RuntimeError as exc:
            raise IngestError(str(exc)) from exc
        pages = cases_from_notion_query(payload, completed_only=False)
        root_path = self._cases_root(root).resolve()
        return pages, scan_cases_root(root_path), root_path

    def _rename_paired_folders(self, renames: list[dict], root_path: Path) -> tuple[list[dict], list[dict]]:
        renamed: list[dict] = []
        skipped: list[dict] = []
        for item in renames:
            target_name = windows_folder_name(item["to_name"])
            if target_name is None:
                skipped.append({**item, "reason": "Notion 案名含有 Windows 不能當資料夾名稱的字元"})
                continue
            source = Path(item["source_path"])
            try:
                source_resolved = source.resolve()
            except OSError as exc:
                skipped.append({**item, "reason": str(exc)})
                continue
            if not source_resolved.is_dir() or not source_resolved.is_relative_to(root_path):
                skipped.append({**item, "reason": "資料夾不在 CASES_ROOT 裡"})
                continue
            parents = source_resolved.parents
            if len(parents) < 3 or parents[2] != root_path:
                skipped.append({**item, "reason": "只改案件夾，不改上層資料夾"})
                continue
            if source_resolved.name == target_name:
                continue
            target = source_resolved.parent / target_name
            if target.exists():
                skipped.append({**item, "reason": "已經有同名資料夾"})
                continue
            try:
                source_resolved.rename(target)
            except OSError as exc:
                skipped.append({**item, "reason": str(exc)})
                continue
            self.store.rewrite_case_directory([source, source_resolved], target, target_name)
            renamed.append(
                {
                    "from_name": item["from_name"],
                    "to_name": target_name,
                    "source_path": str(target),
                    "notion_page_id": item.get("notion_page_id", ""),
                }
            )
        return renamed, skipped

    def list_completed(self) -> list[CompletedCase]:
        path = self.settings.completed_cases_file.strip()
        if path:
            return [case for case in load_case_file(Path(path)) if case.case_name]
        token = self.settings.notion_token.strip()
        database_id = self.settings.notion_database_id.strip()
        if token and database_id:
            return cases_from_notion_query(fetch_notion_database(token, database_id))
        raise IngestError("set RAG_COMPLETED_CASES_FILE or RAG_NOTION_TOKEN and RAG_NOTION_DATABASE_ID")

    def research_completed(
        self,
        case: CompletedCase,
        *,
        search: SearchBackend | None = None,
        delay_seconds: float | None = None,
    ):
        fetcher = self.fetcher
        owns = False
        if fetcher is None:
            fetcher = HttpFetcher(self.settings.crawl_user_agent, self.settings.crawl_timeout_seconds)
            owns = True
        try:
            return research_case(
                case,
                fetcher,
                user_agent=self.settings.crawl_user_agent,
                delay_seconds=self.settings.crawl_delay_seconds if delay_seconds is None else delay_seconds,
                search=search,
            )
        finally:
            if owns:
                fetcher.close()

    def _cases_root(self, root: str) -> Path:
        chosen = root.strip() or self.settings.cases_root.strip()
        if not chosen:
            raise IngestError("CASES_ROOT is not set")
        path = Path(chosen).expanduser()
        if not path.is_dir():
            raise IngestError(f"CASES_ROOT is not a directory: {path}")
        return path

    def _upsert_corpus_case(self, case: CorpusCase) -> str:
        existing = self.store.find_case_by_source_path(str(case.folder))
        if existing is not None:
            return existing["id"]
        return self.store.create_case(
            case.case_name,
            description=f"{case.case_type} / {case.city}",
            stage="in_progress",
            case_type=case.case_type,
            city=case.city,
            source_path=str(case.folder),
        )
