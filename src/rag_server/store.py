"""SQLite file store for cases, documents, and chunk vectors."""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])


def unpack_vector(blob: bytes) -> list[float]:
    count = len(blob) // 4
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}f", blob[: count * 4]))


@dataclass
class ChunkRecord:
    id: str
    case_id: str
    document_id: str
    ordinal: int
    text: str
    page: int | None
    embedding: list[float]
    filename: str | None
    source_url: str | None
    title: str | None
    source_type: str


class CaseStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        self._conn = connection
        self._migrate()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("store is not open")
        return self._conn

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                filename TEXT,
                source_url TEXT,
                page_count INTEGER NOT NULL DEFAULT 0,
                ocr_page_count INTEGER NOT NULL DEFAULT 0,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                page INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                embedding BLOB NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_documents_case ON documents(case_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_case ON chunks(case_id);
            """
        )
        for table, column, ddl in (
            ("cases", "stage", "TEXT NOT NULL DEFAULT 'in_progress'"),
            ("cases", "case_type", "TEXT NOT NULL DEFAULT ''"),
            ("cases", "city", "TEXT NOT NULL DEFAULT ''"),
            ("cases", "source_path", "TEXT NOT NULL DEFAULT ''"),
            ("cases", "notion_page_id", "TEXT NOT NULL DEFAULT ''"),
            ("documents", "doc_kind", "TEXT NOT NULL DEFAULT ''"),
            ("documents", "source_path", "TEXT NOT NULL DEFAULT ''"),
        ):
            self._ensure_column(table, column, ddl)
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row["name"] == column for row in rows):
            return
        self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def create_case(
        self,
        name: str,
        description: str = "",
        *,
        stage: str = "in_progress",
        case_type: str = "",
        city: str = "",
        source_path: str = "",
        notion_page_id: str = "",
    ) -> str:
        case_id = new_id("case")
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO cases (
                    id, name, description, created_at, stage, case_type, city,
                    source_path, notion_page_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    name,
                    description,
                    utc_now(),
                    stage,
                    case_type,
                    city,
                    source_path,
                    notion_page_id,
                ),
            )
            self.connection.commit()
        return case_id

    def find_case_by_source_path(self, source_path: str) -> sqlite3.Row | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM cases WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return row

    def find_document_by_source_path(self, source_path: str) -> sqlite3.Row | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return row

    def get_case(self, case_id: str) -> sqlite3.Row | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM cases WHERE id = ?", (case_id,)
            ).fetchone()
        return row

    def list_cases(self) -> list[sqlite3.Row]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM cases ORDER BY created_at DESC"
            ).fetchall()
        return list(rows)

    def counts_for_case(self, case_id: str) -> tuple[int, int]:
        with self._lock:
            document_count = self.connection.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE case_id = ?", (case_id,)
            ).fetchone()["n"]
            chunk_count = self.connection.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE case_id = ?", (case_id,)
            ).fetchone()["n"]
        return int(document_count), int(chunk_count)

    def count_chunks_for_document(self, document_id: str) -> int:
        with self._lock:
            row = self.connection.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(row["n"])

    def list_documents(self, case_id: str) -> list[sqlite3.Row]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM documents WHERE case_id = ? ORDER BY created_at",
                (case_id,),
            ).fetchall()
        return list(rows)

    def add_document(
        self,
        *,
        case_id: str,
        source_type: str,
        title: str,
        filename: str | None,
        source_url: str | None,
        page_count: int,
        ocr_page_count: int,
        warnings: list[str],
        chunks: list[dict],
        doc_kind: str = "",
        source_path: str = "",
    ) -> str:
        document_id = new_id("doc")
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO documents (
                    id, case_id, source_type, title, filename, source_url,
                    page_count, ocr_page_count, warnings_json, created_at,
                    doc_kind, source_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    case_id,
                    source_type,
                    title,
                    filename,
                    source_url,
                    page_count,
                    ocr_page_count,
                    json.dumps(warnings, ensure_ascii=False),
                    utc_now(),
                    doc_kind,
                    source_path,
                ),
            )
            for chunk in chunks:
                self.connection.execute(
                    """
                    INSERT INTO chunks (
                        id, case_id, document_id, ordinal, text, page,
                        start_char, end_char, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        case_id,
                        document_id,
                        chunk["ordinal"],
                        chunk["text"],
                        chunk["page"],
                        chunk["start_char"],
                        chunk["end_char"],
                        pack_vector(chunk["embedding"]),
                    ),
                )
            self.connection.commit()
        return document_id

    def list_chunks(self, case_id: str) -> list[ChunkRecord]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT
                    c.id, c.case_id, c.document_id, c.ordinal, c.text, c.page,
                    c.embedding, d.filename, d.source_url, d.title, d.source_type
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.case_id = ?
                ORDER BY d.created_at, c.ordinal, c.id
                """,
                (case_id,),
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def rewrite_case_directory(self, old_dirs: list[Path], new_dir: Path, new_name: str) -> None:
        """Point indexed case and document paths at a renamed case folder."""
        old_texts: set[str] = set()
        for old_dir in old_dirs:
            old_texts.update(_path_texts(old_dir))
        new_text = str(new_dir)
        with self._lock:
            case_rows = self.connection.execute("SELECT id, source_path FROM cases").fetchall()
            for row in case_rows:
                if row["source_path"] not in old_texts:
                    continue
                self.connection.execute(
                    "UPDATE cases SET source_path = ?, name = ? WHERE id = ?",
                    (new_text, new_name, row["id"]),
                )
            document_rows = self.connection.execute(
                "SELECT id, source_path FROM documents"
            ).fetchall()
            for row in document_rows:
                updated = _replace_path_prefix(row["source_path"], old_texts, new_text)
                if updated is None or updated == row["source_path"]:
                    continue
                self.connection.execute(
                    "UPDATE documents SET source_path = ? WHERE id = ?",
                    (updated, row["id"]),
                )
            self.connection.commit()


def _path_texts(path: Path) -> set[str]:
    raw = str(path).rstrip("\\/")
    return {raw, raw.replace("\\", "/"), raw.replace("/", "\\")}


def _replace_path_prefix(value: str, old_texts: set[str], new_text: str) -> str | None:
    joiner = "\\" if "\\" in new_text else "/"
    new_root = new_text.rstrip("\\/")
    for old in old_texts:
        if value == old:
            return new_root
        for separator in ("/", "\\"):
            prefix = old.rstrip("/\\") + separator
            if value.startswith(prefix):
                suffix = value[len(prefix) :].replace("/", joiner).replace("\\", joiner)
                return new_root + joiner + suffix
    return None


def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        id=row["id"],
        case_id=row["case_id"],
        document_id=row["document_id"],
        ordinal=row["ordinal"],
        text=row["text"],
        page=row["page"],
        embedding=unpack_vector(row["embedding"]),
        filename=row["filename"],
        source_url=row["source_url"],
        title=row["title"],
        source_type=row["source_type"],
    )


def warnings_from_row(row: sqlite3.Row) -> list[str]:
    raw = row["warnings_json"] or "[]"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []
