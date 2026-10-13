"""Extractive case Q&A. Notion section fill lives in notion_analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_server.embeddings import Embedder, cosine
from rag_server.schemas import (
    AnalysisResponse,
    Citation,
    Claim,
    FieldValue,
    QueryResponse,
    UnknownField,
)
from rag_server.store import ChunkRecord
from rag_server.textutil import clip_text, ngram_overlap, split_sentences

EXCERPT_LIMIT = 240
CLAIM_LIMIT = 300

FIELD_SPECS: dict[str, tuple[str, str, list[str]]] = {
    "industry": (
        "產業",
        "產業別 行業 產業類型",
        [
            r"(?:產業別|产业别|產業類型|产业类型|引進產業|引进产业|行業別|行业别|行業|行业)[:：]\s*([^\n。；;]{2,40})",
        ],
    ),
    "location": (
        "地點",
        "投資地點 基地 位置 座落",
        [
            r"(?:投資地點|投资地点|基地位置|座落地點|座落地点|所在地|地點|地点)[:：]\s*([^\n。；;]{2,40})",
            r"(?:位於|位于)([^\n。，,]{2,30}(?:市|縣|县|區|区|省))",
        ],
    ),
    "incentives": (
        "優惠措施",
        "租稅優惠 獎勵 補貼 減免",
        [
            r"(?:租稅優惠|租税优惠|優惠措施|优惠措施|獎勵措施|奖励措施|土地優惠|土地优惠|補貼|补贴)[:：]\s*([^\n。；;]{2,80})",
        ],
    ),
    "investment_amount": (
        "投資金額",
        "投資金額 投資額 新臺幣 億元",
        [
            r"(?:投資金額|投资金额|投資額度|投资额度|投資額|投资额|總投資金額|总投资金额)[:：]\s*([^\n。；;]{2,60})",
            r"((?:新臺幣|新台币|人民幣|人民币|美元)\s*[0-9][0-9,.]*(?:\s*[億亿萬万元])?)",
        ],
    ),
    "timeline": (
        "時程",
        "時程 期程 截止 開工",
        [
            r"(?:推動時程|推动时程|實施期程|实施期程|招商時程|招商时程|時程|时程|期程)[:：]\s*([^\n。]{2,80})",
        ],
    ),
    "contacts": (
        "聯絡窗口",
        "聯絡人 承辦 電話 email",
        [
            r"(?:聯絡窗口|联络窗口|聯絡人|联络人|承辦人|承办人|聯繫方式|联系方式)[:：]\s*([^\n]{2,80})",
            r"([^\n]{0,40}(?:電話|电话|email|Email)[^\n]{2,60})",
        ],
    ),
}


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    score: float


@dataclass
class _FieldHit:
    value: str
    source_line: str
    chunk: ChunkRecord
    score: float


def excerpt_for(chunk_text: str, claim: str, limit: int = EXCERPT_LIMIT) -> str:
    claim = claim.strip()
    if claim and claim in chunk_text:
        start = chunk_text.find(claim)
        pad = max(0, (limit - len(claim)) // 2)
        begin = max(0, start - pad)
        end = min(len(chunk_text), max(begin + limit, start + len(claim)))
        begin = max(0, end - limit)
        return chunk_text[begin:end].strip()
    return clip_text(chunk_text, limit)


def citation_for(chunk: ChunkRecord, claim: str, score: float | None = None) -> Citation:
    return Citation(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        excerpt=excerpt_for(chunk.text, claim),
        page=chunk.page,
        source_url=chunk.source_url,
        filename=chunk.filename,
        title=chunk.title,
        score=None if score is None else round(score, 4),
    )


def hybrid_score(query: str, query_vector: list[float], chunk: ChunkRecord) -> float:
    lexical = ngram_overlap(query, chunk.text)
    semantic = cosine(query_vector, chunk.embedding)
    return 0.65 * semantic + 0.35 * lexical


class IndexMismatch(RuntimeError):
    """Stored vectors were built with a different embedding width."""


def retrieve(
    chunks: list[ChunkRecord],
    query: str,
    embedder: Embedder,
    top_k: int,
) -> list[RetrievedChunk]:
    if not chunks or top_k <= 0:
        return []
    query_vector = embedder.embed_query(query)
    stored_width = len(chunks[0].embedding)
    if stored_width != len(query_vector):
        raise IndexMismatch(
            "stored embeddings do not match the active embedding backend; "
            "re-ingest the case documents"
        )
    ranked = [
        RetrievedChunk(chunk=chunk, score=hybrid_score(query, query_vector, chunk))
        for chunk in chunks
    ]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def _compiled(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def _find_field(
    chunks: list[ChunkRecord],
    patterns: list[str],
    query: str,
    embedder: Embedder,
) -> _FieldHit | None:
    compiled = _compiled(patterns)
    query_vector = embedder.embed_query(query)
    hits: list[_FieldHit] = []
    for chunk in chunks:
        for pattern in compiled:
            match = pattern.search(chunk.text)
            if not match:
                continue
            value = _clean_value(match.group(1))
            if not value:
                continue
            line = _line_containing(chunk.text, match.start())
            hits.append(
                _FieldHit(
                    value=value,
                    source_line=line,
                    chunk=chunk,
                    score=hybrid_score(query, query_vector, chunk),
                )
            )
            break
    if not hits:
        return None
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[0]


def _clean_value(value: str) -> str:
    return value.strip(" \t：:，,。；;")


def _line_containing(text: str, index: int) -> str:
    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)
    if line_end < 0:
        line_end = len(text)
    return text[line_start:line_end].strip()


def _lead_claim(chunks: list[ChunkRecord]) -> Claim | None:
    if not chunks:
        return None
    ordered = sorted(chunks, key=lambda chunk: (chunk.ordinal, chunk.id))
    chunk = ordered[0]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", chunk.text) if part.strip()]
    lead = paragraphs[0] if paragraphs else chunk.text
    lead = clip_text(lead, CLAIM_LIMIT)
    if not lead:
        return None
    return Claim(text=lead, citations=[citation_for(chunk, lead, None)])


def extractive_query(
    case_id: str,
    question: str,
    chunks: list[ChunkRecord],
    embedder: Embedder,
    top_k: int,
) -> QueryResponse:
    if not chunks:
        return QueryResponse(
            case_id=case_id,
            question=question,
            mode="extractive",
            answer="此案件尚無可供引用的文件內容。",
            claims=[],
            citations=[],
        )
    retrieved = retrieve(chunks, question, embedder, top_k)
    claims = _claims_from_retrieved(question, retrieved)
    if not claims and retrieved:
        fallback = clip_text(retrieved[0].chunk.text, CLAIM_LIMIT)
        claims = [
            Claim(
                text=fallback,
                citations=[citation_for(retrieved[0].chunk, fallback, retrieved[0].score)],
            )
        ]
    citations = _unique_citations(claims)
    answer = "\n".join(claim.text for claim in claims) if claims else "找不到可引用的相關內容。"
    return QueryResponse(
        case_id=case_id,
        question=question,
        mode="extractive",
        answer=answer,
        claims=claims,
        citations=citations,
    )


def _claims_from_retrieved(question: str, retrieved: list[RetrievedChunk]) -> list[Claim]:
    scored: list[tuple[float, float, str, ChunkRecord]] = []
    for item in retrieved:
        for sentence in split_sentences(item.chunk.text):
            overlap = ngram_overlap(question, sentence)
            if overlap <= 0:
                continue
            scored.append((overlap, item.score, sentence, item.chunk))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    claims: list[Claim] = []
    seen: set[str] = set()
    best_overlap = scored[0][0] if scored else 0.0
    for overlap, score, sentence, chunk in scored:
        if overlap < best_overlap * 0.6:
            continue
        text = clip_text(sentence, CLAIM_LIMIT)
        if not text or text in seen:
            continue
        if text not in chunk.text and sentence not in chunk.text:
            continue
        seen.add(text)
        claims.append(Claim(text=text, citations=[citation_for(chunk, text, score)]))
        if len(claims) >= 3:
            break
    return claims


def extractive_analysis(
    case_id: str,
    chunks: list[ChunkRecord],
    embedder: Embedder,
) -> AnalysisResponse:
    summary: list[Claim] = []
    lead = _lead_claim(chunks)
    if lead is not None:
        summary.append(lead)
    fields: dict[str, FieldValue] = {}
    unknowns: list[UnknownField] = []
    for key, (label, query, patterns) in FIELD_SPECS.items():
        hit = _find_field(chunks, patterns, query, embedder) if chunks else None
        if hit is None:
            fields[key] = FieldValue(value=None, status="unknown", label=label, citations=[])
            unknowns.append(UnknownField(field=key, label=label))
            continue
        claim_text = hit.source_line if hit.value in hit.source_line else hit.value
        citation = citation_for(hit.chunk, claim_text, hit.score)
        fields[key] = FieldValue(
            value=hit.value,
            status="found",
            label=label,
            citations=[citation],
        )
        if claim_text not in {claim.text for claim in summary}:
            summary.append(Claim(text=claim_text, citations=[citation]))
    return AnalysisResponse(
        case_id=case_id,
        mode="extractive",
        summary=summary,
        fields=fields,
        unknowns=unknowns,
    )


def analyze_case(
    case_id: str,
    chunks: list[ChunkRecord],
    embedder: Embedder,
    *,
    case_name: str = "",
) -> AnalysisResponse:
    del case_name
    return extractive_analysis(case_id, chunks, embedder)


def answer_question(
    case_id: str,
    question: str,
    chunks: list[ChunkRecord],
    embedder: Embedder,
    top_k: int,
) -> QueryResponse:
    return extractive_query(case_id, question, chunks, embedder, top_k)


def _unique_citations(claims: list[Claim]) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for claim in claims:
        for citation in claim.citations:
            if citation.chunk_id in seen:
                continue
            seen.add(citation.chunk_id)
            citations.append(citation)
    return citations
