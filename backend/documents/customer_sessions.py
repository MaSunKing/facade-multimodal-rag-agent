"""Request-scoped multi-document intake and lightweight question retrieval.

The store is deliberately process-local: customer files are parsed in memory,
are never added to the company knowledge base, and expire automatically.  It
reuses the format-specific parsers that already produce canonical evidence.
"""

from __future__ import annotations

import re
import math
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from backend.document_parsing.file_ingestion import ingest_uploaded_file


MAX_SESSION_FILES = 4
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_SESSION_BYTES = 50 * 1024 * 1024
SESSION_TTL_SECONDS = 60 * 60


@dataclass
class StoredVisualAsset:
    visual_id: str
    document_id: str
    document_name: str
    kind: str
    source: dict[str, Any]
    media_type: str | None
    metadata: dict[str, Any]
    image_bytes: bytes | None
    searchable_text: str


@dataclass
class CustomerDocument:
    document_id: str
    file_name: str
    size_bytes: int
    source_type: str
    parser: str
    chunks: list[dict[str, Any]]
    source_locations: dict[str, dict[str, Any]]
    visuals: list[StoredVisualAsset]
    visual_coverage: dict[str, Any]


@dataclass
class CustomerDocumentSession:
    session_id: str
    documents: list[CustomerDocument] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


_sessions: dict[str, CustomerDocumentSession] = {}
_lock = threading.RLock()


def _purge_expired(now: float | None = None) -> None:
    current = now if now is not None else time.monotonic()
    expired = [key for key, value in _sessions.items() if current - value.updated_at > SESSION_TTL_SECONDS]
    for key in expired:
        del _sessions[key]


def _public_document(document: CustomerDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "file_name": document.file_name,
        "size_bytes": document.size_bytes,
        "source_type": document.source_type,
        "parser": document.parser,
        "chunk_count": len(document.chunks),
        "visual_count": len(document.visuals),
        "ready_visual_count": sum(1 for visual in document.visuals if visual.image_bytes),
        "visual_coverage": document.visual_coverage,
    }


def _table_row_range(row: list[Any]) -> str | None:
    coordinates = [str(cell.coordinate) for cell in row if getattr(cell, "coordinate", None)]
    if not coordinates:
        return None
    return coordinates[0] if len(coordinates) == 1 else f"{coordinates[0]}:{coordinates[-1]}"


def _build_source_locations(intermediate: Any) -> dict[str, dict[str, Any]]:
    locations: dict[str, dict[str, Any]] = {}
    for block in intermediate.blocks:
        locations[block.block_id] = block.source.model_dump(mode="json")
    for table in intermediate.tables:
        base = {
            "source_type": intermediate.source_type,
            "file_name": intermediate.file_name,
            "sheet_name": table.sheet_name,
            "page_number": table.page_number,
            "section_id": table.section_id or table.table_id,
            "section_title": table.title,
            "source_range": table.range,
            "parser": table.parser or intermediate.parser,
            "extraction_confidence": table.extraction_confidence,
        }
        locations[table.table_id] = base
        for row in table.rows:
            if not row:
                continue
            row_index = min(cell.row_index for cell in row)
            locations[f"{table.table_id}:row:{row_index}"] = {
                **base,
                "row_index": row_index,
                "source_range": _table_row_range(row),
            }
    for visual in intermediate.visual_assets:
        locations[visual.visual_id] = visual.source.model_dump(mode="json")
    return locations


def _visual_context(intermediate: Any, visual: Any) -> str:
    source = visual.source
    nearby: list[str] = []
    for block in intermediate.blocks:
        same_page = source.page_number is not None and block.source.page_number == source.page_number
        same_sheet = source.sheet_name and block.source.sheet_name == source.sheet_name
        if (same_page or same_sheet) and block.text:
            nearby.append(block.text[:800])
    return " ".join(
        [
            visual.visual_id,
            visual.kind,
            str(source.model_dump(mode="json")),
            str(visual.metadata),
            *nearby[:3],
        ]
    )


def add_files(files: list[tuple[str, bytes]], session_id: str | None = None) -> dict[str, Any]:
    if not files:
        raise ValueError("at_least_one_file_required")
    if len(files) > MAX_SESSION_FILES:
        raise ValueError("maximum_four_files_per_session")
    if any(len(content) > MAX_FILE_BYTES for _, content in files):
        raise ValueError("single_file_too_large")
    if sum(len(content) for _, content in files) > MAX_SESSION_BYTES:
        raise ValueError("session_files_too_large")

    parsed: list[CustomerDocument] = []
    for file_name, content in files:
        result = ingest_uploaded_file(file_name, content)
        if (
            os.getenv("CUSTOMER_DOCUMENT_OCR_ENABLED", "1").lower() in {"1", "true", "yes"}
            and result.intermediate.source_type in {"pdf", "word"}
            and any(asset.kind == "document_page" and asset.image_bytes for asset in result.intermediate.visual_assets)
        ):
            # CPU PP-Structure supplies searchable page text. It does not
            # consume the 16 GB Qwen GPU and remains a review-only candidate.
            try:
                from backend.document_parsing.document_ocr import augment_with_document_ocr_candidates

                result = augment_with_document_ocr_candidates(result)
            except Exception:
                # OCR is optional. Original page bytes remain available for
                # the question-time local VLM fallback.
                pass
        intermediate = result.intermediate
        chunks = [chunk.model_dump(mode="json") for chunk in intermediate.model_context_chunks]
        source_locations = _build_source_locations(intermediate)
        candidate_by_visual = {
            candidate.visual_id: candidate
            for candidate in result.vision_document_candidates
            if candidate.status == "candidate_ready"
        }
        visuals = [
            StoredVisualAsset(
                visual_id=visual.visual_id,
                document_id=intermediate.document_id or "",
                document_name=file_name,
                kind=visual.kind,
                source=visual.source.model_dump(mode="json"),
                media_type=visual.media_type,
                metadata=dict(visual.metadata),
                image_bytes=visual.image_bytes,
                searchable_text=" ".join(
                    [
                        _visual_context(intermediate, visual),
                        (
                            json.dumps(candidate_by_visual[visual.visual_id].model_dump(mode="json"), ensure_ascii=False)
                            if visual.visual_id in candidate_by_visual
                            else ""
                        ),
                    ]
                ),
            )
            for visual in intermediate.visual_assets
        ]
        parsed.append(
            CustomerDocument(
                document_id=intermediate.document_id or f"doc_{uuid.uuid4().hex[:16]}",
                file_name=file_name,
                size_bytes=len(content),
                source_type=intermediate.source_type,
                parser=intermediate.parser,
                chunks=chunks,
                source_locations=source_locations,
                visuals=visuals,
                visual_coverage=(
                    {
                        "page_count": result.pdf.page_count,
                        "rendered_page_numbers": list(result.pdf.vision_rendered_page_numbers),
                        "next_page_start": result.pdf.vision_next_page_start,
                        "coverage_complete": result.pdf.vision_next_page_start is None,
                    }
                    if hasattr(result, "pdf")
                    else {"coverage_complete": True}
                ),
            )
        )

    with _lock:
        _purge_expired()
        key = session_id or f"upload_{uuid.uuid4().hex}"
        existing = _sessions.get(key)
        documents = list(existing.documents) if existing else []
        by_id = {document.document_id: document for document in documents}
        for document in parsed:
            by_id[document.document_id] = document
        if len(by_id) > MAX_SESSION_FILES:
            raise ValueError("maximum_four_files_per_session")
        session = CustomerDocumentSession(session_id=key, documents=list(by_id.values()))
        _sessions[key] = session
        return {
            "session_id": key,
            "expires_in_seconds": SESSION_TTL_SECONDS,
            "documents": [_public_document(document) for document in session.documents],
            "privacy": "process_memory_only_not_added_to_company_rag",
        }


def get_session(session_id: str) -> CustomerDocumentSession | None:
    with _lock:
        _purge_expired()
        session = _sessions.get(session_id)
        if session:
            session.updated_at = time.monotonic()
        return session


def session_summary(session_id: str) -> dict[str, Any] | None:
    session = get_session(session_id)
    if not session:
        return None
    return {
        "session_id": session.session_id,
        "expires_in_seconds": SESSION_TTL_SECONDS,
        "documents": [_public_document(document) for document in session.documents],
        "privacy": "process_memory_only_not_added_to_company_rag",
    }


def get_visual_asset(session_id: str, document_id: str, visual_id: str) -> StoredVisualAsset | None:
    """Return one in-memory visual only while its private upload session lives."""

    session = get_session(session_id)
    if not session:
        return None
    for document in session.documents:
        if document.document_id != document_id:
            continue
        for visual in document.visuals:
            if visual.visual_id == visual_id and visual.image_bytes:
                return visual
    return None


def delete_session(session_id: str) -> bool:
    with _lock:
        return _sessions.pop(session_id, None) is not None


def _terms(text: str) -> set[str]:
    latin = re.findall(r"[a-z0-9_./%-]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    grams: list[str] = []
    for value in chinese:
        grams.extend(value[index : index + 2] for index in range(max(1, len(value) - 1)))
    return set(latin + chinese + grams)


def _citation(document: CustomerDocument, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source_ref in list(chunk.get("source_refs") or []):
        location = document.source_locations.get(str(source_ref))
        if not location:
            continue
        item = {
            "document_name": document.file_name,
            "source_page": location.get("page_number"),
            "section_heading": location.get("section_title") or location.get("section_id") or location.get("sheet_name"),
            "source_type": "customer_upload",
            "sheet_name": location.get("sheet_name"),
            "source_range": location.get("source_range") or location.get("cell"),
            "row_index": location.get("row_index"),
            "bounding_box": location.get("bounding_box"),
            "parser": location.get("parser") or document.parser,
            "extraction_confidence": location.get("extraction_confidence"),
        }
        key = (item["source_page"], item["sheet_name"], item["source_range"], item["section_heading"])
        if key not in seen:
            seen.add(key)
            citations.append(item)
    if citations:
        return citations[:8]
    return [
        {
            "document_name": document.file_name,
            "source_page": None,
            "section_heading": str(chunk.get("chunk_id")),
            "source_type": "customer_upload",
            "sheet_name": None,
            "source_range": None,
            "row_index": None,
            "bounding_box": None,
            "parser": document.parser,
            "extraction_confidence": None,
        }
    ]


def _bounded_excerpt(text: str, query_terms: set[str], limit: int) -> str:
    if len(text) <= limit:
        return text
    lines = [line for line in text.splitlines() if line.strip()]
    headers = [line for line in lines if line.startswith(("[DOCUMENT", "[TABLE", "[COLUMNS", "[BLOCK"))]
    matches = [line for line in lines if query_terms & _terms(line)]
    chosen: list[str] = []
    used = 0
    for line in [*headers[:3], *matches, *lines]:
        if line in chosen:
            continue
        addition = len(line) + (1 if chosen else 0)
        if used + addition > limit:
            continue
        chosen.append(line)
        used += addition
    return "\n".join(chosen)


def _select_visuals(
    documents: list[CustomerDocument],
    question: str,
    selected: list[dict[str, Any]],
    *,
    max_visuals: int = 1,
) -> list[dict[str, Any]]:
    query_terms = _terms(question)
    selected_pages: set[tuple[str, int]] = set()
    selected_sheets: set[tuple[str, str]] = set()
    for item in selected:
        pages = {citation.get("source_page") for citation in item.get("citations", []) if citation.get("source_page") is not None}
        sheets = {citation.get("sheet_name") for citation in item.get("citations", []) if citation.get("sheet_name")}
        # Do not convert a multi-page packed chunk into a false exact-page
        # label. OCR/metadata similarity must disambiguate it instead.
        if len(pages) == 1:
            selected_pages.add((item["document_id"], int(next(iter(pages)))))
        if len(sheets) == 1:
            selected_sheets.add((item["document_id"], str(next(iter(sheets)))))
    ranked: list[tuple[float, int, StoredVisualAsset]] = []
    order = 0
    for document in documents:
        for visual in document.visuals:
            if not visual.image_bytes:
                continue
            order += 1
            page = visual.source.get("page_number")
            sheet = visual.source.get("sheet_name")
            score = sum(1.0 + math.log1p(len(term)) for term in query_terms & _terms(visual.searchable_text))
            if (document.document_id, page) in selected_pages:
                score += 12.0
            if (document.document_id, sheet) in selected_sheets:
                score += 10.0
            if document.source_type == "image":
                score += 4.0
            ranked.append((score, order, visual))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    output: list[dict[str, Any]] = []
    for score, _, visual in ranked:
        if len(output) >= max_visuals:
            break
        if score <= 0 and visual.source.get("source_type") != "image":
            continue
        output.append(
            {
                "visual_id": visual.visual_id,
                "document_id": visual.document_id,
                "document_name": visual.document_name,
                "kind": visual.kind,
                "source": visual.source,
                "media_type": visual.media_type,
                "metadata": visual.metadata,
                "searchable_text": visual.searchable_text[:6_000],
                "score": round(score, 3),
                "selection_status": "question_relevant" if score > 0 else "single_image_direct",
                "image_bytes": visual.image_bytes,
            }
        )
    return output


def retrieve(session_id: str, question: str, *, max_chunks: int = 8, max_characters: int = 12_000) -> dict[str, Any]:
    """Select source-linked chunks across all uploaded files.

    This is an intentionally transparent lexical baseline.  The canonical
    evidence remains complete in the session; only the per-question snapshot
    is bounded.
    """

    session = get_session(session_id)
    if not session:
        return {"status": "session_not_found", "evidence": [], "documents": []}
    query_terms = _terms(question)
    candidates: list[tuple[float, int, CustomerDocument, dict[str, Any]]] = []
    all_chunk_terms: list[set[str]] = []
    for document in session.documents:
        all_chunk_terms.extend(_terms(str(chunk.get("text") or "")) for chunk in document.chunks)
    document_frequency = {
        term: sum(1 for chunk_terms in all_chunk_terms if term in chunk_terms)
        for term in query_terms
    }
    chunk_total = max(1, len(all_chunk_terms))
    order = 0
    for document in session.documents:
        for chunk in document.chunks:
            order += 1
            text = str(chunk.get("text") or "")
            chunk_terms = _terms(text)
            overlap = query_terms & chunk_terms
            lowered = text.lower()
            score = sum(
                math.log1p((chunk_total + 1) / (document_frequency.get(term, 0) + 1))
                * (1.0 + lowered.count(term.lower()) / (lowered.count(term.lower()) + 1.5))
                for term in overlap
            )
            score /= 1.0 + math.log1p(max(0, len(text) - 1_000) / 1_000)
            if str(chunk.get("kind")) == "document_index":
                score -= 0.25
            candidates.append((score, order, document, chunk))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: list[dict[str, Any]] = []
    used = 0
    represented: set[str] = set()
    # First retain one best chunk from every file, then globally fill the rest.
    ordered_candidates = []
    for item in candidates:
        if item[2].document_id not in represented:
            ordered_candidates.append(item)
            represented.add(item[2].document_id)
    ordered_candidates.extend(item for item in candidates if item not in ordered_candidates)
    seen_chunks: set[tuple[str, str]] = set()
    for score, _, document, chunk in ordered_candidates:
        key = (document.document_id, str(chunk.get("chunk_id")))
        if key in seen_chunks:
            continue
        text = str(chunk.get("text") or "")
        remaining = max_characters - used
        if remaining <= 0 or len(selected) >= max_chunks:
            break
        excerpt = _bounded_excerpt(text, query_terms, min(remaining, 6_000))
        if not excerpt:
            continue
        evidence_id = f"U{len(selected) + 1}"
        selected.append(
            {
                "evidence_id": evidence_id,
                "document_id": document.document_id,
                "document_name": document.file_name,
                "chunk_id": str(chunk.get("chunk_id")),
                "source_refs": list(chunk.get("source_refs") or []),
                "text": excerpt,
                "score": round(score, 3),
                "citations": _citation(document, chunk),
            }
        )
        seen_chunks.add(key)
        used += len(excerpt)
    selected_visuals = _select_visuals(session.documents, question, selected)
    precise_citation_count = sum(
        1
        for item in selected
        for citation in item["citations"]
        if citation.get("source_page") is not None or citation.get("sheet_name") or citation.get("source_range")
    )
    return {
        "status": "ok",
        "evidence": selected,
        "documents": [_public_document(document) for document in session.documents],
        "selected_visuals": selected_visuals,
        "input_snapshot": {
            "selected_chunk_count": len(selected),
            "selected_characters": used,
            "max_characters": max_characters,
            "selected_visual_ids": [visual["visual_id"] for visual in selected_visuals],
            "available_visual_count": sum(len(document.visuals) for document in session.documents),
            "selected_visual_count": len(selected_visuals),
            "visual_coverage_complete": all(
                bool(document.visual_coverage.get("coverage_complete")) for document in session.documents
            ),
            "visual_coverage": {
                document.document_id: document.visual_coverage for document in session.documents
            },
            "precise_citation_count": precise_citation_count,
            "citation_coverage_complete": precise_citation_count > 0 or not selected,
            "canonical_evidence_preserved": True,
        },
    }


@contextmanager
def temporary_visual_files(visuals: list[dict[str, Any]]):
    paths: list[Path] = []
    suffixes = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/tiff": ".tiff"}
    try:
        for visual in visuals:
            content = visual.get("image_bytes")
            if not isinstance(content, bytes) or not content:
                continue
            suffix = suffixes.get(str(visual.get("media_type")), ".png")
            with NamedTemporaryFile(prefix="customer-evidence-", suffix=suffix, delete=False) as file:
                file.write(content)
                paths.append(Path(file.name))
        yield paths
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
