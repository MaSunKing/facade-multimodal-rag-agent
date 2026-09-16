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
import base64
import binascii
import hashlib
import hmac
import secrets
import threading
import time
import uuid
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from backend.document_parsing.file_ingestion import ingest_uploaded_file
from backend.document_parsing.pdf_ingestion import ingest_scanned_pdf_vision_batch


MAX_SESSION_FILES = 4
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_SESSION_BYTES = 50 * 1024 * 1024
SESSION_TTL_SECONDS = 60 * 60
MAX_CUSTOMER_PDF_VISUAL_PAGES = 40
MAX_CUSTOMER_PDF_VISION_BATCHES = 10
CUSTOMER_PDF_VISUAL_BATCH_PAGES = 4
MAX_RESIDENT_SESSIONS = 32
MAX_RESIDENT_BYTES = 256 * 1024 * 1024
_parse_slot = threading.BoundedSemaphore(1)


class AttachmentCapacityError(ValueError):
    pass

# Customer uploads are intentionally kept complete in ``CustomerDocument``.
# These limits only govern the question-time snapshot sent to the language
# model.  They use a deterministic tokenizer-free estimate so retrieval stays
# CPU-only and does not contend with the local Qwen model for GPU memory.
# The local answer model is Qwen3-VL-8B 4-bit on a 16 GB GPU.  Ten thousand
# estimated attachment tokens can still expand materially after JSON framing
# and the system prompt, exhausting the KV-cache during generation.  Canonical
# evidence remains complete; this limit only governs the per-question snapshot.
DEFAULT_CUSTOMER_TEXT_TOKEN_BUDGET = 5_000
ABSOLUTE_MAX_CUSTOMER_TEXT_TOKEN_BUDGET = 24_000
DEFAULT_CUSTOMER_TEXT_WINDOW_TOKENS = 768
DEFAULT_CUSTOMER_TEXT_WINDOW_OVERLAP_TOKENS = 96
DEFAULT_CUSTOMER_DOCUMENT_MIN_TOKENS = 512
DEFAULT_CUSTOMER_MAX_SELECTED_WINDOWS = 24


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
    # ``owner_id`` is an opaque, process-local identity. Logged-in callers use
    # their stable user id; anonymous callers use a hash of the browser client
    # id. It is never returned in the public session payload.
    owner_id: str | None = None
    documents: list[CustomerDocument] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)
    visual_ticket_secret: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)


_sessions: dict[str, CustomerDocumentSession] = {}
_lock = threading.RLock()
_CURRENT_SESSION_OWNER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "facade_customer_document_owner", default=None
)


@contextmanager
def bind_session_owner(owner_id: str | None):
    """Bind one verified HTTP caller to nested attachment operations.

    The answer pipeline calls several helpers indirectly through LangGraph. A
    context variable keeps those calls owner-aware without putting an owner
    field in the customer-controlled request body.
    """

    token = _CURRENT_SESSION_OWNER.set(owner_id)
    try:
        yield
    finally:
        _CURRENT_SESSION_OWNER.reset(token)


def current_session_owner() -> str | None:
    return _CURRENT_SESSION_OWNER.get()


def _effective_owner(owner_id: str | None) -> str | None:
    return owner_id if owner_id is not None else current_session_owner()


def _owner_matches(session: CustomerDocumentSession, owner_id: str | None) -> bool:
    # Ownerless sessions exist only for local parser/unit-test callers. Every
    # HTTP-created session is owner-bound by ``documents.router``.
    return session.owner_id is None or session.owner_id == _effective_owner(owner_id)


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
    from backend.documents.visual_layout import compact_visual_metadata
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
            str(compact_visual_metadata(visual.metadata)),
            *nearby[:3],
        ]
    )


def _with_optional_document_ocr(result: Any) -> Any:
    if not (
        os.getenv("CUSTOMER_DOCUMENT_OCR_ENABLED", "1").lower() in {"1", "true", "yes"}
        and result.intermediate.source_type in {"pdf", "word"}
        and any(asset.kind == "document_page" and asset.image_bytes for asset in result.intermediate.visual_assets)
    ):
        return result
    # CPU PP-Structure supplies searchable page text. It does not consume the
    # local Qwen GPU and remains a review-only candidate.
    try:
        from backend.document_parsing.document_ocr import augment_with_document_ocr_candidates

        return augment_with_document_ocr_candidates(result)
    except Exception:
        # OCR is optional. Original page bytes remain available for the
        # question-time local VLM fallback.
        return result


def add_files(*args, **kwargs):
    # Serialise CPU/OCR/Office parsing. Fail explicitly instead of accumulating
    # unbounded waiting uploads or evicting another active user's documents.
    if not _parse_slot.acquire(blocking=False):
        raise AttachmentCapacityError("附件正在解析，请稍后重试。")
    try:
        return _add_files_serial(*args, **kwargs)
    finally:
        _parse_slot.release()


def estimated_session_bytes(session: CustomerDocumentSession) -> int:
    # Conservative retained-data estimate, NOT a measurement of process RSS.
    # Includes Python Unicode/object overhead and original upload allowance.
    return sum(doc.size_bytes + 4 * len(json.dumps(doc.chunks, ensure_ascii=False))
               + 4 * len(json.dumps(doc.source_locations, ensure_ascii=False))
               + sum(len(v.image_bytes or b"") + 4 * len(v.searchable_text) + 1024 for v in doc.visuals)
               for doc in session.documents)


def _add_files_serial(
    files: list[tuple[str, bytes]],
    session_id: str | None = None,
    *,
    owner_id: str | None = None,
) -> dict[str, Any]:
    if not files:
        raise ValueError("at_least_one_file_required")
    if len(files) > MAX_SESSION_FILES:
        raise ValueError("maximum_four_files_per_session")
    if any(len(content) > MAX_FILE_BYTES for _, content in files):
        raise ValueError("single_file_too_large")
    if sum(len(content) for _, content in files) > MAX_SESSION_BYTES:
        raise ValueError("session_files_too_large")
    with _lock:
        _purge_expired()
        if session_id not in _sessions and len(_sessions) >= MAX_RESIDENT_SESSIONS:
            raise AttachmentCapacityError("附件会话容量已满，请清除不用的附件或稍后重试。")

    # Reject cumulative overflow before running expensive PDF/OCR/Office
    # parsers.  A second exact check below handles content-addressed
    # replacement safely under the session lock.
    if session_id:
        with _lock:
            _purge_expired()
            existing = _sessions.get(session_id)
            if existing is None:
                raise ValueError("attachment_session_not_found")
            if not _owner_matches(existing, owner_id):
                raise PermissionError("attachment_session_owner_mismatch")
            existing_size = sum(document.size_bytes for document in existing.documents) if existing else 0
        if existing_size + sum(len(content) for _, content in files) > MAX_SESSION_BYTES:
            raise ValueError("session_files_too_large")

    parsed: list[CustomerDocument] = []
    for file_name, content in files:
        result = _with_optional_document_ocr(ingest_uploaded_file(file_name, content))
        batches = [result]
        coverage_stop_reason: str | None = None
        if file_name.lower().endswith(".pdf") and hasattr(result, "pdf"):
            next_page_start = result.pdf.vision_next_page_start
            rendered_pages = set(result.pdf.vision_rendered_page_numbers)
            while next_page_start is not None:
                if len(rendered_pages) + CUSTOMER_PDF_VISUAL_BATCH_PAGES > MAX_CUSTOMER_PDF_VISUAL_PAGES:
                    coverage_stop_reason = "visual_page_safety_limit"
                    break
                if len(batches) >= MAX_CUSTOMER_PDF_VISION_BATCHES:
                    coverage_stop_reason = "visual_batch_safety_limit"
                    break
                try:
                    batch = _with_optional_document_ocr(
                        ingest_scanned_pdf_vision_batch(file_name, content, page_start=next_page_start)
                    )
                except Exception as exc:
                    coverage_stop_reason = f"visual_batch_failed:{type(exc).__name__}"
                    break
                new_pages = set(batch.pdf.vision_rendered_page_numbers) - rendered_pages
                batches.append(batch)
                rendered_pages.update(batch.pdf.vision_rendered_page_numbers)
                following_page_start = batch.pdf.vision_next_page_start
                if not new_pages or following_page_start == next_page_start:
                    coverage_stop_reason = "visual_batch_made_no_progress"
                    next_page_start = following_page_start
                    break
                next_page_start = following_page_start

        intermediate = result.intermediate
        chunks_by_id: dict[str, dict[str, Any]] = {}
        source_locations: dict[str, dict[str, Any]] = {}
        for batch in batches:
            for chunk in batch.intermediate.model_context_chunks:
                chunks_by_id.setdefault(chunk.chunk_id, chunk.model_dump(mode="json"))
            source_locations.update(_build_source_locations(batch.intermediate))
        chunks = list(chunks_by_id.values())
        candidate_by_visual = {
            candidate.visual_id: candidate
            for batch in batches
            for candidate in batch.vision_document_candidates
            if candidate.status == "candidate_ready"
        }
        visual_by_id = {
            visual.visual_id: (batch.intermediate, visual)
            for batch in batches
            for visual in batch.intermediate.visual_assets
        }
        visuals = [
            StoredVisualAsset(
                visual_id=visual.visual_id,
                document_id=intermediate.document_id or "",
                document_name=file_name,
                kind=visual.kind,
                source=visual.source.model_dump(mode="json"),
                media_type=visual.media_type,
                metadata={**dict(visual.metadata),
                          "text_visual_binding": "same_container_candidate_not_verified",
                          "requires_layout_verification": True},
                image_bytes=visual.image_bytes,
                searchable_text=" ".join(
                    [
                        _visual_context(batch_intermediate, visual),
                        (
                            json.dumps(candidate_by_visual[visual.visual_id].model_dump(mode="json"), ensure_ascii=False)
                            if visual.visual_id in candidate_by_visual
                            else ""
                        ),
                    ]
                ),
            )
            for batch_intermediate, visual in visual_by_id.values()
        ]
        if hasattr(result, "pdf"):
            rendered_page_numbers = sorted(
                {
                    page_number
                    for batch in batches
                    for page_number in batch.pdf.vision_rendered_page_numbers
                }
            )
            final_next_page_start = batches[-1].pdf.vision_next_page_start
            coverage_complete = final_next_page_start is None and coverage_stop_reason is None
            visual_coverage = {
                "page_count": result.pdf.page_count,
                "rendered_page_numbers": rendered_page_numbers,
                "next_page_start": final_next_page_start,
                "coverage_complete": coverage_complete,
                "batch_count": len(batches),
                "max_visual_pages": MAX_CUSTOMER_PDF_VISUAL_PAGES,
                "stop_reason": coverage_stop_reason,
            }
        else:
            visual_coverage = {"coverage_complete": True}
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
                visual_coverage=visual_coverage,
            )
        )

    with _lock:
        _purge_expired()
        key = session_id or f"upload_{uuid.uuid4().hex}"
        existing = _sessions.get(key)
        if existing is not None and not _owner_matches(existing, owner_id):
            raise PermissionError("attachment_session_owner_mismatch")
        documents = list(existing.documents) if existing else []
        by_id = {document.document_id: document for document in documents}
        for document in parsed:
            by_id[document.document_id] = document
        if len(by_id) > MAX_SESSION_FILES:
            raise ValueError("maximum_four_files_per_session")
        if sum(document.size_bytes for document in by_id.values()) > MAX_SESSION_BYTES:
            raise ValueError("session_files_too_large")
        session = CustomerDocumentSession(
            session_id=key,
            owner_id=existing.owner_id if existing is not None else _effective_owner(owner_id),
            documents=list(by_id.values()),
            visual_ticket_secret=(
                existing.visual_ticket_secret if existing is not None else secrets.token_bytes(32)
            ),
        )
        projected = sum(estimated_session_bytes(value) for name, value in _sessions.items() if name != key)
        if projected + estimated_session_bytes(session) > MAX_RESIDENT_BYTES:
            raise AttachmentCapacityError("附件解析结果超出本地总容量，请减少文件或清除不用的附件。")
        _sessions[key] = session
        return {
            "session_id": key,
            "expires_in_seconds": SESSION_TTL_SECONDS,
            "documents": [_public_document(document) for document in session.documents],
            "privacy": "process_memory_only_not_added_to_company_rag",
        }


def get_session(session_id: str, *, owner_id: str | None = None) -> CustomerDocumentSession | None:
    with _lock:
        _purge_expired()
        session = _sessions.get(session_id)
        if session and _owner_matches(session, owner_id):
            session.updated_at = time.monotonic()
            return session
        return None


def session_summary(session_id: str, *, owner_id: str | None = None) -> dict[str, Any] | None:
    session = get_session(session_id, owner_id=owner_id)
    if not session:
        return None
    return {
        "session_id": session.session_id,
        "expires_in_seconds": SESSION_TTL_SECONDS,
        "documents": [_public_document(document) for document in session.documents],
        "privacy": "process_memory_only_not_added_to_company_rag",
    }


def session_access_status(session_id: str, *, owner_id: str | None = None) -> str:
    """Return ``owned``, ``foreign`` or ``missing`` without exposing it over HTTP."""

    with _lock:
        _purge_expired()
        session = _sessions.get(session_id)
        if session is None:
            return "missing"
        return "owned" if _owner_matches(session, owner_id) else "foreign"


def get_visual_asset(
    session_id: str,
    document_id: str,
    visual_id: str,
    *,
    owner_id: str | None = None,
    access_ticket: str | None = None,
) -> StoredVisualAsset | None:
    """Return one in-memory visual only while its private upload session lives."""

    session = (
        _session_for_visual_ticket(session_id, document_id, visual_id, access_ticket)
        if access_ticket
        else get_session(session_id, owner_id=owner_id)
    )
    if not session:
        return None
    for document in session.documents:
        if document.document_id != document_id:
            continue
        for visual in document.visuals:
            if visual.visual_id == visual_id and visual.image_bytes:
                return visual
    return None


def delete_session(session_id: str, *, owner_id: str | None = None) -> bool:
    with _lock:
        _purge_expired()
        session = _sessions.get(session_id)
        if session is None or not _owner_matches(session, owner_id):
            return False
        del _sessions[session_id]
        return True


def _visual_ticket_payload(session_id: str, document_id: str, visual_id: str, expires_at: int) -> bytes:
    return json.dumps(
        {
            "session_id": session_id,
            "document_id": document_id,
            "visual_id": visual_id,
            "expires_at": expires_at,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def issue_customer_visual_ticket(
    session_id: str,
    document_id: str,
    visual_id: str,
    *,
    lifetime_seconds: int = 300,
) -> str | None:
    """Issue a short-lived capability for an authorised browser ``img`` tag.

    Image elements cannot attach the bearer/client-id headers used by fetch.
    The ticket is scoped to one visual and signed with an in-memory per-session
    secret, so it expires with the upload session and cannot be reused for a
    different attachment.
    """

    session = get_session(session_id)
    if session is None:
        return None
    exists = any(
        document.document_id == document_id
        and any(visual.visual_id == visual_id and visual.image_bytes for visual in document.visuals)
        for document in session.documents
    )
    if not exists:
        return None
    expires_at = int(time.time()) + max(30, min(int(lifetime_seconds), 900))
    payload = _visual_ticket_payload(session_id, document_id, visual_id, expires_at)
    signature = hmac.new(session.visual_ticket_secret, payload, hashlib.sha256).digest()
    encoded_payload = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{encoded_payload}.{encoded_signature}"


def _session_for_visual_ticket(
    session_id: str,
    document_id: str,
    visual_id: str,
    ticket: str,
) -> CustomerDocumentSession | None:
    try:
        encoded_payload, encoded_signature = ticket.split(".", 1)
        payload = base64.urlsafe_b64decode(
            (encoded_payload + "=" * (-len(encoded_payload) % 4)).encode("ascii")
        )
        signature = base64.urlsafe_b64decode(
            (encoded_signature + "=" * (-len(encoded_signature) % 4)).encode("ascii")
        )
        parsed = json.loads(payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            return None
        if (
            parsed.get("session_id") != session_id
            or parsed.get("document_id") != document_id
            or parsed.get("visual_id") != visual_id
            or int(parsed.get("expires_at") or 0) < int(time.time())
        ):
            return None
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        return None
    with _lock:
        _purge_expired()
        session = _sessions.get(session_id)
        if session is None:
            return None
        expected = hmac.new(session.visual_ticket_secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        session.updated_at = time.monotonic()
        return session


def _terms(text: str) -> set[str]:
    latin = re.findall(r"[a-z0-9_./%-]+", text.lower())
    stopwords = {'the','a','an','of','in','on','at','to','for','from','and','or','is','are','was','were','be','by','with','this','that','it','its','what','which','how'}
    latin = [term for term in latin if term not in stopwords]
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    grams: list[str] = []
    for value in chinese:
        grams.extend(value[index : index + 2] for index in range(max(1, len(value) - 1)))
    return set(latin + chinese + grams)


def _semantic_rerank_customer_windows(
    question: str,
    candidates: list[dict[str, Any]],
    *,
    global_document_question: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Model-rerank the lexical high-recall pool without touching raw files."""

    enabled = str(os.getenv("CUSTOMER_ATTACHMENT_SEMANTIC_RERANK", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    pool_size = min(24, len(candidates))
    audit: dict[str, Any] = {
        "enabled": enabled,
        "model": "Qwen3-Reranker-0.6B",
        "candidate_pool_size": pool_size,
        "status": "not_needed" if pool_size < 12 else "pending",
    }
    if global_document_question:
        # A generic whole-document overview has no topical target for a
        # semantic reranker.  Ranking it against words such as "introduce" or
        # "analyse" produced arbitrary high-scoring footnotes.  Parser-aware
        # structural sampling below is the correct signal for this task.
        audit.update(
            {
                "status": "structure_aware_global_sampling",
                "reason": "generic_overview_has_no_semantic_target",
            }
        )
        return candidates, audit
    if not enabled or pool_size < 12:
        if not enabled:
            audit["status"] = "disabled"
        return candidates, audit

    pool = candidates[:pool_size]
    try:
        from backend.sales.dense_retrieval import (
            RETRIEVAL_INFERENCE_LOCK,
            get_reranker_model,
            retrieval_runtime_status,
        )

        runtime = retrieval_runtime_status()
        if str(runtime.get("effective_device")) != "cuda":
            audit.update(
                {
                    "status": "deferred_to_generation_model",
                    "reason": "generation_gpu_reserved",
                    "effective_device": str(runtime.get("effective_device")),
                    "final_selection_model": "Qwen3-VL-8B-Instruct-4bit",
                }
            )
            return candidates, audit

        instruction = (
            "Given a question about one or more private uploaded documents, judge whether the document window "
            "directly supports the answer. Prefer exact fields, values, headings and table rows. For an overall "
            "summary, prefer complete structure and table-overview evidence. Do not infer facts absent from the window."
        )
        with RETRIEVAL_INFERENCE_LOCK:
            reranker = get_reranker_model()
            semantic_scores = reranker.score(
                question,
                [str(candidate["text"]) for candidate in pool],
                batch_size=2,
                max_length=1024,
                instruction=instruction,
            )
        lexical_max = max((float(candidate["score"]) for candidate in pool), default=1.0) or 1.0
        for candidate, semantic_score in zip(pool, semantic_scores):
            candidate["lexical_score"] = float(candidate["score"])
            candidate["semantic_reranker_score"] = float(semantic_score)
            candidate["score"] = 0.85 * float(semantic_score) + 0.15 * (
                float(candidate["lexical_score"]) / lexical_max
            )
        pool.sort(
            key=lambda item: (
                -(
                    1
                    if global_document_question
                    and item["is_document_index"]
                    and int(item.get("document_index_ordinal") or 0)
                    <= 1
                    else 0
                ),
                -float(item["score"]),
                int(item["order"]),
            )
        )
        audit.update(
            {
                "status": "applied",
                "effective_device": str(reranker.device),
                "reranked_count": len(pool),
            }
        )
        return [*pool, *candidates[pool_size:]], audit
    except Exception as exc:
        audit.update({"status": "fallback_lexical", "reason": type(exc).__name__})
        return candidates, audit


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


def _environment_integer(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _estimate_text_tokens(text: str) -> int:
    """Return a conservative, deterministic token estimate without a model.

    Han characters are usually close to one token each for the target model.
    Latin words/numbers are estimated at roughly four characters per token,
    while punctuation receives a small allowance.  This is deliberately an
    estimate: the exact processor count is recorded as unknown until the model
    input is built, but it is much safer across Chinese and English than a raw
    character limit.
    """

    if not text:
        return 0
    han_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    latin_tokens = sum(
        max(1, math.ceil(len(token) / 4))
        for token in re.findall(r"[A-Za-z0-9_]+(?:[./%-][A-Za-z0-9_]+)*", text)
    )
    punctuation_count = len(
        re.findall(r"[^\sA-Za-z0-9_\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text)
    )
    estimate = han_count + latin_tokens + math.ceil(punctuation_count / 2)
    return max(1, estimate)


def _maximum_end_for_token_budget(text: str, start: int, token_budget: int) -> int:
    """Find the longest character interval whose estimated size fits."""

    if start >= len(text) or token_budget <= 0:
        return start
    low, high = start + 1, len(text)
    best = start
    while low <= high:
        middle = (low + high) // 2
        if _estimate_text_tokens(text[start:middle]) <= token_budget:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return max(start + 1, best)


def _prefer_structural_boundary(text: str, start: int, end: int) -> int:
    """Prefer a paragraph/sentence/word boundary near a computed window end."""

    if end >= len(text) or end - start < 32:
        return end
    search_start = start + max(1, int((end - start) * 0.65))
    segment = text[search_start:end]
    boundaries = [
        match.end()
        for match in re.finditer(r"\r?\n+|[。！？!?；;](?:\s+|$)|\s+", segment)
    ]
    return search_start + boundaries[-1] if boundaries else end


def _overlap_start(text: str, start: int, end: int, overlap_tokens: int) -> int:
    if overlap_tokens <= 0 or end <= start:
        return end
    low, high = start, end
    best = end
    while low <= high:
        middle = (low + high) // 2
        if _estimate_text_tokens(text[middle:end]) <= overlap_tokens:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    # Moving forward to the next boundary avoids starting halfway through a
    # word/line while keeping the overlap inside its budget.
    tail = text[best:end]
    boundary = re.search(r"\r?\n+|[。！？!?；;]\s*|\s+", tail)
    if boundary and boundary.end() < len(tail):
        best += boundary.end()
    return min(end, max(start + 1, best))


def _compact_row_formulas(row: str) -> str:
    """Losslessly share repeated self-cell formula templates in a prompt row."""
    pattern = re.compile(r"(\b[A-Z]+\d+)(\[[^\]]*\])?='([^']*)' formula='([^']*)'")
    matches = list(pattern.finditer(row))
    templates: dict[str, list[str]] = {}
    for match in matches:
        cell, formula = match.group(1), match.group(4)
        template = re.sub(r'(?<![A-Za-z0-9_])'+re.escape(cell)+r'(?![A-Za-z0-9_])', '{cell}', formula)
        templates.setdefault(template, []).append(cell)
    shared = {template: f'F{index+1}' for index, (template, cells) in enumerate(templates.items()) if len(cells) > 1}
    if not shared:
        return row
    def replace(match):
        template = re.sub(r'(?<![A-Za-z0-9_])'+re.escape(match.group(1))+r'(?![A-Za-z0-9_])', '{cell}', match.group(4))
        if template not in shared:
            return match.group()
        return f"{match.group(1)}{match.group(2) or ''}='{match.group(3)}' formula_template={shared[template]}"
    return pattern.sub(replace, row) + '\n[FORMULA_TEMPLATES substitute {cell} with each cell address] ' + json.dumps({key:template for template,key in shared.items()},ensure_ascii=False)


def _window_chunk(text: str, chunk_id: str, *, window_tokens: int, overlap_tokens: int, _table_rows: bool = True, header_context: str = "") -> list[dict[str, Any]]:
    """Split a canonical chunk into auditable, overlapping retrieval views."""

    if not text:
        return []
    rows = list(re.finditer(r"(?m)^\[ROW[^\n]*", text))
    if _table_rows and text.startswith("[TABLE ") and ' sheet=' in text.split('\n',1)[0] and len(rows) > 1:
        prefix = text[:rows[0].start()] + header_context
        # Read-only context; keep headers with values without indexing their
        # repeated words as extra relevance. Canonical source is untouched.
        output = []
        for row in rows:
            compact_row = _compact_row_formulas(row.group())
            if _estimate_text_tokens(compact_row) > max(64, window_tokens-_estimate_text_tokens(prefix)):
                compact_row = row.group()
            pieces = _window_chunk(compact_row, chunk_id, window_tokens=max(64, window_tokens-_estimate_text_tokens(prefix)), overlap_tokens=0, _table_rows=False)
            for piece in pieces:
                value = prefix + piece['text']
                output.append({**piece, 'window_id': f'{chunk_id}:row:{len(output)+1}',
                    'text': value, 'ranking_text': piece['text'],
                    'start_character': row.start() if compact_row != row.group() else row.start()+piece['start_character'],
                    'end_character': row.end() if compact_row != row.group() else row.start()+piece['end_character'],
                    'estimated_tokens': _estimate_text_tokens(value), 'windowed': True})
        return output
    if _estimate_text_tokens(text) <= window_tokens:
        return [
            {
                "window_id": f"{chunk_id}:window:1",
                "original_chunk_id": chunk_id,
                "start_character": 0,
                "end_character": len(text),
                "text": text,
                "estimated_tokens": _estimate_text_tokens(text),
                "windowed": False,
            }
        ]

    windows: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        end = _maximum_end_for_token_budget(text, start, window_tokens)
        end = _prefer_structural_boundary(text, start, end)
        if end <= start:
            end = min(len(text), start + 1)
        actual_start = start
        actual_end = end
        while actual_start < actual_end and text[actual_start].isspace():
            actual_start += 1
        while actual_end > actual_start and text[actual_end - 1].isspace():
            actual_end -= 1
        if actual_end > actual_start:
            window_text = text[actual_start:actual_end]
            windows.append(
                {
                    "window_id": f"{chunk_id}:window:{len(windows) + 1}",
                    "original_chunk_id": chunk_id,
                    "start_character": actual_start,
                    "end_character": actual_end,
                    "text": window_text,
                    "estimated_tokens": _estimate_text_tokens(window_text),
                    "windowed": True,
                }
            )
        if end >= len(text):
            break
        following_start = _overlap_start(text, start, end, overlap_tokens)
        start = following_start if following_start > start else end
    return windows


def _question_centered_text(text: str, query_terms: set[str], token_budget: int) -> str:
    """Choose the best local view of auxiliary text without prefix clipping."""

    windows = _window_chunk(
        text,
        "auxiliary",
        window_tokens=max(64, token_budget),
        overlap_tokens=min(64, max(0, token_budget // 8)),
    )
    if not windows:
        return ""
    windows.sort(
        key=lambda window: (
            -len(query_terms & _terms(str(window["text"]))),
            int(window["start_character"]),
        )
    )
    return str(windows[0]["text"])


def _select_visuals(
    documents: list[CustomerDocument],
    question: str,
    selected: list[dict[str, Any]],
    *,
    max_visuals: int = 4,
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
    # Prefer the best relevant asset from each file before spending the
    # remaining budget on additional pages from the same file.  This keeps a
    # four-file request inspectable without allowing duplicate renderings of
    # one page/sheet to consume the entire visual budget.
    ordered: list[tuple[float, int, StoredVisualAsset]] = []
    location_matched = [
        item
        for item in ranked
        if (item[2].document_id, item[2].source.get("page_number")) in selected_pages
        or (item[2].document_id, item[2].source.get("sheet_name")) in selected_sheets
    ]
    ordered.extend(location_matched)
    represented_documents: set[str] = {item[2].document_id for item in location_matched}
    for item in ranked:
        visual = item[2]
        if item not in ordered and visual.document_id not in represented_documents:
            ordered.append(item)
            represented_documents.add(visual.document_id)
    ordered.extend(item for item in ranked if item not in ordered)

    output: list[dict[str, Any]] = []
    seen_locations: set[tuple[Any, ...]] = set()
    for score, _, visual in ordered:
        if len(output) >= max_visuals:
            break
        if score <= 0 and visual.source.get("source_type") != "image":
            continue
        page = visual.source.get("page_number")
        sheet = visual.source.get("sheet_name")
        if page is not None:
            location_key = (visual.document_id, "page", page)
        elif sheet:
            location_key = (visual.document_id, "sheet", sheet)
        else:
            location_key = (visual.document_id, "visual", visual.visual_id)
        if location_key in seen_locations:
            continue
        seen_locations.add(location_key)
        output.append(
            {
                "visual_id": visual.visual_id,
                "document_id": visual.document_id,
                "document_name": visual.document_name,
                "kind": visual.kind,
                "source": visual.source,
                "media_type": visual.media_type,
                "metadata": visual.metadata,
                "searchable_text": _question_centered_text(visual.searchable_text, query_terms, 1_200),
                "score": round(score, 3),
                "selection_status": "question_relevant" if score > 0 else "single_image_direct",
                "image_bytes": visual.image_bytes,
            }
        )
    return output


def retrieve(
    session_id: str,
    question: str,
    *,
    document_scope: str = "unknown",
    max_chunks: int | None = None,
    max_characters: int | None = None,
    max_text_tokens: int | None = None,
) -> dict[str, Any]:
    """Select source-linked windows across all uploaded files.

    The canonical evidence in the session is never shortened.  Long chunks
    are converted to overlapping, question-ranked views for this one request.
    ``max_characters`` remains a backwards-compatible caller override only;
    the formal budget and audit are token estimates.
    """

    session = get_session(session_id)
    if not session:
        return {"status": "session_not_found", "evidence": [], "documents": []}

    safety_token_limit = _environment_integer(
        "CUSTOMER_ATTACHMENT_TEXT_TOKEN_SAFETY_MAX",
        16_000,
        minimum=1_024,
        maximum=ABSOLUTE_MAX_CUSTOMER_TEXT_TOKEN_BUDGET,
    )
    configured_token_budget = _environment_integer(
        "CUSTOMER_ATTACHMENT_TEXT_TOKEN_BUDGET",
        DEFAULT_CUSTOMER_TEXT_TOKEN_BUDGET,
        minimum=512,
        maximum=safety_token_limit,
    )
    requested_token_budget = configured_token_budget if max_text_tokens is None else max_text_tokens
    legacy_character_budget_applied = max_characters is not None and max_text_tokens is None
    if legacy_character_budget_applied:
        # Older internal callers can still request a small smoke-test budget.
        # Treat it as an approximate mixed-language conversion, never as a
        # prefix slice of every chunk.
        requested_token_budget = max(64, math.ceil(max(0, int(max_characters or 0)) / 2))
    text_token_budget = min(safety_token_limit, max(64, int(requested_token_budget)))
    selected_window_limit = min(
        64,
        max(
            1,
            int(max_chunks)
            if max_chunks is not None
            else _environment_integer(
                "CUSTOMER_ATTACHMENT_MAX_SELECTED_WINDOWS",
                DEFAULT_CUSTOMER_MAX_SELECTED_WINDOWS,
                minimum=1,
                maximum=64,
            ),
        ),
    )
    window_token_budget = _environment_integer(
        "CUSTOMER_ATTACHMENT_WINDOW_TOKENS",
        DEFAULT_CUSTOMER_TEXT_WINDOW_TOKENS,
        minimum=128,
        maximum=min(2_048, text_token_budget),
    )
    # A single window must not be larger than one fair-share slot when all
    # four uploads are relevant; otherwise a small smoke budget could make the
    # last relevant file mathematically impossible to include.
    window_token_budget = min(
        window_token_budget,
        max(64, text_token_budget // max(1, len(session.documents))),
    )
    overlap_token_budget = _environment_integer(
        "CUSTOMER_ATTACHMENT_WINDOW_OVERLAP_TOKENS",
        DEFAULT_CUSTOMER_TEXT_WINDOW_OVERLAP_TOKENS,
        minimum=0,
        maximum=max(0, window_token_budget // 3),
    )
    configured_document_minimum = _environment_integer(
        "CUSTOMER_ATTACHMENT_DOCUMENT_MIN_TOKENS",
        DEFAULT_CUSTOMER_DOCUMENT_MIN_TOKENS,
        minimum=64,
        maximum=min(2_048, text_token_budget),
    )

    query_terms = _terms(question)
    # Task scope is supplied by the semantic Planner.  Retrieval must not
    # independently reinterpret the question through a keyword router.
    normalized_document_scope = (
        document_scope
        if document_scope in {"local_lookup", "whole_document", "cross_document", "unknown"}
        else "unknown"
    )
    # Whole-document summaries and cross-document comparisons both need a
    # structural view plus representative content from every attachment.
    # Treating cross-document analysis as a narrow lexical lookup caused broad
    # prompts such as "compare these files and give recommendations" to fall
    # back to document indexes because those words do not occur in data rows.
    global_document_question = normalized_document_scope in {
        "whole_document",
        "cross_document",
    }
    raw_estimated_tokens = 0
    raw_chunk_count = 0
    windowed_chunk_count = 0
    window_records: list[dict[str, Any]] = []
    original_chunks: dict[tuple[str, str], str] = {}
    per_document_raw_tokens: dict[str, int] = {document.document_id: 0 for document in session.documents}
    order = 0
    for document in session.documents:
        table_headers: dict[str, str] = {}
        for source_chunk in document.chunks:
            source_text = str(source_chunk.get('text') or '')
            table = re.match(r'\[TABLE id=([^\n]*?) sheet=', source_text)
            if not table or table.group(1) in table_headers:
                continue
            headers = []
            for row in re.findall(r'(?m)^\[ROW[^\n]*', source_text)[:3]:
                values = re.findall(r"='([^']*)'", row)
                if not values or any(re.fullmatch(r'[\d.,/ -]+', value) for value in values):
                    break
                if _estimate_text_tokens(''.join(headers)+row) > 180:
                    break
                headers.append(row)
            table_headers[table.group(1)] = '\n'.join(headers) + ('\n' if headers else '')
        for chunk in document.chunks:
            chunk_id = str(chunk.get("chunk_id"))
            text = str(chunk.get("text") or "")
            if not text:
                continue
            raw_chunk_count += 1
            chunk_estimated_tokens = _estimate_text_tokens(text)
            raw_estimated_tokens += chunk_estimated_tokens
            per_document_raw_tokens[document.document_id] += chunk_estimated_tokens
            original_chunks[(document.document_id, chunk_id)] = text
            # PDF prose benefits from shorter coherent windows. This does
            # not alter the canonical document or the total prompt budget.
            prose_pdf = document.file_name.lower().endswith('.pdf') and not text.startswith('[TABLE')
            windows = _window_chunk(
                text,
                chunk_id,
                window_tokens=min(window_token_budget,384) if prose_pdf else window_token_budget,
                overlap_tokens=min(overlap_token_budget,64) if prose_pdf else overlap_token_budget,
                header_context=table_headers.get(match.group(1), '') if (match := re.match(r'\[TABLE id=([^\n]*?) sheet=', text)) else '',
            )
            if len(windows) > 1:
                windowed_chunk_count += 1
            for window_index, window in enumerate(windows):
                order += 1
                window_records.append(
                    {
                        **window,
                        "is_last_window": window_index == len(windows) - 1,
                        "order": order,
                        "document": document,
                        "chunk": chunk,
                        "terms": _terms(str(window.get("ranking_text", window["text"]))),
                        "original_chunk_estimated_tokens": chunk_estimated_tokens,
                    }
                )

    document_frequency = {
        term: sum(1 for window in window_records if term in window["terms"])
        for term in query_terms
    }
    window_total = max(1, len(window_records))
    positive_candidates: list[dict[str, Any]] = []
    index_window_ordinal_by_document: dict[str, int] = {}
    reserved_index_windows_per_document = 1
    for window in window_records:
        overlap = query_terms & window["terms"]
        lowered = str(window["text"]).lower()
        score = sum(
            math.log1p((window_total + 1) / (document_frequency.get(term, 0) + 1))
            * (1.0 + lowered.count(term.lower()) / (lowered.count(term.lower()) + 1.5))
            for term in overlap
        )
        sheet_match = re.search(r"\[TABLE[^\n]*? sheet=(.*?) state=", str(window['text']))
        if sheet_match and re.search(r'(?<![\w])'+re.escape(sheet_match.group(1))+r'(?![\w])', question, re.IGNORECASE):
            score += 4.0
        # Exact dates/periods/identifiers are task constraints, not domain rules.
        for identifier in re.findall(r'\b\d{4}[/\-]\d{2,4}\b', question):
            if identifier in str(window.get('ranking_text', window['text'])):
                score += 8.0
        is_document_index = str(window["chunk"].get("kind")) == "document_index"
        if is_document_index:
            document_id = window["document"].document_id
            ordinal = index_window_ordinal_by_document.get(document_id, 0) + 1
            index_window_ordinal_by_document[document_id] = ordinal
            window["document_index_ordinal"] = ordinal
            # Whole-file questions need the first one or two structural
            # windows, but must still leave room for representative content.
            # Treating every index continuation as higher priority previously
            # consumed the whole prompt with sheet names and table metadata.
            score += (
                8.0
                if global_document_question and ordinal <= reserved_index_windows_per_document
                else -1000.0
                if global_document_question
                else -0.25
            )
        elif global_document_question:
            # A request to introduce or analyse a complete attachment has no
            # domain keywords to overlap with its contents.  Admit content
            # windows explicitly, preferring the first coherent window of
            # each parser-generated chunk over arbitrary prefix clipping.
            window_text = str(window.get("text") or "")
            first_window = int(window.get("start_character") or 0) == 0
            last_window = bool(window.get("is_last_window"))
            parser_chunk_kind = str(window.get("chunk", {}).get("kind") or "")
            if "kind=derived_text" in window_text or "LONG_TEXT_IN_DERIVED_CHUNKS" in window_text:
                overview_weight = 0.2
                prioritize_tail = False
            elif (
                parser_chunk_kind == "table"
                or "[TABLE " in window_text
                or "[COLUMNS]" in window_text
                or "[ROW " in window_text
            ):
                overview_weight = 2.5
                prioritize_tail = last_window
            elif re.search(r"kind=(?:heading|title|paragraph|text)", window_text):
                overview_weight = 2.0
                prioritize_tail = False
            elif "[VISUAL " in window_text:
                overview_weight = 0.5
                prioritize_tail = False
            else:
                overview_weight = 1.2
                prioritize_tail = False
            score += (
                overview_weight
                if first_window or prioritize_tail
                else min(0.15, overview_weight / 10)
            )
        # Multi-column prose is sometimes duplicated as Camelot stream
        # tables. Prefer native prose to those long-sentence pseudo cells;
        # retain the candidates and source for actual table-only evidence.
        window_text = str(window['text'])
        pseudo_cells = re.findall(r"='([^']*)'", window_text)
        long_cells = sum(len(re.findall(r'[A-Za-z]+', cell)) >= 14 for cell in pseudo_cells)
        if ('source=pdf;' in window_text or 'parser=camelot-stream' in window_text) and long_cells >= 2:
            score *= 0.25
        window["score"] = score
        window["is_document_index"] = is_document_index
        if score > 0:
            positive_candidates.append(window)
    positive_candidates.sort(
        key=lambda item: (
            -(
                1
                if global_document_question
                and item["is_document_index"]
                and int(item.get("document_index_ordinal") or 0) <= reserved_index_windows_per_document
                else 0
            ),
            -float(item["score"]),
            int(item["order"]),
        )
    )
    retrieval_fallback_reason: str | None = None
    if not positive_candidates:
        # A valid parsed attachment must not be confused with an expired
        # session merely because the question has no lexical overlap.  Supply
        # one parser-generated document index per file and let the final model
        # report the mismatch or ask for a narrower question.
        represented_documents: set[str] = set()
        for window in window_records:
            if str(window["chunk"].get("kind")) != "document_index":
                continue
            document_id = window["document"].document_id
            if document_id in represented_documents:
                continue
            window["score"] = 0.0
            window["is_document_index"] = True
            positive_candidates.append(window)
            represented_documents.add(document_id)
        if positive_candidates:
            retrieval_fallback_reason = "no_positive_query_overlap_used_document_index"
    # Deduplicate parser representations by their cell values, not parser IDs.
    # Canonical chunks remain untouched and can still be inspected.
    unique_candidates = []
    seen_values = set()
    for candidate in positive_candidates:
        text = str(candidate['text'])
        values = re.findall(r"='([^']*)'", text)
        page = re.search(r'\bpage=(\d+)', text)
        same_page_stream = bool(page and ('parser=camelot-stream' in text or 'source=pdf;' in text))
        fingerprint = (candidate['document'].document_id, page.group(1) if page else None, tuple(values))
        if same_page_stream and len(values) >= 4 and fingerprint in seen_values:
            continue
        if same_page_stream and len(values) >= 4:
            seen_values.add(fingerprint)
        unique_candidates.append(candidate)
    positive_candidates = unique_candidates
    positive_candidates, semantic_rerank_audit = _semantic_rerank_customer_windows(
        question,
        positive_candidates,
        global_document_question=global_document_question,
    )

    candidates_by_document: dict[str, list[dict[str, Any]]] = {}
    for candidate in positive_candidates:
        document = candidate["document"]
        candidates_by_document.setdefault(document.document_id, []).append(candidate)
    relevant_document_ids = list(candidates_by_document)
    document_minimum = (
        min(configured_document_minimum, max(64, text_token_budget // len(relevant_document_ids)))
        if relevant_document_ids
        else 0
    )

    selected_candidates: list[dict[str, Any]] = []
    selected_window_ids: set[tuple[str, str]] = set()
    selected_intervals: dict[tuple[str, str], list[tuple[int, int]]] = {}
    selected_tokens_by_document: dict[str, int] = {document.document_id: 0 for document in session.documents}
    selected_token_total = 0

    def overlaps_selected(candidate: dict[str, Any]) -> bool:
        document = candidate["document"]
        key = (document.document_id, str(candidate["original_chunk_id"]))
        start = int(candidate["start_character"])
        end = int(candidate["end_character"])
        length = max(1, end - start)
        for existing_start, existing_end in selected_intervals.get(key, []):
            intersection = max(0, min(end, existing_end) - max(start, existing_start))
            if intersection / min(length, max(1, existing_end - existing_start)) >= 0.65:
                return True
        return False

    def add_candidate(candidate: dict[str, Any]) -> bool:
        nonlocal selected_token_total
        document = candidate["document"]
        identity = (document.document_id, str(candidate["window_id"]))
        candidate_tokens = int(candidate["estimated_tokens"])
        if (
            identity in selected_window_ids
            or len(selected_candidates) >= selected_window_limit
            or candidate_tokens > text_token_budget - selected_token_total
            or overlaps_selected(candidate)
        ):
            return False
        selected_candidates.append(candidate)
        selected_window_ids.add(identity)
        key = (document.document_id, str(candidate["original_chunk_id"]))
        selected_intervals.setdefault(key, []).append(
            (int(candidate["start_character"]), int(candidate["end_character"]))
        )
        selected_tokens_by_document[document.document_id] += candidate_tokens
        selected_token_total += candidate_tokens
        return True

    # Whole-document questions need a structural view from every attachment
    # before detailed rows compete for the remaining budget.  This is a
    # hierarchy-aware reservation, not a prefix slice: every canonical window
    # above was already generated and scored.
    if global_document_question:
        index_by_document: dict[str, list[dict[str, Any]]] = {}
        for candidate in positive_candidates:
            if (
                candidate["is_document_index"]
                and int(candidate.get("document_index_ordinal") or 0) <= reserved_index_windows_per_document
            ):
                index_by_document.setdefault(candidate["document"].document_id, []).append(candidate)
        for document in session.documents:
            candidates = index_by_document.get(document.document_id, [])
            for candidate in candidates:
                add_candidate(candidate)

        # Before any second window from the same parser chunk is considered,
        # reserve one representative content window from every table/section
        # that fits.  This prevents early sheets from consuming the whole
        # overview budget and hiding later sheets such as a final summary.
        represented_chunks: set[tuple[str, str]] = set()
        for candidate in positive_candidates:
            if candidate["is_document_index"]:
                continue
            key = (
                candidate["document"].document_id,
                str(candidate["original_chunk_id"]),
            )
            if key in represented_chunks:
                continue
            if add_candidate(candidate):
                represented_chunks.add(key)

    # Phase 1 reserves a modest minimum only for files with actual lexical
    # evidence.  A zero-relevance upload therefore consumes neither a slot nor
    # token budget.  Phase 2 lets all remaining windows compete globally.
    documents_by_relevance = sorted(
        relevant_document_ids,
        key=lambda document_id: -float(candidates_by_document[document_id][0]["score"]),
    )
    for document_id in documents_by_relevance:
        for candidate in candidates_by_document[document_id]:
            if selected_tokens_by_document[document_id] >= document_minimum:
                break
            add_candidate(candidate)
    for candidate in positive_candidates:
        add_candidate(candidate)

    selected: list[dict[str, Any]] = []
    selected_characters = 0
    for candidate in selected_candidates:
        document = candidate["document"]
        chunk = candidate["chunk"]
        excerpt = str(candidate["text"])
        citations = _citation(document, chunk)
        source_refs = list(chunk.get('source_refs') or [])
        if '[ROW ' in excerpt:
            visible_rows = {int(number) for number in re.findall(r"\b[A-Z]+(\d+)(?:\[[^\]]*\])?=", excerpt)}
            if visible_rows:
                citations = [citation for citation in citations
                             if citation.get('row_index') is None or citation.get('row_index') in visible_rows]
                source_refs = [ref for ref in source_refs
                               if document.source_locations.get(str(ref), {}).get('row_index') is None
                               or document.source_locations.get(str(ref), {}).get('row_index') in visible_rows]
        selected_characters += len(excerpt)
        selected.append(
            {
                "evidence_id": f"U{len(selected) + 1}",
                "document_id": document.document_id,
                "document_name": document.file_name,
                "chunk_id": str(candidate["original_chunk_id"]),
                "original_chunk_id": str(candidate["original_chunk_id"]),
                "window_id": str(candidate["window_id"]),
                "window_offset": {
                    "start_character": int(candidate["start_character"]),
                    "end_character": int(candidate["end_character"]),
                },
                "window_estimated_tokens": int(candidate["estimated_tokens"]),
                "original_chunk_estimated_tokens": int(candidate["original_chunk_estimated_tokens"]),
                "source_refs": source_refs,
                "text": excerpt,
                "score": round(float(candidate["score"]), 3),
                "lexical_score": round(float(candidate.get("lexical_score", candidate["score"])), 3),
                "semantic_reranker_score": (
                    round(float(candidate["semantic_reranker_score"]), 6)
                    if candidate.get("semantic_reranker_score") is not None
                    else None
                ),
                "evidence_scope": "document_index" if candidate["is_document_index"] else "content",
                "citations": citations,
            }
        )

    selected_visuals = _select_visuals(session.documents, question, selected)
    precise_citation_count = sum(
        1
        for item in selected
        for citation in item["citations"]
        if citation.get("source_page") is not None or citation.get("sheet_name") or citation.get("source_range")
    )
    selected_coverage_by_chunk: dict[tuple[str, str], int] = {}
    for key, intervals in selected_intervals.items():
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        selected_coverage_by_chunk[key] = sum(end - start for start, end in merged)
    truncated_chunk_count = sum(
        1
        for key, text in original_chunks.items()
        if selected_coverage_by_chunk.get(key, 0) < len(text)
    )
    per_document_quotas = [
        {
            "document_id": document.document_id,
            "document_name": document.file_name,
            "raw_estimated_tokens": per_document_raw_tokens.get(document.document_id, 0),
            "relevant_window_count": len(candidates_by_document.get(document.document_id, [])),
            "minimum_reserved_tokens": document_minimum if document.document_id in candidates_by_document else 0,
            "selected_estimated_tokens": selected_tokens_by_document.get(document.document_id, 0),
            "selected_window_count": sum(
                1 for candidate in selected_candidates if candidate["document"].document_id == document.document_id
            ),
            "zero_relevance": document.document_id not in candidates_by_document,
        }
        for document in session.documents
    ]
    return {
        "status": "ok",
        "evidence": selected,
        "documents": [_public_document(document) for document in session.documents],
        "selected_visuals": selected_visuals,
        "input_snapshot": {
            "selected_chunk_count": len(selected),
            "raw_chunk_count": raw_chunk_count,
            "raw_estimated_text_tokens": raw_estimated_tokens,
            "selected_estimated_text_tokens": selected_token_total,
            "max_text_tokens": text_token_budget,
            "configured_text_token_budget": configured_token_budget,
            "text_token_safety_limit": safety_token_limit,
            "window_token_budget": window_token_budget,
            "window_overlap_tokens": overlap_token_budget,
            "generated_window_count": len(window_records),
            "selected_window_count": len(selected),
            "unselected_window_count": max(0, len(window_records) - len(selected)),
            "windowed_chunk_count": windowed_chunk_count,
            "truncated_chunk_count": truncated_chunk_count,
            "per_document_quotas": per_document_quotas,
            "selection_policy": "relevant_documents_minimum_then_global_window_ranking",
            "retrieval_scope": "global_document" if global_document_question else "question_local",
            "planner_document_scope": normalized_document_scope,
            "retrieval_fallback_reason": retrieval_fallback_reason,
            "global_document_question": global_document_question,
            "global_reserved_index_windows_per_document": (
                reserved_index_windows_per_document if global_document_question else 0
            ),
            "global_content_sampling_enabled": global_document_question,
            "all_canonical_chunks_scanned": True,
            "scanned_canonical_chunk_count": raw_chunk_count,
            "scanned_retrieval_window_count": len(window_records),
            "selected_document_index_window_count": sum(
                1 for candidate in selected_candidates if candidate["is_document_index"]
            ),
            "selected_content_window_count": sum(
                1 for candidate in selected_candidates if not candidate["is_document_index"]
            ),
            "semantic_rerank": semantic_rerank_audit,
            "final_generation_model_selects_evidence": True,
            "token_count_method": "deterministic_language_aware_estimate_not_model_tokenizer",
            "gold_evidence_input_status": "unknown_at_runtime",
            "gold_evidence_note": (
                "Customer uploads do not include official gold evidence labels; "
                "the snapshot records selected windows but cannot assert that gold evidence was included."
            ),
            "selected_characters": selected_characters,
            "max_characters": max_characters,
            "legacy_character_budget_applied": legacy_character_budget_applied,
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
            "citation_coverage_complete": all(
                any(
                    citation.get("source_page") is not None
                    or citation.get("sheet_name")
                    or citation.get("source_range")
                    for citation in item["citations"]
                )
                for item in selected
            ),
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
