"""HTTP boundary for temporary customer document sessions."""

from __future__ import annotations
import threading

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.documents.customer_sessions import (
    MAX_FILE_BYTES,
    MAX_SESSION_BYTES,
    AttachmentCapacityError,
    add_files,
    delete_session,
    get_visual_asset,
    session_access_status,
    session_summary,
)
from backend.documents.ownership import optional_attachment_owner, required_attachment_owner
from backend.document_parsing.ingestion import FinanceIntakeError


router = APIRouter(prefix="/api/copilot/documents", tags=["copilot-documents"])
_upload_admission = threading.BoundedSemaphore(2)


async def _read_upload_limited(file: UploadFile, *, byte_limit: int) -> bytes:
    """Stream one multipart member and stop before it can fill process memory."""

    content = bytearray()
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(min(chunk_size, byte_limit - len(content) + 1))
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > byte_limit:
            raise ValueError("single_file_too_large")
    return bytes(content)


@router.post("")
async def upload_customer_documents(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(default=None),
    owner_id: str = Depends(required_attachment_owner),
) -> dict:
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="一次最多上传 4 份文件。")
    if not _upload_admission.acquire(blocking=False):
        for file in files:
            await file.close()
        raise HTTPException(status_code=429, detail="本地附件上传繁忙，请稍后重试。")
    payload: list[tuple[str, bytes]] = []
    try:
        existing = session_summary(session_id, owner_id=owner_id) if session_id else None
        if session_id and existing is None:
            if session_access_status(session_id, owner_id=owner_id) == "foreign":
                # Never let a browser turn another owner's id into an append
                # target, or reveal anything about the foreign session.
                raise HTTPException(status_code=404, detail="附件会话不存在、已过期或不属于当前用户。")
            # A backend restart/TTL expiry leaves the browser with a stale id.
            # Start a fresh random session instead of reusing caller input.
            session_id = None
        existing_bytes = sum(
            int(document.get("size_bytes") or 0)
            for document in (existing or {}).get("documents", [])
        )
        remaining_session_bytes = MAX_SESSION_BYTES - existing_bytes
        if remaining_session_bytes <= 0:
            raise ValueError("session_files_too_large")
        for file in files:
            content = await _read_upload_limited(
                file,
                byte_limit=min(MAX_FILE_BYTES, remaining_session_bytes),
            )
            payload.append((file.filename or "uploaded-file", content))
            remaining_session_bytes -= len(content)
            if remaining_session_bytes < 0:
                raise ValueError("session_files_too_large")
        return await run_in_threadpool(add_files, payload, session_id, owner_id=owner_id)
    except AttachmentCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="附件会话不存在、已过期或不属于当前用户。") from exc
    except (FinanceIntakeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _upload_admission.release()
        for file in files:
            await file.close()


@router.get("/{session_id}/visual/{document_id}/{visual_id:path}")
def read_customer_visual(
    session_id: str,
    document_id: str,
    visual_id: str,
    ticket: str | None = Query(default=None, max_length=2000),
    owner_id: str | None = Depends(optional_attachment_owner),
) -> Response:
    """Serve a selected upload visual without persisting it or exposing other sessions."""

    visual = get_visual_asset(
        session_id,
        document_id,
        visual_id,
        owner_id=owner_id,
        access_ticket=ticket,
    )
    if visual is None or not visual.image_bytes:
        raise HTTPException(status_code=404, detail="附件图片不存在、已过期或不属于该会话。")
    return Response(
        content=visual.image_bytes,
        media_type=visual.media_type or "application/octet-stream",
        headers={
            "Cache-Control": "private, no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{session_id}")
def inspect_customer_documents(
    session_id: str,
    owner_id: str = Depends(required_attachment_owner),
) -> dict:
    summary = session_summary(session_id, owner_id=owner_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="附件会话不存在、已过期或不属于当前用户。")
    return summary


@router.delete("/{session_id}")
def remove_customer_documents(
    session_id: str,
    owner_id: str = Depends(required_attachment_owner),
) -> dict[str, bool]:
    if not delete_session(session_id, owner_id=owner_id):
        raise HTTPException(status_code=404, detail="附件会话不存在、已过期或不属于当前用户。")
    return {"deleted": True}
