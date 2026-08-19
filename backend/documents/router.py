"""HTTP boundary for temporary customer document sessions."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.documents.customer_sessions import add_files, delete_session, get_visual_asset, session_summary
from backend.document_parsing.ingestion import FinanceIntakeError


router = APIRouter(prefix="/api/copilot/documents", tags=["copilot-documents"])


@router.post("")
async def upload_customer_documents(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(default=None),
) -> dict:
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="一次最多上传 4 份文件。")
    payload: list[tuple[str, bytes]] = []
    try:
        for file in files:
            payload.append((file.filename or "uploaded-file", await file.read()))
        return await run_in_threadpool(add_files, payload, session_id)
    except (FinanceIntakeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()


@router.get("/{session_id}/visual/{document_id}/{visual_id:path}")
def read_customer_visual(session_id: str, document_id: str, visual_id: str) -> Response:
    """Serve a selected upload visual without persisting it or exposing other sessions."""

    visual = get_visual_asset(session_id, document_id, visual_id)
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
def inspect_customer_documents(session_id: str) -> dict:
    summary = session_summary(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="附件会话不存在或已过期。")
    return summary


@router.delete("/{session_id}")
def remove_customer_documents(session_id: str) -> dict[str, bool]:
    return {"deleted": delete_session(session_id)}
