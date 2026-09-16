"""Request identity for process-local customer attachment sessions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator

from fastapi import Depends, Header, HTTPException

from backend.access_control import Principal, optional_principal
from backend.documents.customer_sessions import bind_session_owner


_CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,80}$")


def attachment_owner_id(
    principal: Principal | None,
    browser_client_id: str | None,
    *,
    required: bool,
) -> str | None:
    """Return a stable identity without storing a bearer token or raw client id."""

    if isinstance(principal, Principal):
        return f"user:{principal.user_id}"
    clean_client_id = str(browser_client_id or "").strip()
    if _CLIENT_ID_PATTERN.fullmatch(clean_client_id):
        digest = hashlib.sha256(clean_client_id.encode("utf-8")).hexdigest()
        return f"browser:{digest}"
    if required:
        raise HTTPException(
            status_code=400,
            detail="附件操作缺少有效的浏览器客户端标识，请刷新页面后重试。",
        )
    return None


def required_attachment_owner(
    principal: Principal | None = Depends(optional_principal),
    x_facade_client_id: str | None = Header(default=None),
) -> str:
    owner_id = attachment_owner_id(principal, x_facade_client_id, required=True)
    assert owner_id is not None
    return owner_id


def optional_attachment_owner(
    principal: Principal | None = Depends(optional_principal),
    x_facade_client_id: str | None = Header(default=None),
) -> str | None:
    return attachment_owner_id(principal, x_facade_client_id, required=False)


async def bind_attachment_request_owner(
    owner_id: str | None = Depends(optional_attachment_owner),
) -> AsyncIterator[str | None]:
    """Keep indirect LangGraph/session reads bound to the verified HTTP owner."""

    with bind_session_owner(owner_id):
        yield owner_id
