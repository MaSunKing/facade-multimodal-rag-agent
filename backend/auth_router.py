"""Authentication and administrator-facing account management endpoints."""

from __future__ import annotations

from datetime import datetime
import sqlite3

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from backend.documents.ownership import required_attachment_owner

from backend.access_control import (
    Principal,
    bearer_token,
    bootstrap_required,
    create_initial_admin,
    create_user,
    list_users,
    login,
    optional_principal,
    public_principal,
    require_admin,
    revoke_token,
    update_user,
)


router = APIRouter(prefix="/api/auth", tags=["access-control"])


class BootstrapRequest(BaseModel):
    setup_token: str = Field(min_length=20, max_length=200)
    username: str = Field(min_length=3, max_length=64)
    display_name: str = Field(default="管理员", max_length=80)
    password: str = Field(min_length=10, max_length=200)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=200)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    display_name: str = Field(max_length=80)
    password: str = Field(min_length=10, max_length=200)
    role: str = Field(default="member")
    can_access_internal: bool = False


class UpdateUserRequest(BaseModel):
    role: str | None = None
    can_access_internal: bool | None = None
    active: bool | None = None


def _principal_payload(principal: Principal) -> dict[str, object]:
    return {
        "user_id": principal.user_id,
        "username": principal.username,
        "display_name": principal.display_name,
        "role": principal.role,
        "can_access_internal": principal.can_access_internal,
        "access_scopes": sorted(principal.access_scopes),
    }


@router.get("/status")
def auth_status(principal: Principal | None = Depends(public_principal)) -> dict[str, object]:
    return {
        "bootstrap_required": bootstrap_required(),
        "authenticated": principal is not None,
        "principal": _principal_payload(principal) if principal else None,
        "anonymous_access_scopes": ["public"],
        "internal_knowledge_status": "empty",
    }


@router.post("/bootstrap")
def bootstrap_admin(request: BootstrapRequest) -> dict[str, object]:
    try:
        create_initial_admin(**request.model_dump())
        token, principal, expires_at = login(request.username, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"access_token": token, "expires_at": expires_at.isoformat(), "principal": _principal_payload(principal)}


@router.post("/login")
def login_account(request: LoginRequest) -> dict[str, object]:
    try:
        token, principal, expires_at = login(request.username, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {"access_token": token, "expires_at": expires_at.isoformat(), "principal": _principal_payload(principal)}


@router.post("/logout")
def logout_account(authorization: str | None = Header(default=None)) -> dict[str, bool]:
    token = bearer_token(authorization)
    if token:
        revoke_token(token)
    return {"logged_out": True}


@router.get("/me")
def current_account(principal: Principal = Depends(optional_principal)) -> dict[str, object]:
    if principal is None:
        raise HTTPException(status_code=401, detail="请先登录。")
    return _principal_payload(principal)


@router.get("/admin/users")
def admin_users(_: Principal = Depends(require_admin)) -> dict[str, object]:
    return {"users": list_users(), "internal_knowledge_status": "empty"}


@router.get('/admin/traces')
def admin_traces(limit: int = Query(default=30, ge=1, le=100), _: Principal = Depends(require_admin)):
    from backend.sales.admin_traces import list_traces
    return list_traces(limit)


@router.get('/traces')
def user_traces(limit: int = Query(default=30, ge=1, le=100),
                owner: str = Depends(required_attachment_owner)):
    from backend.sales.admin_traces import owned_traces
    try:
        return owned_traces(owner,limit)
    except sqlite3.OperationalError:
        raise HTTPException(status_code=503, detail='轨迹存储暂不可用，请稍后刷新；不影响当前任务。',
                            headers={'Retry-After':'3'}) from None


@router.get('/traces/{request_id}')
def user_trace(request_id: str,
               owner: str = Depends(required_attachment_owner)):
    from backend.sales.admin_traces import owned_trace
    try:
        result=owned_trace(owner,request_id)
    except sqlite3.OperationalError:
        raise HTTPException(status_code=503, detail='轨迹存储暂不可用，请稍后刷新；不影响当前任务。',
                            headers={'Retry-After':'3'}) from None
    if result is None:
        raise HTTPException(status_code=404,detail='没有找到属于当前用户的请求轨迹。')
    return result


@router.get('/admin/traces/{request_id}')
def admin_trace(request_id: str, _: Principal = Depends(require_admin)):
    import re
    if not re.fullmatch(r'(?:req_)?[a-f0-9]{32}', request_id):
        raise HTTPException(status_code=400, detail='无效的请求编号')
    from backend.sales.admin_traces import read_trace
    result = read_trace(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail='请求链路不存在')
    return result


@router.post("/admin/users")
def admin_create_user(request: CreateUserRequest, _: Principal = Depends(require_admin)) -> dict[str, object]:
    try:
        return create_user(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/admin/users/{user_id}")
def admin_update_user(user_id: str, request: UpdateUserRequest, _: Principal = Depends(require_admin)) -> dict[str, object]:
    try:
        return update_user(user_id, **request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
