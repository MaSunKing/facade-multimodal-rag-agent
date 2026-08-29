"""Authentication and administrator-facing account management endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from backend.access_control import (
    Principal,
    bearer_token,
    bootstrap_required,
    create_initial_admin,
    create_user,
    list_users,
    login,
    optional_principal,
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
def auth_status(principal: Principal | None = Depends(optional_principal)) -> dict[str, object]:
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

