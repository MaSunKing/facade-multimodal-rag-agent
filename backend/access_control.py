"""Local SQLite authentication and knowledge-access policy.

Anonymous visitors may read only ``public`` knowledge.  The first administrator
is created with a one-time token written to ``runtime/admin_bootstrap_token.txt``;
after that, only an administrator can create accounts or grant internal access.
No customer question, attachment, or chat history is stored by this module.
"""

from __future__ import annotations

import base64
import binascii
import contextvars
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Literal

from fastapi import Header, HTTPException


ROOT = Path(__file__).resolve().parents[1]
STATE_DB_PATH = Path(os.getenv("FACADE_STATE_DB", ROOT / "runtime" / "facade_state.sqlite3"))
BOOTSTRAP_TOKEN_PATH = Path(
    os.getenv("FACADE_ADMIN_BOOTSTRAP_TOKEN_PATH", ROOT / "runtime" / "admin_bootstrap_token.txt")
)
VISUAL_TICKET_SECRET_PATH = Path(
    os.getenv("FACADE_VISUAL_TICKET_SECRET_PATH", ROOT / "runtime" / "visual_ticket_secret.bin")
)
AUTH_SESSION_HOURS = max(1, int(os.getenv("FACADE_AUTH_SESSION_HOURS", "12")))
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]{3,64}$")
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False


@dataclass(frozen=True)
class Principal:
    user_id: str
    username: str
    display_name: str
    role: Literal["admin", "member"]
    can_access_internal: bool

    @property
    def access_scopes(self) -> frozenset[str]:
        return frozenset({"public", "internal"} if self.can_access_internal else {"public"})


ANONYMOUS_ACCESS_SCOPES = frozenset({"public"})
_REQUEST_ACCESS_SCOPES: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "facade_request_access_scopes", default=ANONYMOUS_ACCESS_SCOPES
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STATE_DB_PATH, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with _connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'member')),
                    can_access_internal INTEGER NOT NULL DEFAULT 0 CHECK(can_access_internal IN (0, 1)),
                    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions(expires_at);

                CREATE TABLE IF NOT EXISTS agent_requests (
                    request_id TEXT PRIMARY KEY,
                    attachment_session_id TEXT,
                    query_length INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    principal_kind TEXT NOT NULL,
                    user_id TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    total_latency_ms INTEGER,
                    error_type TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE SET NULL
                );
                CREATE TABLE IF NOT EXISTS tool_runs (
                    tool_run_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms INTEGER,
                    result_count INTEGER,
                    summary_json TEXT,
                    error_type TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES agent_requests(request_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS evidence_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    snapshot_version TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL,
                    snapshot_path TEXT,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES agent_requests(request_id) ON DELETE CASCADE
                );
                """
            )
            now = _iso(_utc_now())
            connection.execute(
                "UPDATE agent_requests SET status='interrupted', finished_at=? WHERE status='running'",
                (now,),
            )
            connection.execute("DELETE FROM auth_sessions WHERE expires_at<=?", (now,))
        _SCHEMA_READY = True
        _ensure_bootstrap_token()


def _user_count() -> int:
    with _connect() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])


def _ensure_bootstrap_token() -> None:
    if _user_count() > 0 or BOOTSTRAP_TOKEN_PATH.exists():
        return
    BOOTSTRAP_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    BOOTSTRAP_TOKEN_PATH.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(BOOTSTRAP_TOKEN_PATH, 0o600)
    except OSError:
        pass


def _visual_ticket_secret() -> bytes:
    VISUAL_TICKET_SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not VISUAL_TICKET_SECRET_PATH.exists():
        VISUAL_TICKET_SECRET_PATH.write_bytes(secrets.token_bytes(32))
        try:
            os.chmod(VISUAL_TICKET_SECRET_PATH, 0o600)
        except OSError:
            pass
    return VISUAL_TICKET_SECRET_PATH.read_bytes()


def bootstrap_required() -> bool:
    ensure_schema()
    return _user_count() == 0


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$16384$8$1$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def _password_matches(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_text, digest_text = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.b64decode(salt_text)
        expected = base64.b64decode(digest_text)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _validate_account_input(username: str, password: str, display_name: str) -> tuple[str, str]:
    clean_username = username.strip()
    clean_display = display_name.strip() or clean_username
    if not USERNAME_PATTERN.fullmatch(clean_username):
        raise ValueError("用户名需为3至64位，可使用字母、数字、点、下划线、@或连字符。")
    if len(password) < 10 or len(password) > 200:
        raise ValueError("密码长度需为10至200位。")
    if len(clean_display) > 80:
        raise ValueError("显示名称不能超过80个字符。")
    return clean_username, clean_display


def create_initial_admin(*, setup_token: str, username: str, password: str, display_name: str) -> Principal:
    ensure_schema()
    if not bootstrap_required():
        raise ValueError("管理员已初始化。")
    expected = BOOTSTRAP_TOKEN_PATH.read_text(encoding="utf-8").strip() if BOOTSTRAP_TOKEN_PATH.exists() else ""
    if not expected or not hmac.compare_digest(setup_token.strip(), expected):
        raise ValueError("一次性初始化令牌无效。")
    clean_username, clean_display = _validate_account_input(username, password, display_name)
    now = _iso(_utc_now())
    principal = Principal(uuid.uuid4().hex, clean_username, clean_display, "admin", True)
    with _connect() as connection:
        connection.execute(
            "INSERT INTO users(user_id, username, display_name, password_hash, role, can_access_internal, active, created_at, updated_at) VALUES(?,?,?,?,?,?,1,?,?)",
            (
                principal.user_id,
                principal.username,
                principal.display_name,
                _password_hash(password),
                principal.role,
                1,
                now,
                now,
            ),
        )
    try:
        BOOTSTRAP_TOKEN_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    return principal


def _row_principal(row: sqlite3.Row) -> Principal:
    return Principal(
        user_id=str(row["user_id"]),
        username=str(row["username"]),
        display_name=str(row["display_name"]),
        role=str(row["role"]),  # type: ignore[arg-type]
        can_access_internal=bool(row["can_access_internal"]),
    )


def login(username: str, password: str) -> tuple[str, Principal, datetime]:
    ensure_schema()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username=? COLLATE NOCASE AND active=1", (username.strip(),)
        ).fetchone()
        if row is None or not _password_matches(password, str(row["password_hash"])):
            raise ValueError("用户名或密码错误。")
        principal = _row_principal(row)
        raw_token = secrets.token_urlsafe(40)
        expires_at = _utc_now() + timedelta(hours=AUTH_SESSION_HOURS)
        connection.execute(
            "INSERT INTO auth_sessions(token_hash, user_id, created_at, expires_at) VALUES(?,?,?,?)",
            (hashlib.sha256(raw_token.encode()).hexdigest(), principal.user_id, _iso(_utc_now()), _iso(expires_at)),
        )
    return raw_token, principal, expires_at


def principal_for_token(raw_token: str) -> Principal | None:
    ensure_schema()
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    now = _iso(_utc_now())
    with _connect() as connection:
        row = connection.execute(
            """SELECT u.* FROM auth_sessions s JOIN users u ON u.user_id=s.user_id
               WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>? AND u.active=1""",
            (token_hash, now),
        ).fetchone()
    return _row_principal(row) if row is not None else None


def revoke_token(raw_token: str) -> None:
    ensure_schema()
    with _connect() as connection:
        connection.execute(
            "UPDATE auth_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
            (_iso(_utc_now()), hashlib.sha256(raw_token.encode()).hexdigest()),
        )


def issue_visual_ticket(principal: Principal, asset_id: str, *, lifetime_seconds: int = 300) -> str:
    expires_at = int(_utc_now().timestamp()) + max(30, min(lifetime_seconds, 900))
    payload = f"{principal.user_id}\n{asset_id}\n{expires_at}".encode("utf-8")
    signature = hmac.new(_visual_ticket_secret(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + b"\n" + signature).decode("ascii").rstrip("=")


def principal_for_visual_ticket(ticket: str, asset_id: str) -> Principal | None:
    try:
        padded = ticket + "=" * (-len(ticket) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        user_id, ticket_asset, expiry_text, signature = decoded.split(b"\n", 3)
        payload = b"\n".join((user_id, ticket_asset, expiry_text))
        if not hmac.compare_digest(signature, hmac.new(_visual_ticket_secret(), payload, hashlib.sha256).digest()):
            return None
        if ticket_asset.decode("utf-8") != asset_id or int(expiry_text) < int(_utc_now().timestamp()):
            return None
        with _connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id=? AND active=1", (user_id.decode("utf-8"),)
            ).fetchone()
        return _row_principal(row) if row is not None else None
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error):
        return None


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="无效的身份凭证。")
    return token.strip()


def optional_principal(authorization: str | None = Header(default=None)) -> Principal | None:
    token = bearer_token(authorization)
    if token is None:
        return None
    principal = principal_for_token(token)
    if principal is None:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录。")
    return principal


def require_principal(authorization: str | None = Header(default=None)) -> Principal:
    principal = optional_principal(authorization)
    if principal is None:
        raise HTTPException(status_code=401, detail="请先登录。")
    return principal


def require_admin(authorization: str | None = Header(default=None)) -> Principal:
    principal = require_principal(authorization)
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可以执行此操作。")
    return principal


def list_users() -> list[dict[str, object]]:
    ensure_schema()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT user_id, username, display_name, role, can_access_internal, active, created_at, updated_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(row) for row in rows]


def create_user(*, username: str, password: str, display_name: str, role: str, can_access_internal: bool) -> dict[str, object]:
    if role not in {"admin", "member"}:
        raise ValueError("无效的角色。")
    clean_username, clean_display = _validate_account_input(username, password, display_name)
    now = _iso(_utc_now())
    user_id = uuid.uuid4().hex
    try:
        with _connect() as connection:
            connection.execute(
                "INSERT INTO users(user_id, username, display_name, password_hash, role, can_access_internal, active, created_at, updated_at) VALUES(?,?,?,?,?,?,1,?,?)",
                (user_id, clean_username, clean_display, _password_hash(password), role, int(can_access_internal), now, now),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("用户名已存在。") from exc
    return next(item for item in list_users() if item["user_id"] == user_id)


def update_user(user_id: str, *, role: str | None, can_access_internal: bool | None, active: bool | None) -> dict[str, object]:
    ensure_schema()
    if role is not None and role not in {"admin", "member"}:
        raise ValueError("无效的角色。")
    with _connect() as connection:
        current = connection.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if current is None:
            raise ValueError("用户不存在。")
        next_role = role or str(current["role"])
        next_internal = bool(current["can_access_internal"]) if can_access_internal is None else can_access_internal
        next_active = bool(current["active"]) if active is None else active
        if str(current["role"]) == "admin" and bool(current["active"]) and (next_role != "admin" or not next_active):
            active_admins = int(
                connection.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]
            )
            if active_admins <= 1:
                raise ValueError("不能停用或降级唯一的管理员。")
        connection.execute(
            "UPDATE users SET role=?, can_access_internal=?, active=?, updated_at=? WHERE user_id=?",
            (next_role, int(next_internal), int(next_active), _iso(_utc_now()), user_id),
        )
        if not next_active:
            connection.execute(
                "UPDATE auth_sessions SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL",
                (_iso(_utc_now()), user_id),
            )
    return next(item for item in list_users() if item["user_id"] == user_id)


def current_access_scopes() -> frozenset[str]:
    return _REQUEST_ACCESS_SCOPES.get()


def begin_agent_request(
    *,
    principal: Principal | None,
    attachment_session_id: str | None,
    query_length: int,
) -> tuple[str, float]:
    """Persist non-content request metadata before Agent execution."""

    ensure_schema()
    request_id = "req_" + uuid.uuid4().hex
    started = _utc_now()
    with _connect() as connection:
        connection.execute(
            """INSERT INTO agent_requests(
                   request_id, attachment_session_id, query_length, status,
                   principal_kind, user_id, started_at
               ) VALUES(?,?,?,?,?,?,?)""",
            (
                request_id,
                attachment_session_id,
                max(0, int(query_length)),
                "running",
                principal.role if isinstance(principal, Principal) else "anonymous",
                principal.user_id if isinstance(principal, Principal) else None,
                _iso(started),
            ),
        )
    return request_id, started.timestamp()


def finish_agent_request(
    request_id: str,
    started_timestamp: float,
    *,
    response: dict[str, object] | None,
    error_type: str | None = None,
) -> None:
    """Store bounded execution trace and a response Evidence snapshot.

    Full user questions and chat history are never written.  Public company
    evidence excerpts can be frozen for reproducibility; private attachment
    excerpts are intentionally redacted until disk-backed TTL sessions exist.
    """

    ensure_schema()
    finished = _utc_now()
    latency_ms = max(0, round((finished.timestamp() - started_timestamp) * 1000))
    status = "failed" if error_type else "completed"
    with _connect() as connection:
        connection.execute(
            "UPDATE agent_requests SET status=?, finished_at=?, total_latency_ms=?, error_type=? WHERE request_id=?",
            (status, _iso(finished), latency_ms, error_type, request_id),
        )
    if response is None:
        return

    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    orchestration = meta.get("orchestration") if isinstance(meta, dict) else {}
    tools = orchestration.get("tools") if isinstance(orchestration, dict) else []
    retrieval = response.get("retrieval") if isinstance(response.get("retrieval"), dict) else {}
    result_count = int(retrieval.get("result_count") or 0) if isinstance(retrieval, dict) else 0
    online_search = meta.get("online_search") if isinstance(meta.get("online_search"), dict) else {}
    now = _iso(_utc_now())
    with _connect() as connection:
        for tool_name in [str(item) for item in (tools or []) if str(item)]:
            tool_status = "completed"
            tool_error_type = None
            tool_result_count = result_count if tool_name in {"company_rag", "customer_documents"} else None
            tool_summary: dict[str, object] = {
                "planned": True,
                "workflow": orchestration.get("workflow"),
            }
            if tool_name == "public_web_search":
                search_status = str(online_search.get("status") or "unknown")
                tool_status = "completed" if search_status == "ok" else "failed"
                tool_error_type = (
                    None
                    if search_status == "ok"
                    else str(online_search.get("error_code") or search_status)
                )
                tool_result_count = len(response.get("online_sources") or [])
                tool_summary["online_search"] = {
                    key: online_search.get(key)
                    for key in (
                        "status",
                        "trigger",
                        "message",
                        "error_code",
                        "retryable",
                        "http_status",
                        "attempts",
                        "cache_hit",
                        "api_calls_for_query",
                        "source_profile",
                    )
                    if online_search.get(key) is not None
                }
            connection.execute(
                """INSERT INTO tool_runs(
                       tool_run_id, request_id, tool_name, status, result_count,
                       summary_json, error_type, created_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    "tool_" + uuid.uuid4().hex,
                    request_id,
                    tool_name,
                    tool_status,
                    tool_result_count,
                    json.dumps(tool_summary, ensure_ascii=False),
                    tool_error_type,
                    now,
                ),
            )

    attachment_session_id = None
    with _connect() as connection:
        row = connection.execute(
            "SELECT attachment_session_id FROM agent_requests WHERE request_id=?", (request_id,)
        ).fetchone()
        attachment_session_id = str(row[0]) if row and row[0] else None
    supporting = retrieval.get("supporting_results") if isinstance(retrieval, dict) else []
    generation_audit = (
        meta.get("generation_input_audit")
        if isinstance(meta, dict) and isinstance(meta.get("generation_input_audit"), dict)
        else {}
    )
    context_audit = (
        generation_audit.get("context_engine")
        if isinstance(generation_audit.get("context_engine"), dict)
        else meta.get("context_engine")
        if isinstance(meta, dict) and isinstance(meta.get("context_engine"), dict)
        else {}
    )
    integrity = (
        generation_audit.get("integrity_check")
        if isinstance(generation_audit.get("integrity_check"), dict)
        else {}
    )
    context_snapshot_audit = {
        "engine": context_audit.get("engine"),
        "candidate_count": context_audit.get("candidate_count"),
        "after_dedup_count": context_audit.get("after_dedup_count"),
        "source_candidate_counts": context_audit.get("source_candidate_counts") or {},
        "protected_relation_counts": context_audit.get("protected_relation_counts") or {},
        "conflict_group_count": len(context_audit.get("conflict_groups") or []),
        "missing_target_term_count": len(context_audit.get("missing_target_terms") or []),
        "coverage_sufficient_before_generation": context_audit.get(
            "coverage_sufficient_before_generation"
        ),
        "max_prompt_tokens": generation_audit.get("max_prompt_tokens"),
        "actual_prompt_tokens": generation_audit.get("actual_prompt_tokens"),
        "candidate_evidence_ids": generation_audit.get("candidate_evidence_ids") or [],
        "kept_evidence_ids": generation_audit.get("kept_evidence_ids") or [],
        "removed_evidence_ids": generation_audit.get("removed_evidence_ids") or [],
        "integrity_valid": integrity.get("valid"),
        "integrity_violation_count": integrity.get("violation_count"),
        "repair_attempted": bool(
            (generation_audit.get("repair_retrieval") or {}).get("attempted")
            if isinstance(generation_audit.get("repair_retrieval"), dict)
            else False
        ),
    }
    snapshot_payload = {
        "snapshot_version": "answer_evidence_v2",
        "request_id": request_id,
        "access_scopes": (meta.get("access_control") or {}).get("access_scopes", ["public"])
        if isinstance(meta, dict)
        else ["public"],
        "attachment_content_redacted": bool(attachment_session_id),
        "citations": response.get("citations") or [],
        "visual_assets": [
            {
                "asset_id": item.get("asset_id"),
                "citation": item.get("citation"),
            }
            for item in (response.get("visual_assets") or [])
            if isinstance(item, dict)
        ],
        "online_sources": response.get("online_sources") or [],
        "online_search": {
            key: online_search.get(key)
            for key in (
                "status",
                "trigger",
                "message",
                "error_code",
                "retryable",
                "http_status",
                "attempts",
                "cache_hit",
                "api_calls_for_query",
                "source_profile",
                "quota",
            )
            if online_search.get(key) is not None
        },
        "context_snapshot_audit": context_snapshot_audit,
        "supporting_results": (
            [
                {
                    "result_id": item.get("result_id"),
                    "document_name": item.get("document_name"),
                    "source_page": item.get("source_page"),
                    "section_heading": item.get("section_heading"),
                    "excerpt": item.get("excerpt"),
                }
                for item in (supporting or [])
                if isinstance(item, dict)
            ]
            if not attachment_session_id
            else [
                {
                    "result_id": item.get("result_id"),
                    "document_name": item.get("document_name"),
                    "source_page": item.get("source_page"),
                    "section_heading": item.get("section_heading"),
                    "content_redacted": True,
                }
                for item in (supporting or [])
                if isinstance(item, dict)
            ]
        ),
    }
    serialized = json.dumps(snapshot_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    snapshot_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    snapshot_dir = ROOT / "runtime" / "request_snapshots" / finished.strftime("%Y%m%d")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{request_id}.json"
    temporary = snapshot_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(snapshot_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(snapshot_path)
    evidence_count = len(snapshot_payload["citations"]) + len(snapshot_payload["supporting_results"])
    with _connect() as connection:
        connection.execute(
            """INSERT INTO evidence_snapshots(
                   snapshot_id, request_id, snapshot_version, snapshot_hash,
                   snapshot_path, evidence_count, total_tokens, created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                "snap_" + uuid.uuid4().hex,
                request_id,
                "answer_evidence_v2",
                snapshot_hash,
                str(snapshot_path.relative_to(ROOT)),
                evidence_count,
                int(generation_audit.get("actual_prompt_tokens") or 0) or None,
                now,
            ),
        )


@contextmanager
def request_access(principal: Principal | None) -> Iterator[None]:
    # Unit tests and internal Python callers invoke FastAPI endpoints directly;
    # in that case an unresolved ``Depends`` default reaches this function.
    # Treat it as an anonymous request rather than granting or crashing.
    if not isinstance(principal, Principal):
        principal = None
    token = _REQUEST_ACCESS_SCOPES.set(principal.access_scopes if principal else ANONYMOUS_ACCESS_SCOPES)
    try:
        yield
    finally:
        _REQUEST_ACCESS_SCOPES.reset(token)
