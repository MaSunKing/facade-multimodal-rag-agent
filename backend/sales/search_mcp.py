"""Real MCP network-search tool with the existing governed Evidence contract.

The backend uses an in-process MCP transport: protocol initialization and
tools/call still occur, but the request budget/cancellation ContextVar stays
local. The identical server can be run over stdio by an external MCP host.
No public HTTP listener, filesystem tool, model loading or cloud registration.
"""
from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel

from backend.request_budget import check_budget, RequestBudgetExceeded
from backend.sales.baidu_search import (
    BaiduSearchError, SearchQuotaExceeded, WebSourceProfile, search_baidu_web,
)


class SearchToolResult(BaseModel):
    ok: bool
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


def build_search_server() -> FastMCP:
    server = FastMCP("QianfanPublicWebSearch")

    @server.tool()
    async def network_search(query: str, source_profile: WebSourceProfile = "auto", bypass_cache: bool = False) -> SearchToolResult:
        """Search public web for the current public query. Never send private evidence,
        attachments or conversation history. Returns admitted references with
        provenance, quality diagnostics and quota; snippets are not verified facts.
        No usable sources is a valid empty result, not permission to invent an answer.
        """
        check_budget()
        try:
            result = await anyio.to_thread.run_sync(partial(
                search_baidu_web, query, source_profile=source_profile, bypass_cache=bypass_cache))
            check_budget()
            return SearchToolResult(ok=True, result=result)
        except BaiduSearchError as exc:
            # Do not expose upstream error bodies/credentials to an MCP client.
            return SearchToolResult(ok=False, error={
                "code": "quota_exhausted" if isinstance(exc, SearchQuotaExceeded) else exc.error_code,
                "retryable": exc.retryable, "http_status": exc.http_status, "attempts": exc.attempts,
            })

    return server


async def _search_via_mcp(query: str, source_profile: str | None, bypass_cache: bool) -> dict:
    check_budget()
    async with create_connected_server_and_client_session(build_search_server()) as session:
        response = await session.call_tool("network_search", {
            "query": query, "source_profile": source_profile or "auto", "bypass_cache": bypass_cache,
        })
    # Validate outside the transport's task group to retain typed failures.
    check_budget()
    envelope = response.structuredContent
    if response.isError or not isinstance(envelope, dict) or not isinstance(envelope.get('ok'), bool):
        raise BaiduSearchError("MCP search returned an invalid tool result.", error_code="mcp_tool_invalid")
    if not envelope['ok']:
        error = envelope.get('error') or {}
        cls = SearchQuotaExceeded if error.get('code') == 'quota_exhausted' else BaiduSearchError
        raise cls("MCP network search failed.", error_code=str(error.get('code') or 'mcp_tool_failed'),
                  retryable=bool(error.get('retryable')), http_status=error.get('http_status'),
                  attempts=int(error.get('attempts') or 1))
    result = envelope.get('result')
    if not isinstance(result, dict) or not isinstance(result.get('sources'), list):
        raise BaiduSearchError("MCP search evidence contract is invalid.", error_code="mcp_evidence_invalid")
    return {**result, "tool_transport": "mcp_in_process", "mcp_tool": "network_search"}


def search_web_via_mcp(query: str, *, source_profile: str | None = None, bypass_cache: bool = False) -> dict:
    """Synchronous workflow entrypoint, called in the existing worker thread."""
    check_budget()
    try:
        return asyncio.run(_search_via_mcp(query, source_profile, bypass_cache))
    except (BaiduSearchError, RequestBudgetExceeded):
        raise
    except Exception as exc:
        # Check cancellation before interpreting transport/task-group errors.
        check_budget()
        raise BaiduSearchError("MCP transport failed.", error_code="mcp_transport_failed") from exc


if __name__ == '__main__':
    build_search_server().run(transport='stdio')
