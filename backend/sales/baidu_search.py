"""Minimal, source-preserving client for Baidu AI Search.

The client sends the customer's current search question exactly as entered
(apart from leading/trailing whitespace).  It never uploads local RAG chunks,
source PDFs, customer images, or the browser conversation to Baidu.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BAIDU_WEB_SEARCH_ENDPOINT = "https://qianfan.baidubce.com/v2/ai_search/web_search"


class BaiduSearchError(RuntimeError):
    """Raised when the configured Baidu Search endpoint cannot return results."""


def is_baidu_search_configured() -> bool:
    return bool(os.getenv("BAIDU_AI_SEARCH_API_KEY", "").strip())


def compact_query(query: str, limit: int = 3000) -> str:
    """Return the full current question without silently dropping keywords.

    ``DraftRequest`` already limits a customer question to 3,000 characters.
    Keeping the same bound makes the outbound request predictable while, most
    importantly, preserving project names and all other wording.  A too-long
    value raises an explicit error instead of truncating it into a different
    search request.
    """

    value = str(query).strip()
    if not value:
        raise BaiduSearchError("The search query is empty.")
    if len(value) > limit:
        raise BaiduSearchError(f"The search query exceeds the {limit}-character limit.")
    return value


def _references_from_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise the documented references response while tolerating wrappers."""

    candidates: Any = payload.get("references")
    if not isinstance(candidates, list):
        data = payload.get("data")
        candidates = data.get("references") if isinstance(data, dict) else []
    if not isinstance(candidates, list):
        return []

    sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {None, "web"}:
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or item.get("web_anchor") or item.get("website") or "网页来源").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        excerpt = str(item.get("content") or item.get("snippet") or "").strip()
        sources.append(
            {
                "source_id": f"W{len(sources) + 1}",
                "title": title[:200],
                "url": url,
                "website": str(item.get("website") or "").strip()[:120],
                "date": str(item.get("date") or "").strip()[:60],
                "excerpt": excerpt[:1200],
            }
        )
        if len(sources) >= 5:
            break
    return sources


def search_baidu_web(query: str) -> dict[str, Any]:
    """Search public web pages and return a small, citation-ready source list."""

    api_key = os.getenv("BAIDU_AI_SEARCH_API_KEY", "").strip()
    if not api_key:
        raise BaiduSearchError("BAIDU_AI_SEARCH_API_KEY is not configured.")

    header_name = os.getenv("BAIDU_AI_SEARCH_AUTH_HEADER", "X-Appbuilder-Authorization").strip()
    if header_name not in {"Authorization", "X-Appbuilder-Authorization"}:
        raise BaiduSearchError("BAIDU_AI_SEARCH_AUTH_HEADER must be Authorization or X-Appbuilder-Authorization.")

    full_question = compact_query(query)
    body = {
        "messages": [{"role": "user", "content": full_question}],
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": 5}],
        "safe_search": True,
    }
    request = Request(
        BAIDU_WEB_SEARCH_ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={header_name: f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=18) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise BaiduSearchError(f"Baidu search returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise BaiduSearchError(f"Baidu search network error: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise BaiduSearchError(f"Baidu search response could not be read: {type(exc).__name__}") from exc

    if payload.get("code"):
        raise BaiduSearchError(str(payload.get("message") or payload["code"]))
    return {"query": full_question, "sources": _references_from_response(payload)}
