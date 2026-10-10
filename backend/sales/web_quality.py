"""Conservative admission checks, not a semantic fact checker or an allowlist."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

QUALITY_VERSION = 1


def is_time_sensitive(query: str) -> bool:
    return bool(re.search(r"今天|今日|现在|当前|实时|最新|本周|本月|today|current|latest|real.?time", query, re.I))


def filter_sources(query: str, sources: list[dict], policy: dict | None = None):
    """Drop obvious unsafe/SEO pollution before fetching, packing or displaying.

    Keyword absence alone is not proof of irrelevance. Only reject weak lexical
    overlap when the title is also unrelated; semantic sufficiency stays with
    Support Audit. Never certify a fact based on this admission decision.
    Rejected diagnostics contain IDs/reasons, not objectionable excerpts.
    """
    from backend.sales.baidu_search import _query_terms, _relevance_score, _host_matches
    policy = policy or {}
    accepted, rejected = [], []
    content_query = re.sub(r"20\d{2}年\d{1,2}月\d{1,2}日|\b20\d{2}-\d{1,2}-\d{1,2}\b", "", query)
    terms = _query_terms(content_query) - {"查询", "概括", "介绍", "最新", "实时", "今天", "今日", "当前", "现在", "请问"}
    adult_query = bool(re.search(r"成人|色情|情色|A片|成人视频|porn|sex", query, re.I))
    for source in sources:
        reason = ""
        try:
            parts = urlsplit(str(source.get("url") or ""))
            host = parts.hostname or ""
            if parts.scheme not in {"http", "https"} or not host or parts.username or parts.password or parts.port not in {None, 80, 443}:
                reason = "unsafe_url"
            try:
                if not ipaddress.ip_address(host).is_global:
                    reason = "non_public_url"
            except ValueError:
                if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
                    reason = "non_public_url"
            if any(_host_matches(host, str(domain)) for domain in policy.get("blocked_domains", []) if domain):
                reason = "blocked_domain"
        except ValueError:
            reason = "unsafe_url"
        title = str(source.get("title") or "")
        snippet = str(source.get("search_excerpt") or "")
        # Unrelated adult landing pages, not reports about adult-content harms.
        if not adult_query and re.search(r"激情视频|成人视频|成人免费.*视频|一级片|色情网站|pornhub|xxx.?video", title, re.I):
            reason = "unrelated_adult_landing_page"
        if not adult_query and re.search(r"成人版", title) and re.search(r"最新版|安卓版|iphone版|下载", title, re.I):
            reason = "seo_download_pollution"
        if not title.strip() and not snippet.strip():
            reason = "empty_source"
        title_related = any(term.casefold() in title.casefold() for term in terms)
        if not reason and terms and not title_related and _relevance_score(content_query, source) == 0:
            reason = "no_query_overlap"
        if reason:
            rejected.append({"source_id": source.get("source_id"), "original_rank": source.get("original_rank"), "reason": reason})
        else:
            accepted.append({**source, "quality_status": "admitted", "quality_is_fact_verification": False})
    return accepted, rejected
