"""Quota-aware, source-governed Baidu public-web search.

Only the current customer question is sent to Baidu. Private RAG chunks,
uploaded files, images and browser history never leave the local service.
One logical query performs at most one Baidu API request; cache lookup,
authority re-ranking and optional public-page verification are local steps.
"""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import math
import os
import re
import socket
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from backend.request_budget import check_budget, reserve_recovery
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, ProxyHandler, build_opener, urlopen
from urllib.robotparser import RobotFileParser
from backend.sales.web_quality import QUALITY_VERSION, filter_sources, is_time_sensitive


BAIDU_WEB_SEARCH_ENDPOINT = "https://qianfan.baidubce.com/v2/ai_search/web_search"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "data" / "sales" / "config" / "web_source_policy_v1.json"
DEFAULT_STATE_DB = ROOT / "runtime" / "sales_web_search.sqlite3"
WebSourceProfile = Literal[
    "auto",
    "construction_standard",
    "public_project",
    "manufacturer_product",
    "industry_news",
    "general_public",
]
VALID_PROFILES = {
    "auto",
    "construction_standard",
    "public_project",
    "manufacturer_product",
    "industry_news",
    "general_public",
}
_DB_LOCK = threading.RLock()
_TRACKING_QUERY_KEYS = {"from", "spm", "source", "ref", "referrer"}
SEARCH_RANKING_VERSION = 2


class BaiduSearchError(RuntimeError):
    """Raised when the configured search endpoint cannot return results."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "baidu_search_failed",
        retryable: bool = False,
        http_status: int | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable
        self.http_status = http_status
        self.attempts = max(1, int(attempts))


class SearchQuotaExceeded(BaiduSearchError):
    """Raised before an API request when the local daily budget is exhausted."""


def is_baidu_search_configured() -> bool:
    return bool(os.getenv("BAIDU_AI_SEARCH_API_KEY", "").strip())


def compact_query(query: str, limit: int = 3000) -> str:
    value = str(query).strip()
    if not value:
        raise BaiduSearchError("The search query is empty.")
    if len(value) > limit:
        raise BaiduSearchError(f"The search query exceeds the {limit}-character limit.")
    return value


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", compact_query(query)).strip()


def classify_source_profile(query: str) -> str:
    """Deterministic fallback when the Agent did not specify a profile."""

    value = normalize_query(query).lower()
    if any(term in value for term in ("规范", "标准", "条文", "国标", "行标", "jgj", "gb/", "gb ", "住建部", "验收规程")):
        return "construction_standard"
    if any(term in value for term in ("项目", "工程", "招标", "中标", "建设单位", "施工单位", "竣工", "开工", "公共资源")):
        return "public_project"
    if any(term in value for term in ("厂家", "厂商", "品牌", "型号", "产品参数", "产品手册", "官网")):
        return "manufacturer_product"
    if any(term in value for term in ("最新", "新闻", "动态", "趋势", "近期", "今天", "本周", "本月")):
        return "industry_news"
    return "general_public"


def resolve_source_profile(query: str, source_profile: str | None = None) -> str:
    profile = str(source_profile or "auto").strip()
    if profile not in VALID_PROFILES:
        profile = "auto"
    return classify_source_profile(query) if profile == "auto" else profile


def _load_policy() -> dict[str, Any]:
    configured = Path(os.getenv("WEB_SOURCE_POLICY_PATH", str(DEFAULT_POLICY_PATH))).expanduser()
    try:
        value = json.loads(configured.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _state_db_path() -> Path:
    value = Path(os.getenv("WEB_SEARCH_STATE_DB", str(DEFAULT_STATE_DB))).expanduser()
    value.parent.mkdir(parents=True, exist_ok=True)
    return value


def _connect_state_db() -> sqlite3.Connection:
    connection = sqlite3.connect(_state_db_path(), timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS web_search_cache (
        cache_key TEXT PRIMARY KEY, query TEXT NOT NULL, source_profile TEXT NOT NULL,
        created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, payload_json TEXT NOT NULL)"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS web_search_usage (
        usage_date TEXT PRIMARY KEY, api_calls INTEGER NOT NULL)"""
    )
    connection.commit()
    return connection


def _china_date() -> str:
    return datetime.now(timezone(timedelta(hours=8))).date().isoformat()


def _daily_limits() -> tuple[int, int]:
    hard_limit = max(1, int(os.getenv("WEB_SEARCH_DAILY_HARD_LIMIT", "50")))
    business_limit = max(1, int(os.getenv("WEB_SEARCH_DAILY_BUSINESS_LIMIT", "45")))
    return min(business_limit, hard_limit), hard_limit


def quota_snapshot() -> dict[str, int]:
    business_limit, hard_limit = _daily_limits()
    with _DB_LOCK, closing(_connect_state_db()) as connection:
        row = connection.execute(
            "SELECT api_calls FROM web_search_usage WHERE usage_date = ?", (_china_date(),)
        ).fetchone()
    used = int(row[0]) if row else 0
    return {
        "business_limit": business_limit,
        "hard_limit": hard_limit,
        "used": used,
        "remaining": max(0, business_limit - used),
        "reserved": max(0, hard_limit - business_limit),
    }


def _reserve_api_call(*, usage_date: str | None = None) -> dict[str, int]:
    business_limit, hard_limit = _daily_limits()
    current_date = usage_date or _china_date()
    with _DB_LOCK, closing(_connect_state_db()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT api_calls FROM web_search_usage WHERE usage_date = ?", (current_date,)
        ).fetchone()
        used = int(row[0]) if row else 0
        if used >= business_limit:
            connection.rollback()
            raise SearchQuotaExceeded("The local daily Baidu search business budget is exhausted.")
        used += 1
        connection.execute(
            """INSERT INTO web_search_usage(usage_date, api_calls) VALUES(?, ?)
            ON CONFLICT(usage_date) DO UPDATE SET api_calls = excluded.api_calls""",
            (current_date, used),
        )
        connection.commit()
    return {
        "business_limit": business_limit,
        "hard_limit": hard_limit,
        "used": used,
        "remaining": max(0, business_limit - used),
        "reserved": max(0, hard_limit - business_limit),
    }


def _release_api_call(*, usage_date: str) -> dict[str, int]:
    """Release a local reservation when no usable API response was received."""

    business_limit, hard_limit = _daily_limits()
    with _DB_LOCK, closing(_connect_state_db()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT api_calls FROM web_search_usage WHERE usage_date = ?", (usage_date,)
        ).fetchone()
        used = max(0, int(row[0]) - 1) if row else 0
        connection.execute(
            """INSERT INTO web_search_usage(usage_date, api_calls) VALUES(?, ?)
            ON CONFLICT(usage_date) DO UPDATE SET api_calls = excluded.api_calls""",
            (usage_date, used),
        )
        connection.commit()
    return {
        "business_limit": business_limit,
        "hard_limit": hard_limit,
        "used": used,
        "remaining": max(0, business_limit - used),
        "reserved": max(0, hard_limit - business_limit),
    }


def _cache_key(query: str, profile: str) -> str:
    return hashlib.sha256(f"quality-v{QUALITY_VERSION}\nranking-v{SEARCH_RANKING_VERSION}\n{profile}\n{normalize_query(query).casefold()}".encode("utf-8")).hexdigest()


def _cache_ttl_seconds(profile: str) -> int:
    defaults = {
        "construction_standard": 30 * 86400,
        "public_project": 24 * 3600,
        "manufacturer_product": 7 * 86400,
        "industry_news": 6 * 3600,
        "general_public": 24 * 3600,
    }
    env_key = f"WEB_SEARCH_CACHE_TTL_{profile.upper()}"
    return max(300, int(os.getenv(env_key, str(defaults.get(profile, 86400)))))


def _read_cache(query: str, profile: str) -> dict[str, Any] | None:
    now = int(time.time())
    key = _cache_key(query, profile)
    with _DB_LOCK, closing(_connect_state_db()) as connection:
        row = connection.execute(
            "SELECT payload_json, expires_at FROM web_search_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        if int(row[1]) <= now:
            connection.execute("DELETE FROM web_search_cache WHERE cache_key = ?", (key,))
            connection.commit()
            return None
    try:
        payload = json.loads(str(row[0]))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _write_cache(query: str, profile: str, payload: dict[str, Any]) -> None:
    now = int(time.time())
    with _DB_LOCK, closing(_connect_state_db()) as connection:
        connection.execute(
            """INSERT OR REPLACE INTO web_search_cache
            (cache_key, query, source_profile, created_at, expires_at, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                _cache_key(query, profile), normalize_query(query), profile, now,
                now + (min(_cache_ttl_seconds(profile), 300) if is_time_sensitive(query) else _cache_ttl_seconds(profile)), json.dumps(payload, ensure_ascii=False),
            ),
        )
        connection.commit()


def _canonical_url(raw_url: str) -> str:
    try:
        parts = urlsplit(raw_url.strip())
    except ValueError:
        return ""
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return ""
    query = [
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    try:
        if parts.username or parts.password:
            return ""
        host = parts.hostname.encode("idna").decode("ascii").lower()
        port = f":{parts.port}" if parts.port and parts.port not in {80, 443} else ""
    except (ValueError, UnicodeError):
        return ""
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    return urlunsplit((parts.scheme.lower(), f"{host}{port}", path, urlencode(query), ""))


def _references_from_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: Any = payload.get("references")
    if not isinstance(candidates, list):
        data = payload.get("data")
        candidates = data.get("references") if isinstance(data, dict) else []
    if not isinstance(candidates, list):
        return []
    sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for provider_rank, item in enumerate(candidates, start=1):
        if not isinstance(item, dict) or item.get("type") not in {None, "web"}:
            continue
        canonical = _canonical_url(str(item.get("url") or ""))
        if not canonical or canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        search_excerpt = re.sub(
            r"\s+", " ", html.unescape(str(item.get("content") or item.get("snippet") or ""))
        ).strip()
        sources.append({
            "source_id": f"W{provider_rank}",
            "original_rank": provider_rank,
            "provider_reference_id": str(item.get("id") or "")[:120],
            "provider_rerank_score": item.get("rerank_score"),
            "provider_authority_score": item.get("authority_score"),
            "title": str(item.get("title") or item.get("web_anchor") or item.get("website") or "网页来源").strip()[:200],
            "url": canonical,
            "website": str(item.get("website") or "").strip()[:120],
            "date": str(item.get("date") or "").strip()[:60],
            "search_excerpt": search_excerpt[:1800],
            "excerpt": search_excerpt[:1800],
            "page_fetch_status": "not_attempted",
            "evidence_level": "unavailable" if not search_excerpt else "unverified_search_excerpt",
        })
        if len(sources) >= 50:
            break
    return sources


def _host_matches(host: str, suffix: str) -> bool:
    value = suffix.lower().lstrip(".")
    return host == value or host.endswith(f".{value}")


def _domain_authority(source: dict[str, Any], profile: str, policy: dict[str, Any]) -> tuple[float, str]:
    host = (urlsplit(str(source.get("url") or "")).hostname or "").lower()
    profile_policy = ((policy.get("profiles") or {}).get(profile) or {}) if isinstance(policy, dict) else {}
    for rule in profile_policy.get("domain_rules") or []:
        suffix = str(rule.get("suffix") or "")
        if suffix and _host_matches(host, suffix):
            return float(rule.get("weight", 0.5)), str(rule.get("tier") or "profile_rule")
    for rule in policy.get("global_domain_rules") or []:
        suffix = str(rule.get("suffix") or "")
        if suffix and _host_matches(host, suffix):
            return float(rule.get("weight", 0.5)), str(rule.get("tier") or "global_rule")
    if host.endswith(".gov.cn") or host == "gov.cn":
        return 0.95, "government"
    if host.endswith(".edu.cn"):
        return 0.72, "education"
    if host.endswith(".org.cn"):
        return 0.62, "organization"
    return float(policy.get("default_domain_weight", 0.42)), "unclassified"


def _query_terms(value: str) -> set[str]:
    latin = re.findall(r"[a-z0-9][a-z0-9._/-]{1,}", value.casefold())
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", value)
    chinese: set[str] = set()
    for run in chinese_runs:
        chinese.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
        chinese.update(run[index : index + 3] for index in range(max(0, len(run) - 2)))
    synonym_groups = (
        {"旧楼改造", "老旧小区改造", "旧改", "城市更新", "建筑改造"},
        {"规范", "标准", "规程", "条文"},
        {"厂家", "厂商", "制造商", "生产企业"},
    )
    for group in synonym_groups:
        if any(term in value for term in group):
            chinese.update(group)
    stop_terms = {"什么", "有什", "么旧", "问一", "一下", "哪些", "怎么", "如何", "是否"}
    return {term for term in [*latin, *chinese] if len(term) >= 2 and term not in stop_terms}


def _relevance_score(query: str, source: dict[str, Any]) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.5
    title = str(source.get("title") or "").casefold()
    excerpt = str(source.get("search_excerpt") or "").casefold()
    weighted_matches = sum(
        1.8 if term.casefold() in title else 1.0 if term.casefold() in excerpt else 0.0
        for term in terms
    )
    return min(1.0, weighted_matches / max(4.0, min(float(len(terms)), 12.0)))


def _recency_score(date_text: str, profile: str) -> float:
    if profile == "construction_standard":
        return 0.75
    years = re.findall(r"20\d{2}", date_text)
    if not years:
        return 0.45
    age = max(0, datetime.now().year - max(int(value) for value in years))
    return max(0.15, math.exp(-age / 3.0))


def _valid_provider_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _rank_sources(query: str, sources: list[dict[str, Any]], profile: str, *, force_original_order: bool = False) -> list[dict[str, Any]]:
    """Use only provider relevance for order; local metrics remain diagnostics.

    Missing/invalid scores switch the entire admitted set to provider order.
    Never silently treat a missing score as zero or mix incomparable rankings.
    """
    policy = _load_policy()
    complete_scores = bool(sources) and all(_valid_provider_score(source.get('provider_rerank_score')) for source in sources)
    use_scores = complete_scores and not force_original_order
    strategy = 'provider_rerank_score_desc_stable' if use_scores else 'provider_original_order_fallback'
    ranked: list[dict[str, Any]] = []
    for source in sources:
        authority, tier = _domain_authority(source, profile, policy)
        relevance = _relevance_score(query, source)
        recency = _recency_score(str(source.get("date") or ""), profile)
        final_score = float(source['provider_rerank_score']) if use_scores else None
        ranked.append({
            **source,
            "authority_score": round(authority, 4),
            "authority_tier": tier,
            "relevance_score": round(relevance, 4),
            "recency_score": round(recency, 4),
            "final_score": final_score,
            "ranking_strategy": strategy,
        })
    ranked.sort(key=lambda item: (-float(item['provider_rerank_score']) if use_scores else 0,
                                 int(item.get("original_rank") or 99)))
    for index, source in enumerate(ranked, start=1):
        source.setdefault("source_id", f"W{int(source.get('original_rank') or index)}")
        source["reranked_position"] = index
    return ranked


def _source_quality_summary(sources: list[dict[str, Any]]) -> dict[str, Any]:
    verified = sum(1 for item in sources if item.get("evidence_level") == "verified_page_content")
    official_excerpt = sum(1 for item in sources if item.get("evidence_level") == "official_search_excerpt")
    low_authority = sum(1 for item in sources if float(item.get("authority_score") or 0) < 0.55)
    authoritative_pages = sum(1 for item in sources if item.get('evidence_level') == 'verified_page_content'
                              and float(item.get('authority_score') or 0) >= 0.85)
    return {
        "source_count": len(sources),
        "verified_page_count": verified,
        "official_excerpt_count": official_excerpt,
        "low_authority_count": low_authority,
        "reliable_source_count": authoritative_pages + official_excerpt,
        "readable_source_count": verified + official_excerpt,
        "page_fetch_is_fact_verification": False,
        "requires_cautious_wording": bool(sources) and authoritative_pages + official_excerpt == 0,
    }


def _assert_public_url(raw_url: str) -> str:
    canonical = _canonical_url(raw_url)
    parts = urlsplit(canonical)
    if not canonical or not parts.hostname or parts.port not in {None, 80, 443}:
        raise ValueError("unsupported_url")
    from backend.sales.web_network import resolve_public_host
    resolve_public_host(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    return canonical


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _assert_public_url(urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            clean = re.sub(r"\s+", " ", data).strip()
            if clean:
                self.parts.append(clean)


def _download_small_public_text(url: str, *, timeout: float, max_bytes: int) -> tuple[str, str, str]:
    safe_url = _assert_public_url(url)
    from backend.sales.web_network import resolve_public_host
    parts = urlsplit(safe_url)
    _, proxy = resolve_public_host(parts.hostname, parts.port or (443 if parts.scheme=='https' else 80))
    # No Fake-IP => normal environment/direct behavior, even when a stale
    # local proxy setting remains after the VPN has been disabled.
    opener = (build_opener(ProxyHandler({'http':proxy,'https':proxy}), _SafeRedirectHandler())
              if proxy else build_opener(_SafeRedirectHandler()))
    request = Request(safe_url, headers={"User-Agent": "FacadeCopilotEvidenceVerifier/1.0"})
    with opener.open(request, timeout=timeout) as response:
        final_url = _assert_public_url(response.geturl())
        content_type = str(response.headers.get_content_type() or "").lower()
        if content_type not in {"text/html", "text/plain"}:
            raise ValueError(f"unsupported_content_type:{content_type or 'unknown'}")
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("page_too_large")
        encoding = response.headers.get_content_charset()
        if not encoding:
            marker = re.search(br'charset\s*=\s*[\"\x27]?([A-Za-z0-9_-]+)', body[:8192], re.I)
            encoding = marker.group(1).decode('ascii') if marker else 'utf-8'
        try:
            decoded = body.decode(encoding, errors='replace')
        except LookupError:
            decoded = body.decode('utf-8', errors='replace')
        return decoded, content_type, final_url


def _robots_permission(url: str, *, timeout: float = 3.0) -> tuple[bool, str]:
    parts = urlsplit(_assert_public_url(url))
    robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
    try:
        body, _, _ = _download_small_public_text(robots_url, timeout=timeout, max_bytes=200_000)
    except HTTPError as exc:
        return (True, "robots_not_present") if exc.code == 404 else (False, f"robots_http_{exc.code}")
    except Exception as exc:
        return False, f"robots_unavailable:{type(exc).__name__}"
    parser = RobotFileParser()
    parser.parse(body.splitlines())
    if not parser.can_fetch('FacadeCopilotEvidenceVerifier/1.0', url):
        return False, "robots_disallowed"
    return True, "robots_allowed"


def _extract_relevant_excerpt(page_text: str, query: str, limit: int = 1800) -> str:
    from backend.sales.web_content import extract_page_text, select_page_windows
    clean, _ = extract_page_text(page_text)
    return select_page_windows(clean, query, limit)[0]


def _verify_source_page(source: dict[str, Any], query: str) -> dict[str, Any]:
    if os.getenv("WEB_PAGE_VERIFICATION_ENABLED", "1").strip().lower() not in {"1", "true", "yes"}:
        return {**source, "page_fetch_status": "disabled"}
    try:
        allowed, robots_status = _robots_permission(str(source.get("url") or ""))
        if not allowed:
            return {**source, "page_fetch_status": robots_status}
        page_text, _, final_url = _download_small_public_text(
            str(source.get("url") or ""),
            timeout=float(os.getenv("WEB_PAGE_FETCH_TIMEOUT_SECONDS", "6")),
            max_bytes=int(os.getenv("WEB_PAGE_FETCH_MAX_BYTES", "1500000")),
        )
        from backend.sales.web_content import extract_page_text, select_page_windows
        parsed, parser_method = extract_page_text(page_text)
        excerpt, extraction_audit = select_page_windows(parsed, query)
        if len(excerpt) < 80:
            return {**source, "page_fetch_status": "insufficient_page_text", "verified_url": final_url}
        return {
            **source, "url": final_url, "verified_url": final_url,
            "verified_excerpt": excerpt, "excerpt": excerpt,
            "parsed_page_text": parsed[:100000],
            "extraction_audit": {**extraction_audit, 'parser': parser_method,
                                 'cache_truncated': len(parsed)>100000},
            "page_fetch_status": "verified", "evidence_level": "verified_page_content",
            "page_fetched_at_china": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        }
    except HTTPError as exc:
        status = f"http_{exc.code}"
    except URLError:
        status = "network_error"
    except (TimeoutError, socket.timeout):
        status = "timeout"
    except ValueError as exc:
        status = str(exc)[:80]
    except Exception as exc:
        status = f"fetch_failed:{type(exc).__name__}"
    return {**source, "page_fetch_status": status}


def _verify_top_sources(sources: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    top_k = max(0, min(3, int(os.getenv("WEB_PAGE_VERIFY_TOP_K", "2"))))
    if not sources or top_k == 0:
        return sources
    with ThreadPoolExecutor(max_workers=min(2, top_k), thread_name_prefix="web-verify") as pool:
        verified = list(pool.map(lambda source: _verify_source_page(source, query), sources[:top_k]))
    merged = [*verified, *sources[top_k:]]
    for source in merged:
        if source.get("evidence_level") == "verified_page_content":
            continue
        if source.get("search_excerpt") and float(source.get("authority_score") or 0) >= 0.85:
            source["evidence_level"] = "official_search_excerpt"
        elif source.get("search_excerpt"):
            source["evidence_level"] = "unverified_search_excerpt"
        else:
            source["evidence_level"] = "unavailable"
    return merged


def _call_baidu(full_question: str) -> dict[str, Any]:
    api_key = os.getenv("BAIDU_AI_SEARCH_API_KEY", "").strip()
    if not api_key:
        raise BaiduSearchError("BAIDU_AI_SEARCH_API_KEY is not configured.")
    header_name = os.getenv("BAIDU_AI_SEARCH_AUTH_HEADER", "X-Appbuilder-Authorization").strip()
    if header_name not in {"Authorization", "X-Appbuilder-Authorization"}:
        raise BaiduSearchError("BAIDU_AI_SEARCH_AUTH_HEADER must be Authorization or X-Appbuilder-Authorization.")
    body = {
        "messages": [{"role": "user", "content": full_question}],
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": max(5, min(50, int(os.getenv('WEB_SEARCH_RECALL_TOP_K', '20'))))}],
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
        raise BaiduSearchError(
            f"Baidu search returned HTTP {exc.code}: {detail}",
            error_code=f"http_{exc.code}",
            retryable=500 <= int(exc.code) < 600,
            http_status=int(exc.code),
        ) from exc
    except URLError as exc:
        raise BaiduSearchError(
            f"Baidu search network error: {exc.reason}",
            error_code="network_error",
            retryable=True,
        ) from exc
    except TimeoutError as exc:
        raise BaiduSearchError(
            "Baidu search response timed out.",
            error_code="timeout",
            retryable=True,
        ) from exc
    except json.JSONDecodeError as exc:
        raise BaiduSearchError(
            "Baidu search response was not valid JSON.",
            error_code="invalid_json",
        ) from exc
    if payload.get("code"):
        raise BaiduSearchError(
            str(payload.get("message") or payload["code"]),
            error_code=f"api_{payload.get('code')}",
        )
    return payload


def _call_baidu_with_retry(full_question: str) -> tuple[dict[str, Any], int]:
    """Retry only transient network, timeout and server-side failures once."""

    max_attempts = max(1, min(2, int(os.getenv("WEB_SEARCH_MAX_ATTEMPTS", "2"))))
    retry_delay = max(0.0, min(2.0, float(os.getenv("WEB_SEARCH_RETRY_DELAY_SECONDS", "0.35"))))
    for attempt in range(1, max_attempts + 1):
        try:
            return _call_baidu(full_question), attempt
        except BaiduSearchError as exc:
            exc.attempts = attempt
            if not exc.retryable or attempt >= max_attempts:
                raise
            if not reserve_recovery('public_web_search','retry_transient_network',minimum_seconds=22):
                raise
            check_budget()
            if retry_delay:
                time.sleep(retry_delay)
    raise AssertionError("unreachable")


def search_baidu_web(query: str, *, source_profile: str | None = None, bypass_cache: bool = False) -> dict[str, Any]:
    """Return admitted public evidence in provider-score/original order."""

    full_question = normalize_query(query)
    profile = resolve_source_profile(full_question, source_profile)
    cached = None if bypass_cache else _read_cache(full_question, profile)
    if cached is not None:
        cached_sources = list(cached.get('sources') or [])
        cached_sources, rejected = filter_sources(full_question, cached_sources, _load_policy())
        # An inadmissible cached result is never re-published or re-fetched.
        if not cached_sources:
            cached = None
    if cached is not None:
        # Refresh extraction after a parser/network-policy upgrade without
        # spending another search API call. Do it once per cached payload.
        if cached.get('page_extraction_version') != 2:
            cached_sources.sort(key=lambda s:int(s.get('original_rank') or 99))
            cached_sources = _verify_top_sources(cached_sources, full_question)
            cached = {**cached, 'sources':cached_sources, 'page_extraction_version':2}
            _write_cache(full_question, profile, cached)
        reranked = _rank_sources(full_question, cached_sources, profile,
                                force_original_order=cached.get('ranking_strategy') == 'provider_original_order_fallback')
        return {
            **cached,
            "sources": reranked,
            "source_quality": _source_quality_summary(reranked),
            "cache_hit": True,
            "quota": quota_snapshot(),
            "api_calls_for_query": 0,
            "quality_version": QUALITY_VERSION,
            "quality_rejections": [*(cached.get('quality_rejections') or []), *rejected],
            "status": "ok",
            "ranking_strategy": reranked[0]['ranking_strategy'] if reranked else 'provider_original_order_fallback',
        }

    usage_date = _china_date()
    quota = _reserve_api_call(usage_date=usage_date)
    try:
        payload, attempt_count = _call_baidu_with_retry(full_question)
    except Exception:
        _release_api_call(usage_date=usage_date)
        raise
    # Preserve Baidu order for deciding which two pages to fetch. Source
    # quality metadata is still attached; do not gate fetching on authority.
    candidates = _references_from_response(payload)
    admitted, rejected = filter_sources(full_question, candidates, _load_policy())
    sources = _rank_sources(full_question, admitted, profile)
    # Fetch selected leading candidates in the same provider-based order.
    sources = _verify_top_sources(sources, full_question)
    sources, post_rejected = filter_sources(full_question, sources, _load_policy())
    rejected.extend(post_rejected)
    sources = _rank_sources(full_question, sources, profile)
    result = {
        "query": full_question,
        "status": "ok" if sources else "no_usable_results",
        "quality_version": QUALITY_VERSION,
        "quality_rejections": rejected,
        "provider_request_id": str(payload.get('request_id') or '')[:120],
        "returned_reference_count": len(payload.get('references') or []),
        "admitted_reference_count": len(sources),
        "provider_score_missing_or_invalid_count": sum(not _valid_provider_score(s.get('provider_rerank_score')) for s in sources),
        "page_extraction_version": 2,
        "source_profile": profile,
        "sources": sources[:5],
        "cache_hit": False,
        "cache_bypassed": bypass_cache,
        "quota": quota,
        "ranking_strategy": sources[0]['ranking_strategy'] if sources else 'provider_original_order_fallback',
        "source_quality": _source_quality_summary(sources[:5]),
        "api_calls_for_query": 1,
        "api_attempts_for_query": attempt_count,
    }
    # Do not let a polluted/empty response survive for a day in the cache.
    if sources:
        _write_cache(full_question, profile, result)
    return result
