from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.sales import baidu_search


class BaiduSearchPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {
                "BAIDU_AI_SEARCH_API_KEY": "test-key",
                "WEB_SEARCH_STATE_DB": str(Path(self.temp_dir.name) / "state.sqlite3"),
                "WEB_SEARCH_DAILY_BUSINESS_LIMIT": "45",
                "WEB_SEARCH_DAILY_HARD_LIMIT": "50",
                "WEB_PAGE_VERIFICATION_ENABLED": "0",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.temp_dir.cleanup()

    def test_profile_classification(self) -> None:
        self.assertEqual(
            baidu_search.classify_source_profile("JGJ 336幕墙规范条文"),
            "construction_standard",
        )
        self.assertEqual(
            baidu_search.classify_source_profile("济南某项目的中标施工单位是谁"),
            "public_project",
        )

    def test_one_api_call_is_cached_and_official_source_is_promoted(self) -> None:
        response = {
            "references": [
                {
                    "type": "web",
                    "title": "个人文章解读",
                    "url": "https://example.com/post?utm_source=test",
                    "content": "幕墙规范的个人转载内容。",
                    "date": "2026-01-01",
                },
                {
                    "type": "web",
                    "title": "住房和城乡建设部正式资料",
                    "url": "https://www.mohurd.gov.cn/gongkai/standard.html",
                    "content": "住房和城乡建设部发布的幕墙规范正式资料。",
                    "date": "2025-01-01",
                },
            ]
        }
        with patch.object(baidu_search, "_call_baidu", return_value=response) as api_call:
            first = baidu_search.search_baidu_web("幕墙规范有什么要求")
            second = baidu_search.search_baidu_web("幕墙规范有什么要求")

        self.assertEqual(api_call.call_count, 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["api_calls_for_query"], 1)
        self.assertEqual(second["api_calls_for_query"], 0)
        self.assertEqual(first["source_profile"], "construction_standard")
        self.assertIn("mohurd.gov.cn", first["sources"][0]["url"])
        self.assertEqual(first["quota"]["used"], 1)

    def test_daily_business_limit_stops_before_second_api_call(self) -> None:
        os.environ["WEB_SEARCH_DAILY_BUSINESS_LIMIT"] = "1"
        response = {"references": []}
        with patch.object(baidu_search, "_call_baidu", return_value=response) as api_call:
            baidu_search.search_baidu_web("第一个不同查询")
            with self.assertRaises(baidu_search.SearchQuotaExceeded):
                baidu_search.search_baidu_web("第二个不同查询")
        self.assertEqual(api_call.call_count, 1)

    def test_transient_failure_retries_once_and_releases_local_quota(self) -> None:
        failure = baidu_search.BaiduSearchError(
            "temporary network failure",
            error_code="network_error",
            retryable=True,
        )
        with patch.object(baidu_search, "_call_baidu", side_effect=failure) as api_call:
            with self.assertRaises(baidu_search.BaiduSearchError) as raised:
                baidu_search.search_baidu_web("哈尔滨今天的天气")

        self.assertEqual(api_call.call_count, 2)
        self.assertEqual(raised.exception.attempts, 2)
        self.assertEqual(baidu_search.quota_snapshot()["used"], 0)

    def test_auth_failure_is_not_retried_and_releases_local_quota(self) -> None:
        failure = baidu_search.BaiduSearchError(
            "authentication failed",
            error_code="http_401",
            retryable=False,
            http_status=401,
        )
        with patch.object(baidu_search, "_call_baidu", side_effect=failure) as api_call:
            with self.assertRaises(baidu_search.BaiduSearchError):
                baidu_search.search_baidu_web("哈尔滨今天的天气")

        self.assertEqual(api_call.call_count, 1)
        self.assertEqual(baidu_search.quota_snapshot()["used"], 0)

    def test_private_network_url_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            baidu_search._assert_public_url("http://127.0.0.1/internal")


if __name__ == "__main__":
    unittest.main()
