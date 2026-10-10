import asyncio
import json
import os
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp.shared.memory import create_connected_server_and_client_session
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from backend.request_budget import RequestBudget, RequestCancelled, budget_scope, current_budget
from backend.sales import baidu_search as search
from backend.sales.search_mcp import build_search_server, search_web_via_mcp
from backend.sales.web_quality import filter_sources


def source(title, text='', url='https://example.com/page'):
    return {'source_id': 'W1', 'title': title, 'search_excerpt': text, 'excerpt': text, 'url': url, 'original_rank': 1}


class QualityTests(unittest.TestCase):
    def test_unrelated_adult_and_sports_results_do_not_become_evidence(self):
        accepted, rejected = filter_sources('查询2026年10月8日北京的实时气温', [
            source('激情视频网站2026最新版', '北京 气温'),
            source('郑钦文晋级中网八强', '2026年10月8日发布。'),
            source('豆包成人版最新版 iphone版', '北京'),
            source('一级片', '北京'),
        ])
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 4)
        self.assertNotIn('激情', json.dumps(rejected, ensure_ascii=False))

    def test_other_subjects_and_real_adult_reports_not_blanket_banned(self):
        for query, title in [('儿童防火安全', '儿童防火安全知识'), ('成人教育', '成人教育通知'),
                             ('手机软件最新版本', '手机软件最新版下载'), ('色情网站治理', '色情网站治理政策')]:
            accepted, _ = filter_sources(query, [source(title)])
            self.assertEqual(len(accepted), 1)

    def test_non_allowlisted_relevant_result_is_kept(self):
        accepted, _ = filter_sources('anchor panel installation', [source('anchor panel installation', 'details')])
        self.assertEqual(len(accepted), 1)
        self.assertFalse(accepted[0]['quality_is_fact_verification'])

    def test_unsafe_urls_and_configured_domains_rejected(self):
        for url in ['http://127.0.0.1/a', 'http://localhost/a', 'https://user:password@example.com/a', 'https://example.com:8080/a']:
            self.assertEqual(filter_sources('panel', [source('panel', url=url)])[0], [])
        self.assertEqual(filter_sources('panel', [source('panel')], {'blocked_domains':['example.com']})[0], [])

    def test_malformed_provider_url_is_not_internal_error(self):
        self.assertEqual(search._canonical_url('https://example.com:bad/a'), '')

    def test_provider_score_descending_and_stable_ties_all_profiles(self):
        rows = [{**source('panel'), 'original_rank':i, 'source_id':f'W{i}', 'provider_rerank_score':score}
                for i, score in enumerate([.4,.9,.9,0],1)]
        for profile in search.VALID_PROFILES:
            ranked = search._rank_sources('panel', rows, profile)
            self.assertEqual([r['source_id'] for r in ranked], ['W2','W3','W1','W4'])

    def test_any_invalid_score_uses_whole_set_original_order(self):
        for invalid in [None, '0.8', True, -1, 1.1, float('nan'), float('inf')]:
            rows = [{**source('panel'), 'original_rank':2,'provider_rerank_score':1},
                    {**source('panel'), 'original_rank':1,'provider_rerank_score':invalid}]
            ranked = search._rank_sources('panel', rows, 'general_public')
            self.assertEqual([r['original_rank'] for r in ranked], [1,2])
            self.assertEqual(ranked[0]['ranking_strategy'], 'provider_original_order_fallback')

    def test_provider_scores_and_raw_rank_survive(self):
        sources = search._references_from_response({'references': [
            {'url':'bad'}, {'id':'ref2','title':'panel','content':'panel','url':'https://example.com', 'rerank_score':0.73, 'authority_score':0.11}]})
        self.assertEqual(sources[0]['original_rank'], 2)
        self.assertEqual(sources[0]['provider_rerank_score'], 0.73)
        ranked = search._rank_sources('panel', sources, 'general_public')
        self.assertEqual(ranked[0]['source_id'], 'W2')

    def test_late_relevant_candidate_survives_polluted_first_five(self):
        raw = {'references': [
            {'url':f'https://example.com/{i}', 'title':'unrelated sports', 'content':'tennis results'} for i in range(5)
        ] + [{'url':'https://example.com/panel', 'title':'panel installation', 'content':'panel instructions'}]}
        admitted, rejected = filter_sources('panel installation', search._references_from_response(raw))
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]['original_rank'], 6)
        self.assertEqual(len(rejected), 5)


class MCPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'WEB_SEARCH_STATE_DB': str(Path(self.temp.name)/'state.sqlite3'),
                                         'WEB_PAGE_VERIFICATION_ENABLED': '0'})
        self.env.start()

    def tearDown(self):
        self.env.stop(); self.temp.cleanup()

    def test_initialize_list_and_call_tool_real_protocol(self):
        async def run():
            async with create_connected_server_and_client_session(build_search_server()) as session:
                tools = await session.list_tools()
                self.assertEqual([t.name for t in tools.tools], ['network_search'])
                self.assertIn('source_profile', tools.tools[0].inputSchema['properties'])
                invalid = await session.call_tool('network_search', {'query':'panel', 'source_profile':'invented'})
                self.assertTrue(invalid.isError)
        asyncio.run(run())

    def test_external_stdio_host_can_initialize_and_discover_tool(self):
        async def run():
            script = Path(__file__).resolve().parents[3]/'scripts/run_search_mcp.py'
            async with stdio_client(StdioServerParameters(command=sys.executable, args=[str(script)])) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.assertEqual([t.name for t in tools.tools], ['network_search'])
        asyncio.run(run())

    def test_mcp_roundtrip_preserves_evidence_and_shared_budget(self):
        budget = RequestBudget(float('inf'))
        def invoke(*args, **kwargs):
            self.assertIs(current_budget.get(), budget)
            return {'sources':[source('panel')], 'query':args[0]}
        with patch('backend.sales.search_mcp.search_baidu_web', side_effect=invoke), budget_scope(budget):
            result = search_web_via_mcp('panel')
        self.assertEqual(result['tool_transport'], 'mcp_in_process')
        self.assertEqual(result['sources'][0]['source_id'], 'W1')

    def test_quota_errors_preserved_no_raw_secret_error_body(self):
        with patch('backend.sales.search_mcp.search_baidu_web', side_effect=search.SearchQuotaExceeded('SECRET')):
            with self.assertRaises(search.SearchQuotaExceeded) as raised:
                search_web_via_mcp('panel')
        self.assertNotIn('SECRET', str(raised.exception))

    def test_cancelled_request_never_calls_upstream(self):
        budget = RequestBudget(float('inf'))
        with budget_scope(budget), patch('backend.sales.search_mcp.search_baidu_web') as call:
            budget.cancelled.set()
            with self.assertRaises(RequestCancelled):
                search_web_via_mcp('panel')
            call.assert_not_called()

    def test_polluted_response_not_fetched_or_cached(self):
        raw = {'references':[{'title':'激情视频网站','content':'气温 北京','url':'https://example.com/porn'}]}
        with patch.object(search, '_call_baidu', return_value=raw) as call, patch.object(search, '_verify_source_page') as fetch:
            first = search_web_via_mcp('北京实时气温')
            second = search_web_via_mcp('北京实时气温')
        self.assertEqual(call.call_count, 2)
        self.assertEqual(first['status'], 'no_usable_results')
        self.assertFalse(second['cache_hit'])
        self.assertEqual(first['sources'], [])
        fetch.assert_not_called()

    def test_realtime_cache_lifetime_is_short(self):
        search._write_cache('北京今天气温', 'general_public', {'sources':[source('北京气温')]})
        with search._connect_state_db() as conn:
            row = conn.execute('SELECT expires_at-created_at FROM web_search_cache').fetchone()
        self.assertLessEqual(row[0], 300)

    def test_selection_and_cached_fallback_remain_top_five(self):
        raw = {'references':[{'title':'panel', 'content':'panel', 'url':f'https://example.com/{i}',
                              **({'rerank_score':i/10} if i<6 else {})} for i in range(1,9)]}
        with patch.object(search, '_call_baidu', return_value=raw) as call:
            first = search_web_via_mcp('panel')
            cached = search_web_via_mcp('panel')
        self.assertEqual(call.call_count, 1)
        self.assertEqual([s['original_rank'] for s in first['sources']], [1,2,3,4,5])
        self.assertEqual([s['original_rank'] for s in cached['sources']], [1,2,3,4,5])
        self.assertEqual(cached['ranking_strategy'], 'provider_original_order_fallback')

    def test_valid_scores_select_highest_five_not_original_first_five(self):
        raw = {'references':[{'title':'panel', 'content':'panel', 'url':f'https://example.com/{i}',
                              'rerank_score':i/10} for i in range(1,9)]}
        with patch.object(search, '_call_baidu', return_value=raw):
            result = search_web_via_mcp('panel')
        self.assertEqual([s['original_rank'] for s in result['sources']], [8,7,6,5,4])

    def test_old_cache_namespace_is_not_used(self):
        self.assertNotEqual(search._cache_key('panel', 'general_public'),
                            __import__('hashlib').sha256(b'general_public\npanel').hexdigest())

    def test_cached_results_are_readmitted_without_publishing_garbage(self):
        search._write_cache('panel', 'general_public', {'page_extraction_version':2, 'sources':[
            source('panel installation'), source('激情视频网站', url='https://example.com/bad')]})
        with patch.object(search, '_call_baidu') as call:
            result = search_web_via_mcp('panel')
        self.assertEqual(len(result['sources']), 1)
        call.assert_not_called()
        self.assertTrue(result['cache_hit'])


if __name__ == '__main__':
    unittest.main()
