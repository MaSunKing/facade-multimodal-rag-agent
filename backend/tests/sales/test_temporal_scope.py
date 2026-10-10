import unittest

from backend.sales.query_constraints import resolve_query_time, mentioned_dates
from backend.sales.web_context import web_evidence_row, prepare_web_context
from backend.sales.task_support import AUDIT_PROMPT, REPLAN_PROMPT, FACT_MEANING_POLICY


class TemporalScopeTests(unittest.TestCase):
    def test_relative_days_across_year_boundary(self):
        self.assertEqual(resolve_query_time('昨天产量', '2026-01-01'), '2025-12-31产量')
        self.assertEqual(resolve_query_time('Tomorrow delivery', '2026-12-31'), '2027-01-01 delivery')

    def test_current_is_as_of_not_a_fact_date(self):
        self.assertEqual(resolve_query_time('最新标准', '2026-10-08'), '最新标准 当前日期 2026-10-08')

    def test_historical_scope_unchanged(self):
        query = '2024年10月报告与2025年10月报告比较'
        self.assertEqual(resolve_query_time(query, '2026-10-08'), query)

    def test_each_atomic_query_retains_its_target(self):
        for target in ['Object A', 'Object B']:
            result = resolve_query_time(f'{target} today output', '2026-10-08')
            self.assertEqual(result, f'{target} 2026-10-08 output')

    def test_literal_formats_do_not_invent_year(self):
        self.assertEqual(mentioned_dates('2026年10月8日 10/07 10/08 10月9日 2026-10-08'),
                         ['2026年10月8日', '10/07', '10/08', '10月9日', '2026-10-08'])

    def test_metadata_does_not_filter_current_row(self):
        source = {'source_id': 'W1', 'date': '2026-09-27', 'url': 'https://example.org',
                  'excerpt': '预报10/07至10/10\n10/08 产量20吨'}
        row = web_evidence_row(source)
        self.assertEqual(row['date_scope'], 'page_metadata_not_fact_date')
        self.assertIn('10/08', row['mentioned_dates'])
        self.assertIn('20吨', row['text'])
        context, _ = prepare_web_context([source], '2026-10-08')
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]['source_date_metadata'], '2026-09-27')

    def test_new_page_does_not_promote_old_fact_date(self):
        row = web_evidence_row({'source_id': 'W1', 'date': '2026-10-08',
                                'excerpt': '2024年10月8日 产量20吨'})
        self.assertEqual(row['mentioned_dates'], ['2024年10月8日'])

    def test_shared_prompts_preserve_scope_and_no_new_metrics(self):
        self.assertIn('所有来源', AUDIT_PROMPT)
        self.assertIn('对应行和值', AUDIT_PROMPT)
        self.assertIn('不改变用户原任务', REPLAN_PROMPT)
        self.assertIn('保留有效目标、维度', REPLAN_PROMPT)
        self.assertIn('区间起始日', FACT_MEANING_POLICY)


if __name__ == '__main__':
    unittest.main()
