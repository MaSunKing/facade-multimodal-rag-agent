import unittest
from backend.sales.evidence_expansion import expand_with_budget, native_provider, needs_expansion
from backend.sales.context_engine import pack_task_context


class EvidenceExpansionTests(unittest.TestCase):
    def test_all_operations_use_budget_first_whole_blocks(self):
        for operation in ('lookup','summary','comparison'):
            rows = [{'evidence_id':'T1','text':'一、安装','section_heading':'一、安装'}]
            provider = lambda row:[{'chunk_id':'step','text':'1. 固定支撑，不得在未验收基层上安装。',
                                   'citations':[{'document_name':'A','source_page':2}]}]
            count = lambda rows:50 + sum(len(r['text']) for r in rows)
            kept,audit = pack_task_context(rows,question='Explain installation',operation=operation,
                expansion_provider=provider,audit_tokens=lambda _:0,generation_tokens=count,ceiling=200)
            self.assertIn('不得',kept[0]['text'])
            self.assertTrue(audit['budget_satisfied'])
            self.assertFalse(kept[0]['expansion_status']['body_complete'])
            self.assertEqual(kept[0]['citations'][0]['source_page'],2)
            self.assertEqual(rows[0]['text'],'一、安装')

    def test_overflow_rolls_back_new_step_not_original(self):
        row = {'evidence_id':'E1','text':'一、固定'}
        rows,audit = expand_with_budget([row],provider=lambda _:[{'chunk_id':'a',
            'text':'1. 紧固前必须确认材料、基层与荷载条件均符合设计要求。'}],
            token_count=lambda rows:sum(len(r['text']) for r in rows),token_limit=12)
        self.assertEqual(rows[0]['text'],row['text'])
        self.assertEqual(audit['decisions'][0]['reason'],'budget_boundary_rollback')
        self.assertTrue(audit['budget_satisfied'])

    def test_full_budget_does_not_read_native_source(self):
        def forbidden(row):
            raise AssertionError('provider must be lazy')
        rows,audit = expand_with_budget([{'evidence_id':'E1','text':'一、固定'}],
            provider=forbidden,token_count=lambda _:100,token_limit=100)
        self.assertEqual(audit['decisions'][0]['reason'],'budget_insufficient')

    def test_admitted_native_source_only_and_next_section_boundary(self):
        docs=[{'id':'h','kind':'text','text':'一、安装','source_refs':[{'document_name':'A','source_page':1,'bbox':[0,0]}]},
              {'id':'s','kind':'text','text':'钢架固定。','source_refs':[{'document_name':'A','source_page':1,'bbox':[0,10]}]},
              {'id':'n','kind':'text','text':'二、清洁','source_refs':[{'document_name':'A','source_page':1,'bbox':[0,20]}]},
              {'id':'p','kind':'text','text':'不可借用。','source_refs':[{'document_name':'B','source_page':1,'bbox':[0,5]}]}]
        source={'company_rag':{'text_evidence':[{'chunk_id':'h'}]}}
        provider=native_provider(source,company_documents=docs)
        self.assertEqual([c['chunk_id'] for c in provider({'evidence_id':'T1'})],['s'])
        self.assertEqual(provider({'evidence_id':'T2'}),[])
        self.assertIsNone(needs_expansion({'text':'一、安装','facts_eligible':False}))

    def test_table_and_numeric_condition_are_atomic(self):
        row={'evidence_id':'U1','text':'一、表格'}
        block='[COLUMNS] 条件 | 值\n[ROW] 温度低于10℃时不得施工 | 20mm'
        rows,audit=expand_with_budget([row],provider=lambda _:[{'text':block}],
            token_count=lambda rows:sum(len(r['text']) for r in rows),token_limit=20)
        self.assertEqual(rows[0]['text'],row['text'])


if __name__ == '__main__':
    unittest.main()
