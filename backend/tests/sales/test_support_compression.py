import copy
import unittest
from backend.sales.support_compression import compress_support
from backend.sales.prepared_support import fit_support_budget


class SupportCompressionTests(unittest.TestCase):
    def test_extract_keeps_complete_numeric_condition_and_source(self):
        text = '无关背景介绍。\n' * 15 + '目标容量为20 L，不适用于浸水。\n仅限干燥环境。\n' + '历史背景材料。\n' * 15
        rows = [{'evidence_id':'E1','document_id':'d1','text':text}]
        original=copy.deepcopy(rows)
        selected,audit=compress_support(rows,'目标容量',max_chars=80)
        self.assertIn('20 L，不适用于浸水',selected[0]['text'])
        self.assertIn('仅限干燥环境',selected[0]['text'])
        self.assertLess(len(selected[0]['text']),len(text))
        self.assertEqual(rows,original)
        self.assertEqual(selected[0]['document_id'],'d1')
        self.assertTrue(audit['decisions'][0]['source_spans'])
        self.assertEqual(audit['model_calls'],0)

    def test_protected_modalities_and_groups_are_unchanged(self):
        for flag in [{'visual_description':True},{'derived_summary':True},
                     {'packing_group_id':'conflict_1'}, {'protected_relation_types':['condition']},
                     {'text':'[COLUMNS] A=Revenue\n[ROW] A1=20\n'*30}]:
            row={'evidence_id':'E1','text':'主题20 mm，不适用于浸水。\n'*30,**flag}
            selected,_=compress_support([row],'主题',max_chars=30)
            self.assertEqual(selected[0]['text'],row['text'])

    def test_summary_target_is_soft_not_silent_file_loss(self):
        rows=[{'evidence_id':str(i),'document_id':str(i),'document_name':str(i),
               'text':'must preserve','required_document_coverage':True,'derived_summary':True} for i in range(4)]
        kept,audit=fit_support_budget(rows,audit_tokens=lambda _:6000,generation_tokens=lambda _:6100,
                                     ceiling=8000,question='overview',target_tokens=4500)
        self.assertEqual(len(kept),4)
        self.assertTrue(audit['budget_satisfied'])
        self.assertFalse(audit['compression_target_satisfied'])

    def test_decimal_is_not_split_and_cross_language_retains_unmatched(self):
        text='Rotation period 11.5 hours. Only in dry conditions.\n'+'Background information.\n'*20
        kept,_=compress_support([{'evidence_id':'E1','text':text}], 'Rotation',max_chars=50)
        self.assertIn('11.5 hours.',kept[0]['text'])
        kept,_=compress_support([{'evidence_id':'E1','text':text}], '自转周期',max_chars=50)
        self.assertIn('11.5 hours.',kept[0]['text'])

    def test_repeated_events_not_removed_as_boilerplate(self):
        text='An event occurred.\n'*5
        kept,_=compress_support([{'evidence_id':'E1','text':text}], 'event')
        self.assertEqual(kept[0]['text'],text)
        kept,_=compress_support([{'evidence_id':'E1','text':text,'repeat_is_boilerplate':True}], 'event')
        self.assertLess(len(kept[0]['text']),len(text))

    def test_empty_metadata_keeps_false_and_zero(self):
        from backend.sales.evidence_packing import nonempty_metadata
        self.assertEqual(nonempty_metadata({'missing':None,'empty':[], 'pixels_supplied':False,'count':0}),
                         {'pixels_supplied':False,'count':0})

    def test_dedup_only_same_source_span_not_conflicting_values(self):
        base={'document_id':'d','original_chunk_id':'chunk','source_refs':['p1'],'text':'Capacity is 20 L.'}
        rows=[{**base,'evidence_id':'E1'},{**base,'evidence_id':'E2','protected_goal_ids':['g2']},
              {**base,'evidence_id':'E3','text':'Capacity is 30 L.'}]
        kept,audit=compress_support(rows,'Capacity')
        self.assertEqual([r['evidence_id'] for r in kept],['E1','E3'])
        self.assertEqual(kept[0]['protected_goal_ids'],['g2'])
        self.assertEqual(audit['decisions'][1]['duplicate_of'],'E1')


if __name__=='__main__':unittest.main()
