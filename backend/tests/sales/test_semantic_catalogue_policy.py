import json
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch
from backend.app import DraftRequest, _CustomerAnswerWorkflowCallbacks
from backend.sales.tool_planner import ToolPlan
from backend.sales.evidence_gate import evidence_preview, budget_audit_payload
from backend.sales.visual_policy import select_visual_support, company_metadata_audit
from backend.sales.visual_labels import build_labels


class SemanticCataloguePolicyTests(unittest.TestCase):
    def test_audit_budget_keeps_whole_rows_and_does_not_mutate_source(self):
        payload={'question':'original task','evidence':[{'evidence_id':'E1','text':'a'*200},{'evidence_id':'E2','text':'b'*200}]}
        packed,audit=budget_audit_payload(payload,lambda value:len(json.dumps(value)),max_tokens=500)
        self.assertEqual(len(payload['evidence']),2)
        self.assertEqual(packed['question'],payload['question'])
        self.assertTrue(all(len(row['text'])==200 for row in packed['evidence']))
        self.assertLessEqual(audit['input_tokens'],500)
        self.assertTrue(audit['omitted_preview_ids'])

    def test_overview_preview_retains_identity_and_all_approved_sections(self):
        docs=json.loads(Path('data/sales/processed/rag_index/lexical_index.json').read_text(encoding='utf8'))['documents']
        evidence=[{'text':d['text'],'sales_playbook_use':d.get('sales_playbook_use'),
                   'citations':d.get('source_refs',[])} for d in docs if d.get('sales_playbook_use')=='approved_product_master_profile']
        rows=evidence_preview([{'text_evidence':evidence}], question='你们是谁，有哪些产品？',
            plan=ToolPlan(product_overview=True,answer_aspects=['介绍河北大自然的公司身份','介绍河北大自然的产品系列']))
        self.assertEqual(len(rows),11)
        self.assertTrue(any('河北大自然石材有限公司' in r['text'] for r in rows))
        self.assertTrue(any('装饰线条' in r['text'] for r in rows))
        self.assertLessEqual(sum(len(r['text']) for r in rows),6000)

    def test_goals_select_late_evidence_not_only_first_six(self):
        rows=evidence_preview([{'text_evidence':[{'text':'Generic background '+str(i)} for i in range(8)]+
                              [{'text':'Panel X thickness is 20 mm.'}]}],question='Panel X thickness?',
                              plan=ToolPlan(answer_aspects=['Panel X thickness']))
        self.assertTrue(any('20 mm' in row['text'] for row in rows))

    def test_healthy_company_gallery_does_not_require_pixels_for_uncertain_display(self):
        asset={'asset_id':'sample','image_labels':{'display_ready':True,'label_issues':[]}}
        selected,audit=select_visual_support([asset],{'visual_need':'uncertain'})
        self.assertEqual(selected,[])
        self.assertEqual(audit['metadata_only_display_asset_ids'],['company:sample'])
        selected,_=select_visual_support([asset],{'visual_need':'required'})
        self.assertEqual(selected,[asset])  # Actual diagram/detail demand overrides display labels.

    def test_disputed_company_labels_can_escalate_to_pixels(self):
        asset={'asset_id':'a','image_labels':{'display_ready':False,'label_issues':['multiple_product_bindings']}}
        selected,_=select_visual_support([asset],{'visual_need':'not_required'})
        self.assertEqual(selected,[asset])
        self.assertEqual(company_metadata_audit([asset])['disputed_label_asset_ids'],['a'])

    def test_real_labels_preserve_unknowns_and_do_not_claim_pixel_review(self):
        with sqlite3.connect('data/sales/processed/catalog/sales_catalog.sqlite3') as connection:
            labels=build_labels(connection)
        self.assertEqual(len(labels),471)
        self.assertTrue(all(not l['pixel_contents_verified'] for l in labels.values()))
        self.assertTrue(all(l['citation'] and l['label_basis'] for l in labels.values()))
        self.assertTrue(any(l['label_issues'] for l in labels.values()))

    def test_changed_query_executes_and_preserves_original_requirements(self):
        request=DraftRequest(customer_question='你们是谁，有哪些产品？')
        request._source_results={'company_rag':{'text':'old'}}
        original=ToolPlan(tools=['company_rag'],retrieval_query='old',product_overview=True,
                          answer_aspects=['企业身份','主要产品系列'])
        result={'action':'execute','retain_source_tools':['company_rag'],
                'revised_plan':{'tools':['company_rag'],'retrieval_query':'new'}}
        with patch('backend.app.evaluate_task_support',return_value=result),patch('backend.app.load_model',side_effect=AssertionError):
            revised=_CustomerAnswerWorkflowCallbacks.replan_from_feedback(request,original,{}, {'answerable':False})
        self.assertTrue(revised.product_overview)
        self.assertEqual(revised.answer_aspects,original.answer_aspects)
        self.assertEqual(request._retained_source_results,{})
        self.assertTrue(request._replan_audit['changed_query_requires_execution'])


if __name__=='__main__':unittest.main()
