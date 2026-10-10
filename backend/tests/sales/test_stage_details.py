import json
import unittest
from types import SimpleNamespace
from backend.request_budget import RequestBudgetExceeded
from backend.sales.runtime_status import ExecutionStatus, current_execution
from backend.sales.stage_details import begin,end,safe
from backend.sales.answer_graph import build_customer_answer_graph
from backend.sales.staged_execution import WorkflowStep
from backend.sales.tool_planner import ToolPlan
from backend.sales.visual_policy import select_visual_support


class StageDetailsTests(unittest.TestCase):
    def test_plan_and_tool_details_survive_later_generation_failure(self):
        class Callbacks:
            def plan_tools(self, request):
                return ToolPlan(tools=['company_rag'],retrieval_query='Complete project query',answer_aspects=['Projects and photos'])
            def open_steps(self,request,plan):
                yield WorkflowStep('company_rag',lambda **kwargs:{'text_evidence':[{'evidence_id':'T1','text':'private attachment body'}]},kwargs={'query':request.customer_question})
                yield WorkflowStep('generate_answer')
                raise RequestBudgetExceeded('request_budget_exceeded')
        journal=ExecutionStatus(); token=current_execution.set(journal)
        try:
            with self.assertRaises(RequestBudgetExceeded):
                build_customer_answer_graph(Callbacks()).invoke({'request':SimpleNamespace(customer_question='Project description and photos')})
            rows=journal.snapshot()['stage_details']
            self.assertEqual(rows[0]['output']['parsed_plan']['retrieval_query'],'Complete project query')
            rag=next(r for r in rows if r['stage']=='company_rag')
            self.assertEqual(rag['input']['query'],'Project description and photos')
            self.assertEqual(rag['output']['result']['text_evidence_count'],1)
            self.assertEqual(rows[-1]['error_code'],'REQUEST_TIMEOUT')
            self.assertNotIn('private attachment body',json.dumps(rows))
        finally: current_execution.reset(token)

    def test_bounded_and_redacted(self):
        journal=ExecutionStatus(); token=current_execution.set(journal)
        try:
            for index in range(65): end(begin('example',{'query':'token=secret-value','image_bytes':b'pixels'}), {'count':index})
            rows=journal.snapshot()['stage_details']
            self.assertEqual(len(rows),60)
            self.assertNotIn('secret-value',json.dumps(rows))
            self.assertNotIn('pixels',json.dumps(rows))
        finally: current_execution.reset(token)

    def test_company_display_does_not_turn_into_pixels_on_generic_required_flag(self):
        asset={'asset_id':'case-photo','image_labels':{'display_ready':True,'label_issues':[]}}
        selected,audit=select_visual_support([asset],{'visual_need':'required','visual_purpose':'display_only'})
        self.assertEqual(selected,[])
        self.assertFalse(audit['required_but_missing'])
        self.assertEqual(audit['metadata_only_display_asset_ids'],['company:case-photo'])

    def test_company_detail_and_label_dispute_never_reopen_pixels(self):
        asset={'asset_id':'node','image_labels':{'display_ready':True,'label_issues':[]}}
        selected,_=select_visual_support([asset],{'visual_need':'required','visual_purpose':'content_inspection'})
        self.assertEqual(selected,[])
        asset['image_labels']['label_issues']=['source_label_conflict']
        selected,_=select_visual_support([asset],{'visual_need':'required','visual_purpose':'label_dispute'})
        self.assertEqual(selected,[])

    def test_customer_image_does_not_inherit_company_label_shortcut(self):
        asset={'document_id':'d','visual_id':'p','image_labels':{'display_ready':True,'label_issues':[]}}
        selected,_=select_visual_support([asset],{'visual_need':'required','visual_purpose':'content_inspection'})
        self.assertEqual(selected,[asset])
