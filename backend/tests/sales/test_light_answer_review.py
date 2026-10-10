import unittest
from unittest.mock import patch
from backend.sales.answer_review import validate_review, review_schema, light_metadata, revision_feedback, ANSWER_AUDIT_PROMPT
from backend.sales.support_wire import audit_wire_payload


class LightReviewTests(unittest.TestCase):
    def test_review_inference_cannot_seed_revision(self):
        result=revision_feedback({'review_policy':'task_coverage_only','decision':'revise',
            'coverage_status':'partial','reason':'共享候选证明共享做法',
            'revision':{'requirement':'比较甲乙','reason':'应写两者均支持干挂'}})
        self.assertNotIn('干挂',str(result))
        self.assertNotIn('共享做法',str(result))
        self.assertEqual(result['requirement'],'比较甲乙')

    def test_candidates_only_have_counts(self):
        self.assertEqual(light_metadata({'candidate_state':'represented','evidence_ids':['T1','T2']}),
            {'candidate_state':'represented','candidate_count':2})

    def test_light_revision_reuses_full_context_once(self):
        import json
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks as C
        from backend.sales.tool_planner import ToolPlan
        from backend.tests.sales.test_audit_error_contract import text_audit_model
        request=DraftRequest(customer_question='比较甲乙')
        request._prepared_evidence=[{'evidence_id':'T1','text':'甲厚度18毫米。'},
                                    {'evidence_id':'T2','text':'乙厚度25毫米。'}]
        response={'customer_reply':'只说甲','meta':{'model_used':True,'answer_review':{
            'decision':'revise','coverage_status':'partial','review_policy':'task_coverage_only',
            'revision':{'requirement':'比较甲乙','reason':'遗漏乙'}}}}
        revised='甲18，乙25毫米。'
        approved={**response,'customer_reply':revised,'meta':{'answer_review':{'decision':'pass'}}}
        with text_audit_model([revised]),patch.object(C,'review_answer',return_value=approved) as review,patch.object(C,'replan_from_feedback',side_effect=AssertionError('no tools')):
            result=C.finish_review(request,ToolPlan(tools=['company_rag']),response)
        self.assertEqual(request._answer_revision_count,1)
        self.assertEqual(len(request._draft_context),2)
        self.assertEqual(result['customer_reply'],revised)
        self.assertEqual(review.call_args.args[2]['customer_reply'], revised)

    def payload(self):
        return {'review_policy':'task_coverage_only','question':'比较单板和一体板',
                'actual_tools':['company_rag'],'visual_candidates':[]}

    def test_pass_does_not_certify_facts(self):
        result=validate_review({'decision':'pass','coverage_status':'complete'},self.payload())
        self.assertFalse(result['fact_support_verified'])
        self.assertNotIn('supported_evidence_ids',result)

    def test_revision_needs_no_grounding_ids(self):
        result=validate_review({'decision':'revise','coverage_status':'partial',
            'revision':{'requirement':'比较单板和一体板','reason':'遗漏一方'}},self.payload())
        self.assertEqual(result['decision'],'revise')

    def test_unproven_execution_gap_is_limited(self):
        result=validate_review({'decision':'recover','coverage_status':'missing',
            'missing_aspects':['一体板'],'recovery_need':'lookup',
            'proof':{'requirement':'一体板','reason':'草稿没有提到',
                     'basis':'execution_gap','next_query':'一体板介绍'}},self.payload())
        self.assertEqual(result['decision'],'limited')

    def test_schema_has_no_source_quotes(self):
        props=review_schema(self.payload())['properties']
        self.assertNotIn('supported_evidence_ids',props)
        self.assertNotIn('support_spans',props)
        self.assertIn('company_capabilities',ANSWER_AUDIT_PROMPT)

    def test_exhausted_execution_budget_cannot_authorize_recovery(self):
        payload={**self.payload(),'execution_gaps':[{'state':'call_budget_exhausted'}]}
        result=validate_review({'decision':'recover','coverage_status':'missing',
            'missing_aspects':['一体板'],'recovery_need':'lookup',
            'proof':{'requirement':'一体板','reason':'执行预算耗尽',
                     'basis':'execution_gap','next_query':'一体板介绍'}},payload)
        self.assertEqual(result['decision'],'limited')
        self.assertEqual(result['reason'],'execution_gap_not_recoverable')

    def test_added_dimension_cannot_force_revision(self):
        result=validate_review({'decision':'revise','coverage_status':'partial',
            'revision':{'requirement':'价格','reason':'还应该加上价格'}},self.payload())
        self.assertEqual(result['decision'],'limited')

    def test_nested_metadata_drops_prose_pixels(self):
        result=light_metadata({'entities':[{'entity_id':'P1','text':'private original',
            'visual_description':'texture','image_path':'secret.png','name':'单板'}]})
        self.assertEqual(result,{'entities':[{'entity_id':'P1','name':'单板'}]})

    def test_wire_preserves_original_question(self):
        payload={**self.payload(),'attachment_task':{'operation':'summary','selection':'all','files':[]},
                 'planned_task':{'sources':['customer_documents']}}
        result=audit_wire_payload(payload,compact=True)
        self.assertEqual(result['question'],payload['question'])
        self.assertNotIn('evidence',result)
        self.assertIn('company_capabilities',result)
