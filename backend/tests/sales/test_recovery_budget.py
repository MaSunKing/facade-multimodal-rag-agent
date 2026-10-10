import unittest
from unittest.mock import patch
from backend.request_budget import RequestBudget, budget_scope
from backend.documents.summaries import SummaryBudget
from backend.sales.recovery_budget import budget_view, recovery_allowed, validate_budget_view
from backend.sales.model_contract import validate_audit, validate_replan
from backend.sales.output_contract import OutputContractError


class RecoveryBudgetTests(unittest.TestCase):
    def resources(self):
        b=RequestBudget.for_answer(has_attachments=True)
        b.summary_state=SummaryBudget(max_calls=1,calls=1)
        return b,budget_view(b, {'budget_exhausted':True},tools=['customer_documents'])

    def test_exhausted_summary_does_not_exhaust_lookup(self):
        _,r=self.resources()
        self.assertFalse(recovery_allowed(r,'summary'))
        self.assertFalse(recovery_allowed(r,'unknown'))
        self.assertTrue(recovery_allowed(r,'lookup'))
        self.assertFalse(recovery_allowed(r,'web'))
        self.assertEqual(r['summary']['remaining_calls'],0)

    def test_reserves_final_time_and_shared_recovery_slot(self):
        import time
        b=RequestBudget(time.monotonic()+80)
        self.assertFalse(budget_view(b,tools=['company_rag'])['recovery_available'])
        b=RequestBudget.for_answer(has_attachments=True)
        b.recoveries=[{'stage':'task_support_audit','action':'semantic_replan'}]
        self.assertFalse(budget_view(b,tools=['company_rag'])['lookup_available'])

    def test_customer_visual_uses_existing_shared_slot_without_direct_upload(self):
        b=RequestBudget.for_answer(has_attachments=True)
        view=budget_view(b,tools=['customer_documents'],customer_visual_available=True)
        self.assertTrue(recovery_allowed(view,'visual'))
        self.assertFalse(budget_view(b,tools=['company_rag'],customer_visual_available=True)['visual_available'])
        b.recoveries=[{'stage':'task_support_audit','action':'semantic_replan'}]
        self.assertFalse(budget_view(b,tools=['customer_documents'],customer_visual_available=True)['visual_available'])

    def test_budget_fields_strict_not_model_authority(self):
        _,r=self.resources()
        for broken in [dict(r,summary_available='false'),dict(r,authority='model'),
                       dict(r,summary={'remaining_calls':True}),dict(r,final_reserve_seconds=float('nan')),
                       dict(r,summary_available=True),dict(r,recovery_available=False)]:
            with self.assertRaises(ValueError): validate_budget_view(broken)

    def test_audit_need_and_extra_budget_fields_checked(self):
        _,r=self.resources()
        base={'status':'partial','supported_evidence_ids':['U1'],'missing_aspects':['condition']}
        for update in [{},{'recovery_need':[]},{'recovery_need':'invalid'},
                       {'recovery_need':'lookup','recovery_budget':r}]:
            with self.assertRaises(OutputContractError): validate_audit(dict(base,**update),['U1'],recovery_budget=r)
        validate_audit(dict(base,recovery_need='lookup'),['U1'],recovery_budget=r)

    def test_replan_summary_blocked_lookup_allowed(self):
        _,r=self.resources()
        plan={'action':'execute','gap_ids':['g'],'revised_plan':{'tools':['customer_documents'],'document_operation':'summary'}}
        with self.assertRaises(OutputContractError):
            validate_replan(plan,allowed_tools=['customer_documents'],gap_ids=['g'],recovery_budget=r)
        plan['revised_plan']['document_operation']='lookup'
        validate_replan(plan,allowed_tools=['customer_documents'],gap_ids=['g'],recovery_budget=r)

    def test_exhaustion_guard_runs_before_qwen_replan(self):
        from backend.app import DraftRequest,_CustomerAnswerWorkflowCallbacks,ToolPlan
        b,_=self.resources()
        request=DraftRequest(customer_question='Summarize all attachments')
        request._summary_audit={'budget_exhausted':True}
        with budget_scope(b),patch('backend.app.evaluate_task_support') as model:
            result=_CustomerAnswerWorkflowCallbacks.replan_from_feedback(request,ToolPlan(tools=['customer_documents']),
                {'status':'partial','recovery_need':'summary'}, {})
        self.assertIsNone(result)
        model.assert_not_called()
        self.assertFalse(request._replan_audit['invoked'])


if __name__=='__main__': unittest.main()
