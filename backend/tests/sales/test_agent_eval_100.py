import queue
import time
import unittest
from unittest.mock import patch
from collections import Counter
from backend.request_budget import RequestBudget, RequestBudgetExceeded, budget_scope
from backend.sales.runtime_status import ExecutionStatus,current_execution
from backend.sales.stage_details import begin,end
from backend.sales.token_usage import usage
from scripts.build_agent_eval_100 import cases
from scripts.run_agent_eval_100 import await_result,run_case


class AgentEvalTests(unittest.TestCase):
    def test_distribution_and_unique_ids(self):
        rows=cases()
        self.assertEqual(len(rows),100)
        self.assertEqual(len({r['id'] for r in rows}),100)
        self.assertEqual(Counter(r['group'] for r in rows),
                         {'company':40,'attachment':20,'web':15,'multiturn':20,'fault':5})
        self.assertTrue(all(r['timeout_seconds']==1200 and r['rubric'] for r in rows))

    def test_normal_chat_unlimited_evaluation_opt_in(self):
        self.assertFalse(RequestBudget.for_answer(has_attachments=False).has_deadline)
        budget=RequestBudget.for_answer(has_attachments=False,timeout_seconds=.001)
        self.assertTrue(budget.has_deadline)
        with patch('backend.request_budget.time.monotonic',return_value=budget.deadline+1):
            with self.assertRaises(RequestBudgetExceeded):
                with budget_scope(budget): pass

    def test_absolute_deadline_not_socket_inactivity(self):
        with self.assertRaises(TimeoutError): await_result(queue.Queue(),time.monotonic()+.001)

    def test_timeout_cancels_before_next_case(self):
        class Client:
            def request(self,*a,**kw): raise TimeoutError('case_total_deadline')
            def cancel_and_wait(self,ids): self.ids=ids; return False
        client=Client()
        row=run_case(client,{'id':'fixture','group':'company','question':'q','rubric':[]},.01)
        self.assertEqual(row['status'],'timeout')
        self.assertFalse(row['cancellation_confirmed'])
        self.assertEqual(client.ids,row['client_task_ids'])

    def test_queue_label_does_not_claim_overlap_when_first_finished(self):
        class Client:
            def request(self,*a,**kw): return {'answerable':True,'customer_reply':'ok'}
            def bounded(self,*a,**kw): return {'active':False}
        row=run_case(Client(),{'id':'fixture','group':'multiturn','question':'q',
                              'turns':['q','r'],'mode':'queue','rubric':[]})
        self.assertEqual(row['status'],'completed')
        self.assertFalse(row['queue_exercised'])

    def test_retries_are_counted_not_only_valid_attempt(self):
        result=usage([{'attempts':[{'input_tokens':10,'output_tokens':5,'status':'invalid'},
                                  {'input_tokens':20,'output_tokens':8,'status':'valid'}]}])
        self.assertEqual((result['input_tokens'],result['output_tokens'],result['total_tokens']), (30,13,43))

    def test_failed_generation_is_not_zero_measured_output(self):
        result=usage([{'attempts':[{'input_tokens':10,'status':'failed'}]}])
        self.assertFalse(result['complete']); self.assertEqual(result['unknown_output_calls'],1)

    def test_stage_usage_cpu_zero_and_model_usage_visible(self):
        journal=ExecutionStatus(); token=current_execution.set(journal)
        try:
            cpu=begin('directory'); end(cpu)
            handle=begin('plan_request')
            journal.model_contracts.append({'attempts':[{'input_tokens':22,'output_tokens':3}]})
            end(handle,{'status':'ok'})
            snapshot=journal.snapshot()
        finally: current_execution.reset(token)
        self.assertEqual(snapshot['stage_details'][0]['token_usage']['total_tokens'],0)
        self.assertEqual(snapshot['stage_details'][1]['output']['token_usage']['total_tokens'],25)
        self.assertEqual(snapshot['token_usage']['total_tokens'],25)
        self.assertNotIn('_contract_start',snapshot['stage_details'][1])

    def test_live_usage_not_a_completed_zero(self):
        journal=ExecutionStatus(); token=current_execution.set(journal)
        try:
            begin('audit')
            journal.model_contracts.append({'attempts':[{'input_tokens':42,'status':'running'}]})
            snapshot=journal.snapshot()
        finally: current_execution.reset(token)
        self.assertFalse(snapshot['stage_details'][0]['output']['token_usage']['complete'])


if __name__=='__main__': unittest.main()
