"""CPU/mocked tests; these must not load any local model."""
import json
import time
import unittest
from contextlib import nullcontext, contextmanager
from pathlib import Path
from unittest.mock import patch

from backend.request_budget import RequestBudget, budget_scope, reserve_recovery
from backend.sales.runtime_status import ExecutionStatus, current_execution, classify, model_tool_errors
from backend.sales.evidence_gate import validate_evaluation
from backend.sales.tool_planner import ToolPlan, retrieval_query_for_attempt


@contextmanager
def text_audit_model(outputs):
    import torch
    from transformers import BatchEncoding
    from backend.tests.sales.test_model_contract import Tokenizer as BaseTokenizer, Model
    class Tokenizer(BaseTokenizer):
        eos_token_id=0
        def apply_chat_template(self,*args,**kwargs): return 'mock'
        def encode(self,text,**kwargs): return [ord(c) for c in text]
        def __call__(self,text,**kwargs): return BatchEncoding(super().__call__(text,**kwargs))
    model=Model(outputs); model.device='cpu'
    with patch('backend.app.load_model',return_value=(Tokenizer(),model)), patch('backend.app.generation_session',side_effect=nullcontext):
        yield model


class AuditErrorContractTests(unittest.TestCase):
    def test_actual_audit_callback_reaches_model_for_all_source_types(self):
        # Exercise payload construction + real packing, not just the lower-level
        # model helper. This catches missing imports before any model invocation.
        from backend.app import DraftRequest, _CustomerAnswerWorkflowCallbacks
        from backend.sales.gap_recovery import gap_state
        for tool, prefix in [('public_web_search','W'), ('company_rag','T'), ('customer_documents','U')]:
            with self.subTest(tool=tool):
                request = DraftRequest(customer_question='Describe current conditions', use_online_search=True,
                                       interrupted_question='Describe previous conditions')
                request._prepared_evidence = [
                    {'evidence_id':prefix+'2', 'text':'Current conditions are recorded.', 'source_type':tool},
                    {'evidence_id':prefix+'1', 'text':'Earlier unrelated conditions.', 'source_type':tool}]
                request._gap_state = gap_state(request.customer_question,
                    {'status':'partial', 'missing_aspects':['Old ID diagnostic']})
                request._replan_audit = {'feedback':{'missing_aspects':['Old ID diagnostic']}}
                plan = ToolPlan(tools=[tool], answer_aspects=['Current conditions'])
                sources = {tool: ([], {}) if tool=='public_web_search' else {}}
                request._source_results=sources
                with text_audit_model([]), patch('backend.app.evaluate_task_support', return_value={
                    'decision':'pass','status':'supported', 'supported_evidence_ids':[prefix+'2']}) as evaluate:
                    result = _CustomerAnswerWorkflowCallbacks.review_answer(request, plan,
                        {'customer_reply':'Current conditions are recorded.', 'meta':{'model_used':True}},2)
                self.assertEqual(result['meta']['answer_review']['decision'],'pass')
                evaluate.assert_called_once()
                payload = evaluate.call_args.args[1]
                self.assertEqual(payload['task_update']['latest_requirement'], request.customer_question)
                self.assertEqual(payload['task_update']['interrupted_question'], request.interrupted_question)
                self.assertNotIn('prior_audit', payload)
                self.assertNotIn('Old ID diagnostic', str(payload))
                self.assertEqual(request._evidence_gate_audit['status'], 'supported')
                self.assertNotIn('error_code', request._evidence_gate_audit)
                # The authoritative draft Context is retained; review guidance
                # is applied only to a bounded revision, not a new retrieval.
                self.assertEqual([r['evidence_id'] for r in request._prepared_evidence], [prefix+'2',prefix+'1'])

    def test_initial_original_and_recovery_complete_query(self):
        question = '介绍我们的山东项目，给项目图片，不要施工图。'
        plan = ToolPlan(retrieval_query='项目 图片', tools=['company_rag'])
        self.assertEqual(retrieval_query_for_attempt(question, plan), question)
        plan = plan.model_copy(update={'retrieval_query':'公司在山东的项目名称、介绍与项目照片，不要施工图',
                                      'recovery_directive':{'action':'semantic_replan','problem':'Missing project photos'}})
        self.assertTrue(retrieval_query_for_attempt(question, plan).startswith(question))
        self.assertEqual(retrieval_query_for_attempt(question, plan, public=True), question)
        plan = plan.model_copy(update={'web_retrieval_query': plan.retrieval_query})
        self.assertEqual(retrieval_query_for_attempt(question, plan, public=True), plan.web_retrieval_query)

    def test_types_are_not_coerced_into_false_feedback(self):
        for extra in [{'missing_aspects':'scope'}, {'missing_aspects':[{}]},
                      {'web_can_resolve':'false'}, {'supported_evidence_ids':{'E1':True}}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_evaluation({'status':'uncertain', **extra}, [{'evidence_id':'E1'}])

    def test_exact_failure_codes_and_no_exception_text_leak(self):
        cases = [('invalid_visual_audit_reference','AUDIT_INVALID_REFERENCE'),
                 ('invalid_evaluation_missing_aspects','AUDIT_SCHEMA_INVALID'),
                 ('audit_input_budget_exceeded','AUDIT_INPUT_BUDGET_EXCEEDED')]
        cases += [('context_budget_exceeded','CONTEXT_BUDGET_EXCEEDED'),
                  ('uploaded_prompt_budget_exceeded','CONTEXT_BUDGET_EXCEEDED')]
        for message, code in cases:
            result = classify(ValueError(message), 'evidence_evaluation')
            self.assertEqual(result['code'], code)
            self.assertNotIn(message, json.dumps(result))

    def test_format_repair_once_then_success_uses_shared_slot(self):
        from backend.app import evaluate_task_support, DraftRequest
        budget = RequestBudget(time.monotonic()+120)
        with budget_scope(budget), text_audit_model(['{"status":', '{"status":"supported","supported_evidence_ids":["VL1"]}']) as model:
            result = evaluate_task_support(DraftRequest(customer_question='Describe layout'), {'evidence':[{'evidence_id':'VL1','text':'visible structure'}]})
            self.assertEqual(len(model.calls),2)
            self.assertEqual(result['status'],'supported')
            self.assertFalse(result['pixels_reviewed'])
            self.assertTrue(reserve_recovery('assess_retry','semantic_replan',minimum_seconds=1))

    def test_invalid_json_is_failure_not_missing_evidence(self):
        from backend.app import evaluate_task_support, DraftRequest
        journal = ExecutionStatus()
        token = current_execution.set(journal)
        try:
            with budget_scope(RequestBudget(time.monotonic()+120)), text_audit_model(['{"status":']*3) as model:
                result = evaluate_task_support(DraftRequest(customer_question='Describe layout'), {})
            self.assertEqual(len(model.calls),3)
            self.assertEqual(result['error_code'], 'OUTPUT_INVALID_JSON')
            self.assertEqual(journal.errors[0]['stage'], 'evidence_evaluation')
            self.assertIn('message', model_tool_errors()[0])
            self.assertEqual(result['status'], 'uncertain')
        finally:
            current_execution.reset(token)

    def test_low_budget_skips_format_repair(self):
        from backend.app import evaluate_task_support, DraftRequest
        with budget_scope(RequestBudget(time.monotonic()+60)), text_audit_model(['{"status":']*3) as model:
            result = evaluate_task_support(DraftRequest(customer_question='Describe layout'), {})
        self.assertEqual(len(model.calls),3)
        self.assertEqual(result['error_code'], 'OUTPUT_INVALID_JSON')

    def test_output_cap_is_reported_as_truncation_not_no_support(self):
        from backend.app import evaluate_task_support, DraftRequest
        with budget_scope(RequestBudget(time.monotonic()+60)), text_audit_model(['{'+' '*(n-1) for n in (384,576,864)]):
            result = evaluate_task_support(DraftRequest(customer_question='Panel features?'), {})
        self.assertEqual(result['error_code'], 'OUTPUT_TOKEN_LIMIT')
        self.assertEqual(result['status'],'uncertain')

    def test_source_error_reasons_survive_recovery_diagnostics(self):
        from backend.sales.recovery_policy import source_diagnostics, assess_source_gaps
        statuses = source_diagnostics({'public_web_search':([], {'status':'failed','error_code':'timeout'})})
        self.assertEqual(statuses['public_web_search']['state'], 'transient_error')
        plan = ToolPlan(tools=['public_web_search'], requires_public_web=True)
        decision = assess_source_gaps(plan, statuses, web_allowed=True)
        self.assertEqual(decision.gaps[0]['error_code'], 'WEB_TIMEOUT')
        self.assertEqual(source_diagnostics({'company_rag':{'text_evidence':'bad'}})['company_rag']['state'],'invalid_input')

    def test_permission_guard_explains_removed_tool(self):
        from backend.sales.tool_planner import guard_plan
        plan = guard_plan({'tools':['public_web_search'], 'requires_public_web':True},
                          has_documents=False,has_image=False,facade_related=False,web_allowed=False)
        self.assertEqual(plan.guard_diagnostics[0]['code'], 'WEB_NOT_AUTHORISED')
        self.assertNotIn('public_web_search', plan.tools)

    def test_network_button_is_a_hard_permission_boundary(self):
        from backend.app import maybe_search_online, DraftRequest
        with patch('backend.app.search_baidu_web', side_effect=AssertionError('no network')):
            sources, meta = maybe_search_online(DraftRequest(customer_question='Project details',use_online_search=False),
                                               'Project details', automatic_named_project_lookup=True)
        self.assertEqual(sources, [])
        self.assertEqual(meta['status'], 'not_requested')

    def test_request_deadline_is_not_converted_to_semantic_gap(self):
        from backend.app import evaluate_task_support, DraftRequest
        from backend.request_budget import RequestBudgetExceeded
        with patch('backend.app.load_model', side_effect=RequestBudgetExceeded('request_budget_exceeded')):
            with self.assertRaises(RequestBudgetExceeded):
                evaluate_task_support(DraftRequest(customer_question='Describe layout'),{})

    def test_oom_during_audit_is_also_technical_failure(self):
        from backend.sales.task_support import audit_execution_failed
        self.assertTrue(audit_execution_failed({'error_code':'MODEL_OOM','reason':'support_audit_failed'}))
        self.assertFalse(audit_execution_failed({'status':'unsupported','problem':'Different entity'}))

    def test_company_first_actual_step_uses_original_query(self):
        from backend.app import DraftRequest, _run_facade_rag_answer
        question = '我们的山东项目有哪些？介绍并给图片。'
        plan = ToolPlan(tools=['company_rag'], retrieval_query='项目 图片')
        cursor = _run_facade_rag_answer.steps(DraftRequest(customer_question=question),tool_plan=plan,allow_public_web=False)
        try:
            step = next(cursor)
            self.assertEqual(step.node, 'company_rag')
            self.assertEqual(step.kwargs['query'], question)
        finally:
            cursor.close()

    def test_attachment_first_actual_step_uses_original_query(self):
        from backend.app import DraftRequest, _run_customer_document_answer
        question = '这份材料限制浸水吗？解释限制条件。'
        cursor = _run_customer_document_answer.steps(DraftRequest(customer_question=question),
                    ToolPlan(tools=['customer_documents'],retrieval_query='材料 浸水'))
        try:
            step = next(cursor)
            self.assertEqual(step.node, 'customer_documents')
            self.assertEqual(step.args[1], question)
        finally:
            cursor.close()
