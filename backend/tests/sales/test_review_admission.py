import copy
import unittest
from backend.sales.review_budget import admit_review, fill_admitted_review, admit_actual_review
from backend.sales.support_wire import audit_wire_text

class ReviewAdmissionTests(unittest.TestCase):
    def test_actual_draft_not_maximum_output_controls_admission(self):
        payload={'question':'q','final_draft':{'customer_reply':'short answer'},
                 'image_labels':[{'labels':{'description':'red','uncertainties':['brand unknown']}}]}
        count=lambda p:len(audit_wire_text(p))
        value,audit=admit_actual_review(payload,count,retry_reserve=20,ceiling=1000)
        self.assertEqual(value,payload)
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual(audit['required_input_tokens'],count(value))
        self.assertEqual(audit['format_retry_reserved_tokens'],0)
        self.assertNotIn('draft_output_reserved_tokens',audit)

    def test_first_review_fits_even_when_hypothetical_retry_would_not(self):
        payload={'final_draft':{'customer_reply':'full draft'},'image_labels':[{'labels':['texture']}]}
        original=copy.deepcopy(payload)
        value,audit=admit_actual_review(payload,lambda _:7550,retry_reserve=617)
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual(audit['required_input_tokens'],7550)
        self.assertEqual(value,original)

    def test_budget_failure_classified_as_budget_not_internal_error(self):
        from backend.sales.runtime_status import classify
        result=classify(ValueError('answer_review_actual_input_budget_exceeded'),stage='answer_audit')
        self.assertEqual(result['code'],'CONTEXT_BUDGET_EXCEEDED')

    def test_actual_overflow_compacts_metadata_without_changing_draft_or_labels(self):
        payload={'final_draft':{'customer_reply':'keep entire answer'},
                 'field_guide':'repeated glossary '*100,
                 'image_labels':[{'labels':{'uncertainties':['unknown identity']}}]}
        count=lambda p:len(audit_wire_text(p))
        value,audit=admit_actual_review(payload,count,retry_reserve=20,ceiling=800)
        self.assertTrue(audit['metadata_compacted'])
        self.assertEqual(value['final_draft'],payload['final_draft'])
        self.assertEqual(value['image_labels'],payload['image_labels'])
        with self.assertRaises(ValueError):
            admit_actual_review({'final_draft':{'customer_reply':'x'*900}},count,retry_reserve=20,ceiling=800)
    def test_display_only_card_does_not_introduce_unseen_catalogue_facts(self):
        from backend.sales.review_execution import review_image_labels
        rows=review_image_labels([], [{'asset_id':'x','title':'panel',
            'image_labels':{'title':'panel','caption':'unseen long catalogue description',
                'uncertainties':['brand not confirmed']}}])
        self.assertEqual(rows[0]['labels']['uncertainties'],['brand not confirmed'])
        self.assertNotIn('caption',rows[0]['labels'])

    def test_label_wire_factoring_preserves_every_value_and_identity(self):
        from backend.sales.support_wire import factor_label_records
        labels=[{'asset_id':str(i),'document_id':'d'+str(i),
            'labels':{'description':'texture '+str(i),'uncertainties':['brand unknown']}} for i in range(10)]
        packed=factor_label_records(labels)
        restored=[]
        for group in packed['groups']:
            for identity,values in group['rows']:
                restored.append({**dict(zip(group['identity_columns'],identity)),
                    'labels':dict(zip(group['label_columns'],values))})
        self.assertEqual(restored,labels)
        self.assertLess(len(str(packed)),len(str(labels)))

    def test_failed_admission_exposes_exact_required_budget(self):
        with self.assertRaises(ValueError) as caught:
            admit_review({'image_labels':[{'description':'x'*8000}]},lambda p:len(audit_wire_text(p)),
                output_limit=3072,retry_reserve=500)
        audit=caught.exception.budget_audit
        self.assertGreater(audit['required_input_tokens'],audit['input_ceiling'])
        self.assertEqual(audit['draft_output_reserved_tokens'],3072)

    def test_general_preflight_receives_plan_before_model_generation(self):
        from unittest.mock import patch
        import torch
        import backend.app as app
        from backend.sales.tool_planner import ToolPlan
        class Inputs(dict):
            def to(self, device): return self
            @property
            def input_ids(self): return self['input_ids']
        class Tokenizer:
            eos_token_id=0
            def encode(self,text,**kw): return [0]*(len(text)//4)
            def apply_chat_template(self,messages,**kw): return str(messages)
            def __call__(self,text,**kw):
                return Inputs(input_ids=torch.zeros((1, max(1,len(text)//4)),dtype=torch.long))
        request=app.DraftRequest(customer_question='a public information request')
        plan=ToolPlan(tools=['public_web_search'])
        request._source_results={'public_web_search':([],{'status':'completed'})}
        # Reject admission deliberately: inference must never precede this guard.
        from types import SimpleNamespace
        with patch.object(app,'load_model',return_value=(Tokenizer(),SimpleNamespace(device='cpu'))), \
             patch.object(app,'prepare_answer_review_admission',side_effect=ValueError('context_budget_exceeded')) as admission, \
             patch.object(app,'checked_generate') as generate:
            app.general_local_chat_answer(request,[],{'status':'completed'},tool_plan=plan)
        admission.assert_not_called()
        generate.assert_called_once()

    def test_customer_display_witness_is_admitted_before_final(self):
        from unittest.mock import patch
        from backend.app import DraftRequest, prepare_answer_review_admission, _CustomerAnswerWorkflowCallbacks as C
        from backend.sales.tool_planner import ToolPlan
        class Tokenizer:
            def encode(self,text,**kw): return [0]*(len(text)//4)
            def apply_chat_template(self,messages,**kw): return str(messages)
        request=DraftRequest(customer_question='show file pictures')
        request._source_results={'customer_documents':{}}
        assets=[{'asset_id':'customer:d:v','document_name':'file'}]
        def build(req,plan,response,round_number,**kw):
            self.assertTrue(kw['prepare_only'])
            self.assertEqual(response['visual_assets'],assets)
            return {'review_policy':'task_coverage_only','final_draft':{},'image_delivery':assets}
        with patch.object(C,'review_answer',side_effect=build):
            prepare_answer_review_admission(request,ToolPlan(tools=['customer_documents']),[],Tokenizer(),display_assets=assets)
        self.assertEqual(request._answer_audit_payload['image_delivery'],assets)
        self.assertTrue(request._answer_audit_admission['budget_satisfied'])

    def test_reserves_complete_answer_and_retry_before_generation(self):
        payload={'review_policy':'task_coverage_only','question':'both objects',
                 'final_draft':{'customer_reply':''},'image_labels':[{'asset_id':'client:1','labels':['red']} ]}
        count=lambda p:len(audit_wire_text(p))
        base,audit=admit_review(payload,count,output_limit=3072,retry_reserve=500,ceiling=5000)
        self.assertTrue(audit['budget_satisfied'])
        filled, actual=fill_admitted_review(base,'x'*3072,count,audit)
        self.assertLessEqual(actual,4500)
        self.assertEqual(filled['image_labels'],payload['image_labels'])
        self.assertEqual(base['final_draft']['customer_reply'],'')

    def test_metadata_overflow_is_rejected_without_dropping_labels(self):
        payload={'question':'q','image_labels':[{'description':'x'*8000}]}
        original=copy.deepcopy(payload)
        with self.assertRaisesRegex(ValueError,'admission_budget'):
            admit_review(payload,lambda p:len(audit_wire_text(p)),output_limit=3072,retry_reserve=500)
        self.assertEqual(payload,original)

    def test_wire_does_not_json_escape_draft_or_mutate_payload(self):
        reply='"quoted"\npath\\value\n'*10
        payload={'review_policy':'task_coverage_only','final_draft':{'customer_reply':reply}}
        original=copy.deepcopy(payload)
        text=audit_wire_text(payload)
        self.assertIn(reply,text)
        self.assertEqual(payload,original)

    def test_exact_guard_never_truncates_or_requests_regeneration(self):
        base,audit=admit_review({},lambda p:len(audit_wire_text(p)),output_limit=100,retry_reserve=20,ceiling=600)
        with self.assertRaisesRegex(ValueError,'invariant_failed'):
            fill_admitted_review(base,'x'*1000,lambda p:len(audit_wire_text(p)),audit)
