"""Protocol tests with CPU tensors only: never load Qwen or use network."""
import json
import unittest
from types import SimpleNamespace
import torch
from backend.sales.model_contract import checked_generate, json_contract, validate_plan, validate_audit, validate_replan, fields_contract
from backend.sales.output_contract import OutputContractError
from backend.sales.runtime_status import ExecutionStatus,current_execution,classify
from backend.request_budget import RequestBudget,budget_scope


class Tokenizer:
    eos_token='<end>'
    def __call__(self,text,**kwargs): return {'input_ids':torch.tensor([[ord(c) for c in text]])}
    def decode(self,tokens,**kwargs): return ''.join(chr(int(t)) for t in tokens)


class Model:
    def __init__(self,outputs): self.outputs=list(outputs);self.calls=[]
    def generate(self,**kwargs):
        self.calls.append(kwargs)
        result=self.outputs.pop(0)
        return torch.cat((kwargs['input_ids'],torch.tensor([[ord(c) for c in result]])),dim=-1)


class ModelContractTests(unittest.TestCase):
    def test_retry_only_compacts_after_actual_feedback_overflows(self):
        model=Model(['{}','{"ok":true}']); tokenizer=Tokenizer()
        contract=fields_contract('test',tokenizer,{'ok':bool})
        contract.max_input_tokens=20
        calls=[]
        def compact(failed,feedback,device):
            calls.append((failed,feedback))
            return {'input_ids':torch.tensor([[1,2,3]],device=device),
                    'attention_mask':torch.ones(1,3,dtype=torch.long,device=device)}
        contract.retry_input_builder=compact
        checked_generate(model,contract=contract,input_ids=torch.tensor([[1,2]]),
            attention_mask=torch.ones(1,2,dtype=torch.long),max_new_tokens=100)
        self.assertEqual(len(model.calls),2)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],'{}')
        self.assertEqual(model.calls[1]['input_ids'].shape,model.calls[1]['attention_mask'].shape)

    def test_retry_still_over_limit_stops_without_another_model_call(self):
        model=Model(['{}']); tokenizer=Tokenizer()
        contract=fields_contract('test',tokenizer,{'ok':bool}); contract.max_input_tokens=20
        contract.retry_input_builder=lambda *_:{'input_ids':torch.zeros(1,21,dtype=torch.long)}
        with self.assertRaises(OutputContractError) as caught:
            checked_generate(model,contract=contract,input_ids=torch.tensor([[1,2]]),max_new_tokens=100)
        self.assertEqual(caught.exception.code,'CONTRACT_RETRY_BUDGET_EXCEEDED')
        self.assertEqual(len(model.calls),1)

    def test_multimodal_format_retry_extends_aligned_text_token_masks(self):
        model=Model(['{}','{"ok":true}']); tokenizer=Tokenizer()
        checked_generate(model,contract=fields_contract('test',tokenizer,{'ok':bool}),
            input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long),
            mm_token_type_ids=torch.tensor([[0,1]]),max_new_tokens=100)
        self.assertEqual(len(model.calls),2)
        retry=model.calls[1]
        self.assertEqual(retry['input_ids'].shape,retry['mm_token_type_ids'].shape)
        self.assertEqual(retry['input_ids'].shape,retry['attention_mask'].shape)
        self.assertTrue(torch.equal(retry['mm_token_type_ids'][0,:2],torch.tensor([0,1])))
        self.assertTrue(bool((retry['mm_token_type_ids'][0,2:]==0).all()))

    def test_lossless_comma_repair_preserves_complete_values(self):
        from backend.sales.model_contract import object_json
        self.assertEqual(object_json('{"action":"execute" "gap_ids":["G1" "G2",],}'),
                         {'action':'execute','gap_ids':['G1','G2']})
        for raw in ('{"action":"execute"', '{"reason":"bad "quote" here"}',
                    '{"value":}', '{"value":[1,2}'):
            with self.assertRaises(OutputContractError):
                object_json(raw)

    def test_retry_reserve_covers_candidate_and_protocol_without_accumulating(self):
        from backend.sales.model_contract import format_retry_reserve
        reserve = format_retry_reserve(len, 'protocol', max_failed_output_tokens=512)
        from backend.sales.model_contract import protocol_feedback
        measured=len(protocol_feedback('OUTPUT_FIELD_VALUE_INVALID','supported_evidence_ids',
            {'json_error':"Expecting ',' delimiter",'line':999,'column':99999,'position':99999})+'protocol\n')
        self.assertEqual(reserve,512+measured+64)
        self.assertLessEqual((6500-reserve)+512+measured+64,6500)

    def test_initial_company_plan_requires_operation_and_category_field(self):
        with self.assertRaises(OutputContractError):
            validate_plan({'tools':['company_rag']}, require_company_scope=True)
        value={'tools':['company_rag'],'knowledge_categories':[], 'company_operation':'summary','company_collection':{'action':'none'}}
        self.assertEqual(validate_plan(value, require_company_scope=True), value)
        # Recovery may inherit unchanged fields rather than restating them.
        self.assertEqual(validate_plan({'tools':['company_rag']}), {'tools':['company_rag']})

    def test_initial_company_plan_requires_operation_and_category_field(self):
        with self.assertRaises(OutputContractError):
            validate_plan({'tools':['company_rag']}, require_company_scope=True)
        value={'tools':['company_rag'],'knowledge_categories':[], 'company_operation':'summary','company_collection':{'action':'none'}}
        self.assertEqual(validate_plan(value, require_company_scope=True), value)
        # Recovery may inherit unchanged fields rather than restating them.
        self.assertEqual(validate_plan({'tools':['company_rag']}), {'tools':['company_rag']})

    def test_audit_real_gap_missing_recovery_need_retries(self):
        from backend.sales.model_contract import support_contract
        from backend.sales.recovery_budget import budget_view
        from backend.sales.task_support import AUDIT_OUTPUT_CONTRACT, AUDIT_RETRY_CONTRACT, AUDIT_PROMPT
        resources=budget_view(None,tools=['company_rag'])
        model=Model(['{"status":"partial","supported_evidence_ids":["T1"],"missing_aspects":["条件"]}',
                     '{"status":"partial","supported_evidence_ids":["T1"],"missing_aspects":["条件"],"recovery_need":"lookup"}'])
        tokenizer=Tokenizer()
        contract=support_contract(tokenizer,{'evidence':[{'evidence_id':'T1'}],'recovery_budget':resources})
        output=checked_generate(model,contract=contract,input_ids=torch.tensor([[1,2]]),max_new_tokens=384)
        self.assertEqual(len(model.calls),2)
        retry=tokenizer.decode(model.calls[1]['input_ids'][0,2:])
        self.assertIn(AUDIT_RETRY_CONTRACT,retry)
        self.assertIn('OUTPUT_FIELD_MISSING',retry)
        self.assertIn('recovery_need',retry)
        self.assertEqual(json.loads(tokenizer.decode(output[0,2:]))['recovery_need'],'lookup')
        self.assertTrue(AUDIT_PROMPT.endswith(AUDIT_OUTPUT_CONTRACT))

    def test_audit_examples_follow_minimal_contract(self):
        import re
        from backend.sales.task_support import AUDIT_PROMPT
        examples=re.findall(r'\{[^{}]+\}',AUDIT_PROMPT)
        self.assertGreaterEqual(len(examples),3)
        for example in examples:
            value=json.loads(example)
            self.assertIn('status',value)
            self.assertIn('supported_evidence_ids',value)
            if value.get('missing_aspects'):
                self.assertIsInstance(value['recovery_need'],str)

    def test_supported_audit_omission_does_not_retry(self):
        model=Model(['{"status":"supported","supported_evidence_ids":["T1"]}'])
        checked_generate(model,contract=json_contract('audit',Tokenizer(),lambda v:validate_audit(v,['T1'])),
                         input_ids=torch.tensor([[1]]),max_new_tokens=100)
        self.assertEqual(len(model.calls),1)

    def test_first_input_budget_blocks_without_call(self):
        model=Model(['{}'])
        with self.assertRaises(OutputContractError) as caught:
            checked_generate(model,contract=json_contract('test',Tokenizer(),max_input_tokens=1),
                             input_ids=torch.tensor([[1,2]]),max_new_tokens=10)
        self.assertEqual(caught.exception.code,'CONTEXT_BUDGET_EXCEEDED')
        self.assertFalse(model.calls)

    def run_model(self,outputs,**limits):
        model=Model(outputs);tokenizer=Tokenizer();journal=ExecutionStatus()
        token=current_execution.set(journal)
        budget=RequestBudget.for_answer(has_attachments=True)
        try:
            with budget_scope(budget):
                output=checked_generate(model,contract=json_contract('test',tokenizer,**limits),
                    input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long),max_new_tokens=512)
        finally: current_execution.reset(token)
        return model,output,journal,budget

    def test_first_valid_is_single_call(self):
        model,output,journal,budget=self.run_model(['{}'])
        self.assertEqual(len(model.calls),1);self.assertEqual(journal.model_contracts[0]['retries_used'],0)
        self.assertEqual(budget.recoveries,[])

    def test_two_retries_same_prefix_preserved(self):
        model,output,journal,budget=self.run_model(['{','[]','{"ok":true}'])
        self.assertEqual(len(model.calls),3)
        self.assertEqual(Tokenizer().decode(output[0,2:]),'{"ok":true}')
        self.assertEqual(model.calls[-1]['input_ids'][0,:2].tolist(),[1,2])
        self.assertEqual(budget.recoveries,[])
        self.assertNotIn('raw_output',json.dumps(journal.snapshot()))

    def test_third_failure_is_specific(self):
        model=Model(['[]']*3)
        with self.assertRaises(OutputContractError) as caught:
            checked_generate(model,contract=json_contract('planner',Tokenizer()),input_ids=torch.tensor([[1]]),max_new_tokens=20)
        self.assertEqual(len(model.calls),3)
        self.assertEqual(caught.exception.code,'OUTPUT_NOT_OBJECT')
        self.assertEqual(classify(caught.exception)['attempts'],3)

    def test_retry_cannot_bypass_input_budget(self):
        with self.assertRaises(OutputContractError) as caught: self.run_model(['{','{}'],max_input_tokens=2)
        self.assertEqual(caught.exception.code,'CONTRACT_RETRY_BUDGET_EXCEEDED')

    def test_image_tensors_preserved_and_position_state_not_reused(self):
        model=Model(['{','{}']);pixels=object();grid=object()
        checked_generate(model,contract=json_contract('visual',Tokenizer()),input_ids=torch.tensor([[1,2]]),
            pixel_values=pixels,image_grid_thw=grid,position_ids=object(),max_new_tokens=20)
        self.assertIs(model.calls[1]['pixel_values'],pixels);self.assertIs(model.calls[1]['image_grid_thw'],grid)
        self.assertNotIn('position_ids',model.calls[1])

    def test_permission_failure_no_format_retry(self):
        model=Model(['{"tools":["public_web_search"]}'])
        with self.assertRaises(OutputContractError) as caught:
            checked_generate(model,contract=json_contract('planner',Tokenizer(),
                lambda v:validate_plan(v,allowed_tools=['company_rag'])),input_ids=torch.tensor([[1]]),max_new_tokens=70)
        self.assertEqual(caught.exception.code,'ACCESS_DENIED');self.assertEqual(len(model.calls),1)

    def test_summary_retry_counts_against_shared_reservation(self):
        reserves=[]
        self.run_model(['{','{}'],retry_guard=lambda i,o:reserves.append((i,o)))
        self.assertEqual(len(reserves),1)

    def test_runtime_source_identity_not_model_writable(self):
        for fields in ({'owner':'a'},{'recovery_directive':{'x':1}},{'planner_cache_hit':True}):
            with self.assertRaises(OutputContractError): validate_plan({'tools':['company_rag'],**fields})
        with self.assertRaises(OutputContractError): validate_plan({'tools':['customer_documents'],'target_document_ids':['foreign']},document_ids=['owned'])

    def test_audit_ids_types_enums_not_semantics(self):
        validate_audit({'status':'partial','supported_evidence_ids':['E1'],'missing_aspects':['specified condition'],'recovery_need':'lookup'},['E1'])
        for v in ({'status':'supported','supported_evidence_ids':['foreign']},
                  {'status':'partial','supported_evidence_ids':['E1'],'web_can_resolve':'yes'},
                  {'status':'supported','supported_evidence_ids':['E1'],'visual_need':'maybe'}):
            with self.assertRaises(OutputContractError): validate_audit(v,['E1'])
        with self.assertRaises(OutputContractError):
            validate_audit({'status':'partial','supported_evidence_ids':[], 'visual_need':'required',
                            'candidate_asset_ids':['customer:a']},[],['customer:a'])
        validate_audit({'status':'supported','supported_evidence_ids':['V1']},['V1'])

    def test_replan_binds_given_gap_ids_and_documents(self):
        v={'action':'execute','gap_ids':['g1'],'revised_plan':{'tools':['customer_documents'],'target_document_ids':['d1']}}
        validate_replan(v,document_ids=['d1'],allowed_tools=['customer_documents'],gap_ids=['g1'])
        for field in ('gap_ids','revised_plan'):
            broken=dict(v);broken[field]=[]
            with self.assertRaises(OutputContractError): validate_replan(broken,document_ids=['d1'],allowed_tools=['customer_documents'],gap_ids=['g1'])

    def test_generic_secondary_schema(self):
        contract=fields_contract('arbitrary_role',Tokenizer(),{'action':str},{'items':list},{'action':['continue','stop']})
        contract.validate('{"action":"continue","items":[]}')
        with self.assertRaises(OutputContractError): contract.validate('{"action":42}')


if __name__=='__main__': unittest.main()
