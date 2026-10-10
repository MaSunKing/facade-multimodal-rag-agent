import unittest
from types import SimpleNamespace
from backend.sales.tool_planner import ToolPlan
from backend.sales.task_delivery import image_delivery, relation_delivery
from backend.sales.content_taxonomy import audit_capability_map, planner_capability_map
from backend.documents.document_tasks import resolve_documents
from backend.sales.model_contract import object_json, validate_plan
from backend.sales.output_contract import OutputContractError
from backend.sales.runtime_status import classify


class TaskDeliveryTests(unittest.TestCase):
    def test_default_all_is_required_even_for_lookup(self):
        docs = [SimpleNamespace(document_id=k, file_name=k) for k in ['a','b']]
        selected, required = resolve_documents(docs, ['a'], scope='local_lookup', operation='lookup', selection='all')
        self.assertEqual([d.document_id for d in selected], ['a','b'])
        self.assertEqual([d['document_id'] for d in required], ['a','b'])

    def test_explicit_selection_and_unknown_acl(self):
        docs = [SimpleNamespace(document_id=k, file_name=k) for k in ['a','b']]
        selected, _ = resolve_documents(docs, ['b'], scope='whole_document', operation='summary', selection='explicit')
        self.assertEqual([d.document_id for d in selected], ['b'])
        with self.assertRaises(ValueError):
            resolve_documents(docs, [], scope='local_lookup', operation='lookup', selection='explicit')
        with self.assertRaises(ValueError):
            resolve_documents(docs, ['foreign'], scope='local_lookup', operation='lookup', selection='all')

    def test_explicit_requires_owned_ids(self):
        with self.assertRaises(OutputContractError):
            validate_plan({'tools':['customer_documents'], 'document_selection':'explicit'})

    def test_policy_guard_preserves_selection(self):
        from backend.sales.tool_planner import guard_plan
        plan=ToolPlan(tools=['customer_documents'], document_selection='explicit', target_document_ids=['b'])
        guarded=guard_plan(plan, has_documents=True, has_image=False, facade_related=False, web_allowed=False)
        self.assertEqual(guarded.document_selection, 'explicit')
        self.assertEqual(guarded.target_document_ids, ['b'])

    def test_summary_requirements_bind_each_file_explicitly(self):
        from backend.documents.document_tasks import file_aspects
        from backend.documents.summary_context import required_document_view
        aspects, bindings=file_aspects(['原问题的维度'], [
            {'document_id':'a','document_name':'一'}, {'document_id':'b','document_name':'二'}], operation='summary')
        self.assertIn('概括主要内容', bindings[1]['requirement'])
        self.assertEqual(required_document_view(bindings)[1]['requirement'], bindings[1]['requirement'])

    def test_image_decision_and_availability_are_separate(self):
        assets=[{'evidence_id':'V1','asset_id':'one','title':'节点图'}]
        self.assertEqual(image_delivery(ToolPlan(wants_visuals=True), assets)['available_count'],1)
        self.assertFalse(image_delivery(ToolPlan(), assets)['requested'])
        self.assertEqual(image_delivery(ToolPlan(wants_visuals=True), [])['available_count'],0)

    def test_shared_map_no_inventory_facts(self):
        self.assertEqual(audit_capability_map()['knowledge_categories'], planner_capability_map()['knowledge_categories'])
        self.assertEqual(len(audit_capability_map()['category_scope']),6)

    def test_relation_delivery_bounded_typed_and_not_fact_proof(self):
        result=relation_delivery({'meta':{'relation_scope':{'relations':[
            {'relation':'source_method_reference','family':'panel','private_raw_text':'do not include'}]*14}}})
        self.assertEqual(result['count'],14)
        self.assertEqual(len(result['links']),12)
        self.assertEqual(result['omitted_link_count'],2)
        self.assertNotIn('private_raw_text',result['links'][0])

    def test_json_syntax_feedback_has_location_not_raw_text(self):
        with self.assertRaises(OutputContractError) as caught:
            # A trailing comma is now losslessly repaired. Unclosed strings
            # remain errors and diagnostics still must not reveal raw values.
            object_json('{"secret_value":"private}')
        details=caught.exception.validation_details
        self.assertGreater(details['column'],1)
        self.assertNotIn('private',str(details))

    def test_replan_input_failures_are_distinct(self):
        self.assertEqual(classify(ValueError('audit_input_budget_exceeded'),'replan_request')['code'], 'REPLAN_INPUT_BUDGET_EXCEEDED')
        self.assertEqual(classify(ValueError('bad input'),'replan_request')['code'], 'REPLAN_INPUT_INVALID')

    def test_retry_receives_failed_candidate_and_syntax_location(self):
        import torch
        from backend.tests.sales.test_model_contract import Model, Tokenizer
        from backend.sales.model_contract import checked_generate, json_contract
        model=Model(['{"tools":[}', '{"tools":[]}'])
        tokenizer=Tokenizer()
        output=checked_generate(model, contract=json_contract('planner', tokenizer),
                                input_ids=torch.tensor([[1,2]]), max_new_tokens=384)
        retry=tokenizer.decode(model.calls[1]['input_ids'][0,2:])
        self.assertIn('{"tools":[}', retry)
        self.assertIn('column', retry)
        self.assertEqual(tokenizer.decode(output[0,2:]), '{"tools":[]}')

    def test_supported_file_scope_requires_citation_identity_coverage(self):
        from backend.tests.sales.test_model_contract import Tokenizer
        from backend.sales.model_contract import support_contract
        contract=support_contract(Tokenizer(), {'required_documents':[{'document_id':'a'},{'document_id':'b'}],
            'evidence':[{'evidence_id':'U1','document_id':'a'},{'evidence_id':'U2','document_id':'b'}]})
        value=contract.validate('{"status":"supported","supported_evidence_ids":["U1"]}')
        self.assertEqual(value['status'],'partial')
        self.assertEqual(value['supported_evidence_ids'],['U1'])
        self.assertEqual(value['recovery_need'],'lookup')
        contract.validate('{"status":"supported","supported_evidence_ids":["U1","U2"]}')
        contract.validate('{"status":"partial","supported_evidence_ids":["U1"],"missing_aspects":["b内容不足"],"recovery_need":"lookup"}')
