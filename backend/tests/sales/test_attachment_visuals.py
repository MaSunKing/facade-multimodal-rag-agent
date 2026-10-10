import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from contextlib import nullcontext
from backend.sales.tool_planner import ToolPlan
from backend.sales.attachment_visuals import balanced_assets, wants_inspection, recovery_available, display_evidence, asset_key
from backend.documents.document_tasks import attachment_task_table, validate_document_support
from backend.sales.output_contract import OutputContractError


class AttachmentVisualTests(unittest.TestCase):
    def test_audit_and_final_share_resolved_scope_without_losing_constraints(self):
        from backend.documents.document_tasks import attachment_delivery_question
        task=attachment_task_table([{'document_id':'a','document_name':'first'},
            {'document_id':'b','document_name':'second'}],[],operation='summary',selection='all')
        raw='summarize first, keep the exclusions and payment conditions'
        audit=attachment_delivery_question({'question':raw,'attachment_task':task},['customer_documents'])
        final=attachment_delivery_question({'customer_question':raw,'attachment_task':task},['customer_documents'])
        self.assertEqual(audit['question'],final['customer_question'])
        self.assertEqual(final['original_question'],raw)
        for sources in (['company_rag'],['customer_documents','public_web_search']):
            original={'customer_question':raw,'attachment_task':task}
            self.assertIs(attachment_delivery_question(original,sources),original)

    def test_summary_obligations_do_not_inherit_synthesized_named_lookup_facets(self):
        from backend.documents.document_tasks import attachment_support_envelope
        task=attachment_task_table([
            {'document_id':'a','document_name':'one'},
            {'document_id':'b','document_name':'two'}],[],operation='summary',selection='all')
        source={'question':'summarize one with its limitations','attachment_task':task,
            'proposed_requirements':['one thickness'],
            'planned_task':{'aspects':['one thickness'],'sources':['customer_documents']},
            'retrieval_requirements':[{'target_kind':'document','aspect':'one thickness'},
                {'target_kind':'entity','aspect':'company product'}]}
        result=attachment_support_envelope(source)
        self.assertEqual(result['original_question'],source['question'])
        self.assertIn(source['question'],result['question'])
        self.assertIn('one',result['question'])
        self.assertIn('two',result['question'])
        self.assertEqual(attachment_support_envelope(result)['original_question'],source['question'])
        self.assertEqual(source['proposed_requirements'],['one thickness'])
        self.assertEqual(result['proposed_requirements'],[])
        self.assertEqual([r.get('target_id') for r in result['retrieval_requirements']],[None,'a','b'])
        self.assertNotIn('aspects',result['planned_task'])
        for selection in ('explicit',):
            self.assertIs(attachment_support_envelope({**source,'attachment_task':{**task,'selection':selection}}).get('planned_task'),source['planned_task'])

    def test_tool_pixels_only_when_requested_and_failure_is_local(self):
        from backend import app
        request=SimpleNamespace(image_data_url=None,document_session_id=None,customer_question='describe',
            _attachment_visual_states={},_visual_evidence_cache={},_visual_evidence_rows=[])
        assets=[{'document_id':d,'visual_id':'v','image_bytes':d.encode(),'document_name':d} for d in ('A','B')]
        paths=[MagicMock(),MagicMock()]
        for path,asset in zip(paths,assets): path.read_bytes.return_value=asset['image_bytes']
        description={'image_id':'image1','labels':['shape'],'description':'shape',
            'visible_text':[],'relations':[],'uncertainties':[]}
        with patch.object(app,'temporary_uploaded_image',return_value=nullcontext(None)), \
             patch.object(app,'temporary_visual_files',side_effect=lambda _:nullcontext(paths)), \
             patch.object(app,'check_budget'), \
             patch.object(app,'generate_multi_visual_response') as generate:
            rows=app.prepare_visual_text_evidence(request,ToolPlan(tools=['customer_documents'],wants_visuals=True),
                {'customer_documents':{'selected_visuals':assets}})
            self.assertEqual(rows,[])
            generate.assert_not_called()
            generate.side_effect=[RuntimeError('fixture failure'),json.dumps({'images':[description]})]
            rows=app.prepare_visual_text_evidence(request,ToolPlan(tools=['customer_documents'],document_visual_required=True),
                {'customer_documents':{'selected_visuals':assets}})
            self.assertEqual([r['document_id'] for r in rows],['B'])
            self.assertEqual(request._attachment_visual_states[asset_key(assets[0])]['status'],'failed')
            self.assertEqual(generate.call_count,2)
            rows=app.prepare_visual_text_evidence(request,ToolPlan(tools=['customer_documents']),
                {'customer_documents':{'selected_visuals':assets}})
            self.assertEqual([r['document_id'] for r in rows],['B'])
            self.assertEqual(generate.call_count,2)

    def test_display_does_not_request_pixels(self):
        for operation in ('lookup','summary','comparison'):
            self.assertFalse(wants_inspection(ToolPlan(tools=['customer_documents'],
                document_operation=operation,wants_visuals=True)))
        self.assertTrue(wants_inspection(ToolPlan(tools=['customer_documents'],document_visual_required=True)))
        self.assertFalse(wants_inspection(ToolPlan(tools=['company_rag'],document_visual_required=True)))
        self.assertTrue(wants_inspection(ToolPlan(tools=['customer_documents'],
            recovery_directive={'customer_visual_recovery':True})))

    def test_balancing_dedup_and_origin(self):
        a={'document_id':'A','visual_id':'1'}
        b={'document_id':'B','visual_id':'1'}
        assets=[a,a,{'document_id':'A','visual_id':'2'},b,{'asset_id':'company','document_id':'C','visual_id':'3'}]
        self.assertEqual(balanced_assets(assets,2),[a,b])
        rows=display_evidence(assets)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(not r['facts_eligible'] for r in rows))

    def test_shared_visual_budget_and_failed_assets(self):
        a={'document_id':'A','visual_id':'1','image_bytes':b'image'}
        self.assertTrue(recovery_available([a],{}))
        self.assertFalse(recovery_available([a],{asset_key(a):{'status':'failed','pixel_calls':1}}))
        self.assertFalse(recovery_available([a],{'other':{'pixel_calls':4}}))

    def test_per_file_audit_does_not_invent_support(self):
        task=attachment_task_table([{'document_id':'A','document_name':'a'},
            {'document_id':'B','document_name':'b'}],
            [{'document_id':'A','evidence_id':'U1'},{'document_id':'B','evidence_id':'U2'}],operation='summary',selection='all')
        value={'status':'supported','supported_evidence_ids':['U1','U2'],
            'document_support':{'A':['U1'],'B':['U2']},'missing_aspects':[]}
        validate_document_support(value,task)
        reduced={**value,'supported_evidence_ids':['U1']}
        validate_document_support(reduced,task)
        self.assertEqual(reduced['supported_evidence_ids'],['U1','U2'])
        alternate={**value,'document_support':[
            {'document_id':'A','evidence_ids':['U1']},{'document_id':'B','evidence_ids':'U2'}]}
        validate_document_support(alternate,task)
        self.assertEqual(alternate['document_support'],value['document_support'])
        with self.assertRaises(OutputContractError):
            validate_document_support({**value,'document_support':{'A':['U1'],'B':['U1']}},task)
        with self.assertRaises(OutputContractError):
            validate_document_support({**value,'document_support':{'A':['U1'],'B':[]}},task)
        validate_document_support({**value,'status':'partial','document_support':{'A':['U1'],'B':[]},
            'missing_aspects':['B content missing']},task)

    def test_minimal_audit_output_builds_file_table_in_code(self):
        from backend.sales.model_contract import support_contract
        from backend.tests.sales.test_model_contract import Tokenizer
        required=[{'document_id':'A','document_name':'a'},{'document_id':'B','document_name':'b'}]
        rows=[{'document_id':'A','evidence_id':'U1'},{'document_id':'B','evidence_id':'U2'}]
        task=attachment_task_table(required,rows,operation='summary',selection='all')
        contract=support_contract(Tokenizer(),{'evidence':rows,'attachment_task':task,'required_documents':required},'audit')
        result=contract.validate(json.dumps({'status':'supported','supported_evidence_ids':['U1','U2']}))
        self.assertEqual(result['document_support'],{'A':['U1'],'B':['U2']})
        incomplete=contract.validate(json.dumps({'status':'supported','supported_evidence_ids':['U1']}))
        self.assertEqual(incomplete['status'],'partial')
        self.assertEqual(incomplete['supported_evidence_ids'],['U1'])
        self.assertEqual(incomplete['recovery_need'],'lookup')
        result=contract.validate(json.dumps({'status':'partial','supported_evidence_ids':['U1'],
            'missing_aspects':['B has no support'],'recovery_need':'lookup'}))
        self.assertEqual(result['document_support'],{'A':['U1'],'B':[]})

    def test_audit_consumes_coverage_gap_without_format_retries(self):
        from backend.app import DraftRequest, evaluate_task_support
        from backend.tests.sales.test_audit_error_contract import text_audit_model
        payload={'question':'summarize selected files',
            'required_documents':[{'document_id':'A','document_name':'a'},{'document_id':'B','document_name':'b'}],
            'evidence':[{'document_id':'A','evidence_id':'U1','text':'A contents'},
                        {'document_id':'B','evidence_id':'U2','text':'B contents'}]}
        with text_audit_model(['{"status":"supported","supported_evidence_ids":["U1"]}']) as model:
            result=evaluate_task_support(DraftRequest(customer_question='summarize'),payload)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['supported_evidence_ids'],['U1'])
        self.assertEqual(result['recovery_need'],'lookup')
        self.assertEqual(len(model.calls),1)

    def test_recovery_preserves_user_scope_and_original_task_operation(self):
        from backend.sales.model_contract import validate_replan
        from backend.documents.document_tasks import attachment_task_operation
        value={'action':'execute','gap_ids':['g'],'revised_plan':{
            'sources':['customer_documents'],'operation':'lookup','wants_images':False,'document_visual_required':False,
            'queries':[{'source':'customer_documents','query':'b.txt'}],'categories':['unused customer category']}}
        result=validate_replan(value,document_ids=['a','b'],allowed_tools=['customer_documents'],gap_ids=['g'],
            previous_plan={'sources':['customer_documents'],'document_selection':'all','target_document_ids':[]},require_compact=True)
        self.assertEqual(result['revised_plan']['document_selection'],'all')
        self.assertEqual(result['revised_plan']['target_document_ids'],[])
        self.assertEqual(attachment_task_operation(ToolPlan(document_operation='lookup',
            recovery_directive={'original_document_operation':'summary'})),'summary')
        with self.assertRaises(OutputContractError):
            validate_replan({**value,'revised_plan':{**value['revised_plan'],'target_document_ids':['foreign']}},
                document_ids=['a','b'],allowed_tools=['customer_documents'],gap_ids=['g'],
                previous_plan={'sources':['customer_documents'],'document_selection':'all'},require_compact=True)
