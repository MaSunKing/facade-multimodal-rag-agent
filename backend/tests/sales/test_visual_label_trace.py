import unittest
from unittest.mock import patch
from backend.sales.runtime_status import ExecutionStatus, current_execution
from backend.sales import admin_traces
from backend.tests.sales.test_admin_traces import AdminTraceTests


class VisualLabelTraceTests(AdminTraceTests):
    def test_owner_only_labels_survive_completed_request_and_restart(self):
        journal=ExecutionStatus()
        label={'asset_id':'customer:a','document_id':'a','evidence_id':'VL1',
               'source_type':'customer_document','label_basis':'pixel_tool_observation',
               'in_draft_context':True,'labels':{'labels':['瓷砖'],
               'description':'可见矩形瓷砖拼接。' * 80,
               'visible_text':[], 'relations':['相邻拼接'],
               'uncertainties':['不能确认品牌或保温层'],
               'image_bytes':'FORBIDDEN_PIXELS','path':'FORBIDDEN_PATH'}}
        own={'X-Facade-Client-ID':'browser-test-owner-12345'}
        other={'X-Facade-Client-ID':'browser-test-other-12345'}
        admin_traces.register_live(journal)
        admin_traces.bind_trace_owner(journal.request_id,'browser:browser-test-owner-12345')
        from backend.documents.ownership import attachment_owner_id
        admin_traces.bind_trace_owner(journal.request_id,attachment_owner_id(None,own['X-Facade-Client-ID'],required=True))
        token=current_execution.set(journal)
        try:
            for phase in ('visual_tool','final_context','answer_audit_input'):
                admin_traces.record_visual_labels(phase,[label],round_number=1)
            self.assertNotIn('visual_label_versions',str(admin_traces.active_traces()))
            journal.finish(True)
        finally:
            current_execution.reset(token)
            admin_traces.unregister_live(journal.request_id)
        # All content is now read from SQL, not the in-memory attachment cache.
        response=self.client.get('/api/auth/traces/'+journal.request_id,headers=own)
        self.assertEqual(response.status_code,200)
        versions=response.json()['visual_label_versions']
        self.assertEqual(len(versions),3)
        self.assertEqual(versions[-1]['labels'][0]['labels']['description'],label['labels']['description'])
        self.assertEqual(versions[0]['labels'][0]['labels']['uncertainties'],['不能确认品牌或保温层'])
        self.assertNotIn('FORBIDDEN',response.text)
        self.assertEqual(self.client.get('/api/auth/traces/'+journal.request_id,headers=other).status_code,404)

    def test_redaction_does_not_lose_full_description_or_mutate_labels(self):
        journal=ExecutionStatus();token=current_execution.set(journal)
        text='可见纹理'*200+' token=private-credential C:/private/location'
        original={'asset_id':'a','labels':{'description':text,'labels':['面板'],'uncertainties':['品牌未知']}}
        try:
            admin_traces.record_visual_labels('final_context',[original])
        finally:current_execution.reset(token)
        saved=journal.visual_label_versions[0]['labels'][0]
        self.assertGreater(len(saved['labels']['description']),600)
        self.assertNotIn('private-credential',saved['labels']['description'])
        self.assertNotIn('C:/private',saved['labels']['description'])
        self.assertEqual(original['labels']['description'],text)


class VisualBoundaryPromptTests(unittest.TestCase):
    def test_label_conflict_witness_is_grounded_and_cannot_pass(self):
        from backend.sales.answer_review import validate_light_review,review_schema
        payload={'question':'比较这些材料','review_policy':'task_coverage_only',
            'image_labels':[{'asset_id':'a','labels':{'uncertainties':['无法确认品牌']}}],
            'final_draft':{'customer_reply':'确定不是该品牌产品。'}}
        self.assertEqual(review_schema(payload)['required'][0],'label_consistency')
        value={'decision':'pass','coverage_status':'complete','label_consistency':{
            'status':'conflict','conflicts':[{'asset_id':'a','uncertainty_quote':'无法确认品牌',
                'draft_quote':'确定不是该品牌产品'}]}}
        self.assertEqual(validate_light_review(value,payload)['decision'],'revise')
        value['label_consistency']['conflicts'][0]['uncertainty_quote']='虚构标签'
        from backend.sales.output_contract import OutputContractError
        with self.assertRaises(OutputContractError):validate_light_review(value,payload)

    def test_shared_scope_and_visual_rules_reach_final_and_revision(self):
        from backend.sales.visual_evidence import VISUAL_DESCRIPTION_PROMPT, validate_descriptions
        from backend.sales.generation_policy import RELATION_EVIDENCE_POLICY,EVIDENCE_COMPLETENESS_POLICY
        from backend.sales.answer_review import ANSWER_AUDIT_PROMPT,REVISION_PROMPT,revision_feedback
        validate_descriptions({'images':[{'image_id':'image1','labels':['瓷砖'],
            'description':'矩形瓷砖拼接','visible_text':[],'relations':[],
            'uncertainties':['无法确认品牌或芯材']}]},['image1'])
        self.assertIn('允许依据清晰可见特征识别对象类别',VISUAL_DESCRIPTION_PROMPT)
        self.assertIn('无法确认属于某对象不等于确认不属于',RELATION_EVIDENCE_POLICY)
        self.assertIn('局部例子',EVIDENCE_COMPLETENESS_POLICY)
        self.assertIn(RELATION_EVIDENCE_POLICY,REVISION_PROMPT)
        self.assertIn('标签边界检查不是事实认证',ANSWER_AUDIT_PROMPT)
        feedback=revision_feedback({'decision':'revise','review_policy':'task_coverage_only',
            'revision':{'requirement':'比较这些对象','reason':'模型推断出的新事实'}})
        self.assertNotIn('模型推断出的新事实',str(feedback))
        self.assertIn('不将未知写成肯定或否定',feedback['instruction'])
        from backend.sales.support_wire import audit_wire_text
        wire=audit_wire_text({'review_policy':'task_coverage_only',
            'final_draft':{'customer_reply':'草稿'},'image_labels':[{'labels':{'uncertainties':['品牌未知']}}]})
        self.assertGreater(wire.index('品牌未知'),wire.index('</final_draft_data>'))


if __name__=='__main__':unittest.main()
