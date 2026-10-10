import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend import access_control
from backend.auth_router import router
from backend.sales import admin_traces
from backend.sales.runtime_status import ExecutionStatus, install_error_protocol, current_execution


class AdminTraceTests(unittest.TestCase):
    def test_long_running_request_remains_visible_with_summary_progress(self):
        import time
        journal = ExecutionStatus()
        journal.enter('document_summary')
        journal.summary_progress = {'file_count': 4, 'packet_index': 2}
        with patch.object(admin_traces, '_active', {journal.request_id: (journal, '2026-10-06T00:00:00+00:00', time.monotonic() - 1800)}):
            rows = admin_traces.active_traces()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['execution']['summary_progress']['file_count'], 4)
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.temp.name)
        self.patches = [patch.object(access_control, 'ROOT', root),
                        patch.object(access_control, 'STATE_DB_PATH', root / 'state.sqlite3'),
                        patch.object(access_control, 'BOOTSTRAP_TOKEN_PATH', root / 'bootstrap.txt'),
                        patch.object(access_control, '_SCHEMA_READY', False)]
        for p in self.patches: p.start()
        access_control.ensure_schema()
        admin = access_control.create_initial_admin(setup_token=access_control.BOOTSTRAP_TOKEN_PATH.read_text().strip(),
            username='admin-test', password='test-password-long', display_name='Test admin')
        self.admin_token = access_control.login(admin.username, 'test-password-long')[0]
        access_control.create_user(username='member-test', password='test-password-long', display_name='Member', role='member', can_access_internal=False)
        self.member_token = access_control.login('member-test', 'test-password-long')[0]
        app = FastAPI()
        app.include_router(router)
        install_error_protocol(app, root / 'errors.jsonl')
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def headers(self, token): return {'Authorization': 'Bearer ' + token}

    def saved_request(self):
        request_id, started = access_control.begin_agent_request(principal=None, attachment_session_id=None, query_length=10)
        response = {'answerable': False, 'customer_reply': 'PRIVATE_CONTENT', 'citations': [{'evidence_id': 'T1'}],
                    'meta': {'mode': 'evidence_audit_failed', 'execution': {'request_id': 'a' * 32, 'state': 'failed', 'stage':'evidence_evaluation'},
                      'evidence_gate': {'status':'uncertain', 'error_code':'AUDIT_FAILED', 'problem':'PRIVATE_CONTENT'},
                      'orchestration': {'tools':['company_rag'], 'tool_rounds':1,
                        'tool_plan_details':[{'tools':['company_rag'], 'retrieval_query':'PRIVATE_CONTENT', 'reason':'PRIVATE_CONTENT'}],
                        'node_trace':[{'node':'evidence_evaluation','elapsed_ms':110,'round':1}]}}}
        access_control.finish_agent_request(request_id, started, response=response)
        return request_id

    def test_admin_only_both_endpoints(self):
        for headers, expected in [({},401), (self.headers(self.member_token),403)]:
            self.assertEqual(self.client.get('/api/auth/admin/traces',headers=headers).status_code,expected)
            self.assertEqual(self.client.get('/api/auth/admin/traces/'+'a'*32,headers=headers).status_code,expected)
        self.assertEqual(self.client.get('/api/auth/admin/traces',headers=self.headers(self.admin_token)).status_code,200)

    def test_public_own_history_and_agent_alias_not_other_browser(self):
        from backend.documents.ownership import attachment_owner_id
        own={'X-Facade-Client-ID':'browser-test-owner-12345'}
        other={'X-Facade-Client-ID':'browser-test-other-12345'}
        request_id=self.saved_request()
        admin_traces.bind_trace_owner('a'*32,attachment_owner_id(None,own['X-Facade-Client-ID'],required=True),'client-task')
        for identifier in ('a'*32,request_id,'client-task'):
            self.assertEqual(self.client.get('/api/auth/traces/'+identifier,headers=own).status_code,200)
            self.assertEqual(self.client.get('/api/auth/traces/'+identifier,headers=other).status_code,404)
        rows=self.client.get('/api/auth/traces',headers=own).json()['requests']
        self.assertEqual([r['request_id'] for r in rows],[request_id])
        self.assertEqual(self.client.get('/api/auth/traces',headers=other).json()['requests'],[])

    def test_members_and_guests_separate_trace_identity(self):
        from backend.documents.ownership import attachment_owner_id
        guest={'X-Facade-Client-ID':'browser-test-owner-12345'}
        member={**guest,**self.headers(self.member_token)}
        journal=ExecutionStatus();admin_traces.register_live(journal)
        try:
            admin_traces.bind_trace_owner(journal.request_id,attachment_owner_id(None,guest['X-Facade-Client-ID'],required=True))
            self.assertEqual(self.client.get('/api/auth/traces/'+journal.request_id,headers=guest).status_code,200)
            self.assertEqual(self.client.get('/api/auth/traces/'+journal.request_id,headers=member).status_code,404)
        finally: admin_traces.unregister_live(journal.request_id)

    def test_unowned_legacy_trace_not_exposed_to_guests(self):
        request_id=self.saved_request()
        response=self.client.get('/api/auth/traces/'+request_id,headers={'X-Facade-Client-ID':'browser-test-owner-12345'})
        self.assertEqual(response.status_code,404)

    def test_storage_error_is_retryable_not_content_leak(self):
        import sqlite3
        headers={'X-Facade-Client-ID':'browser-test-owner-12345'}
        with patch.object(admin_traces,'owned_traces',side_effect=sqlite3.OperationalError('PRIVATE_DB_PATH')):
            response=self.client.get('/api/auth/traces',headers=headers)
        self.assertEqual(response.status_code,503)
        self.assertEqual(response.headers['Retry-After'],'3')
        self.assertNotIn('PRIVATE_DB_PATH',response.text)
        with patch.object(admin_traces,'owned_trace',side_effect=sqlite3.OperationalError('PRIVATE_DB_PATH')):
            response=self.client.get('/api/auth/traces/'+'a'*32,headers=headers)
        self.assertEqual(response.status_code,503)

    def test_persisted_detail_is_projected_without_private_content(self):
        request_id = self.saved_request()
        response = self.client.get('/api/auth/admin/traces/'+request_id,headers=self.headers(self.admin_token))
        self.assertEqual(response.status_code,200)
        self.assertNotIn('PRIVATE_CONTENT',response.text)
        data=response.json()['summary']
        self.assertEqual(data['evidence_gate']['error_code'],'AUDIT_FAILED')
        self.assertEqual(data['node_trace'][0]['node'],'evidence_evaluation')
        self.assertEqual(admin_traces.read_trace('a'*32)['request_id'],request_id)

    def test_failed_audit_not_saved_as_completed(self):
        request_id = self.saved_request()
        with access_control._connect() as db:
            row = db.execute('SELECT status,error_type FROM agent_requests WHERE request_id=?', (request_id,)).fetchone()
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['error_type'], 'AUDIT_FAILED')

    def test_live_and_early_failure_retained(self):
        journal=ExecutionStatus(); admin_traces.register_live(journal)
        try:
            journal.enter('plan_request')
            self.assertEqual(admin_traces.read_trace(journal.request_id)['execution']['stage'],'plan_request')
            journal.finish(False)
        finally: admin_traces.unregister_live(journal.request_id)
        data=admin_traces.read_trace(journal.request_id)
        self.assertEqual(data['status'],'failed')
        self.assertTrue(any(r['request_id']==journal.request_id for r in admin_traces.list_traces()['requests']))

    def test_interrupted_agent_keeps_stage_details_under_both_ids(self):
        from backend.sales.stage_details import begin,end
        request_id, started = access_control.begin_agent_request(principal=None,attachment_session_id=None,query_length=20)
        journal=ExecutionStatus(agent_request_id=request_id)
        token=current_execution.set(journal)
        admin_traces.register_live(journal)
        try:
            end(begin('plan_request'),{'parsed_plan':{'tools':['company_rag'],'retrieval_query':'Projects and photos'}})
            begin('generate_answer')
            journal.finish(False)
            access_control.finish_agent_request(request_id,started,response=None,error_type='REQUEST_TIMEOUT')
        finally:
            admin_traces.unregister_live(journal.request_id)
            current_execution.reset(token)
        for identifier in (request_id,journal.request_id):
            data=admin_traces.read_trace(identifier)
            self.assertEqual(data['summary']['stage_details'][0]['output']['parsed_plan']['tools'],['company_rag'])
            self.assertEqual(data['summary']['stage_details'][-1]['status'],'running')

    def test_legacy_snapshot_and_path_escape(self):
        request_id=self.saved_request()
        with access_control._connect() as db:
            db.execute('DELETE FROM admin_request_traces WHERE request_id=?',(request_id,))
        self.assertEqual(admin_traces.read_trace(request_id)['summary']['node_trace'][0]['node'],'evidence_evaluation')
        escaped=access_control.ROOT/'outside.json'; escaped.write_text(json.dumps({'secret':'PRIVATE_CONTENT'}))
        with access_control._connect() as db:
            db.execute('UPDATE evidence_snapshots SET snapshot_path=? WHERE request_id=?',('outside.json',request_id))
        self.assertNotIn('summary',admin_traces.read_trace(request_id))

    def test_limit_and_request_id_validation(self):
        headers=self.headers(self.admin_token)
        self.assertEqual(self.client.get('/api/auth/admin/traces?limit=1000',headers=headers).status_code,422)
        self.assertEqual(self.client.get('/api/auth/admin/traces/not-valid',headers=headers).status_code,400)
        self.assertEqual(self.client.get('/api/auth/admin/traces/'+'b'*32,headers=headers).status_code,404)

    def test_private_draft_versions_survive_rejection_with_existing_acl(self):
        from backend.sales.runtime_status import current_execution
        from backend.documents.ownership import attachment_owner_id
        request_id,started=access_control.begin_agent_request(principal=None,attachment_session_id=None,query_length=10)
        journal=ExecutionStatus(request_id='c'*32)
        token=current_execution.set(journal)
        try:
            draft=admin_traces.record_answer_version({'customer_reply':'INTERNAL_DRAFT',
                'citations':[{'evidence_id':'T1'}],'prompt':'DO_NOT_STORE','meta':{}},1)
            admin_traces.record_answer_review(draft,{'decision':'revise','reason':'missing scope'})
            revision=admin_traces.record_answer_version({'customer_reply':'REVISED_DRAFT',
                'meta':{'answer_revision_count':1}},1)
            admin_traces.record_answer_review(revision,{'decision':'limited','reason':'still incomplete'})
            self.assertNotIn('INTERNAL_DRAFT',json.dumps(journal.snapshot()))
            access_control.finish_agent_request(request_id,started,response={'customer_reply':'withheld','meta':{'execution':{'request_id':journal.request_id}}})
        finally: current_execution.reset(token)
        own={'X-Facade-Client-ID':'browser-version-owner-12345'}
        other={'X-Facade-Client-ID':'browser-version-other-12345'}
        admin_traces.bind_trace_owner(journal.request_id,attachment_owner_id(None,own['X-Facade-Client-ID'],required=True))
        value=self.client.get('/api/auth/traces/'+request_id,headers=own).json()
        self.assertEqual([v['customer_reply'] for v in value['answer_versions']],['INTERNAL_DRAFT','REVISED_DRAFT'])
        self.assertEqual(value['answer_versions'][1]['publication'],'withheld')
        self.assertNotIn('DO_NOT_STORE',json.dumps(value))
        self.assertEqual(self.client.get('/api/auth/traces/'+request_id,headers=other).status_code,404)
        self.assertEqual(self.client.get('/api/auth/admin/traces/'+request_id,headers=self.headers(self.member_token)).status_code,403)

    def test_finished_http_is_not_duplicated_in_request_list(self):
        request_id=self.saved_request()
        journal=ExecutionStatus(request_id='a'*32); admin_traces.register_live(journal)
        journal.finish(True); admin_traces.unregister_live(journal.request_id)
        rows=admin_traces.list_traces()['requests']
        self.assertEqual([r['request_id'] for r in rows],[request_id])


if __name__ == '__main__': unittest.main()
