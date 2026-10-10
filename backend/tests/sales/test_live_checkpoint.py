import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import access_control
from backend import task_cancellation as cancel
from backend.sales import admin_traces as traces
from backend.sales.runtime_status import ExecutionStatus,classify
from backend.request_budget import RequestBudget,ModelCallTimeout


class LiveCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.patches=[patch.object(access_control,'STATE_DB_PATH',self.root/'state.sqlite3'),
            patch.object(access_control,'BOOTSTRAP_TOKEN_PATH',self.root/'bootstrap.txt'),
            patch.object(access_control,'_SCHEMA_READY',False),patch.object(cancel,'CONTROL_DIRECTORY',self.root/'controls')]
        for p in self.patches: p.start()
        access_control.ensure_schema()

    def tearDown(self):
        cancel._request_budgets.clear()
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def test_unfinished_attempt_survives_without_unregister(self):
        journal=ExecutionStatus(request_id='a'*32)
        journal.enter('evidence_evaluation')
        journal.model_contracts=[{'role':'support_audit','status':'running','attempts':[
            {'attempt':2,'input_tokens':5500,'status':'running','started_unix_seconds':time.time()-12}]}]
        traces.persist_live_checkpoint(journal,'2026-10-07T00:00:00+00:00')
        result=traces.read_trace(journal.request_id)
        self.assertEqual(result['status'],'interrupted')
        self.assertEqual(result['execution']['stage'],'evidence_evaluation')
        attempt=result['execution']['model_contracts'][0]['attempts'][0]
        self.assertGreaterEqual(attempt['elapsed_seconds'],12)
        self.assertEqual(attempt['attempt'],2)
        self.assertEqual(journal.state,'executing')

    def test_checkpoint_stays_owner_scoped(self):
        journal=ExecutionStatus(request_id='b'*32)
        traces.bind_trace_owner(journal.request_id,'owner-a')
        traces.persist_live_checkpoint(journal,'2026-10-07T00:00:00+00:00')
        self.assertIsNotNone(traces.owned_trace('owner-a',journal.request_id))
        self.assertIsNone(traces.owned_trace('owner-b',journal.request_id))

    def test_local_cancel_only_matching_valid_control(self):
        rid='c'*32; budget=RequestBudget.for_answer(has_attachments=True)
        cancel.bind_request_budget(rid,budget)
        directory=cancel.CONTROL_DIRECTORY; directory.mkdir()
        path=directory/('cancel-'+rid+'.json')
        path.write_text(json.dumps({'request_id':'d'*32,'action':'cancel'}))
        self.assertFalse(cancel.consume_local_cancellation(rid))
        self.assertFalse(budget.cancelled.is_set())
        path.write_text(json.dumps({'request_id':rid,'action':'cancel'}))
        self.assertTrue(cancel.consume_local_cancellation(rid))
        self.assertTrue(budget.cancelled.is_set())
        self.assertFalse(path.exists())
        cancel.unbind_request_budget(rid,budget)

    def test_model_timeout_specific_and_not_format_retried(self):
        from backend.sales.model_contract import checked_generate,json_contract
        from backend.tests.sales.test_model_contract import Tokenizer,Model
        import torch
        model=Model(['{}'])
        with patch('backend.sales.model_contract.generation_guard',return_value=(lambda *a,**k:False,
            lambda:(_ for _ in ()).throw(ModelCallTimeout('model_call_timeout')))):
            with self.assertRaises(ModelCallTimeout):
                checked_generate(model,contract=json_contract('test',Tokenizer()),input_ids=torch.tensor([[1]]),max_new_tokens=20)
        self.assertEqual(model.calls,[])
        self.assertEqual(classify(ModelCallTimeout('model_call_timeout'))['code'],'MODEL_CALL_TIMEOUT')


if __name__=='__main__': unittest.main()
