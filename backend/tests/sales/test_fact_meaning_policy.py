import copy
import json
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from contextlib import nullcontext
import torch
from transformers import BatchEncoding

from backend import app
from backend.request_budget import RequestBudget, budget_scope
from backend.sales.goal_reranking import rerank_goal_evidence
from backend.sales.task_support import FACT_MEANING_POLICY, AUDIT_PROMPT


class FactMeaningPolicyTests(unittest.TestCase):
    def test_audit_receives_request_clock_without_mutating_source_dates(self):
        payload = {'current_date_china':'1900-01-01', 'evidence':[
            {'evidence_id':'W2','text':'Recorded maximum as of 6am: 29.6 degrees',
             'source_date':'2026-09-21','date_scope':'page_metadata_not_fact_date'}]}
        original = copy.deepcopy(payload)
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = 'prompt'
        tokenizer.encode.return_value = [1]
        tokenizer.return_value = BatchEncoding({'input_ids':torch.tensor([[1]])})
        tokenizer.decode.return_value = json.dumps({'status':'supported','supported_evidence_ids':['W2']})
        model = Mock(device='cpu')
        with patch.object(app, 'current_china_date', return_value='2026-10-05'), \
             patch.object(app, 'load_model', return_value=(tokenizer, model)), \
             patch.object(app, 'generation_session', return_value=nullcontext()), \
             patch('backend.sales.generation_profile.profiled_generate', return_value=torch.tensor([[1,2]])):
            app.evaluate_task_support(app.DraftRequest(customer_question='Describe the supplied record'),
                payload)
        sent = json.loads(tokenizer.apply_chat_template.call_args.args[0][1]['content'])
        self.assertEqual(sent['current_date_china'], '2026-10-05')
        self.assertEqual(sent['evidence'], original['evidence'])
        self.assertEqual(payload, original)

    def test_generation_paths_share_meaning_preservation_without_extra_call(self):
        from backend.sales.draft_prompt import COMMON
        for name in ('GENERAL_CHAT_SYSTEM_PROMPT','COMPANY_RAG_SYSTEM_PROMPT',
                     'COMPANY_RAG_VISUAL_SYSTEM_PROMPT','CUSTOMER_DOCUMENT_SYSTEM_PROMPT',
                     ):
            self.assertIn(COMMON, getattr(app, name))
        self.assertIn(FACT_MEANING_POLICY, app.GROUNDED_SYSTEM_PROMPT)
        self.assertIn('不认证整篇正文', AUDIT_PROMPT)
        self.assertIn('历史数据不能支持当前结论', AUDIT_PROMPT)

    def test_joint_ranking_no_longer_skips_at_former_80_second_gate(self):
        scorer = Mock(); scorer.score.return_value = [.8]
        with budget_scope(RequestBudget(time.monotonic()+70)):
            _, audit = rerank_goal_evidence('Full original question',
                [{'evidence_id':'T1','text':'fact'}], goals=['full dimension'], scorer=scorer)
        scorer.score.assert_called_once()
        self.assertTrue(audit['applied'])
        self.assertEqual(audit['answer_time_reserve_seconds'], 60)


if __name__ == '__main__':
    unittest.main()
