import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.sales.task_memory import TaskMemory, MemoryUpdate, RETENTION_SECONDS
from backend.documents.visual_inputs import merge_visual_paths


class MemoryTests(unittest.TestCase):
    def test_memory_cancellation_is_not_hidden_as_lexical_fallback(self):
        from backend.request_budget import RequestBudget, RequestCancelled, budget_scope
        budget = RequestBudget(float('inf'))
        with budget_scope(budget):
            budget.cancelled.set()
            with self.assertRaises(RequestCancelled):
                self.store.semantic_scores([], 'current task')

    def test_semantic_paraphrase_cache_and_unrelated_filter(self):
        import numpy as np
        class Encoder:
            def __init__(self): self.calls = []
            def encode(self, texts, **kwargs):
                self.calls.append(list(texts))
                return np.array([[1.,0.] if ('价格' in t or '费用' in t) else [0.,1.] for t in texts])
        encoder = Encoder()
        store = TaskMemory(Path(self.temp.name)/'semantic.sqlite3', semantic_encoder=encoder)
        store.commit('a','t','之前的价格是十万')
        store.commit('a','t','请介绍游戏玩法')
        found = store.recall('a','t','费用是多少')
        self.assertEqual([x['text'] for x in found['related_history']],['之前的价格是十万'])
        self.assertEqual(found['related_history'][0]['match_method'],'semantic')
        self.assertFalse(found['related_history'][0]['technical_evidence'])
        again = store.recall('a','t','费用是多少')
        self.assertEqual(again['audit']['new_embeddings'],0)
        self.assertEqual(store.recall('b','t','费用是多少')['related_history'],[])
        store.forget('a','t')
        with store.connect() as db:
            self.assertEqual(db.execute('select count(*) from task_memory_vectors').fetchone()[0],0)

    def test_planner_history_is_not_a_current_user_message(self):
        from backend.app import DraftRequest, planner_context_input
        request = DraftRequest(customer_question='请讲讲国际象棋', conversation_context=[
            {'role':'user','content':'介绍保温板'}, {'role':'assistant','content':'旧产品回答'}])
        request._task_memory = {'related_history':[{'text':'老项目条件'}],'task_state':[]}
        payload = planner_context_input(request)
        self.assertEqual(payload['current_question'],'请讲讲国际象棋')
        self.assertEqual(len(payload['previous_exchange']),2)
        self.assertEqual(payload['related_history'][0]['text'],'老项目条件')
        self.assertNotIn('conversation_context',payload)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = TaskMemory(Path(self.temp.name) / "memory.sqlite3", semantic_encoder=False)

    def update(self, value, quote, **kwargs):
        return MemoryUpdate(key="project_a.installation", value=value, quote=quote, mode="asserted", **kwargs)

    def test_old_relevant_history_survives_recent_noise(self):
        self.store.commit("alice", "project_a", "黄金麻预算是100万元")
        for i in range(20):
            self.store.commit("alice", "project_a", f"无关话题{i}")
        result = self.store.recall("alice", "project_a", "黄金麻预算多少")
        self.assertIn("100万元", result["related_history"][0]["text"])

    def test_owner_and_thread_isolation(self):
        self.store.commit("alice", "project_a", "黄金麻预算是100万元")
        for owner, thread in [("bob", "project_a"), ("alice", "project_b")]:
            self.assertEqual(self.store.recall(owner, thread, "黄金麻")["audit"]["event_count"], 0)

    def test_revision_and_stale_history(self):
        self.store.commit("a", "t", "不做干挂", [self.update("不干挂", "不做干挂")])
        self.store.commit("a", "t", "改为干挂", [self.update("干挂", "改为干挂")])
        result = self.store.recall("a", "t", "干挂")
        self.assertEqual(result["task_state"][0]["value"], "干挂")
        self.assertNotIn("不做干挂", str(result["related_history"]))

    def test_hypothesis_and_fabricated_quote_rejected(self):
        self.store.commit("a", "t", "如果干挂呢", [MemoryUpdate(key="method", value="干挂", quote="如果干挂呢", mode="hypothetical")])
        self.store.commit("a", "t", "你好", [self.update("干挂", "不是用户说的")])
        self.assertEqual(self.store.recall("a", "t", "干挂")["task_state"], [])

    def test_retraction(self):
        self.store.commit("a", "t", "不做干挂", [self.update("不干挂", "不做干挂")])
        self.store.commit("a", "t", "取消这个条件", [self.update("", "取消这个条件", operation="retract")])
        self.assertEqual(self.store.recall("a", "t", "条件")["task_state"], [])

    def test_expiry_and_deletion(self):
        with patch("backend.sales.task_memory.time.time", return_value=1):
            self.store.commit("a", "t", "old")
        self.assertEqual(self.store.recall("a", "t", "old")["audit"]["event_count"], 0)
        self.store.commit("a", "t", "new")
        self.store.commit("b", "t", "new")
        self.store.forget("a", "t")
        self.assertEqual(self.store.recall("a", "t", "new")["audit"]["event_count"], 0)
        self.assertGreater(self.store.recall("b", "t", "new")["audit"]["event_count"], 0)

    def test_assistant_history_is_not_technical_evidence(self):
        self.store.commit("a", "t", "列举方案", assistant_text="第二种方案是干挂")
        found = self.store.recall("a", "t", "第二种方案")["related_history"][0]
        self.assertEqual(found["role"], "assistant")
        self.assertFalse(found["technical_evidence"])

    def test_http_memory_lifecycle(self):
        from fastapi.testclient import TestClient
        from backend import app
        from unittest.mock import MagicMock
        seen = []
        def reply(state, config=None):
            seen.append(state["request"]._task_memory)
            return {"response": {"intent": "general_chat", "normalized_terms": [], "answerable": True,
                "customer_reply": "测试回答", "key_points": [], "citations": [], "missing_information": [],
                "risk_warnings": [], "next_action": "", "meta": {}, "visual_assets": [], "retrieval": {}}}
        graph = MagicMock()
        graph.invoke.side_effect = reply
        headers = {"x-facade-client-id": "memory_test_client_123456789"}
        with patch.object(app, "TaskMemory", return_value=self.store), patch.object(app, "customer_answer_graph", return_value=graph), \
             patch.object(app, "begin_agent_request", return_value=(None, None)), \
             patch.object(app, "load_model", side_effect=AssertionError("unit_test_must_not_load_model")):
            client = TestClient(app.app)
            body = {"customer_question": "黄金麻预算100万", "memory_enabled": True, "conversation_id": "test_memory_123456"}
            first = client.post("/api/copilot/answer", json=body, headers=headers)
            self.assertEqual(first.status_code, 200, first.text)
            self.assertTrue(first.json()["meta"]["task_memory"]["enabled"])
            body["customer_question"] = "黄金麻预算是多少"
            second = client.post("/api/copilot/answer", json=body, headers=headers)
            self.assertEqual(second.status_code, 200, second.text)
            self.assertIn("100万", str(seen[-1]))
            current = client.get("/api/copilot/memory/test_memory_123456", headers=headers)
            self.assertEqual(current.status_code, 200)
            self.assertGreater(current.json()["audit"]["event_count"], 0)
            foreign = client.get("/api/copilot/memory/test_memory_123456", headers={"x-facade-client-id": "other_test_client_123456789"})
            self.assertEqual(foreign.json()["audit"]["event_count"], 0)
            cleared = client.delete("/api/copilot/memory/test_memory_123456", headers=headers)
            self.assertEqual(cleared.status_code, 200)
            from backend.documents.ownership import attachment_owner_id
            owner = attachment_owner_id(None, headers["x-facade-client-id"], required=True)
            self.assertEqual(self.store.recall(owner, "test_memory_123456", "预算")["audit"]["event_count"], 0)

    def test_task_context_is_bounded(self):
        from backend.app import DraftRequest, compact_conversation_context
        request = DraftRequest(customer_question="介绍一下")
        request._task_memory = {"task_state": [{"value": "字"*200} for _ in range(8)],
            "related_history": [{"text": "字"*450} for _ in range(3)]}
        context = compact_conversation_context(request)
        self.assertLess(len(context[0]["content"]), 2300)

    def test_recent_context_keeps_one_exchange_with_or_without_memory(self):
        from backend.app import DraftRequest, compact_conversation_context
        request = DraftRequest(customer_question="继续", conversation_context=[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn_{i}"}
            for i in range(6)
        ])
        expected = [{"role": "user", "content": "turn_4"},
                    {"role": "assistant", "content": "turn_5"}]
        self.assertEqual(compact_conversation_context(request), expected)
        request._task_memory = {"task_state": [], "related_history": []}
        context = compact_conversation_context(request)
        self.assertEqual(context[1:], expected)
        self.assertIn("NOT technical evidence", context[0]["content"])


class VisualTests(unittest.TestCase):
    def test_merge_dedup_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for i, data in enumerate([b"original", b"original", b"pdf_page", b"word", b"chart", b"overflow"]):
                path = Path(directory) / f"{i}.png"
                path.write_bytes(data)
                files.append(path)
            paths, audit = merge_visual_paths(files[0], files[1:])
            self.assertEqual(paths, [files[0], *files[2:5]])
            self.assertEqual(audit["candidates"][1]["status"], "duplicate")
            self.assertFalse(audit["coverage_complete"])

    def test_no_direct_image_preserves_retrieved_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            path.write_bytes(b"page")
            paths, audit = merge_visual_paths(None, [path])
            self.assertEqual(paths, [path])
            self.assertTrue(audit["coverage_complete"])
