"""History is planning context, never raw retrieval text."""
import unittest

from backend.app import DraftRequest, conversation_retrieval_hint, compact_conversation_context, TOOL_PLANNER_SYSTEM_PROMPT


class FollowupQueryBoundaryTests(unittest.TestCase):
    def make_request(self, question):
        return DraftRequest(customer_question=question, conversation_context=[
            {"role": "user", "content": "为什么这个方案不适合潮湿环境？"},
            {"role": "assistant", "content": "未经核验的旧结论：产品不适合潮湿环境。"},
        ])

    def test_new_goal_does_not_receive_old_search_text(self):
        request = self.make_request("有相关的配件安装教程吗？")
        self.assertEqual(conversation_retrieval_hint(request), "")
        self.assertEqual(len(compact_conversation_context(request)), 2)

    def test_topic_switch_does_not_receive_old_search_text(self):
        self.assertEqual(conversation_retrieval_hint(self.make_request("今天青岛天气如何？")), "")

    def test_memory_block_is_not_a_search_query(self):
        request = self.make_request("安装教程呢？")
        request._task_memory = {"task_state": [{"value": "旧项目预算80万元"}], "related_history": []}
        self.assertEqual(conversation_retrieval_hint(request), "")
        self.assertIn("旧项目预算", str(compact_conversation_context(request)))

    def test_planner_contract_separates_current_goal_and_old_claims(self):
        self.assertIn("当前问题决定任务", TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn("task_update.user_instructions是同一未完成任务的全部要求", TOOL_PLANNER_SYSTEM_PROMPT)
        self.assertIn("先前助手说法未经验证", TOOL_PLANNER_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
