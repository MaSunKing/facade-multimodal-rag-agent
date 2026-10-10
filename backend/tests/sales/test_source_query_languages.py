import unittest
from backend.sales.tool_planner import ToolPlan, retrieval_query_for_attempt, guard_plan


class SourceQueryLanguageTests(unittest.TestCase):
    def test_english_question_chinese_local_sources_english_web(self):
        question = 'Compare the two panels; do not assume fire certification.'
        plan = ToolPlan(tools=['customer_documents','company_rag','public_web_search'],
            company_retrieval_query='比较两种板材，不得假定通过防火认证',
            document_retrieval_queries=['比较附件中的两种板材，不得假定通过防火认证'],
            web_retrieval_query=question, answer_language='English')
        self.assertEqual(retrieval_query_for_attempt(question, plan, source='company_rag'), plan.company_retrieval_query)
        self.assertEqual(retrieval_query_for_attempt(question, plan, source='customer_documents'), plan.document_retrieval_queries[0])
        self.assertEqual(retrieval_query_for_attempt(question, plan, public=True), question)

    def test_chinese_question_english_attachment_stays_separate(self):
        plan = ToolPlan(company_retrieval_query='板厚是多少？',
                        document_retrieval_queries=['What is the panel thickness?'], answer_language='Chinese')
        self.assertEqual(retrieval_query_for_attempt('板厚是多少？', plan, source='company_rag'), '板厚是多少？')
        self.assertEqual(retrieval_query_for_attempt('板厚是多少？', plan, source='customer_documents'), 'What is the panel thickness?')
        self.assertEqual(retrieval_query_for_attempt('板厚是多少？', plan, public=True), '板厚是多少？')

    def test_mixed_language_documents_preserve_complete_queries(self):
        plan = ToolPlan(document_retrieval_queries=['厚度不得超过20毫米', 'Thickness must not exceed 20 mm'])
        query = retrieval_query_for_attempt('Compare limits', plan, source='customer_documents')
        self.assertIn('不得超过20毫米', query)
        self.assertIn('must not exceed 20 mm', query)

    def test_company_never_silently_uses_english_only_query(self):
        plan = ToolPlan(company_retrieval_query='What products do you sell?')
        self.assertEqual(retrieval_query_for_attempt('你们有哪些产品？', plan, source='company_rag'), '你们有哪些产品？')
        with self.assertRaisesRegex(ValueError, 'company_chinese_query_missing'):
            retrieval_query_for_attempt('What products do you sell?', plan, source='company_rag')

    def test_recovery_does_not_export_local_query_to_web(self):
        plan = ToolPlan(retrieval_query='private local facts', company_retrieval_query='企业产品资料',
                        web_retrieval_query='Public panel maintenance comparison',
                        recovery_directive={'gap':'maintenance'})
        self.assertEqual(retrieval_query_for_attempt('Compare maintenance', plan, public=True), plan.web_retrieval_query)

    def test_guard_preserves_source_fields_and_permissions(self):
        proposed = ToolPlan(tools=['company_rag','public_web_search'],
            company_retrieval_query='公司产品有哪些？', web_retrieval_query='Which products?', answer_language='English')
        plan = guard_plan(proposed, has_documents=False, has_image=False, web_allowed=False, facade_related=True)
        self.assertEqual(plan.company_retrieval_query, proposed.company_retrieval_query)
        self.assertEqual(plan.answer_language, 'English')
        self.assertNotIn('public_web_search', plan.tools)


if __name__ == '__main__':
    unittest.main()
