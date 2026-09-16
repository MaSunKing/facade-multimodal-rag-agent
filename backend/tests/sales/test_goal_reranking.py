import time
import unittest
from unittest.mock import Mock

from backend.request_budget import RequestBudget, budget_scope
from backend.sales.goal_reranking import rerank_goal_evidence
from backend.sales.context_engine import optimise_evidence_context
from backend.sales.tool_planner import ToolPlan, guard_plan


class GoalRerankingTests(unittest.TestCase):
    def test_semantic_support_beats_shared_words_and_preserves_two_documents(self):
        items = [
            dict(evidence_id='A1', document_name='a.pdf', evidence_scope='content', text='planet rotation axis and seasons'),
            dict(evidence_id='A2', document_name='a.pdf', evidence_scope='content', text='One complete turn takes 11 hours.'),
            dict(evidence_id='B1', document_name='b.html', evidence_scope='content', text='A day lasts 11 hours.'),
        ]
        scorer = Mock()
        # Pool interleaves documents: A1, B1, A2.
        scorer.score.return_value = [0.02, 0.90, 0.98]
        scored, audit = rerank_goal_evidence('compare rotation period', items, goals=['rotation period'], scorer=scorer)
        ranked, context = optimise_evidence_context('rotation period', scored, target_terms=['rotation period'])
        self.assertTrue(audit['applied'])
        self.assertEqual(ranked[0]['evidence_id'], 'A2')
        self.assertEqual({i['evidence_id'] for i in ranked if i.get('protected_goal_ids')}, {'A2','B1'})
        self.assertEqual(context['goal_coverage_method'], 'semantic_relevance_not_verified_fact')
        self.assertFalse(audit['score_is_semantic_gold'])

    def test_missing_budget_skips_model_without_consuming_recovery(self):
        scorer = Mock()
        budget = RequestBudget(time.monotonic()+20)
        with budget_scope(budget):
            _, audit = rerank_goal_evidence('clause', [dict(evidence_id='A',text='clause')], goals=['clause'], scorer=scorer)
        scorer.score.assert_not_called()
        self.assertEqual(budget.recoveries, [])
        self.assertFalse(audit['applied'])

    def test_bad_scorer_keeps_original_evidence(self):
        items = [dict(evidence_id='A',text='contract term')]
        scorer = Mock(); scorer.score.side_effect = ValueError('failed')
        restored, audit = rerank_goal_evidence('contract', items, goals=['contract'], scorer=scorer)
        self.assertEqual(restored, items)
        self.assertFalse(audit['applied'])

    def test_goals_survive_guard(self):
        plan = ToolPlan(tools=['customer_documents'],answer_goals=['Explain the policy and exceptions'])
        safe = guard_plan(plan,has_documents=True,has_image=False,facade_related=False,web_allowed=False)
        self.assertEqual(safe.answer_goals,plan.answer_goals)

    def test_pool_bounded_and_documents_interleaved(self):
        items = [dict(evidence_id=f'A{i}',document_name='a',text='text') for i in range(50)] + [dict(evidence_id='B',document_name='b',text='text')]
        scorer=Mock(); scorer.score.return_value=[0.2]*16
        _,audit=rerank_goal_evidence('summary',items,goals=['summary']*7,scorer=scorer)
        self.assertEqual(audit['candidate_count'],16)
        self.assertIn('B',audit['scores'])
        scorer.score.assert_called_once()

    def test_semantic_goal_absence_remains_a_retrieval_hint(self):
        items=[dict(evidence_id='A',document_name='a.pdf',text='unrelated material')]
        scorer=Mock(); scorer.score.return_value=[0.001]
        scored,_=rerank_goal_evidence('payment terms',items,goals=['payment terms'],scorer=scorer)
        _,audit=optimise_evidence_context('payment terms',scored)
        self.assertEqual(audit['missing_answer_goals'],['payment terms'])
        self.assertFalse(audit['coverage_sufficient_before_generation'])

    def test_scored_fact_is_not_reordered_by_table_parser_penalty(self):
        from backend.app import compact_grounded_payload_for_generation
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs): return messages[-1]['content']
            def __call__(self,text,**kwargs): return {'input_ids':text.split()}
        items=[dict(evidence_id='T',document_name='a.pdf',evidence_scope='content',
                    text='[ROW source=pdf;page=2] One turn takes 11 hours.',
                    goal_rerank_applied=True,protected_goal_ids=['a:period'],protected_goal_terms=['rotation period']),
               dict(evidence_id='P',document_name='a.pdf',evidence_scope='content',
                    text='rotation axis '*120,goal_rerank_applied=True)]
        _,audit=compact_grounded_payload_for_generation({'evidence':items,'customer_question':'period'},Tokenizer(),max_prompt_tokens=130)
        self.assertIn('T',audit['kept_evidence_ids'])
        self.assertEqual(audit['missing_packed_goal_ids'],[])

    def test_numeric_audit_accepts_parser_escaped_whitespace_not_other_values(self):
        from backend.app import evidence_support_audit
        evidence={'T': {'text': "[ROW] c2='Duration \\n27.12 hr'"}}
        result={'customer_reply':'Duration is 27.12 hours.', 'citations':[{'evidence_id':'T'}]}
        self.assertEqual(evidence_support_audit(result,evidence)['unsupported_numeric_claims'],[])
        result['customer_reply']='Duration is 27.13 hours.'
        self.assertEqual(evidence_support_audit(result,evidence)['unsupported_numeric_claims'],['27.13'])

    def test_two_documents_of_same_source_type_can_form_candidate_conflict(self):
        items=[dict(evidence_id='U1',document_name='a.pdf',source_type='customer_document',text='Panel width: 6mm'),
               dict(evidence_id='U2',document_name='b.docx',source_type='customer_document',text='Panel width: 8mm')]
        _,audit=optimise_evidence_context('Panel width',items)
        self.assertEqual(len(audit['conflict_groups']),1)
        self.assertEqual(set(audit['conflict_groups'][0]['evidence_ids']),{'U1','U2'})
