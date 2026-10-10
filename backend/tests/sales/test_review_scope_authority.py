import copy
import unittest

from backend.sales.answer_review import ANSWER_AUDIT_PROMPT, validate_light_review
from backend.sales.support_wire import audit_wire_payload


class ReviewScopeAuthorityTests(unittest.TestCase):
    def payload(self):
        return {
            'review_policy': 'task_coverage_only',
            'question': 'List registered cases in Region A.',
            'task_update': {'user_instructions': ['Keep the original condition.']},
            'planned_task': {'sources': ['company_rag'], 'operation': 'summary',
                             'categories': ['05_project_case'], 'wants_images': False,
                             'targets': ['Region A'], 'query': 'rewritten query',
                             'aspects': ['name', 'location', 'type'], 'filters': {}},
            'actual_tools': ['company_rag'],
            'actual_execution_scope': {'status': 'collection', 'filters': {},
                'entity_targets': [{'id': 'P'+str(i), 'name': 'case '+str(i)} for i in range(12)],
                'matches': [{'entity_id': 'P'+str(i), 'basis': 'category_collection'} for i in range(12)]},
            'directory_coverage': [{'target_count': 12, 'retained_count': 7, 'status': 'partial'}],
            'requirement_coverage': {'requirements': [], 'candidate_count': 12},
            'final_draft': {'customer_reply': 'Region A has three registered cases.', 'answerable': True},
            'visual_candidates': [],
            'image_labels': [{'labels': {'description': 'observed square', 'uncertainties': ['material unknown']}}],
        }

    def test_candidate_collection_is_not_expected_answer_population(self):
        payload = self.payload()
        original = copy.deepcopy(payload)
        wire = audit_wire_payload(payload, compact=True)
        self.assertEqual(payload, original)
        self.assertFalse(wire['actual_execution_scope']['answer_scope_verified'])
        self.assertEqual(wire['actual_execution_scope']['population_role'],
                         'retrieval_candidates_not_required_answer_members')
        self.assertNotIn('entity_targets', wire['actual_execution_scope'])
        self.assertNotIn('matches', wire['actual_execution_scope'])
        self.assertEqual(wire['actual_execution_scope']['resolved_referents'], [])
        self.assertNotIn('directory_coverage', wire)

    def test_explicit_referents_remain_but_ambiguous_candidates_cannot_become_omissions(self):
        payload = self.payload()
        payload['actual_execution_scope']['matches'][0]['basis'] = 'exact_name'
        payload['actual_execution_scope']['matches'][1]['basis'] = 'semantic_candidate'
        wire = audit_wire_payload(payload, compact=True)
        self.assertEqual(wire['actual_execution_scope']['resolved_referents'],
                         [{'id': 'P0', 'name': 'case 0'}])
        self.assertIn('semantic_candidate', wire['actual_execution_scope']['matching_methods'])
        self.assertNotIn('case 1', str(wire['actual_execution_scope']))

    def test_planner_optional_aspects_do_not_become_user_obligations(self):
        wire = audit_wire_payload(self.payload(), compact=True)
        self.assertEqual(wire['planned_task'], {'sources': ['company_rag'], 'operation': 'summary',
                                               'categories': ['05_project_case'], 'wants_images': False})
        self.assertNotIn('planner_suggestions', wire)
        self.assertNotIn('requirement_coverage', wire)
        self.assertEqual(wire['question'], self.payload()['question'])
        self.assertEqual(wire['task_update'], self.payload()['task_update'])
        self.assertEqual(wire['final_draft'], self.payload()['final_draft'])

    def test_idempotent_projection_preserves_labels_and_unscoped_metadata(self):
        wire = audit_wire_payload(self.payload(), compact=True)
        self.assertEqual(audit_wire_payload(wire, compact=True), wire)
        self.assertEqual(wire['image_labels'], self.payload()['image_labels'])
        self.assertNotIn('evidence', wire)

    def test_scope_authority_is_source_agnostic(self):
        for source in ('company_rag', 'customer_documents', 'public_web_search'):
            payload = self.payload()
            payload['actual_tools'] = [source]
            payload['planned_task']['sources'] = [source]
            wire = audit_wire_payload(payload, compact=True)
            self.assertEqual(wire['actual_tools'], [source])
            self.assertEqual(wire['task_scope_contract']['answer_cardinality'],
                             'not_established_by_navigation_or_candidate_counts')

    def test_real_missing_explicit_dimension_can_still_require_revision(self):
        wire = audit_wire_payload(self.payload(), compact=True)
        result = validate_light_review({'decision': 'revise', 'coverage_status': 'partial',
            'label_consistency':{'status':'consistent','conflicts':[]},
            'revision': {'requirement': 'Region A', 'reason': 'The draft answers a different region.'}}, wire)
        self.assertEqual(result['decision'], 'revise')

    def test_legacy_evidence_audit_not_changed(self):
        payload = self.payload()
        payload.pop('review_policy')
        wire = audit_wire_payload(payload, compact=True)
        self.assertNotIn('task_scope_contract', wire)
        self.assertIn('aspects', wire['planned_task'])

    def test_prompt_prohibits_candidate_count_and_optional_dimension_rejections(self):
        self.assertIn('不是符合用户条件的结果总数', ANSWER_AUDIT_PROMPT)
        self.assertIn('不提供Planner拆出的目标、条件或回答维度', ANSWER_AUDIT_PROMPT)
        self.assertIn('仍检查真正漏答', ANSWER_AUDIT_PROMPT)


if __name__ == '__main__':
    unittest.main()
