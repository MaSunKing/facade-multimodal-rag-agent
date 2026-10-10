import copy
import json
import unittest

from backend.sales.final_input import preflight_final_input


class FinalInputTests(unittest.TestCase):
    def test_review_budget_selects_before_final_even_when_final_fits(self):
        rows=[{'evidence_id':str(i),'text':'ordinary background '+str(i)} for i in range(5)]
        selected,text,wire,audit=self.prepare(rows,{},ceiling=5000,
            review_tokens=lambda items:4000+1000*len(items),review_limit=8000)
        self.assertTrue(wire['budget_satisfied'])
        self.assertLessEqual(len(selected),4)
        self.assertLessEqual(audit['audit_prompt_tokens'],8000)

    def test_review_wrapper_overflow_is_explicit_without_generation(self):
        _,_,wire,audit=self.prepare([{'evidence_id':'A','text':'fact'}],{},
            review_tokens=lambda _:9000,review_limit=8000)
        self.assertFalse(wire['budget_satisfied'])
        self.assertEqual(audit['blocking_budget'],'answer_review')

    def render(self, payload, limit):
        from backend.app import compact_grounded_payload_for_generation
        from backend.tests.sales.test_context_engine import _CharacterTokenizer
        return compact_grounded_payload_for_generation(payload, _CharacterTokenizer(),
            max_prompt_tokens=limit, system_prompt='SYSTEM + image and attachment policy')

    def prepare(self, rows, template, ceiling=5000, **kw):
        return preflight_final_input(rows,
            build_payload=lambda selected: {**template, 'evidence':selected,
                'audited_evidence_ids':[r['evidence_id'] for r in selected]},
            render=self.render, ceiling=ceiling, question='compare sources', **kw)

    def test_complete_envelope_and_frozen_wire_equal_actual_chat_count(self):
        rows=[{'evidence_id':'A','text':'thickness 20 mm', 'source_type':'company_rag'}]
        template={'customer_question':'Compare', 'attachment_task':{'instructions':'all files '*40},
                  'attachment_visuals':{'labels':'visible texture '*50},
                  'company_relations':{'text':'linked method '*40}}
        selected, text, wire, audit=self.prepare(rows, template)
        from backend.tests.sales.test_context_engine import _CharacterTokenizer
        tokenizer=_CharacterTokenizer()
        prompt=tokenizer.apply_chat_template([
            {'role':'system','content':'SYSTEM + image and attachment policy'},
            {'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        self.assertEqual(len(tokenizer(prompt,add_special_tokens=False)['input_ids']),wire['preflight_prompt_tokens'])
        self.assertEqual(audit['actual_prompt_tokens'],wire['actual_prompt_tokens'])
        self.assertEqual(selected, rows)
        template['late_runtime']={'large':'x'*10000}
        self.assertNotIn('late_runtime',json.loads(text))

    def test_overflow_compresses_before_generation_and_keeps_both_visual_sources(self):
        rows=[{'evidence_id':'A','text':'company facts '*300,'source_type':'company_rag'},
              *[{'evidence_id':str(i),'text':'visible red round shape', 'source_type':'customer_document',
                  'document_id':str(i),'visual_description':True} for i in range(2)]]
        original=copy.deepcopy(rows)
        selected,text,wire,audit=self.prepare(rows,{},ceiling=2200,
            required_document_ids=['0','1'],operation='comparison',scope='cross_document')
        self.assertTrue(audit['compression_applied'])
        self.assertTrue(wire['budget_satisfied'])
        self.assertLessEqual(wire['actual_prompt_tokens'],2200)
        self.assertEqual({r.get('document_id') for r in selected if r.get('document_id')},{'0','1'})
        self.assertEqual(rows,original)

    def test_oversized_wrapper_fails_preflight_not_by_raising_limit(self):
        _,_,wire,audit=self.prepare([{'evidence_id':'A','text':'fact'}],
            {'customer_question':'x'*3000},ceiling=1000)
        self.assertFalse(audit['budget_satisfied'])
        self.assertFalse(wire['budget_satisfied'])
        self.assertEqual(wire['max_prompt_tokens'],1000)

    def test_real_composer_preflights_complete_fields(self):
        from unittest.mock import patch
        from backend.app import DraftRequest, prepare_complete_final_input
        from backend.sales.tool_planner import ToolPlan
        from backend.tests.sales.test_context_engine import _CharacterTokenizer
        request=DraftRequest(customer_question='compare')
        request._prepared_evidence=[{'evidence_id':'A','text':'visible texture','source_type':'customer_document','document_id':'d'}]
        request._required_documents=[{'document_id':'d','document_name':'image'}]
        plan=ToolPlan(tools=['customer_documents'],operation='comparison')
        payload={'evidence':request._prepared_evidence,'customer_question':'compare',
            'attachment_task':{},'attachment_visuals':{'labels':'texture '*30},'text_evidence':['A']}
        with patch('backend.sales.evidence_expansion.native_provider',return_value=None):
            selected,text,wire=prepare_complete_final_input(request,plan,payload,_CharacterTokenizer(),
                system_prompt='system + scope',max_prompt_tokens=5000)
        self.assertTrue(wire['budget_satisfied'])
        self.assertEqual(request._support_budget_audit['actual_prompt_tokens'],wire['actual_prompt_tokens'])
        self.assertIn('attachment_task',json.loads(text))
        self.assertIn('attachment_visuals',json.loads(text))
        self.assertEqual(request._prepared_evidence,selected)


if __name__=='__main__':unittest.main()
