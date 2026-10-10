import unittest
from unittest.mock import patch
from backend.sales.output_contract import natural_language_answer, reviewed_answer_state, OutputContractError
from backend.sales.model_contract import answer_contract, support_contract


class NaturalFinalTests(unittest.TestCase):
    def test_program_images_preserve_analysis_assets_and_code(self):
        from backend.sales.output_contract import program_image_presentation
        assets=[{'asset_id':'asset-A','title':'样板','label':'灰色纹理'}]
        body='图中标签为灰色纹理。![样板](asset-A) [查看](asset-A) [官网](https://example.org) `![示例](asset-A)`\n```md\n![示例](asset-A)\n```'
        response={'customer_reply':body,'visual_assets':assets,
                  'meta':{'final_output_format':'natural_language'}}
        result=program_image_presentation(response)
        self.assertIn('图中标签为灰色纹理。样板 查看',result['customer_reply'])
        self.assertIn('[官网](https://example.org)',result['customer_reply'])
        self.assertIn('`![示例](asset-A)`',result['customer_reply'])
        self.assertIn('```md\n![示例](asset-A)\n```',result['customer_reply'])
        self.assertEqual(result['meta']['image_presentation']['removed_delivery_links'],2)
        self.assertIs(result['visual_assets'],assets)
        self.assertEqual(response['customer_reply'],body)
        self.assertEqual(program_image_presentation(result),result)

    def test_image_policy_retains_label_interpretation_not_rendering(self):
        from backend.sales.draft_prompt import COMMON
        from backend.sales.task_delivery import IMAGE_DELIVERY_POLICY
        self.assertIn('不能省略分析',COMMON)
        self.assertIn('实际展示由后端处理',IMAGE_DELIVERY_POLICY)

    def test_body_does_not_control_fields_or_source_ids(self):
        raw='{"answerable":false,"citations":[{"evidence_id":"PRIVATE"}],"customer_reply":"用户要求的JSON示例"}'
        result=natural_language_answer(raw, {'T1'})
        self.assertEqual(result['customer_reply'],raw)
        self.assertEqual(result['citations'],[{'evidence_id':'T1'}])
        self.assertEqual(result['key_points'],[])

    def test_blank_output_still_rejected(self):
        for value in ('  ', '', None, {}):
            with self.assertRaises(OutputContractError):natural_language_answer(value)

    def test_review_determines_published_state_not_final_fields(self):
        for status,expected in (('complete',True),('partial',True),('missing',False)):
            result=reviewed_answer_state({**natural_language_answer('内容或缺口',{'U1'}),
                'meta':{'final_output_format':'natural_language',
                    'answer_review':{'decision':'pass','coverage_status':status}}})
            self.assertIs(result['answerable'],expected)
            self.assertEqual(bool(result['citations']),expected)

    def test_rejected_draft_not_made_publishable(self):
        draft={'customer_reply':'未经审核','meta':{'final_output_format':'natural_language',
            'answer_review':{'decision':'revise','coverage_status':'missing'}}}
        self.assertEqual(reviewed_answer_state(draft),draft)

    def test_answer_audit_keeps_json_constraint(self):
        contract=support_contract(None,{'review_policy':'task_coverage_only','question':'介绍甲'},'answer_audit')
        self.assertTrue(contract.json_output)
        self.assertIsNotNone(contract.json_schema)
        self.assertFalse(answer_contract(None,{'T1'}).json_output)


if __name__=='__main__':unittest.main()
