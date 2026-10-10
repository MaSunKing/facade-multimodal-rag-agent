import unittest
from backend.sales.entity_directory import resolve_entities


class DirectoryLocationRankingTests(unittest.TestCase):
    def test_project_description_includes_registered_province_and_city(self):
        cases = [
            {'case_id':'p1', 'project_name':'港口工程', 'province':'甲省', 'city':'乙市'},
            {'case_id':'p2', 'project_name':'公共建筑', 'province':None, 'city':None},
        ]
        observed = []
        def scorer(query, descriptions):
            observed.extend(descriptions)
            return [0.9, 0.2]
        _, selected, meta = resolve_entities([], cases, '区域项目查询',
            categories=['05_project_case'], targets=['模糊对象'], scorer=scorer)
        self.assertEqual(observed, ['港口工程 甲省 乙市', '公共建筑  '])
        self.assertEqual(selected[0]['case_id'], 'p1')
        self.assertTrue(meta['reranker_used'])
        self.assertNotIn('None', ''.join(observed))

    def test_product_description_and_structured_location_filter_unchanged(self):
        observed = []
        def scorer(query, descriptions):
            observed.extend(descriptions)
            return [0.8]
        resolve_entities([{'product_id':'x', 'name':'材料甲', 'family':'板材体系'}], [],
            '不明确的材料', categories=['02_product_information'], targets=['模糊对象'], scorer=scorer)
        self.assertEqual(observed, ['材料甲 板材体系'])
        _, selected, meta = resolve_entities([], [
            {'case_id':'p', 'project_name':'港口工程', 'province':'甲省', 'city':'乙市'}],
            '地区查询', categories=['05_project_case'], filters={'locations':['甲省']}, scorer=scorer)
        self.assertEqual([r['case_id'] for r in selected], ['p'])
        self.assertFalse(meta['reranker_used'])


if __name__ == '__main__':
    unittest.main()
