import unittest
from backend.sales.gallery_ranking import rank_gallery_candidates


class GalleryRankingTests(unittest.TestCase):
    def test_relevance_beats_asset_id_then_diversity_and_hash_dedup(self):
        labels = {
            'a': {'variant_or_code': 'A', 'sha256': 'a'},
            'z': {'variant_or_code': 'A', 'sha256': 'z'},
            'b': {'variant_or_code': 'B', 'sha256': 'b'},
            'c': {'variant_or_code': 'C', 'sha256': 'z'},
        }
        items = [(1, {'asset_id': 'a'}), (9, {'asset_id': 'z'}),
                 (3, {'asset_id': 'b'}), (2, {'asset_id': 'c'})]
        result = rank_gallery_candidates(items, labels.get)
        self.assertEqual([item[1]['asset_id'] for item in result], ['z', 'b', 'a'])
        self.assertEqual(len(items), 4)

    def test_missing_variant_is_not_collapsed_into_one_group(self):
        items = [(2, {'asset_id': 'a'}), (1, {'asset_id': 'b'})]
        self.assertEqual(rank_gallery_candidates(items, lambda key: {}), items)

    def test_registered_projects_are_diverse_before_repeated_views(self):
        items=[(9,{'asset_id':'a'}),(8,{'asset_id':'b'}),(2,{'asset_id':'c'})]
        labels={'a':{'project_names':['Alpha']},'b':{'project_names':['Alpha']},
                'c':{'project_names':['Beta']}}
        self.assertEqual([a['asset_id'] for _,a in rank_gallery_candidates(items,labels.get)],
                         ['a','c','b'])
