import unittest
from unittest.mock import patch
from backend.sales.vector_store import QdrantStore, point_id, collection_name


class VectorStoreTests(unittest.TestCase):
    def test_loopback_only(self):
        for url in ('https://example.com','http://10.0.0.1:6333','http://user:pass@localhost:6333'):
            with self.assertRaises(ValueError):QdrantStore(url)

    def test_empty_scope_never_searches_all(self):
        store=QdrantStore()
        with patch.object(store,'call') as call:
            self.assertEqual(store.search([1.0],fingerprint='a'*64,allowed_ids=[]),[])
            call.assert_not_called()

    def test_filter_and_output_recheck(self):
        store=QdrantStore()
        result={'points':[{'id':point_id('a'),'payload':{'evidence_id':'a','fingerprint':'a'*64},'score':.9},
                          {'id':point_id('private'),'payload':{'evidence_id':'private','fingerprint':'a'*64},'score':1},
                          {'id':point_id('b'),'payload':{'evidence_id':'b','fingerprint':'old'},'score':.8}]}
        with patch.object(store,'call',return_value=result) as call:
            self.assertEqual(store.search([1.],fingerprint='a'*64,allowed_ids=['a','b']),[('a',.9)])
            wire=call.call_args.args[2]
            self.assertEqual(set(wire['filter']['must'][0]['has_id']),{point_id('a'),point_id('b')})

    def test_stable_id_and_versioned_collection(self):
        self.assertEqual(point_id('original-ID'),point_id('original-ID'))
        self.assertNotEqual(collection_name('a'*64),collection_name('b'*64))


if __name__=='__main__':unittest.main()
