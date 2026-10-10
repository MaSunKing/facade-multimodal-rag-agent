import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from backend.sales.catalog_store import query_product_gallery, choose_gallery_scope_ids


class ResolvedGalleryScopeTests(unittest.TestCase):
    def test_alias_reuses_directory_family_but_unmatched_exact_scope_does_not(self):
        resolution = {'entity_targets':[{'id':'family:体系甲', 'name':'体系甲',
                                        'kind':'family', 'aliases':['简称甲']}]}
        self.assertEqual(choose_gallery_scope_ids([], ['a','b'], scope='exact',
            target='简称甲', resolution=resolution), ['a','b'])
        self.assertEqual(choose_gallery_scope_ids([], ['a','b'], scope='exact',
            target='不存在的型号', resolution=resolution), [])
        self.assertEqual(choose_gallery_scope_ids(['other'], ['a','b'], scope='category',
            target='简称甲', resolution=resolution), ['other'])
        self.assertEqual(choose_gallery_scope_ids(None, ['a'], scope='unknown',
            target='', resolution=resolution), ['a'])

    def test_family_relations_resolve_all_public_children_and_enforce_both_acls(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder)/'catalog.sqlite3'
            with closing(sqlite3.connect(db)) as c, c:
                c.executescript('CREATE TABLE products(product_id TEXT,family TEXT,access_scope TEXT);'
                    'CREATE TABLE product_visuals(product_id TEXT,asset_id TEXT);'
                    'CREATE TABLE visual_assets(asset_id TEXT,category TEXT,access_scope TEXT);'
                    'CREATE TABLE visual_asset_labels(asset_id TEXT,labels_json TEXT);')
                rows = [('a','F','public','public'),('b','F','public','public'),
                        ('c','F','public','public'),('d','F','internal','public'),
                        ('e','F','public','internal'),('f','G','public','public')]
                for key, family, product_acl, image_acl in rows:
                    c.execute('INSERT INTO products VALUES(?,?,?)',(key,family,product_acl))
                    c.execute('INSERT INTO product_visuals VALUES(?,?)',(key,key))
                    c.execute('INSERT INTO visual_assets VALUES(?,?,?)',(key,'product_sample',image_acl))
                    c.execute('INSERT INTO visual_asset_labels VALUES(?,?)',(key,json.dumps({
                        'product_name':key,'sha256':key,'product_gallery_eligible':True,
                        'display_ready':True,'label_issues':[]})))
            self.assertEqual(query_product_gallery(entity_ids=['family:F'],database_path=db),['a','b','c'])
            self.assertEqual(query_product_gallery(entity_ids=['b'],database_path=db),['b'])
            self.assertEqual(query_product_gallery(entity_ids=['family:unknown'],database_path=db),[])
            self.assertEqual(query_product_gallery(entity_ids=["family:' OR 1=1 --"],database_path=db),[])
            self.assertEqual(query_product_gallery(entity_ids=['family:F'],database_path=db,
                allowed_access_scopes=[]),[])

    def test_revision_policy_preserves_other_supported_components(self):
        from backend.sales.answer_review import REVISION_PROMPT
        self.assertIn('修改是修补原答案', REVISION_PROMPT)
        self.assertIn('不因缺图把整题改成answerable=false', REVISION_PROMPT)
        self.assertIn('不拼回未经审核的原草稿', REVISION_PROMPT)


if __name__ == '__main__':
    unittest.main()
