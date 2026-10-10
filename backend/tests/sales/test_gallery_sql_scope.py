"""Structured gallery scope uses actual catalogue fields, never query keywords."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from backend.sales.catalog_store import query_product_gallery
from backend.sales.retriever import LocalRagRetriever


class GallerySqlScopeTests(unittest.TestCase):
    def test_backend_entity_scope_uses_links_and_acl_not_model_gallery_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'catalog.sqlite3'
            with closing(sqlite3.connect(db)) as con, con:
                con.executescript('CREATE TABLE visual_assets(asset_id TEXT,category TEXT,access_scope TEXT);'
                    'CREATE TABLE visual_asset_labels(asset_id TEXT,labels_json TEXT);'
                    'CREATE TABLE products(product_id TEXT,access_scope TEXT);'
                    'CREATE TABLE product_visuals(product_id TEXT,asset_id TEXT);')
                for key,pid,access in [('a','p1','public'),('b','p2','public'),('c','secret','internal')]:
                    con.execute('INSERT INTO products VALUES(?,?)',(pid,access))
                    con.execute('INSERT INTO product_visuals VALUES(?,?)',(pid,key))
                    con.execute('INSERT INTO visual_assets VALUES(?,?,?)',(key,'product_sample','public'))
                    con.execute('INSERT INTO visual_asset_labels VALUES(?,?)',(key,json.dumps({
                        'display_ready':True,'label_issues':[],'product_gallery_eligible':True,'sha256':key})))
            self.assertEqual(query_product_gallery(entity_ids=['p1'],database_path=db),['a'])
            self.assertEqual(query_product_gallery(entity_ids=['p1','p2'],database_path=db),['a','b'])
            self.assertEqual(query_product_gallery(entity_ids=['secret'],database_path=db),[])
            self.assertEqual(query_product_gallery(entity_ids=[],database_path=db),[])
            self.assertEqual(query_product_gallery(entity_ids=["' OR 1=1 --"],database_path=db),[])

    def test_real_broad_query_uses_original_visual_query_without_name_match(self):
        retriever=LocalRagRetriever()
        question='你们单板的各类图片给我一下'
        result=retriever.retrieve(question+'\n请提供真岩®石单板的各类产品图片，包含不同配色和型号，来自企业产品目录。',
            visual_query=question,visual_k=5,wants_visuals=True,visual_scope='product',
            product_overview_request=False,visual_product_family='decorative_panel',visual_gallery_scope='category')
        self.assertEqual(len(result['visual_assets']),5)
        self.assertTrue(all(a['image_labels']['product_family_key']=='decorative_panel' for a in result['visual_assets']))
        self.assertTrue(all(a['selection_reason'] in {'sql_catalog_category','sql_entity_relation'} for a in result['visual_assets']))

    def test_real_exact_code_never_substitutes_other_products(self):
        ids=query_product_gallery(scope='exact',target='GHM2517')
        self.assertTrue(ids)
        from backend.sales.visual_labels import labels_for_asset
        self.assertTrue(all('GHM2517' in (labels_for_asset(key).get('variant_or_code') or '') for key in ids))
        self.assertEqual(query_product_gallery(scope='exact',target='NONEXISTENT-SKU-91837'),[])

    def test_sql_parameters_acl_and_diversity_before_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'catalog.sqlite3'
            with closing(sqlite3.connect(db)) as con, con:
                con.executescript('CREATE TABLE visual_assets(asset_id TEXT,category TEXT,access_scope TEXT);'
                    'CREATE TABLE visual_asset_labels(asset_id TEXT,labels_json TEXT);')
                for key,variant,access in [('a','Finish A','public'),('b','Finish A','public'),
                                            ('c','Finish B','public'),('d','Secret','internal')]:
                    labels={'display_ready':True,'label_issues':[],'product_gallery_eligible':True,
                        'product_family_key':'decorative_panel','variant_or_code':variant,'sha256':key,
                        'visual_role':'product_variant'}
                    con.execute('INSERT INTO visual_assets VALUES(?,?,?)',(key,'product_sample',access))
                    con.execute('INSERT INTO visual_asset_labels VALUES(?,?)',(key,json.dumps(labels)))
            ids=query_product_gallery(scope='category',family='decorative_panel',database_path=db)
            self.assertEqual(ids,['a','c','b'])
            self.assertEqual(query_product_gallery(scope='exact',target="' OR 1=1 --",database_path=db),[])
            self.assertEqual(query_product_gallery(scope='category',family='decorative_panel',
                allowed_access_scopes=[],database_path=db),[])
            self.assertIsNone(query_product_gallery(scope='category',family='made_up',database_path=db))
            self.assertIsNone(query_product_gallery(scope='category',family='unknown',database_path=db))


if __name__=='__main__':unittest.main()
