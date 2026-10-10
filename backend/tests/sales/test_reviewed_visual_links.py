import json
import sqlite3
import unittest
from unittest.mock import patch
from backend.sales.reviewed_visual_links import apply_reviewed_links


class ReviewedLinksTests(unittest.TestCase):
    def test_idempotent_preserves_original_metadata(self):
        c=sqlite3.connect(':memory:')
        c.executescript('''CREATE TABLE visual_assets(asset_id TEXT PRIMARY KEY,title TEXT,metadata_json TEXT);
          CREATE TABLE products(product_id TEXT PRIMARY KEY);
          CREATE TABLE product_visuals(product_id TEXT,asset_id TEXT,role TEXT,display_order INTEGER);
          INSERT INTO visual_assets VALUES('a','Original title','{"section_heading":"original heading"}');
          INSERT INTO products VALUES('p');''')
        payload={'schema_version':'reviewed_visual_links_v1','links':[{'asset_id':'a','product_id':'p',
          'title':'Reviewed title','review_basis':'explicit user confirmation'}]}
        with patch('pathlib.Path.is_file',return_value=True),patch('pathlib.Path.read_text',return_value=json.dumps(payload)):
            apply_reviewed_links(c)
            apply_reviewed_links(c)
        self.assertEqual(c.execute('SELECT count(*) FROM product_visuals').fetchone()[0],1)
        title,m=c.execute('SELECT title,metadata_json FROM visual_assets').fetchone()
        self.assertEqual(title,'Reviewed title')
        m=json.loads(m)
        self.assertEqual(m['section_heading'],'original heading')
        self.assertEqual(m['original_title_before_review'],'Original title')
        c.close()

    def test_category_only_review_does_not_invent_product_binding(self):
        c=sqlite3.connect(':memory:')
        c.executescript('''CREATE TABLE visual_assets(asset_id TEXT PRIMARY KEY,title TEXT,metadata_json TEXT);
          CREATE TABLE products(product_id TEXT PRIMARY KEY);
          CREATE TABLE product_visuals(product_id TEXT,asset_id TEXT,role TEXT,display_order INTEGER);
          INSERT INTO visual_assets VALUES('a','Source title','{"catalog":{"visual_role":"product_variant"}}');''')
        payload={'schema_version':'reviewed_visual_links_v1','links':[{'asset_id':'a',
          'title':'Accessory','review_basis':'user approval','directory_category':'installation_accessory',
          'visual_role':'accessory','product_gallery_eligible':False,'product_family_key':'unknown'}]}
        with patch('pathlib.Path.is_file',return_value=True),patch('pathlib.Path.read_text',return_value=json.dumps(payload)):
            apply_reviewed_links(c)
        self.assertEqual(c.execute('SELECT count(*) FROM product_visuals').fetchone()[0],0)
        m=json.loads(c.execute('SELECT metadata_json FROM visual_assets').fetchone()[0])
        self.assertFalse(m['catalog']['product_gallery_eligible'])
        self.assertEqual(m['original_catalog_before_review']['visual_role'],'product_variant')
        c.close()
