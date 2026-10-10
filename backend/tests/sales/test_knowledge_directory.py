import json
import sqlite3
import unittest
from backend.sales.knowledge_directory import classify_directory


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        self.c=sqlite3.connect(':memory:')
        self.c.executescript('''
          CREATE TABLE visual_assets(asset_id TEXT PRIMARY KEY,title TEXT,category TEXT,
            document_name TEXT,source_page INTEGER,metadata_json TEXT);
          CREATE TABLE knowledge_documents(document_key TEXT PRIMARY KEY,document_name TEXT);
          CREATE TABLE knowledge_text_evidence(evidence_id TEXT PRIMARY KEY,text TEXT);
          CREATE TABLE knowledge_evidence_sources(evidence_id TEXT,document_key TEXT,source_page INTEGER,source_key TEXT);
          CREATE TABLE products(product_id TEXT PRIMARY KEY,name TEXT);
          CREATE TABLE project_cases(case_id TEXT PRIMARY KEY,project_name TEXT);
          CREATE TABLE product_visuals(product_id TEXT,asset_id TEXT);
          CREATE TABLE project_case_visuals(case_id TEXT,asset_id TEXT);
          INSERT INTO knowledge_documents VALUES('d','manual');
          INSERT INTO knowledge_text_evidence VALUES('e','Long Product usage');
          INSERT INTO knowledge_evidence_sources VALUES('e','d',2,'s');
          INSERT INTO products VALUES('p','Long Product');
          INSERT INTO product_visuals VALUES('p','a');
          INSERT INTO visual_assets VALUES('a','Node diagram','node','manual',2,'{}');
          INSERT INTO visual_assets VALUES('b',NULL,'unknown','manual',3,'{}');
        ''')

    def tearDown(self):
        self.c.close()

    def test_labels_categories_and_context_not_certification(self):
        r=classify_directory(self.c)
        self.assertEqual(r['assets'],2)
        self.assertEqual(r['labels_missing'],0)
        self.assertEqual(r['classification_pending'],1)
        self.assertEqual(self.c.execute("SELECT category FROM visual_directory WHERE asset_id='a'").fetchone()[0],'node')
        self.assertEqual(self.c.execute('SELECT relation FROM visual_evidence').fetchone()[0],'same_page_context')
        states={r[0] for r in self.c.execute('SELECT review_status FROM product_evidence')}
        self.assertEqual(states,{'context_only','needs_review'})

    def test_idempotent_and_no_raw_mutation(self):
        before=self.c.execute('SELECT * FROM visual_assets').fetchall()
        a=classify_directory(self.c)
        b=classify_directory(self.c)
        self.assertEqual(a,b)
        self.assertEqual(before,self.c.execute('SELECT * FROM visual_assets').fetchall())

    def test_ambiguous_document_not_linked(self):
        self.c.execute("INSERT INTO knowledge_documents VALUES('d2','manual')")
        r=classify_directory(self.c)
        self.assertEqual(r['visual_document_links'],0)
        self.assertEqual(r['visual_evidence'],0)

    def test_multi_image_entity(self):
        self.c.execute("INSERT INTO visual_assets VALUES('c','Another node','node','manual',2,'{}')")
        self.c.execute("INSERT INTO product_visuals VALUES('p','c')")
        classify_directory(self.c)
        self.assertEqual(self.c.execute('SELECT count(*) FROM visual_evidence').fetchone()[0],2)
        self.assertEqual(self.c.execute("SELECT count(*) FROM product_evidence WHERE relation='linked_image_source_context'").fetchone()[0],1)


if __name__=='__main__':
    unittest.main()
