import sqlite3
import unittest
from backend.sales.project_identity import merge_projects

class ProjectIdentityTests(unittest.TestCase):
    def make_db(self):
        db=sqlite3.connect(':memory:')
        db.executescript('''PRAGMA foreign_keys=ON;
        CREATE TABLE project_cases(case_id TEXT PRIMARY KEY,project_name TEXT,access_scope TEXT,image_count INT);
        INSERT INTO project_cases VALUES('old','Old alias','public',1),('main','Selected name','public',1);
        CREATE TABLE project_case_visuals(case_id TEXT REFERENCES project_cases,asset_id TEXT,PRIMARY KEY(case_id,asset_id));
        INSERT INTO project_case_visuals VALUES('old','A'),('old','B'),('main','B');
        CREATE TABLE restored_source_facts(fact_id TEXT PRIMARY KEY,case_id TEXT REFERENCES project_cases);
        INSERT INTO restored_source_facts VALUES('F','old');''')
        return db

    def test_union_edges_preserve_facts_and_alias_without_duplicate_project(self):
        db=self.make_db()
        with db: self.assertEqual(merge_projects(db,[('old','main')]),1)
        self.assertEqual(db.execute('select count(*) from project_cases').fetchone()[0],1)
        self.assertEqual(db.execute('select asset_id from project_case_visuals order by asset_id').fetchall(),[('A',),('B',)])
        self.assertEqual(db.execute('select * from restored_source_facts').fetchall(),[('F','main')])
        self.assertEqual(db.execute('select * from project_name_aliases').fetchall(),[('main','Old alias')])
        self.assertEqual(db.execute('pragma foreign_key_check').fetchall(),[])
        self.assertEqual(merge_projects(db,[('old','main')]),0)

    def test_acl_crossing_is_not_permitted(self):
        db=self.make_db()
        db.execute("update project_cases set access_scope='internal' where case_id='main'")
        with self.assertRaisesRegex(ValueError,'invalid_project_identity'):
            merge_projects(db,[('old','main')])
