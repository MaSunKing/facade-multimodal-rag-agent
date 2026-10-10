import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from backend.sales.knowledge_scope_notes import rebuild_scope_notes, enrich_scope_notes


class ScopeNoteTests(unittest.TestCase):
    def fixture(self, c):
        c.execute('CREATE TABLE knowledge_text_evidence(evidence_id TEXT PRIMARY KEY,text TEXT,access_scope TEXT)')
        c.execute('CREATE TABLE knowledge_evidence_sources(evidence_id TEXT,reference_json TEXT)')
        c.executemany('INSERT INTO knowledge_text_evidence VALUES(?,?,?)',
                      [('target','原介绍','public'),('proof','已核对样品','public'),('private','私有资料','internal')])
        for eid, ref in [('target',{'source_url':'https://example.org/item','document_name':'公开对象'}),
                         ('proof',{'document_id':'report-A','source_page':4,'document_name':'样品记录'}),
                         ('private',{'source_url':'https://example.org/private'})]:
            c.execute('INSERT INTO knowledge_evidence_sources VALUES(?,?)',(eid,json.dumps(ref)))

    def policy(self, supports=None):
        return {'basis':'reviewed_source_scope','notes':[{'id':'sample-scope','targets':['target'],
            'supports':supports or ['proof'],'text':'只说明送检样品，不外推所有组合。'}]}

    def test_idempotent_source_rows_unchanged_and_missing_anchors_skipped(self):
        with sqlite3.connect(':memory:') as c:
            self.fixture(c)
            before=c.execute('SELECT * FROM knowledge_text_evidence').fetchall()
            policy=self.policy()
            first=rebuild_scope_notes(c,policy)
            self.assertEqual(first,rebuild_scope_notes(c,policy))
            self.assertEqual(first,{'notes':1,'targets':1,'skipped':0})
            self.assertEqual(before,c.execute('SELECT * FROM knowledge_text_evidence').fetchall())
            self.assertEqual(rebuild_scope_notes(c,self.policy(['missing']))['skipped'],1)

    def test_enrichment_id_url_and_acl_no_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'catalog.sqlite3'
            with closing(sqlite3.connect(db)) as c, c:
                self.fixture(c); rebuild_scope_notes(c,self.policy())
            rows=[{'chunk_id':'target','text':'原介绍','citations':[]},
                  {'chunk_id':'sql-record','text':'SQL资料','citations':[{'source_url':'https://example.org/item'}]},
                  {'chunk_id':'else','text':'其他对象','citations':[]},
                  {'chunk_id':'target','text':'内部对象','access_scope':'internal'}]
            enriched=enrich_scope_notes(rows,db)
            self.assertEqual(rows[0]['text'],'原介绍')
            for row in enriched[:2]:
                self.assertIn('只说明送检样品',row['text'])
                self.assertEqual(row['scope_note_ids'],['sample-scope'])
                self.assertTrue(any(r.get('document_id')=='report-A' for r in row['citations']))
            self.assertEqual(enriched[2:],rows[2:])
            self.assertEqual(enrich_scope_notes(enriched,db),enriched)
            with closing(sqlite3.connect(db)) as c, c:
                rebuild_scope_notes(c,self.policy(['private']))
            self.assertEqual(enrich_scope_notes(rows,db),rows)

    def test_legacy_and_invalid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'old.sqlite3'
            rows=[{'text':'没有新表'}]
            self.assertEqual(enrich_scope_notes(rows,db),rows)
            with closing(sqlite3.connect(db)) as c, c:
                self.assertEqual(rebuild_scope_notes(c,self.policy())['notes'],0)
                self.fixture(c)
                bad=self.policy(); bad['notes'][0]['supports']=[]
                with self.assertRaises(ValueError):rebuild_scope_notes(c,bad)
            self.assertEqual(enrich_scope_notes(rows,db),rows)

    def test_final_and_revision_share_generic_boundary(self):
        from backend.sales.draft_prompt import company_prompt, document_prompt, general_prompt
        from backend.sales.answer_review import REVISION_PROMPT
        from backend.sales.generation_policy import RELATION_EVIDENCE_POLICY
        for prompt in (company_prompt('公司身份'),document_prompt(),general_prompt(),REVISION_PROMPT):
            self.assertIn(RELATION_EVIDENCE_POLICY,prompt)
            self.assertIn('没有某个组成部分，不等于相应性能为零',prompt)
            self.assertIn('不同对象或条件的值分别说明',prompt)


if __name__=='__main__':unittest.main()
