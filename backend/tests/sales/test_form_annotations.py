import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from backend.sales.form_annotations import annotate_text, build_annotations, apply_annotations, text_hash


class FormAnnotationsTest(unittest.TestCase):
    def section(self, **kw):
        return dict(section_id='S',title='单板检测报告 AB12345678',source_sha256='abcd'*16,
                    families=['真岩®无机仿石装饰板'],evidence_ids=['linked'],**kw)

    def test_form_is_not_guessed_from_material_thickness_or_method(self):
        self.assertEqual(build_annotations([{'id':'x','kind':'text','text':'人造无机石材装饰板，13mm，干挂'}],[]),[])
        self.assertEqual(annotate_text('保温一体板面层为人造无机石材装饰板','单板'),
                         '保温一体板面层为人造无机石材装饰板')

    def test_same_source_record_and_direct_membership(self):
        docs=[{'id':'cover','kind':'text','text':'AB12345678 样品人造无机石材装饰板（真岩石）',
               'source_refs':[{'document_id':'restored_'+'abcd'*4}]},
              {'id':'other','kind':'text','text':'AB12345678 人造无机石材装饰板',
               'source_refs':[{'document_id':'different'}]},
              {'id':'linked','kind':'text','text':'检测日期及送检规格'}]
        rows=build_annotations(docs,[self.section()])
        self.assertEqual({r['evidence_id'] for r in rows},{'cover','linked'})
        self.assertIn('人造无机石材装饰板（真岩石）（单板）',rows[0]['annotated_text'])
        self.assertIn('板型标注：单板',rows[1]['annotated_text'])

    def test_shared_section_and_stale_source_do_not_apply(self):
        section=self.section(); section['families'].append('真岩®保温装饰一体板')
        doc={'id':'linked','kind':'text','text':'人造无机石材装饰板'}
        self.assertEqual(build_annotations([doc],[section]),[])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'test.sqlite3'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE evidence_form_annotations(evidence_id,source_text_hash,product_form,annotated_text,provenance_json)')
                db.execute('INSERT INTO evidence_form_annotations VALUES (?,?,?,?,?)',
                    ('linked',text_hash('old source'),'单板','wrong annotation','{}'))
            self.assertEqual(apply_annotations([doc],str.split,path),[doc])

    def test_overlay_preserves_source_and_id(self):
        doc={'id':'linked','kind':'text','text':'人造无机石材装饰板','access_scope':'internal'}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'test.sqlite3'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE evidence_form_annotations(evidence_id,source_text_hash,product_form,annotated_text,provenance_json)')
                db.execute('INSERT INTO evidence_form_annotations VALUES (?,?,?,?,?)',
                    ('linked',text_hash(doc['text']),'单板','人造无机石材装饰板（单板）',json.dumps({'section_ids':['S']})))
            row=apply_annotations([doc],str.split,path)[0]
            self.assertEqual(row['source_text'],doc['text'])
            self.assertEqual(doc['text'],'人造无机石材装饰板')
            self.assertEqual(row['access_scope'],'internal')
            self.assertEqual(row['id'],'linked')

if __name__=='__main__': unittest.main()
