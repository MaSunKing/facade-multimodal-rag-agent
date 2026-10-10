import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.sales.relation_scope import resolve_relation_scope, category_candidate_ids
from backend.sales.requirements import build_requirements


class DirectoryLevelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'catalog.sqlite3'
        with sqlite3.connect(self.path) as c:
            c.executescript('''
                CREATE TABLE products(product_id TEXT,name TEXT,family TEXT,surface_finish TEXT,access_scope TEXT);
                CREATE TABLE project_cases(case_id TEXT,project_name TEXT,access_scope TEXT);
                CREATE TABLE knowledge_text_evidence(evidence_id TEXT,text TEXT,access_scope TEXT,metadata_json TEXT);
                CREATE TABLE knowledge_evidence_sources(evidence_id TEXT,document_key TEXT);
                CREATE TABLE technical_family_aliases(alias TEXT,family TEXT);
                CREATE TABLE product_evidence(product_id TEXT,evidence_id TEXT);
                CREATE TABLE technical_sections(section_id TEXT,title TEXT,evidence_ids_json TEXT,asset_ids_json TEXT,provenance_json TEXT,access_scope TEXT);
                CREATE TABLE family_section_links(family TEXT,section_id TEXT,relation TEXT);
                INSERT INTO products VALUES('a1','Red A','System Alpha','','public');
                INSERT INTO products VALUES('a2','Blue A','System Alpha','','public');
                INSERT INTO products VALUES('b1','White B','System Beta','','public');
                INSERT INTO technical_family_aliases VALUES('alpha','System Alpha');
                INSERT INTO technical_family_aliases VALUES('beta','System Beta');
                INSERT INTO technical_sections VALUES('sa','Alpha detail','["na"]','[]','{}','public');
                INSERT INTO technical_sections VALUES('sb','Beta detail','["nb"]','[]','{}','public');
                INSERT INTO family_section_links VALUES('System Alpha','sa','node_reference');
                INSERT INTO family_section_links VALUES('System Beta','sb','node_reference');
            ''')
            for eid,text,access,category in [('na','Alpha joint','public','04_node_atlas'),
                    ('nb','Beta joint','public','04_node_atlas'),
                    ('hidden','Private joint','private','04_node_atlas'),
                    ('intro','Company identity','public','06_enterprise_information')]:
                c.execute('INSERT INTO knowledge_text_evidence VALUES(?,?,?,?)',
                    (eid,text,access,json.dumps({'knowledge_domains':[category]})))
                c.execute('INSERT INTO knowledge_evidence_sources VALUES(?,?)',(eid,'manual'))

    def tearDown(self):
        self.tmp.cleanup()

    def test_family_comparison_keeps_two_families_not_three_variants(self):
        value=resolve_relation_scope(self.path,'compare alpha beta joints',
            categories=['04_node_atlas'],targets=['alpha','beta'],operation='comparison')
        self.assertEqual(len(value['entity_targets']),2)
        self.assertEqual({r['kind'] for r in value['entity_targets']},{'family'})
        req=build_requirements(value['entity_targets'],['structure','installation'],operation='comparison')
        self.assertEqual(len(req),4)
        scopes={g['entity_id']:g['ids'] for g in value['evidence_groups'] if g['relation']=='family_reference'}
        self.assertEqual(scopes['family:System Alpha'],['na'])
        self.assertEqual(scopes['family:System Beta'],['nb'])
        self.assertEqual(set(value['directory_member_ids']),{'a1','a2','b1'})

    def test_explicit_child_is_not_replaced_with_its_family(self):
        value=resolve_relation_scope(self.path,'alpha Red A joint',
            categories=['04_node_atlas'],targets=['Red A'],operation='lookup')
        self.assertEqual([t['id'] for t in value['entity_targets']],['a1'])
        self.assertEqual(value['entity_targets'][0]['kind'],'entity')

    def test_product_filter_does_not_promote_related_projects_to_node_targets(self):
        with sqlite3.connect(self.path) as c:
            c.execute('ALTER TABLE project_cases ADD COLUMN product_name TEXT')
            c.execute("INSERT INTO project_cases VALUES('project1','Unrelated project','public','alpha')")
        value=resolve_relation_scope(self.path,'compare alpha beta joints',
            categories=['04_node_atlas'],targets=['alpha','beta'],
            filters={'products':['alpha','beta']},operation='comparison')
        self.assertEqual({t['id'] for t in value['entity_targets']},
                         {'family:System Alpha','family:System Beta'})
        req=build_requirements(value['entity_targets'],['joints'],operation='comparison')
        self.assertEqual(len(req),2)

    def test_category_pool_is_classified_and_public_only(self):
        self.assertEqual(category_candidate_ids(self.path,['04_node_atlas']),{'na','nb'})
        self.assertEqual(category_candidate_ids(self.path,['06_enterprise_information']),{'intro'})
        self.assertEqual(category_candidate_ids(self.path,['unknown']),set())

    def test_unique_registered_suffix_keeps_omitted_target_bindable(self):
        with sqlite3.connect(self.path) as c:
            c.execute("UPDATE technical_family_aliases SET alias='qualified alpha' WHERE alias='alpha'")
        value=resolve_relation_scope(self.path,'alpha joints',categories=['04_node_atlas'],
            targets=['alpha'],operation='lookup')
        self.assertEqual(value['entity_targets'][0]['id'],'family:System Alpha')
        req=build_requirements(value['entity_targets'],[],operation='lookup',
            requests=[{'targets':['alpha'],'query':'joint detail'}])
        self.assertEqual(req[0]['target_kind'],'family')
        self.assertEqual(req[0]['target_id'],'family:System Alpha')

    def test_private_family_alias_is_not_exposed_as_a_target(self):
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT INTO products VALUES('secret','Hidden item','Private System','','private')")
            c.execute("INSERT INTO technical_family_aliases VALUES('hiddenfamily','Private System')")
        value=resolve_relation_scope(self.path,'hiddenfamily joint',categories=['04_node_atlas'],
            targets=['hiddenfamily'],operation='lookup')
        self.assertEqual(value['entity_targets'],[])
        self.assertNotIn('Private System',value['directory_resolution'].get('family_filter',[]))

    def test_ambiguous_suffix_does_not_choose_one_family(self):
        with sqlite3.connect(self.path) as c:
            c.execute("DELETE FROM technical_family_aliases")
            c.execute("INSERT INTO technical_family_aliases VALUES('qualified alpha','System Alpha')")
            c.execute("INSERT INTO technical_family_aliases VALUES('other alpha','System Beta')")
        value=resolve_relation_scope(self.path,'alpha joints',categories=['04_node_atlas'],
            targets=['alpha'],operation='lookup')
        self.assertEqual(value['entity_targets'],[])


if __name__=='__main__':
    unittest.main()
