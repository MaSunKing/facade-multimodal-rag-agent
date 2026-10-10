import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from backend.sales.catalog_store import retrieve_catalog_evidence
from backend.sales.directory_evidence import collection_packets, directory_coverage
from backend.sales.prepared_support import fit_support_budget
from backend.sales.evidence_packing import model_evidence_view
from backend.sales.model_contract import validate_replan, support_contract
from backend.sales.output_contract import OutputContractError


class DirectoryEvidenceTests(unittest.TestCase):
    def test_compaction_keeps_requested_attributes_and_aggregate_values(self):
        from backend.sales.directory_evidence import compact_directory_packets, process_collection
        members=[{'id':str(i),'name':'Entry '+str(i),'text':'Entry '+str(i)+'；region：North；extra：verbose',
            'attributes':{'region':'North','extra':'verbose'},
            'attribute_labels':{'region':'region','extra':'extra'}} for i in range(3)]
        rows=collection_packets('05_project_case',members,packet_chars=40)
        compact=compact_directory_packets(rows,'list by region')
        self.assertEqual(compact[0]['text'].count('region：North'),3)
        self.assertNotIn('extra：',compact[0]['text'])
        counts=process_collection('05_project_case',members,{'action':'count','group_by':['region']})
        self.assertEqual(compact_directory_packets(counts,'count'),counts)
        grouped=process_collection('05_project_case',members,{'action':'list','group_by':['region']})
        combined=compact_directory_packets(grouped,'list')
        self.assertEqual(combined[0]['text'].count('region：North'),3)
        self.assertNotIn('collection_group_count',combined[0])

    def test_fitting_context_and_noncollection_paths_are_unchanged(self):
        rows=collection_packets('05_project_case',[{'id':'one','text':'Entry'}])
        for action,ceiling in [('list',10000),('none',1)]:
            count=lambda selected:sum(len(r['text']) for r in selected)
            kept,audit=fit_support_budget(rows,question='Find',operation='lookup',collection_action=action,
                audit_tokens=count,generation_tokens=count,ceiling=ceiling)
            self.assertFalse(audit['directory_compaction']['applied'])
            self.assertEqual(kept[0]['text'],rows[0]['text'])

    def test_membership_before_detail_across_categories(self):
        import copy
        from backend.sales.content_taxonomy import LABELS
        for category in LABELS:
            members=[{'id':str(i),'name':'Object '+str(i),
                'text':'Object '+str(i)+'；extra：'+'detail '*60,
                'citations':[{'document_name':'source '+str(i)}]} for i in range(12)]
            rows=collection_packets(category,members,packet_chars=600)
            original=copy.deepcopy(rows)
            count=lambda selected:sum(len(r['text']) for r in selected)
            kept,audit=fit_support_budget(rows,question='List objects',operation='summary',collection_action='list',
                audit_tokens=count,generation_tokens=count,ceiling=300)
            self.assertTrue(audit['budget_satisfied'])
            self.assertEqual(directory_coverage(rows,kept)[0]['retained_count'],12)
            self.assertTrue(all(m['name'] in kept[0]['text'] for m in members))
            self.assertEqual(len(kept[0]['citations']),12)
            self.assertNotIn('directory_members',model_evidence_view(kept[0]))
            self.assertEqual(rows,original)

    def test_minimal_list_overflow_is_honest_partial_membership(self):
        rows=collection_packets('05_project_case',[
            {'id':str(i),'name':'Long project '+str(i)+' x'*40,'text':'Long project '+str(i)+' x'*40}
            for i in range(8)])
        count=lambda selected:sum(len(r['text']) for r in selected)
        kept,audit=fit_support_budget(rows,question='List',collection_action='list',operation='summary',
            audit_tokens=count,generation_tokens=count,ceiling=220)
        self.assertTrue(audit['budget_satisfied'])
        coverage=directory_coverage(rows,kept)[0]
        self.assertEqual(coverage['status'],'partial')
        self.assertLess(coverage['retained_count'],8)
        self.assertIn('未展示',kept[0]['text'])

    def test_multi_category_summary_resolves_union_without_fuzzy_topk(self):
        from backend.sales.entity_directory import resolve_entities
        from unittest.mock import Mock
        products=[{'product_id':str(i),'name':'Item '+str(i)} for i in range(7)]
        cases=[{'case_id':str(i),'project_name':'Site '+str(i)} for i in range(5)]
        scorer=Mock(side_effect=AssertionError('collection must not rerank entities'))
        for categories, expected in [
            (['02_product_information','06_enterprise_information'],(7,0)),
            (['02_product_information','05_project_cases'],(7,5)),
            (['05_project_case','01_standard_specification'],(0,5))]:
            p,c,status=resolve_entities(products,cases,'Overview',categories=categories,
                operation='summary',scorer=scorer)
            self.assertEqual((len(p),len(c)),expected)
            self.assertFalse(status['reranker_used'])
        scorer.assert_not_called()

    def test_navigation_target_does_not_narrow_summary_to_fuzzy_top_three(self):
        from backend.sales.entity_directory import resolve_entities
        from unittest.mock import Mock
        products=[{'product_id':str(i),'name':'Item '+str(i)} for i in range(7)]
        for target in ('产品','产品资料','products','产品系列'):
            scorer=Mock(side_effect=AssertionError('category is not an entity'))
            rows,_,audit=resolve_entities(products,[],'Overview',targets=[target],
                categories=['02_product_information'],operation='summary',scorer=scorer)
            self.assertEqual(len(rows),7)
            self.assertFalse(audit['reranker_used'])
            scorer.assert_not_called()
        # A real entity named like a navigation class still takes precedence.
        rows,_,_=resolve_entities([{'product_id':'p','name':'产品'}],[],'Overview',targets=['产品'],
            categories=['02_product_information'],operation='summary')
        self.assertEqual([r['product_id'] for r in rows],['p'])

    def test_ambiguous_summary_ranks_collection_without_lookup_truncation(self):
        from backend.sales.entity_directory import resolve_entities
        products=[{'product_id':str(i),'name':'Entry '+str(i)} for i in range(7)]
        for operation, count in [('summary',7),('lookup',3),('comparison',3)]:
            rows,_,audit=resolve_entities(products,[],'Unresolved brand',targets=['Unresolved brand'],
                categories=['02_product_information'],operation=operation,
                scorer=lambda query, descriptions:list(range(len(descriptions))))
            self.assertEqual(len(rows),count)
            self.assertEqual(audit['status'],'ambiguous')
            self.assertTrue(audit['reranker_used'])
            if operation=='summary':
                self.assertEqual(audit['selection_scope'],'category_candidates_unresolved_target')

    def test_original_question_uses_registered_region_without_planner_filters(self):
        from backend.sales.entity_directory import resolve_entities
        cases=[{'case_id':'a','project_name':'Site A','province':'山东省','city':'济南市'},
               {'case_id':'b','project_name':'Site B','province':'山东省','city':'青岛市'},
               {'case_id':'c','project_name':'Site C','province':'河北省','city':'保定市'}]
        _,rows,audit=resolve_entities([],cases,'山东有哪些项目',categories=['05_project_case'],operation='summary')
        self.assertEqual({r['case_id'] for r in rows},{'a','b'})
        self.assertEqual(audit['query_derived_filters'],{'locations':['山东']})
        _,rows,audit=resolve_entities([],cases,'不含山东的项目',categories=['05_project_case'],operation='summary')
        self.assertEqual(audit['query_derived_filters'],{})

    def test_entity_and_topic_scopes_coexist_with_public_acl(self):
        from backend.sales.relation_scope import resolve_relation_scope
        from backend.sales.content_taxonomy import LABELS
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'catalog.sqlite3'; self.fixture(path)
            scope=resolve_relation_scope(path,'Overview',categories=list(LABELS),operation='summary')
            self.assertEqual(set(scope['entity_ids']),{'p1','c1'})
            self.assertTrue({'e0','e1','e2','e3'} <= set(scope['candidate_ids']))
            self.assertNotIn('secret',scope['candidate_ids'])
            rows=retrieve_catalog_evidence('Overview',operation='summary',knowledge_categories=list(LABELS),
                directory_scope=scope,database_path=path)
            self.assertEqual({r['directory_category'] for r in rows},set(LABELS))
            # Real retrieval handoff intentionally strips the large ID list.
            lean_scope={k:v for k,v in scope.items() if k not in ('candidate_ids','evidence_groups')}
            handed=retrieve_catalog_evidence('Overview',operation='summary',knowledge_categories=list(LABELS),
                directory_scope=lean_scope,database_path=path)
            self.assertEqual({r['directory_category'] for r in handed},set(LABELS))
            self.assertNotIn('PRIVATE',json.dumps(handed))
            for operation in ['lookup','comparison']:
                narrow=resolve_relation_scope(path,'p1',categories=list(LABELS),
                    targets=['p1'],operation=operation)
                self.assertFalse({'e0','e1','e2','e3'} & set(narrow['candidate_ids']))

    def test_explicit_lookup_and_compare_do_not_expand_to_all_entities(self):
        from backend.sales.entity_directory import resolve_entities
        products=[{'product_id':str(i),'name':'Specific item '+str(i)} for i in range(4)]
        for operation, targets in [('lookup',['Specific item 1']),
                                   ('comparison',['Specific item 1','Specific item 2'])]:
            p,_,_=resolve_entities(products,[],'question',targets=targets,
                categories=['02_product_information','06_enterprise_information'],operation=operation)
            self.assertEqual(len(p),len(targets))

    def fixture(self, path):
        c=sqlite3.connect(path)
        c.executescript('''
          CREATE TABLE products(product_id TEXT,name TEXT,family TEXT,surface_finish TEXT,
             insulation_type TEXT,introduction TEXT,standard_specification TEXT,panel_thickness TEXT,
             weight_kg_m2 TEXT,insulation_thickness TEXT,application_scope TEXT,source_url TEXT,
             main_visual_asset_id TEXT,access_scope TEXT);
          CREATE TABLE project_cases(case_id TEXT,project_name TEXT,province TEXT,city TEXT,
             project_category TEXT,product_name TEXT,product_specificity TEXT,installation_method TEXT,
             area_m2 TEXT,completion_year TEXT,image_count INTEGER,source_url TEXT,source_text TEXT,access_scope TEXT);
          CREATE TABLE knowledge_text_evidence(evidence_id TEXT,text TEXT,access_scope TEXT,metadata_json TEXT);
          CREATE TABLE knowledge_documents(document_key TEXT,document_name TEXT);
          CREATE TABLE knowledge_evidence_sources(evidence_id TEXT,document_key TEXT,section_heading TEXT,reference_json TEXT);
        ''')
        for key, scope in [('p1','public'),('private','internal')]:
            c.execute('INSERT INTO products VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (key,key,'family','finish',None,None,None,None,None,None,None,'https://example.test/'+key,None,scope))
        c.execute('INSERT INTO project_cases VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                ('c1','Project','Province','City','category','p1','source','method',None,None,0,'https://example.test/c1','source','public'))
        for i,domain in enumerate(['01_standard_specification','03_construction_method','04_node_atlas','06_enterprise_information']):
            key='e'+str(i)
            c.execute('INSERT INTO knowledge_text_evidence VALUES(?,?,?,?)',(key,'actual content','public',json.dumps({'knowledge_domains':[domain]})))
            c.execute('INSERT INTO knowledge_documents VALUES(?,?)',(key,'Document '+str(i)))
            c.execute('INSERT INTO knowledge_evidence_sources VALUES(?,?,?,?)',(key,key,'Heading '+str(i),json.dumps({'document_name':key,'source_page':1})))
        c.execute('INSERT INTO knowledge_text_evidence VALUES(?,?,?,?)',('secret','PRIVATE','internal',json.dumps({'knowledge_domains':['03_construction_method']})))
        c.execute('INSERT INTO knowledge_documents VALUES(?,?)',('secret','PRIVATE'))
        c.execute('INSERT INTO knowledge_evidence_sources VALUES(?,?,?,?)',('secret','secret','PRIVATE','{}'))
        c.commit(); c.close()

    def test_all_six_domains_consume_scope_without_series_alias(self):
        from backend.sales.content_taxonomy import LABELS
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'catalog.sqlite3'; self.fixture(path)
            scope={'entity_ids':['p1','c1','private'],'candidate_ids':['e0','e1','e2','e3','secret']}
            rows=retrieve_catalog_evidence('Unseen question',operation='summary',knowledge_categories=list(LABELS),
                directory_scope=scope,database_path=path)
            self.assertEqual({r['directory_category'] for r in rows}, set(LABELS))
            self.assertNotIn('PRIVATE',json.dumps(rows))
            self.assertNotIn('private',{key for r in rows for key in r['directory_member_ids']})
            self.assertTrue(all(r['citations'] for r in rows))

    def test_empty_or_missing_scope_does_not_scan_all_products(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'catalog.sqlite3'; self.fixture(path)
            rows=retrieve_catalog_evidence('Unseen',operation='summary',knowledge_categories=['02_product_information'],
                directory_scope={'entity_ids':[]},database_path=path)
            self.assertEqual(rows,[])

    def test_bounded_packets_keep_whole_members_and_report_partial(self):
        rows=collection_packets('02_product_information',[{'id':str(i),'text':'Member '+str(i)} for i in range(4)],packet_chars=10)
        count=lambda selected:sum(len(r['text']) for r in selected)
        kept, status=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,ceiling=count(rows[:1])+5)
        self.assertTrue(status['budget_satisfied'])
        coverage=status['directory_coverage'][0]
        self.assertEqual((coverage['target_count'],coverage['status']),(4,'partial'))
        self.assertGreater(coverage['retained_count'],1)
        self.assertLess(coverage['retained_count'],4)
        self.assertTrue(status['budget_satisfied'])
        self.assertEqual(len(coverage['missing_member_ids']),4-coverage['retained_count'])
        self.assertNotIn('directory_expected_ids',model_evidence_view(kept[0]))
        self.assertEqual(directory_coverage(rows,rows)[0]['status'],'complete')

    def test_directory_packet_is_not_excerpted_while_claiming_full_membership(self):
        from backend.sales.support_compression import compress_support
        rows=collection_packets('02_product_information',[
            {'id':str(i),'text':'Entry '+str(i)+' description '*30} for i in range(5)],packet_chars=10000)
        kept,_=compress_support(rows,'Entry 1',max_chars=100)
        self.assertEqual(kept[0]['text'],rows[0]['text'])
        self.assertEqual(kept[0]['directory_member_ids'],rows[0]['directory_member_ids'])

    def test_lookup_is_not_replaced_with_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'catalog.sqlite3'; self.fixture(path)
            rows=retrieve_catalog_evidence('p1',target_terms=['p1'],operation='lookup',
                directory_scope={'entity_ids':['p1']},include_project_cases=False,database_path=path)
            self.assertTrue(rows)
            self.assertFalse(any(r.get('directory_collection') for r in rows))

    def test_recovery_ids_have_specific_error_and_allowlist_feedback(self):
        value={'action':'execute','gap_ids':['bad'],'revised_plan':{'tools':['company_rag'],
            'knowledge_categories':['05_project_case'],'company_operation':'lookup'}}
        with self.assertRaises(OutputContractError) as failed:
            validate_replan(value,allowed_tools=['company_rag'],gap_ids=['gap_allowed'])
        self.assertEqual(failed.exception.code,'RECOVERY_GAP_ID_UNKNOWN')
        self.assertEqual(failed.exception.validation_details['allowed_gap_ids'],['gap_allowed'])
        value['gap_ids']=[]
        with self.assertRaises(OutputContractError) as failed:
            validate_replan(value,allowed_tools=['company_rag'],gap_ids=['gap_allowed'])
        self.assertEqual(failed.exception.code,'RECOVERY_GAP_IDS_MISSING')
        contract=support_contract(None,{'available':{'company_rag':True},'gap_state':{'gaps':[{'gap_id':'gap_allowed'}]}},purpose='replan')
        self.assertIn('gap_allowed',contract.retry_instructions)
