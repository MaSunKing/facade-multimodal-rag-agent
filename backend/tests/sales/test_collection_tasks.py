import unittest
from backend.sales.directory_evidence import process_collection, directory_coverage
from backend.sales.model_contract import validate_plan, validate_audit, validate_replan
from backend.sales.output_contract import OutputContractError
from backend.sales.gap_recovery import new_operations
from backend.sales.tool_planner import ToolPlan, guard_plan


class CollectionTaskTests(unittest.TestCase):
    def plan(self, **extra):
        return dict(tools=['company_rag'], knowledge_categories=['05_project_case'],
                    company_operation='summary', **({'company_collection':{'action':'none'}} | extra))

    def test_group_count_deduplicates_and_keeps_unknown(self):
        members=[{'id':str(i),'text':'x','attributes':{'province':v}} for i,v in enumerate(['R','R','S',None])]
        rows=process_collection('05_project_case',members+[members[0]],{'action':'count','group_by':['province']})
        self.assertEqual(sorted(g['count'] for r in rows for g in r['collection_groups']),[1,1,2])
        self.assertTrue(all(r['collection_total_count']==4 for r in rows))
        self.assertTrue(any(r['collection_missing_fields']==['province'] for r in rows))
        self.assertEqual(directory_coverage(rows,rows)[0]['status'],'complete')
        self.assertEqual(directory_coverage(rows,[])[0]['status'],'partial')

    def test_product_and_technical_groups_use_same_executor(self):
        for category,field in [('02_product_information','family'),('04_node_atlas','document_name'),
                               ('01_standard_specification','section_heading'),('06_enterprise_information','category'),
                               ('03_construction_method','document_name')]:
            rows=process_collection(category,[{'id':'a','text':'a','attributes':{field:'X'}}],
                                    {'action':'count','group_by':[field]})
            self.assertEqual(rows[0]['collection_group_count'],1)
            validate_plan(dict(tools=['company_rag'],knowledge_categories=[category],company_operation='summary',
                company_collection={'action':'count','group_by':[field]}),require_company_scope=True)

    def test_no_group_total_and_no_scope_is_not_zero(self):
        self.assertEqual(process_collection('05_project_case',[],{'action':'count'}),[])
        row=process_collection('05_project_case',[{'id':'a','text':'x'}],{'action':'count'})[0]
        self.assertEqual(row['collection_total_count'],1)

    def test_grouped_list_keeps_all_whole_members(self):
        rows=process_collection('02_product_information',[
            {'id':str(i),'text':'member '+str(i),'attributes':{'family':'F'}} for i in range(9)],
            {'action':'list','group_by':['family']},packet_chars=20)
        self.assertEqual(len({i for r in rows for i in r['directory_member_ids']}),9)

    def test_invalid_fields_and_incompatible_task_rejected(self):
        for collection in [{'action':'count','group_by':['province','province']},
                           {'action':'count','group_by':['arbitrary_sql']},
                           {'action':'count','group_by':[],'sql':'SELECT *'}]:
            with self.assertRaises(OutputContractError):
                validate_plan(self.plan(company_collection=collection))
        with self.assertRaises(OutputContractError):
            validate_plan(dict(tools=['company_rag'],knowledge_categories=['02_product_information'],
                company_operation='summary',company_collection={'action':'count','group_by':['province']}))

    def test_recovery_requires_scope_and_keeps_collection_contract(self):
        for revised in [{'tools':['company_rag']},dict(tools=['company_rag'],knowledge_categories=['05_project_case'],company_operation='summary')]:
            with self.assertRaises(OutputContractError):
                validate_replan({'action':'execute','gap_ids':['g'],'revised_plan':revised},
                    gap_ids=['g'],allowed_tools=['company_rag'],previous_plan=self.plan(company_collection={'action':'count'}))
        value={'action':'execute','gap_ids':['g'],'revised_plan':self.plan(company_collection={'action':'count','group_by':['city']})}
        validate_replan(value,gap_ids=['g'],allowed_tools=['company_rag'])

    def test_repeat_not_new_due_to_default_task_type_or_explanation(self):
        old=ToolPlan(**self.plan(),task_type='factual_lookup',retrieval_query='query')
        new=ToolPlan(**self.plan(),retrieval_query='query',reason='different explanation')
        self.assertEqual(new_operations(old,new),[])
        new.company_collection.action='count'
        self.assertEqual(new_operations(old,new),['company_rag'])

    def test_summary_must_explicitly_choose_collection_operation(self):
        with self.assertRaises(OutputContractError) as error:
            validate_plan(dict(tools=['company_rag'],knowledge_categories=['05_project_case'],company_operation='summary'),
                          require_company_scope=True)
        self.assertEqual(error.exception.code,'OUTPUT_FIELD_MISSING')

    def test_scope_mismatch_advice_is_validated(self):
        value={'status':'unsupported','supported_evidence_ids':[], 'missing_aspects':['requested set'],
               'recovery_need':'lookup','gap_type':'scope_mismatch','suggested_categories':['05_project_case']}
        self.assertEqual(validate_audit(value,[])['gap_type'],'scope_mismatch')
        value['suggested_categories']=['not_registered']
        with self.assertRaises(OutputContractError):
            validate_audit(value,[])

    def test_guard_keeps_collection_and_shared_audit_metadata(self):
        from backend.sales.evidence_gate import compact_audit_evidence
        plan=ToolPlan(**self.plan(company_collection={'action':'count','group_by':['city']}))
        guarded=guard_plan(plan,has_documents=False,has_image=False,facade_related=True,web_allowed=False)
        self.assertEqual(guarded.company_collection.model_dump(),plan.company_collection.model_dump())
        row=process_collection('05_project_case',[{'id':'a','text':'x','attributes':{'city':'C'}}],
                               plan.company_collection.model_dump())[0]
        self.assertEqual(compact_audit_evidence([row])[0]['catalog_record_type'],'directory_aggregate')

    def test_multi_attribute_and_multi_category_contract(self):
        rows=process_collection('05_project_case',[
            {'id':str(i),'text':'x','attributes':{'province':'R','project_category':v}} for i,v in enumerate(['A','B'])],
            {'action':'count','group_by':['province','project_category']})
        self.assertEqual(len(rows[0]['collection_groups']),2)
        validate_plan(dict(tools=['company_rag'],knowledge_categories=['02_product_information','05_project_case'],
            company_operation='summary',company_collection={'action':'count','group_by':['category']}))
        with self.assertRaises(OutputContractError):
            validate_plan(dict(tools=['company_rag'],knowledge_categories=['02_product_information','05_project_case'],
                company_operation='summary',company_collection={'action':'count','group_by':['province']}))

    def test_aggregate_scope_respects_acl_and_resolved_subset(self):
        import tempfile, sqlite3
        from pathlib import Path
        from backend.tests.sales.test_directory_evidence import DirectoryEvidenceTests
        from backend.sales.catalog_store import retrieve_catalog_evidence
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'catalog.sqlite3'
            DirectoryEvidenceTests().fixture(path)
            rows=retrieve_catalog_evidence('any',database_path=path,operation='summary',
                knowledge_categories=['05_project_case'],directory_scope={'entity_ids':['c1','private']},
                collection={'action':'count','group_by':['province']})
            aggregates=[r for r in rows if r.get('collection_action')=='count']
            self.assertEqual(sum(r['collection_group_count'] for r in aggregates),1)
            self.assertNotIn('private',str(aggregates))


if __name__=='__main__':
    unittest.main()
