import copy
import unittest
from unittest.mock import patch

from backend.sales.support_wire import audit_wire_payload, audit_system_prompt, audit_wire_text
from backend.sales.prepared_support import fit_support_budget


class SupportWireTests(unittest.TestCase):
    def test_incidental_source_diversity_is_soft_but_explicit_file_is_hard(self):
        rows=[{'evidence_id':'A','text':'requested details','source_type':'company_rag',
               'goal_support_scores':{'detail':1}},
              {'evidence_id':'B','text':'background '*100,'source_type':'catalog_text',
               'document_id':'incidental'}]
        count=lambda rs:sum(len(r['text']) for r in rs)
        kept,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,
            ceiling=100,question='details',collection_action='none')
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual([r['evidence_id'] for r in kept],['A'])
        _,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,
            ceiling=100,question='details',collection_action='none',required_document_ids=['incidental'])
        self.assertFalse(audit['budget_satisfied'])

    def test_technical_comparison_does_not_hard_reserve_color_catalog(self):
        rows=[{'evidence_id':'T1','source_type':'company_rag','text':'actual joint detail',
               'goal_support_scores':{'joint':1.0}},
              {'evidence_id':'C1','source_type':'company_rag','text':'color list '*100,
               'catalog_record_type':'product_collection','entity_coverage_ids':['family:A']}]
        count=lambda rs:sum(len(r['text']) for r in rs)
        kept,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,ceiling=100,
            question='joint comparison',operation='comparison',collection_action='none')
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual([r['evidence_id'] for r in kept],['T1'])
        _,listed=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,ceiling=100,
            question='list colors',operation='summary',collection_action='list')
        self.assertFalse(listed['budget_satisfied'])

    def test_execution_delta_and_display_references_do_not_rewrite_facts(self):
        payload={'planned_task':{'operation':'comparison','categories':['04_node_atlas'],'targets':[]},
                 'execution_task':{'operation':'comparison','categories':['04_node_atlas'],'targets':['A','B']},
                 'evidence':[{'evidence_id':'G1','gallery_collection':True,'evidence_scope':'display_labels',
                              'text':'original labels','citations':['page1','page2','page3']}]}
        original=copy.deepcopy(payload)
        wire=audit_wire_payload(payload,compact=True)
        self.assertEqual(payload,original)
        self.assertEqual(wire['execution_task'],{'inherits':'planned_task','targets':['A','B']})
        self.assertEqual(wire['evidence'][0]['text'],'original labels')
        self.assertEqual(wire['evidence'][0]['citation_reference_count'],3)

    def test_empty_defaults_removed_but_clearing_condition_and_false_preserved(self):
        wire=audit_wire_payload({'planned_task':{'targets':[],'wants_images':False,
                'filters':{'locations':['region']}},
            'execution_task':{'targets':[],'wants_images':False,'filters':{}}},compact=True)
        self.assertNotIn('targets',wire['planned_task'])
        self.assertIs(wire['planned_task']['wants_images'],False)
        self.assertEqual(wire['execution_task']['filters'],{})

    def test_compact_json_preserves_whitespace_inside_evidence(self):
        import json
        raw={'evidence':[{'evidence_id':'T1','text':'A  B\nnot C'}]}
        self.assertEqual(audit_wire_text(raw),json.dumps(raw,ensure_ascii=False))
        wire=audit_wire_payload(raw,compact=True)
        self.assertEqual(json.loads(audit_wire_text(wire)),wire)
        self.assertLess(len(audit_wire_text(wire)),len(json.dumps(wire,ensure_ascii=False)))

    def test_fitting_input_is_not_compressed_or_reordered(self):
        rows=[{'evidence_id':'a','text':'first'}, {'evidence_id':'b','text':'second',
            'goal_support_scores':{'goal':0.99}}]
        with patch('backend.sales.support_compression.compress_support') as compress:
            kept,audit=fit_support_budget(rows,audit_tokens=lambda _:200,
                generation_tokens=lambda _:300,ceiling=400,question='goal',target_tokens=100)
        compress.assert_not_called()
        self.assertEqual(kept,rows)
        self.assertFalse(audit['compression_applied'])
        self.assertTrue(audit['budget_satisfied'])

    def test_final_budget_also_triggers_single_compression(self):
        rows=[{'evidence_id':'a','source_type':'company_rag','text':'abc',
               'goal_support_scores':{'goal':0.9}},
              {'evidence_id':'b','source_type':'company_rag','text':'def',
               'goal_support_scores':{'goal':0.1}}]
        counter=lambda rs:len(rs)*100
        kept,audit=fit_support_budget(rows,audit_tokens=lambda _:50,
            generation_tokens=counter,ceiling=150,question='goal',operation='lookup')
        self.assertEqual([r['evidence_id'] for r in kept],['a'])
        self.assertTrue(audit['compression_applied'])
        self.assertTrue(audit['budget_satisfied'])

    def test_technical_summary_does_not_lock_whole_navigation_directory(self):
        rows=[{'evidence_id':'T1','source_type':'company_rag','text':'actual installation steps'},
            {'evidence_id':'D1','text':'unrelated headings '*100,'directory_collection':True,
             'directory_category':'03_construction_method','directory_member_ids':['x'],
             'directory_expected_ids':['x']}]
        count=lambda rs:sum(len(r['text']) for r in rs)
        kept,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,
            ceiling=100,question='installation steps',operation='summary',collection_action='none')
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual([r['evidence_id'] for r in kept],['T1'])
        self.assertEqual(audit['directory_coverage'][0]['status'],'partial')

    def test_incidental_conflict_group_can_be_evicted_only_as_a_whole(self):
        rows=[{'evidence_id':'main','text':'requested fact','source_type':'company_rag',
            'goal_support_scores':{'goal':0.9}},
            *[{'evidence_id':i,'text':'incidental numeric background '*40,'source_type':'company_rag',
                'packing_group_id':'conflict_incidental','protected_relation_types':['conflict_group'],
                'requirement_ids':['r'+i],'retrieval_requirements':[{'requirement_id':'r'+i,'aspect':'dimension'}]}
                for i in ['a','b']]]
        count=lambda rs:sum(len(r['text']) for r in rs)
        kept,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,ceiling=100,
            question='requested fact',operation='comparison')
        self.assertEqual([r['evidence_id'] for r in kept],['main'])
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual(set(audit['removed_evidence_ids']),{'a','b'})
        self.assertTrue(all(r['candidate_state']=='missing' for r in audit['requirement_coverage']['requirements']))

    def test_optional_null_string_is_omission_not_a_guessed_target(self):
        from backend.sales.planning_spec import expand_planning_spec
        value={'sources':['company_rag'],'operation':'lookup','categories':['03_construction_method'],
            'wants_images':False,'image_target':None,'query':'installation'}
        plan=expand_planning_spec(value)
        self.assertEqual(plan['gallery_target'],'')
        self.assertEqual(plan['retrieval_query'],'installation')

    def test_provisional_source_representative_does_not_lock_incidental_group(self):
        rows=[{'evidence_id':i,'text':'incidental numeric background '*40,
                'source_type':'company_rag','packing_group_id':'conflict_incidental',
                'protected_relation_types':['conflict_group']}
                for i in ['a','b']]
        rows.append({'evidence_id':'G1','text':'display labels','facts_eligible':False,
            'gallery_collection':True,'evidence_scope':'display_labels'})
        count=lambda rs:sum(len(r['text']) for r in rs)
        kept,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,
            ceiling=100,question='requested comparison',operation='comparison')
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual([r['evidence_id'] for r in kept],['G1'])
        self.assertEqual(set(audit['removed_evidence_ids']),{'a','b'})
        rows[0]['required_document_coverage']=True
        _,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,
            ceiling=100,question='requested comparison',operation='comparison')
        self.assertFalse(audit['budget_satisfied'])

    def test_singleton_packing_id_is_not_a_multi_source_conflict_lock(self):
        rows=[{'evidence_id':i,'packing_group_id':i,'text':'whole paragraph '+i,
            'protected_relation_types':['number_unit','condition'],
            'source_type':'company_rag','requirement_ids':['r'+i],
            'retrieval_requirements':[{'requirement_id':'r'+i,'aspect':i}]}
            for i in ['a','b','c','d']]
        count=lambda rs:len(rs)*100
        kept,audit=fit_support_budget(rows,audit_tokens=count,generation_tokens=count,
            ceiling=200,question='multiple dimensions',operation='comparison')
        self.assertTrue(audit['budget_satisfied'])
        self.assertEqual(len(kept),2)
        self.assertEqual(sum(r['candidate_state']=='missing'
            for r in audit['requirement_coverage']['requirements']),2)

    def test_company_recovery_pointer_is_ignored_but_external_pointer_is_validated(self):
        import json
        from backend.sales.model_contract import support_contract
        from backend.sales.output_contract import OutputContractError
        payload={'company_visual_policy':{'company_only':True},'evidence':[{'evidence_id':'G1','text':'labels'}]}
        raw={'status':'partial','supported_evidence_ids':['G1'],'missing_aspects':['real text gap'],
            'recovery_need':'lookup','candidate_asset_ids':['company:unknown']}
        contract=support_contract(None,payload,max_input_tokens=6500)
        value=contract.validate(json.dumps(raw))
        self.assertEqual(value['candidate_asset_ids'],[])
        self.assertEqual(value['missing_aspects'],raw['missing_aspects'])
        self.assertEqual(value['status'],'partial')
        self.assertEqual(contract.max_input_tokens,6500)
        payload['company_visual_policy']['company_only']=False
        with self.assertRaises(OutputContractError):
            support_contract(None,payload).validate(json.dumps(raw))

    def test_wire_is_idempotent_preserves_task_and_customer_visual_recovery(self):
        raw={'question':'A only; not B; today',
             'planned_task':{'operation':'comparison','targets':['A','B'],'aspects':['cost'],
                'query':'duplicate navigation query'},
             'recovery_budget':{'remaining_rounds':1},
             'visual_candidates':[{'asset_id':'company:img','context':'company label'},
                 {'asset_id':'customer:doc:img','context':'nearby text','pixels_supplied':False}],
             'evidence':[{'evidence_id':'VL1','text':'red triangle; unreadable numbers',
                 'visual_description':True,'description_basis':'pixel_tool_observation'}]}
        saved=copy.deepcopy(raw)
        wire=audit_wire_payload(raw,current_date='2026-10-09',compact=True)
        self.assertEqual(wire,audit_wire_payload(wire,current_date='2026-10-09',compact=True))
        self.assertEqual(raw,saved)
        self.assertEqual(wire['evidence'],raw['evidence'])
        self.assertEqual(wire['question'],raw['question'])
        self.assertEqual(wire['visual_candidates'],raw['visual_candidates'][1:])
        self.assertEqual(wire['recovery_budget'],raw['recovery_budget'])

    def test_directory_provenance_compacted_not_deleted_from_server(self):
        sources=[{'document_name':str(i),'page':i} for i in range(30)]
        raw={'evidence':[{'evidence_id':'D1','text':'same catalog facts',
            'catalog_record_type':'directory_collection','sources':sources}]}
        wire=audit_wire_payload(raw,compact=True)
        self.assertEqual(len(wire['evidence'][0]['sources']),2)
        self.assertEqual(wire['evidence'][0]['omitted_source_references'],28)
        self.assertEqual(raw['evidence'][0]['sources'],sources)
        self.assertEqual(wire['evidence'][0]['text'],raw['evidence'][0]['text'])

    def test_grid_compaction_preserves_all_targets_aspects_and_missing_states(self):
        requirements=[{'requirement_id':'r'+str(i),'target_id':'opaque'+str(i),
            'target_name':name,'aspect':aspect,'candidate_state':'represented' if i==0 else 'missing',
            'evidence_ids':['T1'] if i==0 else []}
            for i,(name,aspect) in enumerate([('A','cost'),('B','cost'),('A','install'),('B','install')])]
        raw={'retrieval_requirements':requirements,'evidence':[{'evidence_id':'T1','text':'A cost',
            'requirement_ids':['r0']}]}
        wire=audit_wire_payload(raw,compact=True)
        grid=wire['retrieval_requirements']
        self.assertEqual(grid['cell_columns'],['target_index','candidate_state','evidence_ids'])
        self.assertEqual({(r['aspect'],grid['targets'][t[0]]) for r in grid['aspects'] for t in r['cells']},
            {('cost','A'),('cost','B'),('install','A'),('install','B')})
        self.assertEqual(grid['aspects'][0]['cells'][0][2],['T1'])
        self.assertEqual(grid['aspects'][0]['cells'][1][1],'missing')
        self.assertEqual(wire,audit_wire_payload(wire,compact=True))
        self.assertEqual(raw['retrieval_requirements'],requirements)

    def test_scope_system_same_for_count_and_execution(self):
        payload={'attachment_task':{'operation':'summary','selection':'all','files':[
            {'document_id':'a','document_name':'a','task':'summary a'},
            {'document_id':'b','document_name':'b','task':'summary b'}]}}
        wire=audit_wire_payload(payload)
        self.assertEqual(audit_system_prompt('system',wire),
            audit_system_prompt('system',audit_wire_payload(wire)))


if __name__=='__main__':
    unittest.main()
