import json
import unittest

from backend.sales.planning_spec import planning_output_schema
from backend.sales.review_execution import execution_witnesses, review_image_labels, compact_review_metadata, review_citations
from backend.sales.requirements import rank_requirement_candidates
from backend.sales.entity_directory import resolve_entities
from backend.sales.answer_review import review_schema


class GenericRepairTests(unittest.TestCase):
    def test_cross_evidence_relationship_rule_shared_by_all_generators(self):
        from backend.app import GROUNDED_SYSTEM_PROMPT,CUSTOMER_DOCUMENT_SYSTEM_PROMPT,COMPANY_RAG_SYSTEM_PROMPT
        from backend.sales.answer_review import REVISION_PROMPT
        from backend.sales.generation_policy import RELATION_EVIDENCE_POLICY
        for prompt in (GROUNDED_SYSTEM_PROMPT,CUSTOMER_DOCUMENT_SYSTEM_PROMPT,
                       COMPANY_RAG_SYSTEM_PROMPT,REVISION_PROMPT):
            self.assertIn(RELATION_EVIDENCE_POLICY,prompt)

    def test_materialized_provenance_does_not_repeat_draft_evidence(self):
        citations=[{'evidence_id':'D1','document_name':str(i),'text':'private'} for i in range(66)]
        self.assertEqual(review_citations(citations),[{'evidence_id':'D1','source_reference_count':66}])
        self.assertEqual(len(citations),66)

    def test_review_metadata_factoring_preserves_task_labels_and_all_targets(self):
        payload={'question':'all objects','final_draft':{'customer_reply':'draft'},
            'image_labels':[{'labels':{'uncertainties':['unconfirmed']}}],
            'actual_execution_scope':{'entity_targets':[{'id':'A','name':'Alpha','kind':'entity','aliases':['a']}],
                'matches':[{'entity_id':'A','basis':'category_collection'}]},
            'requirement_coverage':{'requirements':[{'target_name':'Alpha','aspect':'size',
                'candidate_state':'missing','candidate_count':0}]}}
        packed=compact_review_metadata(payload)
        for key in ('question','final_draft','image_labels'):
            self.assertEqual(packed[key],payload[key])
        self.assertEqual(packed['actual_execution_scope']['entity_targets']['rows'],[['Alpha','entity']])
        self.assertEqual(packed['requirement_coverage']['requirements']['rows'],[[0,0,'missing',0]])
        self.assertEqual(packed['requirement_coverage']['requirements']['target_names'],['Alpha'])
        self.assertIn('aliases',payload['actual_execution_scope']['entity_targets'][0])

    def test_planner_schema_restricts_sources_and_separates_roles(self):
        schema=planning_output_schema(['company_rag','customer_documents'])
        self.assertEqual(schema['properties']['sources']['items']['enum'], ['company_rag','customer_documents'])
        self.assertIn('aspects',schema['required'])
        self.assertIn('targets',schema['required'])
        self.assertEqual(schema['properties']['targets']['maxItems'],8)

    def test_topic_query_does_not_create_fuzzy_product_targets(self):
        products=[{'product_id':'P','name':'Alpha','family':'F','surface_finish':''}]
        def reject(*args): raise AssertionError('no entity referent')
        selected,_,meta=resolve_entities(products,[], 'test records',
            categories=['02_product_information'],operation='lookup',scorer=reject)
        self.assertEqual(selected,[])
        self.assertFalse(meta['reranker_used'])
        selected,_,meta=resolve_entities(products,[], 'warm-looking object',
            categories=['02_product_information'],targets=['warm-looking object'],scorer=lambda q,d:[.5])
        self.assertEqual(len(selected),1)
        self.assertEqual(meta['status'],'ambiguous')

    def test_facet_complement_is_not_blocked_by_identity_overlap(self):
        rows=[{'id':'relation','kind':'text','terms':['Alpha','Alpha','background']},
              {'id':'facet','kind':'text','terms':['composition','composition']},
              {'id':'outside','kind':'text','terms':['composition']*10}]
        def score(terms,row): return sum(row['terms'].count(t) for t in terms)
        chosen=rank_requirement_candidates(rows,{'target_name':'Alpha','aspect':'composition'},
            scope={'relation'},category_ids={'relation','facet'},tokenize=str.split,score=score)
        self.assertEqual(chosen[0][1]['id'],'facet')
        self.assertNotIn('outside',[d['id'] for _,d in chosen])

    def test_navigation_class_is_not_a_fuzzy_entity_for_collection(self):
        cases=[{'case_id':'A','project_name':'alpha'}, {'case_id':'B','project_name':'beta'}]
        def reject(*args): raise AssertionError('category label must not rerank entities')
        _,selected,meta=resolve_entities([],cases,'group registered cases',
            categories=['05_project_case'],targets=['项目案例'],operation='summary',scorer=reject)
        self.assertEqual(len(selected),2)
        self.assertEqual(meta['status'],'collection')

    def test_registered_document_topics_do_not_choose_arbitrary_entities(self):
        from backend.sales.content_taxonomy import TOPICS
        products=[{'product_id':'P','name':'Alpha','family':'F'}]
        def reject(*args): raise AssertionError('document topic is not an entity')
        for topic in TOPICS['02_product_information']:
            selected,_,meta=resolve_entities(products,[],topic,categories=['02_product_information'],
                targets=[topic],operation='lookup',scorer=reject)
            self.assertEqual(selected,[])
            self.assertFalse(meta['reranker_used'])

    def test_only_recorded_skips_are_execution_gaps_across_sources(self):
        coverage={'requirements':[{'requirement_id':'R1','target_kind':'unresolved',
                    'target_name':'A','aspect':'size','candidate_state':'missing'}],
                  'unexecuted_due_to_scope':['R1']}
        for source,output in [('company_rag',{'text_evidence':[], 'meta':{'requirement_coverage':coverage}}),
                              ('customer_documents',{'evidence':[], 'input_snapshot':{'requirement_coverage':coverage}})]:
            value=execution_witnesses({'tools':[source]}, {source:output})
            self.assertEqual(value['execution_gaps'][0]['state'],'scope_unresolved')
            del coverage['unexecuted_due_to_scope']
            self.assertEqual(execution_witnesses({'tools':[source]}, {source:output})['execution_gaps'],[])
            coverage['unexecuted_due_to_scope']=['R1']

    def test_image_labels_have_no_pixels_or_unrelated_source_body(self):
        description={'labels':['triangle'],'description':'green triangle',
                     'visible_text':[],'relations':[], 'uncertainties':['not a fire test'],
                     'image_bytes':'forbidden','image_path':'secret'}
        context=[{'evidence_id':'V1','visual_id':'img','evidence_scope':'visual_description',
                  'description_basis':'pixel_tool_observation','text':json.dumps(description)},
                 {'evidence_id':'T1','text':'unrelated source body'}]
        value=review_image_labels(context,[])
        wire=json.dumps(value)
        self.assertIn('not a fire test',wire)
        for forbidden in ('forbidden','secret','unrelated source body','image_path','image_bytes'):
            self.assertNotIn(forbidden,wire)

    def test_company_labels_are_not_reinterpreted_or_pixel_loaded(self):
        value=review_image_labels([], [{'asset_id':'company1','image_labels':{
            'title':'sample','surface_finish':'rough','image_path':'private'},'image_bytes':b'pixels'}])
        self.assertEqual(value[0]['labels']['surface_finish'],'rough')
        self.assertFalse(value[0]['in_draft_context'])
        self.assertNotIn('private',str(value))

    def test_audit_requirement_schema_uses_verbatim_task(self):
        question='Compare A and B, only their payment terms.'
        schema=review_schema({'review_policy':'task_coverage_only','question':question})
        self.assertEqual(schema['properties']['revision']['properties']['requirement']['enum'],[question])


if __name__=='__main__': unittest.main()
