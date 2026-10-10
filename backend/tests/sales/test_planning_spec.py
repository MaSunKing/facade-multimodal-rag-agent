import unittest
from backend.sales.planning_spec import PlanningSpec, planner_view
from backend.sales.tool_planner import ToolPlan, guard_plan
from backend.sales.model_contract import validate_plan, validate_replan, repair_plan_query_fields
from backend.sales.output_contract import OutputContractError


def task(**changes):
    return {'sources':['company_rag'], 'operation':'lookup', 'categories':['02_product_information'],
            'targets':['对象甲'], 'aspects':['厚度'], 'wants_images':False,
            'document_visual_required':False, **changes}


class PlanningSpecTests(unittest.TestCase):
    def test_recovery_execution_does_not_replace_delivery_for_any_source(self):
        from backend.sales.planning_spec import delivery_view, delivery_operation
        root=task(sources=['company_rag','customer_documents','public_web_search'],operation='summary',
            source_operations={'public_web_search':'lookup'},collection={'action':'list'})
        plan=ToolPlan(tools=root['sources'],document_operation='lookup',company_operation='lookup',
            recovery_directive={'original_delivery_task':root})
        self.assertEqual(delivery_operation(plan,'company_rag'),'summary')
        self.assertEqual(delivery_operation(plan,'customer_documents'),'summary')
        self.assertEqual(delivery_operation(plan,'public_web_search'),'lookup')
        self.assertEqual(delivery_view(plan)['collection'],{'action':'list'})
        self.assertEqual(planner_view(plan)['operation'],'lookup')

    def test_summary_removes_only_provisional_source_bindings(self):
        from backend.sales.planning_spec import clear_summary_lookup_bindings
        root=task(sources=['company_rag','customer_documents','public_web_search'],operation='lookup',
            source_operations={'customer_documents':'summary'})
        plan=ToolPlan(tools=root['sources'],recovery_directive={'original_delivery_task':root})
        requirements=[{'requirement_id':'d','target_kind':'document'},
            {'requirement_id':'e','target_kind':'entity'},{'requirement_id':'w','target_kind':'request'}]
        rows=[{'evidence_id':'U1','text':'unchanged source','document_id':'a',
            'requirement_ids':['d','e','w'],'retrieval_requirements':requirements}]
        cleaned, kept=clear_summary_lookup_bindings(rows,requirements,plan)
        self.assertEqual(cleaned[0]['requirement_ids'],['e','w'])
        self.assertEqual(cleaned[0]['text'],rows[0]['text'])
        self.assertEqual([r['requirement_id'] for r in kept],['e','w'])
        self.assertEqual(rows[0]['requirement_ids'],['d','e','w'])
        plan.recovery_directive={}
        self.assertIs(clear_summary_lookup_bindings(rows,requirements,plan)[0],rows)

    def test_single_operation_lowers_once(self):
        for operation in ('lookup','summary','comparison'):
            value = {k:v for k,v in task(operation=operation, collection={'action':'none'}).items()
                     if k not in {'targets','aspects','filters'}}
            plan = ToolPlan.model_validate(validate_plan(value, require_compact=True, require_company_scope=True))
            self.assertEqual(plan.company_operation, operation)
            self.assertEqual(plan.task_type, 'comparison' if operation == 'comparison' else 'factual_lookup')
            self.assertEqual(plan.target_terms, [])
            self.assertEqual(plan.answer_aspects, plan.answer_goals)

    def test_mixed_sources_inherit_and_override(self):
        value=task(sources=['company_rag','customer_documents','public_web_search'],
                   operation='lookup', source_operations={'customer_documents':'summary'})
        plan=ToolPlan.model_validate(value)
        self.assertEqual(plan.company_operation,'lookup')
        self.assertEqual(plan.document_operation,'summary')
        self.assertEqual(plan.document_scope,'cross_document')
        self.assertTrue(plan.requires_public_web)
        self.assertEqual(planner_view(plan)['source_operations'], {'customer_documents':'summary'})

    def test_unknown_never_silently_becomes_lookup_on_new_wire(self):
        for operation in ('unknown', None, 'count'):
            with self.assertRaises(OutputContractError):
                validate_plan(task(operation=operation), require_compact=True)
        with self.assertRaises(OutputContractError):
            validate_plan(task(source_operations={'company_rag':'unknown'}))

    def test_legacy_remains_internal_compatibility_only(self):
        legacy={'tools':['company_rag'], 'company_operation':'lookup'}
        self.assertEqual(ToolPlan.model_validate(legacy).tools,['company_rag'])
        with self.assertRaises(OutputContractError) as ctx:
            validate_plan(legacy, require_compact=True)
        self.assertEqual(ctx.exception.field,'sources')

    def test_conflicting_old_fields_rejected(self):
        for name, value in [('company_operation','summary'),('task_type','comparison'),
                            ('answer_goals',['另一目标']),('tools',['public_web_search']),
                            ('canonical_task',{})]:
            with self.assertRaises(OutputContractError):
                validate_plan(task(**{name:value}))

    def test_unselected_source_override_rejected(self):
        with self.assertRaises(OutputContractError):
            validate_plan(task(source_operations={'customer_documents':'summary'}))

    def test_web_permission_remains_backend_owned(self):
        plan=ToolPlan.model_validate(task(sources=['public_web_search']))
        self.assertTrue(plan.requires_public_web)
        with self.assertRaises(OutputContractError) as ctx:
            validate_plan(task(sources=['public_web_search']),allowed_tools=['general_chat'])
        self.assertEqual(ctx.exception.code,'ACCESS_DENIED')
        safe=guard_plan(plan,has_documents=False,has_image=False,facade_related=False,web_allowed=False)
        self.assertNotIn('public_web_search',safe.tools)
        self.assertNotIn('public_web_search',planner_view(safe)['sources'])

    def test_attachment_selection_and_acl_survive_adapter(self):
        value=task(sources=['customer_documents'],document_selection='explicit',target_document_ids=['D1'])
        lowered=validate_plan(value,document_ids=['D1'])
        safe=guard_plan(lowered,has_documents=True,has_image=False,facade_related=False,web_allowed=False)
        self.assertEqual(safe.document_selection,'explicit')
        self.assertEqual(safe.target_document_ids,['D1'])
        with self.assertRaises(OutputContractError):
            validate_plan(value,document_ids=['D2'])

    def test_request_bindings_keep_source_target_and_time(self):
        queries=[{'source':'public_web_search','query':q,'targets':[target]}
                 for q,target in [('甲地 2026-10-08 天气','甲地'),('乙地 2026-10-08 天气','乙地')]]
        plan=ToolPlan.model_validate(validate_plan(task(sources=['public_web_search'],targets=['甲地','乙地'],
            queries=queries),require_scoped_requests=True))
        self.assertEqual([r.model_dump() for r in plan.retrieval_requests],queries)
        self.assertEqual(plan.web_retrieval_query,'')  # Never silently select just the first requirement.

    def test_missing_queries_error_uses_new_field_name(self):
        with self.assertRaises(OutputContractError) as ctx:
            validate_plan(task(targets=['甲','乙']),require_scoped_requests=True)
        self.assertEqual(ctx.exception.field,'queries')

    def test_collection_is_not_inferred_from_question_words(self):
        plan=ToolPlan.model_validate(validate_plan(task(operation='summary',query='有哪些产品')))
        self.assertEqual(plan.company_collection.action,'none')
        self.assertEqual(plan.company_operation,'summary')
        plan=ToolPlan.model_validate(validate_plan(task(operation='summary',collection={'action':'list'}),require_company_scope=True))
        self.assertEqual(plan.company_collection.action,'list')

    def test_image_navigation_is_derived_but_legacy_fields_still_work(self):
        from backend.sales.planning_spec import planning_output_schema
        schema=planning_output_schema()
        for field in ('image_scope','image_target','image_match'):
            self.assertNotIn(field,schema['properties'])
        plan=ToolPlan.model_validate(task(targets=['对象甲'],wants_images=True))
        self.assertEqual(plan.visual_scope,'product')
        self.assertEqual(plan.gallery_target,'对象甲')
        self.assertEqual(plan.gallery_scope,'descriptive')

    def test_images_remain_display_not_pixel_analysis(self):
        plan=ToolPlan.model_validate(task(wants_images=True,image_scope='product',image_family='decorative_panel'))
        self.assertEqual(plan.tools,['company_rag'])
        self.assertTrue(plan.wants_visuals)
        self.assertEqual(plan.gallery_product_family,'decorative_panel')

    def test_eight_document_queries_survive(self):
        queries=[{'source':'customer_documents','query':f'问题{i}','targets':[]} for i in range(8)]
        plan=ToolPlan.model_validate(validate_plan(task(sources=['customer_documents'],queries=queries)))
        self.assertEqual(len(plan.retrieval_requests),8)
        self.assertEqual(len(plan.document_retrieval_queries),8)

    def test_multilingual_company_seed_keeps_every_requirement(self):
        from backend.sales.tool_planner import retrieval_query_for_attempt
        queries=[{'source':'company_rag','query':query,'targets':[name]} for query,name in
                 [('甲板厚度是多少','甲板'),('乙板安装要求是什么','乙板')]]
        plan=ToolPlan.model_validate(task(query='Compare the requested facts',queries=queries))
        self.assertEqual(retrieval_query_for_attempt('Compare the requested facts',plan,source='company_rag'),'甲板厚度是多少')
        self.assertEqual([r.query for r in plan.retrieval_requests],['甲板厚度是多少','乙板安装要求是什么'])

    def test_recovery_view_tracks_actual_updated_scope_not_stale_wire(self):
        plan=ToolPlan.model_validate(task()).model_copy(update={'company_operation':'summary',
            'answer_aspects':['新维度'],'target_document_ids':['D2']})
        view=planner_view(plan)
        self.assertEqual(view['source_operations'],{'company_rag':'summary'})
        self.assertEqual(view['aspects'],['新维度'])
        self.assertEqual(view['target_document_ids'],['D2'])
        self.assertFalse(set(view)&{'task_type','retrieval_mode','company_operation','answer_goals'})
        PlanningSpec.model_validate(view)

    def test_replan_budget_checks_use_lowered_new_wire(self):
        from backend.sales.recovery_budget import budget_view
        from backend.request_budget import RequestBudget
        from backend.documents.summaries import SummaryBudget
        budget=RequestBudget.for_answer(has_attachments=True)
        budget.summary_state=SummaryBudget(max_calls=1,calls=1)
        resources=budget_view(budget,{'budget_exhausted':True},tools=['customer_documents'])
        value={'action':'execute','gap_ids':['g'],'revised_plan':task(sources=['customer_documents'],operation='summary')}
        for field in ('targets','filters','aspects'):
            value['revised_plan'].pop(field,None)
        with self.assertRaises(OutputContractError):
            validate_replan(value,allowed_tools=['customer_documents'],gap_ids=['g'],recovery_budget=resources,require_compact=True)
        value['revised_plan']['operation']='lookup'
        validate_replan(value,allowed_tools=['customer_documents'],gap_ids=['g'],recovery_budget=resources,require_compact=True)

    def test_runtime_metadata_cannot_be_injected(self):
        with self.assertRaises(OutputContractError):
            repair_plan_query_fields({'tools':['general_chat'],'canonical_task':{'operation':'summary'}})


if __name__ == '__main__':
    unittest.main()
