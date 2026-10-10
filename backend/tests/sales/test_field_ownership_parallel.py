import threading
import unittest
from contextvars import ContextVar
from unittest.mock import patch
import numpy as np
from backend.request_budget import RequestBudget, RequestCancelled, budget_scope
from backend.sales.parallel_reads import parallel_reads
from backend.sales.planning_spec import planning_output_schema, BACKEND_PLAN_FIELDS
from backend.sales.model_contract import validate_plan, validate_replan
from backend.sales.output_contract import OutputContractError


def task(**fields):
    return {'sources':['customer_documents'], 'operation':'lookup', 'categories':[],
            'wants_images':False,
            'document_visual_required':False, **fields}


class FieldOwnershipTests(unittest.TestCase):
    def test_schema_excludes_program_owned_fields(self):
        schema = planning_output_schema(['company_rag','general_chat'])
        self.assertFalse(BACKEND_PLAN_FIELDS & set(schema['properties']))
        self.assertNotIn('document_names', schema['properties'])
        for key in BACKEND_PLAN_FIELDS:
            with self.assertRaises(OutputContractError):
                validate_plan(task(**{key: [] if key == 'target_document_ids' else 'unknown'}), require_compact=True)

    def test_backend_derives_modes_and_binds_authorized_names(self):
        directory = [{'file':'方案甲', 'document_id':'authorized-1'}]
        result = validate_plan(task(document_names=['方案甲']),
            document_ids=['authorized-1'],document_directory=directory, require_compact=True)
        self.assertEqual(result['target_document_ids'], ['authorized-1'])
        self.assertEqual(result['document_selection'], 'explicit')
        self.assertEqual(result['document_scope'], 'local_lookup')
        self.assertEqual(validate_plan(task(operation='summary'))['document_scope'], 'cross_document')
        self.assertEqual(validate_plan(task(sources=['general_chat']))['answer_basis'], 'conversation')

    def test_unknown_document_rejected_same_names_select_complete_set(self):
        with self.assertRaises(OutputContractError):
            validate_plan(task(document_names=['同名']),document_directory=[],require_compact=True)
        directory=[{'file':'同名','display_name':'同名（1）','document_id':'a'},
                   {'file':'同名','display_name':'同名（2）','document_id':'b'}]
        for name, expected in [('同名',['a','b']),('同名（2）',['b'])]:
            result=validate_plan(task(document_names=[name]),document_ids=['a','b'],
                document_directory=directory,require_compact=True)
            self.assertEqual(result['target_document_ids'],expected)

    def test_replan_cannot_change_original_file_selection(self):
        value = {'action':'execute','gap_ids':['g'],'revised_plan':task()}
        result = validate_replan(value, document_ids=['a'], allowed_tools=['customer_documents'],gap_ids=['g'],
            previous_plan=task(document_selection='explicit',target_document_ids=['a']),require_compact=True)
        self.assertEqual(result['revised_plan']['target_document_ids'], ['a'])
        self.assertEqual(result['revised_plan']['document_selection'], 'explicit')

    def test_final_schema_only_contains_answer_not_execution_metadata(self):
        from backend.sales.model_contract import answer_contract
        contract = answer_contract(None, {'E1'})
        self.assertIsNone(contract.json_schema)
        self.assertFalse(contract.json_output)
        result = contract.validate('正文')
        self.assertEqual(result['customer_reply'], '正文')
        self.assertNotIn('memory_updates', result)


class ParallelReadTests(unittest.TestCase):
    def test_independent_reads_overlap_and_preserve_context(self):
        owner = ContextVar('test_owner',default=None)
        owner.set('session-1')
        barrier = threading.Barrier(2)
        def read():
            barrier.wait(timeout=3)
            return owner.get(), threading.get_ident()
        values,audit=parallel_reads({'a':read,'b':read})
        self.assertEqual(values['a']['value'][0], 'session-1')
        self.assertEqual(values['b']['value'][0], 'session-1')
        self.assertNotEqual(values['a']['value'][1],values['b']['value'][1])
        self.assertEqual(audit['max_workers'],2)

    def test_one_failed_read_preserves_success_without_replay(self):
        calls=[]
        def good(): calls.append('read');return [1]
        def bad(): raise ValueError('private body not exposed')
        values,audit=parallel_reads({'a':good,'b':bad})
        self.assertEqual(values['a']['value'],[1])
        self.assertEqual(calls,['read'])
        self.assertEqual(audit['branches']['b']['error_type'],'ValueError')
        self.assertNotIn('private',str(audit))

    def test_cancellation_is_not_a_retrieval_fallback(self):
        budget=RequestBudget.for_answer(has_attachments=False)
        barrier=threading.Barrier(2)
        def read():
            barrier.wait(timeout=3);budget.cancelled.set();return 1
        with self.assertRaises(RequestCancelled), budget_scope(budget):
            parallel_reads({'a':read,'b':read})

    def test_serial_and_parallel_hybrid_keep_scope_and_ranks(self):
        from backend.sales.retriever import LocalRagRetriever
        from types import SimpleNamespace
        # No model semantics claimed: deterministic protocol/equivalence test.
        retriever=object.__new__(LocalRagRetriever)
        documents=[{'id':'allowed','text':'evidence'},{'id':'outside','text':'other'}]
        retriever.documents=documents
        retriever._dense_ids=['allowed','outside']
        retriever._dense_vectors=np.asarray([[1,0],[1,0]],dtype=np.float32)
        retriever._hybrid_validation={'ready':True,'lexical_fingerprint':'f'}
        retriever._lexical_candidates=lambda *a,**kw: ([(1,documents[0])],[(1,documents[0])])
        embedding=SimpleNamespace(encode=lambda *a,**kw:np.asarray([[1,0]],dtype=np.float32))
        options=dict(node_atlas_request=False,standard_request=False,procedure_request=False,
            product_overview_request=False,candidate_ids={'allowed'},defer_rerank=True)
        with patch('backend.sales.dense_retrieval.get_embedding_model',return_value=embedding), \
             patch('backend.sales.dense_retrieval.retrieval_runtime_status',return_value={}):
            with patch.dict('os.environ',{'RAG_PARALLEL_RECALL':'0'}):
                serial=retriever._hybrid_scored('q',[],**options)
            with patch.dict('os.environ',{'RAG_PARALLEL_RECALL':'1'}):
                parallel=retriever._hybrid_scored('q',[],**options)
        self.assertEqual([(s,d['id']) for s,d in serial],[(s,d['id']) for s,d in parallel])
        self.assertEqual([d['id'] for _,d in parallel],['allowed'])
        self.assertEqual(parallel[0][1]['_recall_execution']['max_workers'],2)
