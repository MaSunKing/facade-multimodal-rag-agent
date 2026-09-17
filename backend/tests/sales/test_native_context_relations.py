import unittest
from backend.sales.fact_normalization import disagreement_groups
from backend.documents.table_context import native_table_contexts, row_context_cells
from backend.documents.customer_sessions import _window_chunk


class NativeContextRelationTests(unittest.TestCase):
    def test_field_and_value_binding_beats_repeated_navigation_words(self):
        from backend.sales.context_engine import optimise_evidence_context
        items,_=optimise_evidence_context('Widget-Q price',[
            dict(evidence_id='heading',text="[ROW] A1='Widget-Q price'\n[TABLE_CONTEXT native_cells_not_inferred] A1='Widget-Q price'"),
            dict(evidence_id='data',text="[ROW] A4='Widget-Q' | B4='24'\n[TABLE_CONTEXT native_cells_not_inferred] B2='Price (EUR)'",
                 ranking_text="[ROW] A4='Widget-Q' | B4='24'")])
        self.assertEqual(items[0]['evidence_id'],'data')

    def test_ranked_context_is_not_reordered_by_chunk_tail(self):
        import json
        from backend.app import compact_grounded_payload_for_generation
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return '\n'.join(str(m['content']) for m in messages)
            def __call__(self, text, **kwargs):
                return dict(input_ids=list(range(len(text.split()))))
        evidence=[dict(evidence_id=e, original_chunk_id=c, evidence_scope='content',
                       semantic_context_score=s, text='[ROW] value '+e)
                  for e,c,s in [('A','table1',3),('B','table2',2),('C','table1',1)]]
        wire,_=compact_grounded_payload_for_generation(dict(customer_question='value',evidence=evidence),Tokenizer(),max_prompt_tokens=2000)
        self.assertEqual([i['evidence_id'] for i in json.loads(wire)['evidence']],['A','B','C'])

    def test_parser_wrapper_does_not_hide_explicit_entity(self):
        items = [dict(evidence_id=str(n),document_id=str(n), text=f'[BLOCK id=txt:{n} kind=text source=txt] Beam-Z thickness: {value} mm')
                 for n,value in enumerate((23,24))]
        self.assertEqual(len(disagreement_groups(items)),1)
        self.assertEqual(disagreement_groups([dict(i,text='[unknown] thickness: 20 mm') for i in items]),[])
        self.assertEqual(disagreement_groups([dict(items[0],text='[BLOCK id=t kind=text] thickness: 20 mm'),dict(items[1],text='[BLOCK id=u kind=text] thickness: 22 mm')]),[])

    def test_unit_header_is_carried_across_chunks_and_resets_at_section(self):
        prefix='[TABLE id=T sheet=Stats state=visible range=A1:D20 parser=test]\n[MERGED_RANGES count=2] A2:D2, A9:D9\n'
        chunks=[dict(text=prefix+"[ROW] A2='Output (MWh)'\n[ROW] B3='Year 2027' | C3='Year 2026'\n[ROW] A4='Product A' | B4='120'\n"),
                dict(text=prefix+"[ROW] A8='Product B' | B8='140'\n[ROW] A9='Costs (EUR)'\n[ROW] A10='Product B' | B10='80'")]
        contexts=native_table_contexts(chunks)['T']
        self.assertIn("A2='Output (MWh)'",row_context_cells(contexts[8],['B']))
        self.assertNotIn("A2='Output (MWh)'",row_context_cells(contexts[10],['B']))
        windows=_window_chunk(chunks[1]['text'],'chunk2',window_tokens=384,overlap_tokens=64,table_contexts=contexts)
        found=next(w for w in windows if "B8='140'" in w['text'])
        self.assertIn("A2='Output (MWh)'",found['text'])
        self.assertIn("B3='Year 2027'",found['text'])
        self.assertIn(2,found['context_source_row_indices'])
        self.assertNotIn("B3='Year 2027'",found['ranking_text'])
        self.assertEqual(chunks[1]['text'].count('MWh'),0)

    def test_merged_column_units_remain_scoped(self):
        text="[TABLE id=T sheet=S state=visible range=A1:E8 parser=test]\n[MERGED_RANGES count=2] B2:C2, D2:E2\n[ROW] B2='Length (mm)' | D2='Price (USD)'\n[ROW] A3='Item' | B3='2027' | D3='2027'\n[ROW] A4='Widget' | B4='180' | D4='22'"
        context=native_table_contexts([dict(text=text)])['T'][4]
        self.assertIn("B2='Length (mm)'",row_context_cells(context,['C']))
        self.assertNotIn("D2='Price (USD)'",row_context_cells(context,['B']))
        self.assertIn("D2='Price (USD)'",row_context_cells(context,['E']))

    def test_numeric_rows_are_not_promoted_to_headers(self):
        text="[TABLE id=T sheet=S state=visible range=A1:C8 parser=test]\n[ROW] A1='Product' | B1='Length (mm)'\n[ROW] A2='Item A' | B2='23 mm'\n[ROW] A3='Item B' | B3='2.5e2'\n[ROW] A4='Item C' | B4='30'"
        contexts=native_table_contexts([dict(text=text)])['T']
        values=row_context_cells(contexts[4],['A','B'])
        self.assertIn("B1='Length (mm)'",values)
        self.assertNotIn("A2='Item A'",values)
        self.assertNotIn("B3='2.5e2'",values)
