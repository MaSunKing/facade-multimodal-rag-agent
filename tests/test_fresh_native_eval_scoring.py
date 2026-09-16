"""Scoring-contract checks, not production/model performance tests."""
import unittest
from scripts.run_fresh_english_native_eval import support, support_set


class NativeScoringTests(unittest.TestCase):
    def test_packed_pdf_keeps_block_locator(self):
        sample={'file':'a.pdf','anchors':['5-digit'],'native_locator':{'page':6}}
        item={'document_name':'a.pdf','text':'[BLOCK id=pdf:page:6 kind=page_text] unique 5-digit'}
        self.assertTrue(support(item,sample))
        self.assertFalse(support(dict(item,text=item['text'].replace('page:6','page:60')),sample))

    def test_list_can_span_windows_but_not_native_tables(self):
        sample={'file':'a.docx','anchors':['Children','Elderly'],'native_locator':{'table':3}}
        items=[{'document_name':'a.docx','text':'[TABLE id=word:table:4] Children'},
               {'document_name':'a.docx','text':'[TABLE id=word:table:4] Elderly'}]
        self.assertTrue(support_set(items,sample))
        items[1]['text']='[TABLE id=word:table:5] Elderly'
        self.assertFalse(support_set(items,sample))

    def test_native_cell_not_just_same_number(self):
        sample={'file':'a.xlsx','anchors':['16.48'],'native_locator':{'sheet':'Price','cell':'B15'}}
        item={'document_name':'a.xlsx','text':"[TABLE sheet=Price]\n[ROW] B15='16.48'"}
        self.assertTrue(support(item,sample))
        self.assertFalse(support(dict(item,text=item['text'].replace('B15=','B115=')),sample))
        self.assertFalse(support(dict(item,evidence_role='navigation'),sample))


if __name__=='__main__': unittest.main()
