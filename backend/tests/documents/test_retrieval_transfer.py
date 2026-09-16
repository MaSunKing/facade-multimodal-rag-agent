"""Format-level regression tests, independent of downloaded smoke documents."""
import unittest
from io import BytesIO
from backend.documents.customer_sessions import _window_chunk, _compact_row_formulas, add_files, get_session, delete_session, retrieve


class RetrievalTransferTests(unittest.TestCase):
    def test_formula_template_preserves_exact_expression(self):
        row = "[ROW] B7='12' formula='=IF(Work!B7=0,0,Work!B7)' | C7='14' formula='=IF(Work!C7=0,0,Work!C7)'"
        result = _compact_row_formulas(row)
        self.assertIn("B7='12' formula_template=F1", result)
        self.assertIn("C7='14' formula_template=F1", result)
        self.assertIn('=IF(Work!{cell}=0,0,Work!{cell})', result)

    def test_table_rows_remain_individually_retrievable(self):
        text = '[TABLE id=alpha sheet=Metrics state=visible]\n[COLUMNS] A=Year | B=Count\n'
        text += '\n'.join(f"[ROW source=excel] A{i}='20{i:02}/21' | B{i}='{i*13}'" for i in range(1,30))
        windows = _window_chunk(text, 'alpha', window_tokens=200, overlap_tokens=32)
        self.assertEqual(len(windows),29)
        self.assertTrue(all('[COLUMNS]' in w['text'] for w in windows))
        self.assertTrue(all(w['text'].count('[ROW')==1 for w in windows))
        self.assertIn("'299'", windows[22]['text'])

    def test_text_pdf_retains_native_picture(self):
        import fitz
        from PIL import Image
        image=BytesIO(); Image.new('RGB',(300,200),'purple').save(image,format='PNG')
        doc=fitz.open(); page=doc.new_page()
        page.insert_text((50,50),'Native searchable text with a diagram and explanatory paragraphs. '*3)
        page.insert_image(fitz.Rect(50,100,450,400),stream=image.getvalue())
        content=doc.tobytes(); doc.close()
        value=add_files([('diagram.pdf',content)])
        try:
            parsed=get_session(value['session_id']).documents[0]
            self.assertGreater(len(parsed.visuals),0)
            self.assertTrue(any(v.image_bytes for v in parsed.visuals))
            self.assertTrue(any('Native searchable' in c['text'] for c in parsed.chunks))
        finally: delete_session(value['session_id'])

    def test_period_constraint_selects_late_rows(self):
        from openpyxl import Workbook
        book=Workbook(); sheet=book.active; sheet.title='Statistics'
        sheet.append(['Year','North','South'])
        for i in range(30): sheet.append([f'{1990+i}/{str(1991+i)[2:]}',i*11,i*17])
        stream=BytesIO(); book.save(stream)
        value=add_files([('counts.xlsx',stream.getvalue())])
        try:
            data=retrieve(value['session_id'],'Statistics 2018/19 North South',max_chunks=4)
            text='\n'.join(item['text'] for item in data['evidence'])
            self.assertIn('2018/19',text)
            self.assertIn('308',text)
            self.assertIn('476',text)
        finally: delete_session(value['session_id'])

if __name__=='__main__': unittest.main()
