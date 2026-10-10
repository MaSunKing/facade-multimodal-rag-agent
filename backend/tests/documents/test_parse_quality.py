from types import SimpleNamespace
import unittest
from backend.documents.parse_quality import summarize_parse_quality
from backend.documents.customer_sessions import add_files, delete_session


class ParseQualityTests(unittest.TestCase):
    def document(self, **updates):
        fields = dict(document_id='doc', chunks=[], source_locations={},
                      visual_coverage={}, visuals=[])
        fields.update(updates)
        return SimpleNamespace(**fields)

    def test_index_is_not_answer_text(self):
        summary = summarize_parse_quality(self.document(
            chunks=[{'kind': 'document_index', 'text': 'Navigation'}]))
        self.assertFalse(summary['text_available'])
        self.assertIn('no_extracted_answer_text', summary['warnings'])

    def test_mixed_modalities_and_unverified_ocr(self):
        visual = SimpleNamespace(visual_id='v1', kind='document_page',
                                 source={'page_number': 2}, image_bytes=b'pixels',
                                 metadata={'ocr_literal_text': 'candidate'})
        summary = summarize_parse_quality(self.document(
            chunks=[{'kind': 'evidence', 'text': '20 mm'}], visuals=[visual],
            source_locations={'row': {'row_index': 1, 'extraction_confidence': .6}},
            visual_coverage={'page_count': 3, 'coverage_complete': False,
                             'ocr_index_pending_page_numbers': [3]}))
        self.assertTrue(summary['text_available'])
        self.assertEqual(summary['structured_row_count'], 1)
        self.assertEqual(summary['ready_visual_count'], 1)
        self.assertIn('ocr_candidates_unverified', summary['warnings'])
        self.assertIn('incomplete_page_rendering', summary['warnings'])
        self.assertNotIn('image_bytes', str(summary))
        self.assertNotIn('20 mm', str(summary))

    def test_real_text_intake_attaches_summary(self):
        result = add_files([('test.txt', b'Useful native text')])
        try:
            summary = result['documents'][0]['parse_quality']
            self.assertTrue(summary['text_available'])
            self.assertEqual(summary['ready_visual_count'], 0)
        finally:
            delete_session(result['session_id'])

    def test_bounded_page_and_image_samples(self):
        v = SimpleNamespace(visual_id='v', kind='embedded_image', source={},
                            image_bytes=b'x', metadata={})
        summary = summarize_parse_quality(self.document(visuals=[v] * 100,
            visual_coverage={'ocr_index_pending_page_numbers': list(range(100))}))
        self.assertEqual(len(summary['visual_samples']), 8)
        self.assertEqual(len(summary['ocr_pending_page_samples']), 8)
        self.assertEqual(summary['ocr_pending_page_count'], 100)


if __name__ == '__main__':
    unittest.main()
