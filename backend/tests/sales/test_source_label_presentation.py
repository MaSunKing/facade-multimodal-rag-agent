import unittest
from backend.app import sanitize_customer_document_presentation, remove_unsupported_numeric_sentences, compact_grounded_payload_for_generation


class SourceLabelPresentationTests(unittest.TestCase):
    def test_local_lookup_does_not_interleave_unrelated_sheets(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs): return str(messages)
            def __call__(self,text,**kwargs): return {'input_ids':list(range(len(text)//4))}
        payload={'attachment_context':{'global_document_question':False},'evidence':[
            {'evidence_id':eid,'document_name':'test.xlsx','source_group':sheet,'original_chunk_id':sheet,'evidence_scope':'content','text':f'row {eid}'}
            for eid,sheet in [('U1','Target'),('U2','Target'),('U3','Other')]
        ]}
        _,audit=compact_grounded_payload_for_generation(payload,Tokenizer(),max_prompt_tokens=10000,system_prompt='test')
        self.assertEqual(audit['kept_evidence_ids'][:2],['U1','U2'])

    def test_reject_promotional_clause_without_replacing_supported_answer(self):
        response={'customer_reply':'这是外墙饰面板。它可以免维护。','key_points':[], 'missing_information':[], 'risk_warnings':[], 'next_action':''}
        value=remove_unsupported_numeric_sentences(response,['免维护'])
        self.assertIn('外墙饰面板',value['customer_reply'])
        self.assertNotIn('免维护',value['customer_reply'])

    def test_distinct_documents_keep_names(self):
        response={'customer_reply':'U1是申请表；U2是产品说明。', 'missing_information':[], 'analysis':[], 'recommendations':[]}
        sources={'U1':{'document_name':'申请.docx','text':'姓名'}, 'U2':{'document_name':'介绍.pdf','text':'产品'}}
        result=sanitize_customer_document_presentation(response,sources)
        self.assertIn('《申请.docx》',result['customer_reply'])
        self.assertIn('《介绍.pdf》',result['customer_reply'])
        self.assertNotIn('相关证据',result['customer_reply'])

    def test_unknown_reference_not_given_fabricated_filename(self):
        response={'customer_reply':'U99有数据。', 'missing_information':[], 'analysis':[], 'recommendations':[]}
        result=sanitize_customer_document_presentation(response,{})
        self.assertNotIn('《',result['customer_reply'])

if __name__=='__main__': unittest.main()
