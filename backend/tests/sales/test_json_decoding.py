import unittest
import torch
from backend.sales.json_decoding import json_prefix, tokenizer_data


class AsciiTokenizer:
    all_special_ids = [127]
    eos_token_id = 127
    def __len__(self): return 128
    def encode(self, text, **kwargs): return list(map(ord, text))
    def decode(self, tokens): return ''.join(chr(int(i)) for i in tokens)


class JsonDecodingTests(unittest.TestCase):
    def tearDown(self): tokenizer_data.cache_clear()

    def test_syntax_filters_missing_comma_and_early_eos(self):
        tokenizer = AsciiTokenizer()
        prefix = json_prefix(tokenizer)
        prompt = [65]
        allowed = prefix(0, torch.tensor(prompt))
        self.assertIn(ord('{'), allowed)
        self.assertNotIn(ord('['), allowed)
        self.assertNotIn(127, allowed)
        for char in '{"a":"x"':
            self.assertIn(ord(char), prefix(0, torch.tensor(prompt)))
            prompt.append(ord(char))
        allowed = prefix(0, torch.tensor(prompt))
        self.assertIn(ord(','), allowed)
        self.assertIn(ord('}'), allowed)
        self.assertNotIn(ord('b'), allowed)
        prompt.append(ord('}'))
        self.assertIn(127, prefix(0, torch.tensor(prompt)))

    def test_each_attempt_has_new_parser_but_reuses_tokenizer_data(self):
        tokenizer = AsciiTokenizer()
        first = json_prefix(tokenizer)
        first(0, torch.tensor([65]))
        first(0, torch.tensor([65, ord('{')]))
        second = json_prefix(tokenizer)
        self.assertIn(ord('{'), second(0, torch.tensor([65])))
        self.assertEqual(tokenizer_data.cache_info().misses, 1)

    def test_answer_review_schema_blocks_unknown_ids_and_wrong_span_shape(self):
        from backend.sales.answer_review import review_schema
        tokenizer=AsciiTokenizer()
        prefix=json_prefix(tokenizer,review_schema({'evidence':[{'evidence_id':'T7'}],
            'planned_task':{'operation':'comparison'}}))
        sent=[65]
        def feed(value):
            for char in value:
                self.assertIn(ord(char),prefix(0,torch.tensor(sent)),value)
                sent.append(ord(char))
        feed('{"decision":"pass","status":"supported","supported_evidence_ids":["')
        self.assertNotIn(ord('X'),prefix(0,torch.tensor(sent)))
        feed('T7"],"support_spans":')
        allowed=prefix(0,torch.tensor(sent))
        self.assertIn(ord('{'),allowed)
        self.assertNotIn(ord('['),allowed)
        feed('{"T7":["source quote"]}}')
        self.assertIn(127,prefix(0,torch.tensor(sent)))

    def test_empty_context_schema_allows_limited_without_invented_support(self):
        from backend.sales.answer_review import review_schema
        tokenizer=AsciiTokenizer()
        prefix=json_prefix(tokenizer,review_schema({'evidence':[]}))
        sent=[65]
        for char in '{"decision":"limited","status":"uncertain","supported_evidence_ids":[':
            self.assertIn(ord(char),prefix(0,torch.tensor(sent)))
            sent.append(ord(char))
        self.assertIn(ord(']'),prefix(0,torch.tensor(sent)))
        for char in ']}':
            self.assertIn(ord(char),prefix(0,torch.tensor(sent)))
            sent.append(ord(char))
        self.assertIn(127,prefix(0,torch.tensor(sent)))
        # The installed decoder does not fully enforce maxItems=0; authoritative
        # validation remains mandatory even when decoding uses a field schema.
        from backend.sales.answer_review import validate_review
        from backend.sales.output_contract import OutputContractError
        with self.assertRaises(OutputContractError):
            validate_review({'decision':'limited','status':'uncertain',
                'supported_evidence_ids':['invented']},{'evidence':[]})

    def test_schema_requires_verdict_before_optional_fields(self):
        from backend.sales.answer_review import review_schema
        prefix=json_prefix(AsciiTokenizer(),review_schema({'evidence':[{'evidence_id':'T7'}]}))
        sent=[65]
        for char in '{"':
            self.assertIn(ord(char),prefix(0,torch.tensor(sent)))
            sent.append(ord(char))
        allowed=prefix(0,torch.tensor(sent))
        self.assertIn(ord('d'),allowed)
        self.assertNotIn(ord('r'),allowed)
        # Do not impose a one-space cap: the installed parser applies it even
        # inside strings and would alter exact original source quotes.

    def test_schema_dead_end_cannot_force_early_eos(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        class DeadEnd:
            def __init__(self, data, parser): pass
            def get_allowed_tokens(self, sent):
                return SimpleNamespace(allowed_tokens=[127])
        from lmformatenforcer import TokenEnforcer
        created=[]
        def build(data,parser):
            result=DeadEnd(data,parser) if not created else TokenEnforcer(data,parser)
            created.append(result)
            return result
        with patch('lmformatenforcer.TokenEnforcer',side_effect=build):
            prefix=json_prefix(AsciiTokenizer(),{'type':'object'})
        allowed=prefix(0,torch.tensor([65]))
        self.assertIn(ord('{'),allowed)
        self.assertNotIn(127,allowed)
        self.assertEqual(prefix.schema_fallback_steps,1)

    def test_unicode_string_does_not_empty_parser_stack_or_allow_eos(self):
        class UnicodeTokenizer(AsciiTokenizer):
            def __len__(self): return 129
            def decode(self,tokens):
                return ''.join('汉' if int(i)==128 else chr(int(i)) for i in tokens)
        for schema in (None,{'type':'object','properties':{'text':{'type':'string'}},
                             'required':['text'],'additionalProperties':False}):
            tokenizer_data.cache_clear()
            prefix=json_prefix(UnicodeTokenizer(),schema)
            sent=[65]
            for token in list(map(ord,'{"text":"'))+[128]:
                self.assertIn(token,prefix(0,torch.tensor(sent)))
                sent.append(token)
            allowed=prefix(0,torch.tensor(sent))
            self.assertNotIn(127,allowed)
            self.assertIn(ord('"'),allowed)
            for token in map(ord,'"}'):
                self.assertIn(token,prefix(0,torch.tensor(sent)))
                sent.append(token)
            self.assertIn(127,prefix(0,torch.tensor(sent)))
            self.assertEqual(prefix.schema_fallback_steps,0)

    def test_short_review_schema_only_requests_quotes_for_comparison(self):
        from backend.sales.answer_review import review_schema
        for operation in ('lookup','summary'):
            schema=review_schema({'planned_task':{'operation':operation}})
            self.assertNotIn('support_spans',schema['properties'])
        schema=review_schema({'planned_task':{'operation':'comparison'}})
        self.assertIn('support_spans',schema['required'])

    def test_comparison_quote_choices_are_original_whole_units(self):
        from backend.sales.answer_review import review_schema
        schema=review_schema({'planned_task':{'operation':'comparison'},'evidence':[
            {'evidence_id':'T1','text':'厚度18毫米。\n安装  条件保留原空格。'}]})
        choices=schema['properties']['support_spans']['properties']['T1']['items']['enum']
        self.assertEqual(choices,['厚度18毫米。','安装  条件保留原空格。'])
        schema=review_schema({'planned_task':{'operation':'comparison'},'evidence':[
            {'evidence_id':'T1','text':'长'*101}]})
        self.assertNotIn('enum',schema['properties']['support_spans']['properties']['T1']['items'])

    def test_revision_citations_are_an_array_not_a_single_object(self):
        from backend.sales.answer_review import revision_schema
        prefix=json_prefix(AsciiTokenizer(),revision_schema({'T1'}))
        sent=[65]
        for char in '{"answerable":true,"citations":':
            self.assertIn(ord(char),prefix(0,torch.tensor(sent)))
            sent.append(ord(char))
        allowed=prefix(0,torch.tensor(sent))
        self.assertIn(ord('['),allowed)
        self.assertNotIn(ord('{'),allowed)
        for char in '[{"evidence_id":"T1"}],"customer_reply":"answer"}':
            self.assertIn(ord(char),prefix(0,torch.tensor(sent)))
            sent.append(ord(char))
        self.assertIn(127,prefix(0,torch.tensor(sent)))


if __name__ == '__main__': unittest.main()
