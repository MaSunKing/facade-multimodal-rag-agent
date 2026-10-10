"""JSON decoding constraints; authoritative ACL/semantic validators stay separate."""
from functools import lru_cache


@lru_cache(maxsize=1)
def tokenizer_data(tokenizer):
    # Use the library core rather than its older Transformers adapter, whose
    # import path is incompatible with Transformers 5. No package monkeypatch.
    from lmformatenforcer import TokenEnforcerTokenizerData
    zero = tokenizer.encode('0', add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for index in range(len(tokenizer)):
        if index in special:
            continue
        after = tokenizer.decode([zero, index])[1:]
        single = tokenizer.decode([index])
        regular.append((index, after, len(after) > len(single)))
    return TokenEnforcerTokenizerData(regular,
        lambda tokens:tokenizer.decode(tokens).rstrip('\ufffd'),
        tokenizer.eos_token_id, False, len(tokenizer))


def json_prefix(tokenizer, schema=None):
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer, CharacterLevelParserConfig
    # Schema roles use a stable field order and request compact separators. This does
    # not choose a verdict; it prevents spending the output budget formatting
    # or repeatedly selecting optional protocol fields before required ones.
    data = tokenizer_data(tokenizer)
    config = CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet,
        force_json_field_order=True) if schema else CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet)
    # JsonSchemaParser caches its string alphabet at construction. Supplying
    # it only after TokenEnforcer initialization is too late for Unicode text.
    parser = JsonSchemaParser(schema or {'type':'object'}, config=config)
    enforcer = TokenEnforcer(data, parser)
    # TokenEnforcer initializes the parser's config itself using the tokenizer
    # alphabet. Apply caller options afterwards, preserving that alphabet.
    if schema:
        parser.config = config
    syntax = TokenEnforcer(data, JsonSchemaParser({'type':'object'},
        config=CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet))) if schema else None
    def allowed(batch_id, sent):
        choices = enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
        if syntax is not None:
            legal = syntax.get_allowed_tokens(sent.tolist()).allowed_tokens
            intersection = set(choices).intersection(legal)
            # Schema parser support varies by library version. It must never
            # permit an early EOS / malformed JSON rejected by the syntax
            # parser. Keep syntax legal when schema reaches a dead end;
            # authoritative field/ID validators still run and may reject it.
            if not intersection:
                allowed.schema_fallback_steps += 1
                return legal
            return sorted(intersection)
        return choices
    allowed.schema_fallback_steps = 0
    allowed.json_schema = schema
    return allowed
