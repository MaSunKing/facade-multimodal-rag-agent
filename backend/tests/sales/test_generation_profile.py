import torch
import unittest
from unittest.mock import patch
from backend.sales.generation_profile import TokenTimer, profiled_generate, generation_kwargs


def test_token_timer_does_not_count_prompt():
    audit = {}
    timer = TokenTimer(audit)
    timer.put(torch.tensor([[1, 2, 3]]))
    timer.put(torch.tensor([4]))
    timer.put(torch.tensor([5]))
    timer.end()
    assert audit['output_tokens'] == 2
    assert audit['time_to_first_token_seconds'] >= 0


def test_profile_uses_shared_config_without_real_model():
    class Model:
        hf_device_map = {'decoder': 'cpu'}
        def generate(self, input_ids, streamer, **kw):
            assert kw['do_sample'] is False
            streamer.put(input_ids)
            streamer.put(torch.tensor([4]))
            streamer.end()
            return torch.tensor([[1, 2, 4]])
    audit = {}
    output = profiled_generate(Model(), {'input_ids': torch.tensor([[1, 2]])}, audit=audit,
        **generation_kwargs(5, 0, []))
    assert output.shape[-1] == 3
    assert audit['input_tokens'] == 2 and audit['output_tokens'] == 1
    assert audit['device_map'] == {'decoder': 'cpu'}
    assert audit['cpu_or_disk_offload'] is True


def test_failed_generation_retains_partial_timing_not_content():
    class Model:
        hf_device_map = {'decoder': 'cuda:0'}
        def generate(self, input_ids, streamer, **kw):
            streamer.put(input_ids)
            streamer.put(torch.tensor([4]))
            raise TimeoutError('stage timeout')
    audit = {}
    with patch('torch.cuda.is_available', return_value=False):
        try:
            profiled_generate(Model(), {'input_ids': torch.tensor([[1, 2]])}, audit=audit,
                              **generation_kwargs(5, 0, []))
        except TimeoutError:
            pass
        else:
            raise AssertionError('timeout must propagate')
    assert audit['error_type'] == 'TimeoutError'
    assert audit['output_tokens'] == 1
    assert audit['time_to_first_token_seconds'] is not None
    assert audit['generation_seconds'] >= 0
    assert audit['cpu_or_disk_offload'] is False
    assert not any(key in audit for key in ('prompt', 'output_text', 'input_ids'))


def test_each_format_attempt_keeps_its_own_timing():
    from backend.tests.sales.test_model_contract import Model, Tokenizer
    from backend.sales.model_contract import checked_generate, fields_contract
    from backend.sales.runtime_status import ExecutionStatus, current_execution
    class StreamingModel(Model):
        def generate(self, **kwargs):
            out = super().generate(**kwargs)
            streamer = kwargs['streamer']
            streamer.put(kwargs['input_ids'])
            for token in out[0, kwargs['input_ids'].shape[-1]:]:
                streamer.put(token.unsqueeze(0))
            streamer.end()
            return out
    journal = ExecutionStatus()
    binding = current_execution.set(journal)
    try:
        model = StreamingModel(['{}', '{"ok":true}'])
        checked_generate(model, contract=fields_contract('test', Tokenizer(), {'ok': bool}),
                         input_ids=torch.tensor([[1,2]]), max_new_tokens=100)
        attempts = journal.model_contracts[0]['attempts']
        assert len(attempts) == 2
        assert attempts[0]['runtime_profile']['output_tokens'] == 2
        assert attempts[1]['runtime_profile']['output_tokens'] == 11
        for row in attempts:
            profile = row['runtime_profile']
            assert profile['token_stream_observed']
            assert profile['time_to_first_token_seconds'] >= 0
            assert profile['decode_seconds'] >= 0
            assert not any(k in profile for k in ('prompt', 'output_text', 'input_ids'))
    finally:
        current_execution.reset(binding)


def test_attempt_timer_preserves_caller_stream():
    from backend.sales.generation_profile import AttemptTimer
    class Sink:
        count = 0
        def put(self, value): self.count += value.numel()
        def end(self): pass
    sink = Sink()
    timer = AttemptTimer({}, sink)
    timer.put(torch.tensor([[1,2]]))
    timer.put(torch.tensor([3]))
    timer.end()
    assert sink.count == 3
    assert timer.audit['output_tokens'] == 1


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith('test_') and callable(fn))
