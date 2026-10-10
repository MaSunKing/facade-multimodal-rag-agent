import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from fastapi.testclient import TestClient
from backend.sales import model_transport as rpc
from backend.sales import model_service as worker


class ModelServiceTest(unittest.TestCase):
    def test_rpc_profile_is_per_call_and_observes_transport_not_contents(self):
        model=rpc.RemoteModel(SimpleNamespace(eos_token_id=9))
        model.last_runtime_profile={'stale':True}
        events=[{'kind':'tokens','ids':[9]}, {'kind':'result',
                'tensors':rpc.pack({'output':torch.tensor([[1,9]])}),
                'runtime_profile':{'worker_inference_seconds':0.0}}]
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __iter__(self): return iter(json.dumps(e).encode() for e in events)
        with patch.object(rpc,'urlopen',return_value=Response()), patch.object(rpc,'headers',return_value={}):
            model.generate(input_ids=torch.tensor([[1]]),max_new_tokens=2)
        profile=model.last_runtime_profile
        self.assertNotIn('stale',profile)
        self.assertGreaterEqual(profile['rpc_to_first_token_seconds'],0)
        self.assertGreaterEqual(profile['rpc_total_seconds'],profile['rpc_to_first_token_seconds'])
        self.assertGreaterEqual(profile['tensor_serialization_seconds'],0)
        self.assertNotIn('tensors',profile)

    def test_tensor_roundtrip_including_bfloat_and_pixels(self):
        values={'input_ids':torch.tensor([[1,2,3]]),'mask':torch.ones(1,3,dtype=torch.int64),
                'pixels':torch.randn(2,8,dtype=torch.bfloat16)}
        out=rpc.unpack(rpc.pack(values))
        for k in values:
            self.assertEqual(out[k].dtype,values[k].dtype)
            self.assertTrue(torch.equal(out[k],values[k]))

    def test_schema_transport_preserves_streaming_and_validation_options(self):
        model=rpc.RemoteModel(SimpleNamespace(eos_token_id=9))
        prefix=SimpleNamespace(json_schema={'type':'object'},schema_fallback_steps=0)
        with patch.object(model,'_run',return_value=torch.tensor([[1,9]])) as run:
            model.generate(input_ids=torch.tensor([[1]]),max_new_tokens=7,do_sample=False,
                stopping_criteria=[],prefix_allowed_tokens_fn=prefix)
        self.assertIs(run.call_args.args[3],prefix)
        self.assertEqual(run.call_args.args[2]['max_new_tokens'],7)
        with self.assertRaises(ValueError):model.generate(input_ids=torch.tensor([[1]]),arbitrary_option='unsafe')

    def test_health_is_authenticated_and_does_not_touch_idle_time(self):
        from backend import app as runtime
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'token';path.write_text('test-secret')
            with patch.object(worker,'TOKEN_PATH',path),patch.object(runtime,'model_idle_seconds',return_value=1199),\
                 patch.object(runtime,'mark_model_activity') as activity:
                client=TestClient(worker.app)
                self.assertEqual(client.get('/status').status_code,403)
                value=client.get('/status',headers={'Authorization':'Bearer test-secret'}).json()
                self.assertEqual(value['idle_seconds'],1199)
                self.assertEqual(value['idle_unload_seconds'],1200)
                activity.assert_not_called()
                self.assertNotIn('test-secret',json.dumps(value))

    def test_business_shutdown_does_not_unload_service(self):
        from backend import app as runtime
        with patch.dict(os.environ,{'FACADE_MODEL_RUNTIME':'service'}),\
             patch.object(rpc,'status',return_value={'model_loaded':True,'idle_seconds':41,'pid':77}),\
             patch.object(runtime,'_model',None):
            self.assertFalse(runtime.unload_model_if_idle())
            runtime.stop_idle_model_reaper()
            self.assertTrue(runtime.ready()['model_loaded'])
            self.assertEqual(runtime.ready()['model_service_pid'],77)

    def test_idle_unload_boundary_and_activity_reset(self):
        from backend import app as runtime
        with patch.dict(os.environ,{'FACADE_MODEL_RUNTIME':'embedded'}),\
             patch.object(runtime,'_model',object()),patch.object(runtime,'_processor',object()),\
             patch.object(runtime,'_tokenizer',object()),patch.object(runtime,'MODEL_IDLE_UNLOAD_SECONDS',1200),\
             patch.object(runtime,'_model_last_used_monotonic',100.0),\
             patch.object(runtime.time,'monotonic',return_value=1299.9):
            self.assertFalse(runtime.unload_model_if_idle())
            runtime.mark_model_activity()
            self.assertEqual(runtime._model_last_used_monotonic,1299.9)
            with patch.object(runtime.time,'monotonic',return_value=2500),\
                 patch('torch.cuda.is_available',return_value=False),\
                 patch('backend.sales.dense_retrieval.restore_after_generation_model'):
                self.assertTrue(runtime.unload_model_if_idle())
                self.assertIsNone(runtime._model)

    def test_cancel_signals_only_matching_operation(self):
        a,b=threading.Event(),threading.Event()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'token';path.write_text('test-secret')
            with patch.object(worker,'TOKEN_PATH',path),patch.object(worker,'_pending',{'a':a,'b':b}):
                client=TestClient(worker.app)
                client.post('/cancel',json={'operation_id':'a'},headers={'Authorization':'Bearer test-secret'})
                self.assertTrue(a.is_set());self.assertFalse(b.is_set())

if __name__=='__main__':unittest.main()
