"""Loopback-only, authenticated tensor RPC. No pickle, prompts in logs, or cloud."""
import base64
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TOKEN_PATH=ROOT/'runtime/model_service.token'
URL='http://127.0.0.1:8001'
MAX_WIRE_BYTES=128*1024*1024


def enabled(): return os.getenv('FACADE_MODEL_RUNTIME','embedded')=='service'


def headers():
    return {'Authorization':'Bearer '+TOKEN_PATH.read_text(encoding='utf8').strip(),
            'Content-Type':'application/json'}


def call(path, value=None):
    data=None if value is None else json.dumps(value,separators=(',',':')).encode()
    with urlopen(Request(URL+path,data=data,headers=headers()),timeout=5) as response:
        return json.load(response)


def status():
    try: return call('/status')
    except (OSError,ValueError): return {'model_loaded':False,'status':'unavailable'}


def pack(tensors):
    import torch
    arrays,types={},{}
    for name,value in tensors.items():
        if not isinstance(value,torch.Tensor): raise ValueError('non_tensor_input')
        value=value.detach().cpu().contiguous()
        types[name]=str(value.dtype).removeprefix('torch.')
        arrays[name]=(value.view(torch.uint16) if value.dtype==torch.bfloat16 else value).numpy()
    stream=io.BytesIO(); np.savez(stream,**arrays)
    if stream.tell()>MAX_WIRE_BYTES//2: raise ValueError('tensor_payload_too_large')
    return {'arrays':base64.b64encode(stream.getvalue()).decode('ascii'),'dtypes':types}


def unpack(value):
    import torch
    encoded=value['arrays']
    if not isinstance(encoded,str) or len(encoded)>MAX_WIRE_BYTES: raise ValueError('tensor_payload_too_large')
    data=base64.b64decode(encoded,validate=True)
    result={}
    with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
        if len(arrays.files)>16: raise ValueError('too_many_tensors')
        for key in arrays.files:
            array=arrays[key]
            if array.dtype.hasobject or array.nbytes>MAX_WIRE_BYTES//2: raise ValueError('unsafe_tensor')
            tensor=torch.from_numpy(array.copy())
            dtype=value['dtypes'][key]
            if dtype=='bfloat16': tensor=tensor.view(torch.bfloat16)
            elif str(tensor.dtype).removeprefix('torch.')!=dtype: raise ValueError('tensor_dtype_mismatch')
            result[key]=tensor
    return result


class RemoteModel:
    """Transformers-shaped adapter; validators/retries remain in business process."""
    is_remote_model=True
    device='cpu'
    hf_device_map={}  # Adapter placement is not proof of worker parameter placement.
    dtype='bfloat16'

    def __init__(self,tokenizer):
        self.generation_config=SimpleNamespace(eos_token_id=tokenizer.eos_token_id)
        self.last_runtime_profile={}

    def _run(self,action,tensors,options=None,prefix=None,streamer=None):
        import torch
        from backend.request_budget import check_budget
        check_budget()
        self.last_runtime_profile = {}
        transport_profile = {}
        serialization_started = time.monotonic()
        op=uuid.uuid4().hex; stop=threading.Event()
        from backend.request_budget import current_budget
        budget=current_budget.get()
        def watch():
            while not stop.wait(.2):
                if budget is not None and budget.expired():
                    try: call('/cancel',{'operation_id':op})
                    except (OSError,ValueError): pass
        thread=threading.Thread(target=watch,daemon=True);thread.start()
        payload={'operation_id':op,'action':action,'tensors':pack(tensors),
                 'options':options or {},'json_constraint':prefix is not None,
                 'json_schema':getattr(prefix,'json_schema',None)}
        transport_profile['tensor_serialization_seconds'] = round(time.monotonic()-serialization_started,4)
        if streamer is not None: streamer.put(tensors['input_ids'])
        try:
            data=json.dumps(payload,separators=(',',':')).encode()
            rpc_started = time.monotonic()
            first_token_at = last_token_at = None
            # No model-stage wall-clock cutoff. Caller cancellation/deadline
            # travels separately to /cancel while streaming/loading continues.
            with urlopen(Request(URL+'/run',data=data,headers=headers()),timeout=None) as response:
                transport_profile['response_headers_seconds'] = round(time.monotonic()-rpc_started,4)
                for line in response:
                    event=json.loads(line)
                    if event['kind']=='tokens':
                        last_token_at = time.monotonic()
                        if first_token_at is None:
                            first_token_at = last_token_at
                            transport_profile['rpc_to_first_token_seconds'] = round(first_token_at-rpc_started,4)
                        if streamer is not None:
                            streamer.put(torch.tensor(event['ids']))
                    elif event['kind']=='error':
                        check_budget()
                        if event.get('code')=='OutOfMemoryError': raise torch.cuda.OutOfMemoryError('model_service_oom')
                        failure=RuntimeError('model_service_'+event.get('code','failed'))
                        failure.worker_frames=event.get('frames',[])
                        raise failure
                    elif event['kind']=='result':
                        transport_profile['rpc_total_seconds'] = round(time.monotonic()-rpc_started,4)
                        if last_token_at is not None:
                            transport_profile['rpc_post_last_token_seconds'] = round(time.monotonic()-last_token_at,4)
                        self.last_runtime_profile={**event.get('runtime_profile',{}), **transport_profile}
                        worker_seconds = self.last_runtime_profile.get('worker_inference_seconds')
                        if isinstance(worker_seconds,(int,float)):
                            self.last_runtime_profile['unattributed_rpc_overhead_seconds'] = round(max(0,
                                transport_profile['rpc_total_seconds']-worker_seconds),4)
                        if prefix is not None: prefix.schema_fallback_steps=event.get('schema_fallback_steps',0)
                        check_budget()
                        return unpack(event['tensors'])['output']
            raise RuntimeError('model_service_incomplete_stream')
        finally:
            self.last_runtime_profile.update(transport_profile)
            stop.set()
            if streamer is not None: streamer.end()

    def generate(self,**kwargs):
        import torch
        tensors={k:v for k,v in kwargs.items() if isinstance(v,torch.Tensor)}
        supported={'max_new_tokens','do_sample','pad_token_id','eos_token_id','temperature','top_p','top_k','repetition_penalty','use_cache'}
        ignored={'stopping_criteria','prefix_allowed_tokens_fn','streamer'}
        unknown=set(kwargs)-set(tensors)-supported-ignored
        if unknown: raise ValueError('unsupported_remote_generation_option')
        options={k:v for k,v in kwargs.items() if k in supported}
        return self._run('generate',tensors,options,kwargs.get('prefix_allowed_tokens_fn'),kwargs.get('streamer'))

    def get_image_features(self,**kwargs):
        # Existing caller mean-pools this single-row mean feature again.
        output=self._run('image_features',kwargs)
        return SimpleNamespace(pooler_output=[output])
