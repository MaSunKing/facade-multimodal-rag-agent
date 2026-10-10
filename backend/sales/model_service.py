"""Independent Qwen lifecycle; business API restarts never own this process."""
import asyncio
import json
import os
import queue
import threading
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from backend.sales.model_transport import TOKEN_PATH, ROOT, unpack, pack, MAX_WIRE_BYTES

app=FastAPI()
_pending={}
_pending_lock=threading.Lock()


def authorize(request):
    import hmac
    expected='Bearer '+TOKEN_PATH.read_text(encoding='utf8').strip()
    if not hmac.compare_digest(request.headers.get('authorization',''),expected):
        raise HTTPException(403,'model_service_access_denied')


@app.on_event('startup')
def startup():
    from backend import app as runtime
    runtime.start_idle_model_reaper()


@app.on_event('shutdown')
def shutdown():
    from backend import app as runtime
    runtime.stop_idle_model_reaper()


@app.get('/status')
def status(request:Request):
    authorize(request)
    import sys
    from backend import app as runtime
    idle=runtime.model_idle_seconds()
    with _pending_lock: active=len(_pending)
    return {'status':'ready' if runtime._model is not None else 'cold','model_loaded':runtime._model is not None,
            'idle_seconds':idle,'idle_unload_seconds':runtime.MODEL_IDLE_UNLOAD_SECONDS,
            'active_operations':active,'pid':os.getpid(),'executable':sys.executable,'cwd':str(ROOT),
            'implementation':'facade_qwen_worker_v1',
            'attention_implementation':os.getenv('FACADE_ATTENTION_IMPLEMENTATION','sdpa')}


@app.post('/cancel')
async def cancel(request:Request):
    authorize(request); value=await request.json()
    with _pending_lock:
        event=_pending.get(value.get('operation_id'))
        if event: event.set()
    return {'cancel_requested':event is not None}


@app.post('/run')
async def run(request:Request):
    authorize(request)
    # Bound bytes before JSON/base64/tensor allocation. Do not log payload.
    chunks=[]; size=0
    async for chunk in request.stream():
        size+=len(chunk)
        if size>MAX_WIRE_BYTES: raise HTTPException(413,'model_input_too_large')
        chunks.append(chunk)
    value=json.loads(b''.join(chunks))
    import re
    op=value.get('operation_id','')
    if not re.fullmatch('[a-f0-9]{32}',op) or value.get('action') not in {'generate','image_features'}:
        raise HTTPException(400,'invalid_model_operation')
    cancel_event=threading.Event(); events=queue.Queue()
    with _pending_lock:
        if op in _pending: raise HTTPException(409,'duplicate_model_operation')
        if len(_pending)>=8: raise HTTPException(429,'model_queue_full')
        _pending[op]=cancel_event

    def execute():
        import torch
        from transformers import StoppingCriteria,StoppingCriteriaList
        from backend import app as runtime
        class Cancel(StoppingCriteria):
            def __call__(self,*args,**kwargs): return cancel_event.is_set()
        class Tokens:
            seen=False
            def put(self,tensor):
                if not self.seen: self.seen=True;return
                events.put({'kind':'tokens','ids':tensor.detach().cpu().flatten().tolist()})
            def end(self): pass
        try:
            # Queue/load/inference protect one lifecycle lock. Health does not
            # refresh activity; only actual work does, including its completion.
            with runtime.generation_session(),torch.inference_mode():
                if cancel_event.is_set(): raise RuntimeError('cancelled')
                tokenizer,model=runtime.load_model()
                if cancel_event.is_set(): raise RuntimeError('cancelled')
                tensors={k:v.to(model.device) for k,v in unpack(value['tensors']).items()}
                prefix=None
                if value.get('json_constraint'):
                    from backend.sales.json_decoding import json_prefix
                    prefix=json_prefix(tokenizer,value.get('json_schema'))
                torch.cuda.reset_peak_memory_stats()
                started=time.monotonic()
                if value['action']=='generate':
                    options=value.get('options',{})
                    allowed={'max_new_tokens','do_sample','pad_token_id','eos_token_id','temperature','top_p','top_k','repetition_penalty','use_cache'}
                    if set(options)-allowed: raise ValueError('invalid_options')
                    if not 1<=options.get('max_new_tokens',1)<=8192: raise ValueError('invalid_output_budget')
                    output=model.generate(**tensors,**options,streamer=Tokens(),
                        stopping_criteria=StoppingCriteriaList([Cancel()]),
                        **({'prefix_allowed_tokens_fn':prefix} if prefix is not None else {}))
                else:
                    features=model.get_image_features(**tensors).pooler_output[0].float()
                    output=features.mean(dim=0,keepdim=True)
                profile={'model_service_pid':os.getpid(),'gpu_peak_allocated':torch.cuda.max_memory_allocated(),
                         'gpu_allocated_after':torch.cuda.memory_allocated(),'gpu_reserved_after':torch.cuda.memory_reserved(),
                         'worker_inference_seconds':round(time.monotonic()-started,4)}
                events.put({'kind':'result','tensors':pack({'output':output}),
                    'runtime_profile':profile,'schema_fallback_steps':getattr(prefix,'schema_fallback_steps',0)})
        except Exception as exc:
            import traceback
            from pathlib import Path
            frames=[{'file':Path(frame.filename).name,'function':frame.name,'line':frame.lineno}
                    for frame in traceback.extract_tb(exc.__traceback__)[-8:]]
            events.put({'kind':'error','code':type(exc).__name__,'frames':frames})
        finally:
            with _pending_lock: _pending.pop(op,None)
            events.put(None)
    threading.Thread(target=execute,daemon=True).start()
    async def stream():
        try:
            while True:
                try: event=events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(.05);continue
                if event is None: break
                yield json.dumps(event,separators=(',',':'))+'\n'
        finally: cancel_event.set()
    return StreamingResponse(stream(),media_type='application/x-ndjson')
