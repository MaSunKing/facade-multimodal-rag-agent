"""Sequential, resumable benchmark; one absolute 1200s deadline per case.

No automatic semantic pass from substring hits. A timed-out worker is cancelled
and confirmed idle before another case; failed cancellation stops the run.
"""
import argparse
import os
from collections import Counter
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import uuid
from collections import deque

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/sales/evaluation/agent_eval_100_v1'


def async_call(fn):
    results=queue.Queue()
    def work():
        try: results.put((True,fn()))
        except Exception as exc: results.put((False,exc))
    threading.Thread(target=work,daemon=True).start()
    return results


def await_result(results, deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0: raise TimeoutError('case_total_deadline')
    try: ok,value=results.get(timeout=remaining)
    except queue.Empty: raise TimeoutError('case_total_deadline')
    if not ok: raise value
    return value


class TransportFailure(Exception):
    def __init__(self, status, path, code=None, retry_after=None):
        super().__init__('http_request_failed')
        self.details = {'http_status':status, 'endpoint':path,
                        'error_code':code, 'retry_after_seconds':retry_after}


class Client:
    def __init__(self, base):
        self.base=base.rstrip('/')
        self.headers={'X-Facade-Client-ID':'eval100-'+uuid.uuid4().hex}
        self.admission_lock=threading.Lock()
        self.admissions={'gpu':deque(), 'upload':deque()}

    def admit(self, path, deadline):
        group='gpu' if path in {'/api/copilot/answer','/api/copilot/draft'} else 'upload' if path=='/api/copilot/documents' else None
        if group is None: return
        # Same-IP endpoint limits are security policy, not per-case identities.
        # No rotating IPs, disabling limits or whole-workflow retry on 429.
        with self.admission_lock:
            history=self.admissions[group]; limit=4 if group=='gpu' else 8
            now=time.monotonic()
            while history and now-history[0]>=61: history.popleft()
            if len(history)>=limit:
                wait=max(0,61-(now-history[0]))
                if now+wait>=deadline: raise TimeoutError('admission_wait_exceeds_case_deadline')
                until=now+wait
                while time.monotonic()<until:
                    time.sleep(min(30,until-time.monotonic()))
                now=time.monotonic()
                while history and now-history[0]>=61: history.popleft()
            history.append(now)

    def request(self,path,payload=None,timeout=20,method=None,content_type=None):
        deadline=time.monotonic()+timeout
        if payload is not None and method not in {'GET','DELETE'}:
            self.admit(path,deadline)
        body=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None
        headers=dict(self.headers)
        if body is not None: headers['Content-Type']=content_type or 'application/json'
        try:
            with urlopen(Request(self.base+path,data=body,headers=headers,method=method),timeout=max(.1,deadline-time.monotonic())) as response:
                return json.loads(response.read().decode())
        except HTTPError as exc:
            if exc.code==504: raise TimeoutError('server_request_deadline') from exc
            try:
                value=json.loads(exc.read(16384).decode())
                detail=value.get('error') or value.get('detail') or {}
                code=value.get('code') or (detail.get('code') if isinstance(detail,dict) else None)
                # Persist machine codes, never arbitrary provider bodies/HTML.
                code=code if isinstance(code,str) and code.replace('_','').isalnum() and len(code)<80 else None
            except Exception: code=None
            retry=exc.headers.get('Retry-After') if exc.headers else None
            raise TransportFailure(exc.code,path,code,int(retry) if retry and retry.isdigit() else None) from exc

    def bounded(self,path,deadline,payload=None,**kw):
        return await_result(async_call(lambda:self.request(path,payload,timeout=max(.1,deadline-time.monotonic()),**kw)),deadline)

    def upload(self,files,task_id,deadline):
        boundary='eval-'+uuid.uuid4().hex
        chunks=[f'--{boundary}\r\nContent-Disposition: form-data; name="client_task_id"\r\n\r\n{task_id}\r\n'.encode()]
        for name in files:
            path=DATA/'fixtures'/name
            chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(),path.read_bytes(),b'\r\n']
        chunks.append(f'--{boundary}--\r\n'.encode())
        return self.bounded('/api/copilot/documents',deadline,b''.join(chunks),content_type='multipart/form-data; boundary='+boundary)

    def cancel_and_wait(self,ids,grace=30):
        until=time.monotonic()+grace
        for task in ids:
            self.bounded('/api/copilot/tasks/cancel',until,{'client_task_id':task})
        # Pending cancellation also catches a request admitted after the client
        # timeout. Two idle polls, including global health, prevent slot reuse.
        idle=0
        while time.monotonic()<until:
            statuses=[self.bounded(f'/api/copilot/tasks/{task}/status',until) for task in ids]
            health=self.bounded('/health/ready',until)
            idle=idle+1 if not any(s['active'] for s in statuses) and health.get('active_answer_tasks')==0 else 0
            if idle>=2: return True
            time.sleep(.5)
        return False


def protocol_diagnostics(response):
    meta=response.get('meta') or {}
    execution=meta.get('execution') or {}
    return {'answer_nonempty':bool(str(response.get('customer_reply') or '').strip()),
            'error_codes':[e.get('code') for e in execution.get('errors',[])],
            'answerable':response.get('answerable'),
            'semantic_verdict':'pending_human_review'}


def response_status(response):
    diagnostics=protocol_diagnostics(response)
    codes=set(diagnostics['error_codes'])
    meta=response.get('meta') or {}
    if isinstance(meta.get('error'),dict) and meta['error'].get('code'):
        codes.add(meta['error']['code'])
    if codes & {'WEB_QUOTA_EXHAUSTED','WEB_RATE_LIMITED','SERVICE_BUSY','RATE_LIMITED'}:
        return 'resource_limited'
    if codes - {'WEB_EMPTY'}:
        return 'technical_failure'
    if response.get('answerable') is False:
        return 'insufficient_evidence'
    return 'completed'


def run_case(client,case,seconds=1200):
    started=time.monotonic(); deadline=started+seconds
    ids=[]; trace_ids=[]; responses=[]; session=None
    record={'id':case['id'],'group':case['group'],'question':case['question'],
            'rubric':case['rubric'],'timeout_seconds':seconds,'semantic_verdict':'pending_human_review'}
    try:
        if case.get('mode')=='fault':
            result=subprocess.run([sys.executable,'-m','unittest',case['test_module']],cwd=ROOT,
                capture_output=True,text=True,timeout=max(.1,deadline-time.monotonic()))
            record.update(status='protocol_pass' if result.returncode==0 else 'protocol_fail',
                          protocol_output=result.stdout+result.stderr,semantic_verdict='not_applicable_protocol_only')
            return record
        if case.get('files'):
            task=uuid.uuid4().hex; ids.append(task)
            upload=client.upload(case['files'],task,deadline)
            session=upload['session_id']
        turns=case.get('turns') or [case['question']]
        history=[]; chain=[]
        conversation='eval100-'+uuid.uuid4().hex
        for i,turn in enumerate(turns):
            task=uuid.uuid4().hex; ids.append(task)
            payload={'customer_question':turn,'client_task_id':task,
                'request_timeout_seconds':min(1200,max(.1,deadline-time.monotonic())),
                'use_online_search':bool(case.get('use_online_search')),
                'memory_enabled':False,'conversation_id':conversation,
                'conversation_context':history[-2:]}
            if session: payload['document_session_id']=session
            if case.get('mode')=='steer': payload['task_instructions']=list(chain)
            pending=async_call(lambda p=payload:client.request('/api/copilot/answer',p,timeout=max(.1,deadline-time.monotonic())))
            if case.get('mode') in {'steer','queue'} and i<len(turns)-1:
                # Wait for registration, not an arbitrary delay into inference.
                while time.monotonic()<deadline:
                    if not pending.empty(): break
                    if client.bounded(f'/api/copilot/tasks/{task}/status',deadline).get('active'): break
                    time.sleep(.1)
                if case['mode']=='steer':
                    if not client.cancel_and_wait([task]): raise RuntimeError('steering_cancel_not_confirmed')
                    chain.append(turn)
                    record.setdefault('interrupted_task_ids',[]).append(task)
                    continue
                # Queue a second request while first is active, then consume in
                # order. Each queue scenario has exactly two turns.
                next_task=uuid.uuid4().hex; ids.append(next_task)
                next_payload={**payload,'customer_question':turns[i+1],'client_task_id':next_task,
                    'conversation_context':[{'role':'user','content':turn}],
                    'request_timeout_seconds':min(1200,max(.1,deadline-time.monotonic()))}
                queued=async_call(lambda:client.request('/api/copilot/answer',next_payload,timeout=max(.1,deadline-time.monotonic())))
                overlap=False
                while queued.empty() and pending.empty():
                    next_state=client.bounded(f'/api/copilot/tasks/{next_task}/status',deadline)
                    if next_state.get('active'):
                        overlap=bool(client.bounded(f'/api/copilot/tasks/{task}/status',deadline).get('active'))
                        break
                    time.sleep(.1)
                responses.extend([await_result(pending,deadline),await_result(queued,deadline)])
                record['queue_exercised']=overlap
                break
            response=await_result(pending,deadline); responses.append(response)
            history=[{'role':'user','content':turn},{'role':'assistant','content':str(response.get('customer_reply',''))[:800]}]
            chain.append(turn)
        final=responses[-1]
        for response in responses:
            request_id=((response.get('meta') or {}).get('execution') or {}).get('request_id')
            if request_id: trace_ids.append(request_id)
        traces=[]
        for task in ids:
            try: traces.append(client.bounded('/api/auth/traces/'+task,deadline))
            except Exception: pass  # Uploads have no answer trace; never invent one.
        record.update(status=response_status(final),transport_completed=True,responses=responses,traces=traces,trace_ids=trace_ids,
            diagnostics=protocol_diagnostics(final),
            lexical_hits_diagnostic={p:p in str(final.get('customer_reply','')) for p in case.get('expected_points',[])})
    except (TimeoutError,subprocess.TimeoutExpired) as exc:
        record.update(status='timeout',error_type=type(exc).__name__,case_processing_seconds=round(time.monotonic()-started,3))
        try: record['cancellation_confirmed']=client.cancel_and_wait(ids) if ids else True
        except Exception: record['cancellation_confirmed']=False
    except Exception as exc:
        record.update(status='resource_limited' if isinstance(exc,TransportFailure) and exc.details['http_status']==429 else 'technical_failure',
                      error_type=type(exc).__name__,case_processing_seconds=round(time.monotonic()-started,3))
        if isinstance(exc,TransportFailure): record['transport_error']=exc.details
        try: record['cancellation_confirmed']=client.cancel_and_wait(ids) if ids else True
        except Exception: record['cancellation_confirmed']=False
    finally:
        record.setdefault('case_processing_seconds',round(time.monotonic()-started,3))
        record.update(latency_seconds=round(time.monotonic()-started,3),client_task_ids=ids)
        if record.get('status') in {'timeout','technical_failure','resource_limited'} and not record.get('traces'):
            record['traces']=[]
            for task in ids:
                try: record['traces'].append(client.request('/api/auth/traces/'+task,timeout=3))
                except Exception: pass
        # Cleanup is outside the scoring deadline; cannot turn failure into pass.
        if session and record.get('cancellation_confirmed',True):
            try: client.request('/api/copilot/documents/'+session,method='DELETE',timeout=5)
            except Exception: record['session_cleanup_failed']=True
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/agent_eval_100_v1')
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--group',choices=['company','attachment','web','multiturn','fault'])
    parser.add_argument('--ids',help='Comma-separated frozen case IDs; does not change questions or scoring')
    parser.add_argument('--run',action='store_true',help='Without this, only validate the frozen suite')
    parser.add_argument('--wait-for-idle',action='store_true',help='Wait for active work before starting; never cancel it')
    args=parser.parse_args()
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    raw=(DATA/'questions.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['question_sha256']: raise ValueError('dataset_hash_changed')
    for name,digest in manifest['fixture_sha256'].items():
        if hashlib.sha256((DATA/'fixtures'/name).read_bytes()).hexdigest()!=digest: raise ValueError('fixture_hash_changed')
    cases=[json.loads(line) for line in raw.decode().splitlines() if line]
    assert len(cases)==100 and len({c['id'] for c in cases})==100
    print(json.dumps(dict(Counter(c['group'] for c in cases))))
    if not args.run: print('Validated; use --run to execute.'); return
    args.output.mkdir(parents=True,exist_ok=True)
    run_lock=(args.output/'runner.lock').open('a+')
    try:
        if os.name=='nt':
            import msvcrt
            run_lock.seek(0); msvcrt.locking(run_lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError: raise SystemExit('Another runner owns this output directory; no duplicate run started.')
    previous_manifest=args.output/'run_manifest.json'
    if previous_manifest.exists():
        old=json.loads(previous_manifest.read_text(encoding='utf-8'))
        if old['manifest']!=manifest: raise ValueError('resume_dataset_changed_use_new_output_directory')
    records=args.output/'results.jsonl'
    previous=[json.loads(l) for l in records.read_text(encoding='utf-8').splitlines() if l] if records.exists() else []
    done={r['id'] for r in previous}; client=Client(args.base_url)
    requested_ids=set(args.ids.split(',')) if args.ids else None
    if requested_ids and requested_ids - {c['id'] for c in cases}:
        raise ValueError('unknown_frozen_case_id')
    selected=[c for c in cases if (not args.group or c['group']==args.group)
        and (requested_ids is None or c['id'] in requested_ids)][:args.limit]
    if any(c.get('mode')!='fault' and c['id'] not in done for c in selected):
        while True:
            try: health=client.request('/health/ready',timeout=10)
            except Exception:
                if not args.wait_for_idle: raise
                print('Waiting for backend readiness...',flush=True); time.sleep(5); continue
            if health.get('active_answer_tasks')==0: break
            if not args.wait_for_idle or health.get('active_answer_tasks') is None:
                raise SystemExit('Backend busy or lacks active-task count; no user task cancelled.')
            print('Waiting for active work to finish...',flush=True); time.sleep(5)
    else: health={}
    run_meta={'manifest':manifest,'selected_ids':[c['id'] for c in selected],
              'started_unix':time.time(),'case_timeout_seconds':1200,'resume_policy':'skip_all_recorded_cases',
              'runtime_health':health,
              'code_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                  ['backend/app.py','backend/sales/task_support.py','backend/sales/requirements.py',
                   'backend/sales/retriever.py','backend/sales/token_usage.py','scripts/run_agent_eval_100.py']}}
    (args.output/'run_manifest.json').write_text(json.dumps(run_meta,ensure_ascii=False,indent=2),encoding='utf-8')
    for case in selected:
        if case['id'] in done: continue
        row=run_case(client,case)
        with records.open('a',encoding='utf-8') as out: out.write(json.dumps(row,ensure_ascii=False)+'\n')
        previous.append(row)
        (args.output/'summary.json').write_text(json.dumps({'status_counts':dict(Counter(r['status'] for r in previous)),
            'recorded_cases':len(previous),'semantic_accuracy':None,
            'note':'Completion/protocol/lexical diagnostics are not semantic correctness.'},ensure_ascii=False,indent=2),encoding='utf-8')
        print(case['id'],row['status'],row['latency_seconds'],flush=True)
        if row.get('cancellation_confirmed') is False: raise SystemExit('Worker exit unconfirmed; stopped before next case.')


if __name__=='__main__': main()
