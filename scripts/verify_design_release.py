"""Static documentation/data checks; never claim formal model evaluation."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def main():
    missing=[]
    for file in [ROOT/'README.md', *sorted((ROOT/'docs').rglob('*.md'))]:
        text=file.read_text(encoding='utf8')
        # Code snapshots may contain Markdown strings: only check authored docs.
        if 'prompts' in file.relative_to(ROOT).parts:
            continue
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if link.startswith(('http:','https:','#','mailto:')): continue
            path=link.split('#')[0]
            if path and not (file.parent/path).exists():
                missing.append({'file':str(file.relative_to(ROOT)),'link':path})
    if missing: raise ValueError('broken_links:'+json.dumps(missing))
    source_files=list((ROOT/'backend').rglob('*.py'))
    for file in source_files: ast.parse(file.read_text(encoding='utf8'))
    unresolved=[]
    for file in (ROOT/'backend').rglob('*.py'):
        if 'tests' in file.relative_to(ROOT).parts: continue
        tree=ast.parse(file.read_text(encoding='utf8'))
        for n in ast.walk(tree):
            name=n.module if isinstance(n,ast.ImportFrom) else None
            if name and name.startswith('backend.'):
                target=ROOT/Path(*name.split('.'))
                if not target.with_suffix('.py').exists() and not (target/'__init__.py').exists():
                    unresolved.append({'file':str(file.relative_to(ROOT)),'module':name})
    if unresolved: raise ValueError('missing_modules:'+json.dumps(unresolved))
    data=ROOT/'data/sales/evaluation/agent_eval_100_v1'
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    raw=(data/'questions.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['question_sha256']
    cases=[json.loads(x) for x in raw.decode().splitlines() if x]
    assert len(cases)==100 and len({c['id'] for c in cases})==100
    assert Counter(c['group'] for c in cases)=={'company':40,'attachment':20,'multiturn':20,'web':15,'fault':5}
    for case in cases:
        if case.get('test_module'):
            assert (ROOT/Path(*case['test_module'].split('.'))).with_suffix('.py').exists()
    for name,digest in manifest['fixture_sha256'].items():
        assert hashlib.sha256((data/'fixtures'/name).read_bytes()).hexdigest()==digest
    for file in (ROOT/'examples/traces').glob('*.json'):
        value=json.loads(file.read_text(encoding='utf8'))
        assert value['kind']=='illustrative_not_executed'
        for n in value.get('nodes',[]):
            assert all(n[k] is None for k in ('input_tokens','output_tokens','elapsed_ms','model_calls'))
    print(json.dumps({'links_valid':True,'python_syntax_valid':True,'frozen_questions':100,
                     'illustrative_branches':len(list((ROOT/'examples/traces').glob('*.json')))-1,
                     'formal_model_evaluation':'pending'},ensure_ascii=False))

if __name__=='__main__':main()
