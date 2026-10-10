"""Durable, reviewed catalogue choices; original evidence is never rewritten."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT/'data/sales/catalog_repair_policy.json'

def policy():
    return json.loads(POLICY.read_text(encoding='utf8')) if POLICY.is_file() else {}

def filter_documents(documents):
    excluded = set(policy().get('excluded_visual_assets', {}))
    return [d for d in documents if not (d.get('kind')=='visual' and d['id'].removeprefix('visual:') in excluded)]

def apply_index_choices(documents, formatter, tokenizer):
    choices={}
    for decision in policy().get('field_decisions',[]):
        choices.setdefault(decision['case_id'],{})[decision['field']]=decision['value']
    for d in documents:
        if d.get('kind')!='project_case':
            continue
        record=d.get('case') or {}
        cid=record.get('case_id') or d['id']
        if cid not in choices:
            continue
        d.setdefault('source_case_before_catalog_choice',dict(record))
        record.update(choices[cid]); d['case']=record
        d['search_text']=formatter(record);d['tokens']=tokenizer(d['search_text'])

def apply_catalog_choices(c):
    config=policy()
    tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    c.execute('CREATE TABLE IF NOT EXISTS catalog_field_decisions(case_id TEXT,field TEXT,selected_value TEXT,alternatives_json TEXT,reason TEXT,PRIMARY KEY(case_id,field))')
    c.execute('CREATE TABLE IF NOT EXISTS visual_retrieval_review(asset_id TEXT PRIMARY KEY,status TEXT,reason TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS relation_definitions(relation TEXT PRIMARY KEY,meaning TEXT,does_not_imply TEXT)')
    for relation,meaning in config.get('relation_definitions',{}).items():
        c.execute('INSERT OR REPLACE INTO relation_definitions VALUES(?,?,?)',(relation,meaning,'不自动证明性能、合规、工程部位对应或来源原文已确认'))
    columns={r[1] for r in c.execute('PRAGMA table_info(project_cases)')}
    for decision in config.get('field_decisions',[]):
        field=decision['field']
        if field not in {'product_name','project_category','area_m2','completion_year'}:
            raise ValueError('unsupported_catalog_decision_field:'+field)
        cid=decision['case_id']
        if field not in columns or not c.execute('SELECT 1 FROM project_cases WHERE case_id=?',(cid,)).fetchone():
            continue
        c.execute(f'UPDATE project_cases SET {field}=? WHERE case_id=?',(decision['value'],cid))
        c.execute('INSERT OR REPLACE INTO catalog_field_decisions VALUES(?,?,?,?,?)',
            (cid,field,str(decision['value']),json.dumps(decision['alternatives'],ensure_ascii=False),decision['reason']))
        if 'case_conflicts' in tables:
            c.execute('DELETE FROM case_conflicts WHERE case_id=? AND field=?',(cid,field))
            if field=='product_name': c.execute("DELETE FROM case_conflicts WHERE case_id=? AND field='product'",(cid,))
    from backend.sales.project_identity import merge_projects, choices
    merge_projects(c, choices()['merges'])
    c.execute("UPDATE catalog_field_decisions SET alternatives_json='[]'")
    for aid,reason in config.get('excluded_visual_assets',{}).items():
        c.execute('INSERT OR REPLACE INTO visual_retrieval_review VALUES(?,?,?)',(aid,'excluded',reason))
        row=c.execute('SELECT metadata_json FROM visual_assets WHERE asset_id=?',(aid,)).fetchone() if 'visual_assets' in tables else None
        if row:
            metadata=json.loads(row[0]); metadata.update(retrieval_eligible=False,retrieval_exclusion_reason=reason)
            c.execute('UPDATE visual_assets SET metadata_json=? WHERE asset_id=?',(json.dumps(metadata,ensure_ascii=False),aid))
    if 'case_search' in tables:
        c.execute('DELETE FROM case_search')
        c.execute('INSERT INTO case_search SELECT case_id,project_name,province,city,project_category,product_name,source_text FROM project_cases')
    if {'product_evidence','knowledge_text_evidence','products'} <= tables:
        for color in config.get('color_evidence_links',[]):
            if c.execute('SELECT 1 FROM knowledge_text_evidence WHERE evidence_id=?',(color['evidence_id'],)).fetchone() and c.execute('SELECT 1 FROM products WHERE product_id=?',(color['product_id'],)).fetchone():
                c.execute('INSERT OR REPLACE INTO product_evidence VALUES(?,?,?,?,?)',
                    (color['product_id'],color['evidence_id'],'catalog_color_definition','reviewed_source',json.dumps(color['provenance'],ensure_ascii=False)))
