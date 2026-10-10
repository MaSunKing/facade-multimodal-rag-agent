"""Reviewed identity choices, never runtime fuzzy merging or fact inference."""
import json
from pathlib import Path

CHOICES = Path(__file__).resolve().parents[2] / 'data/sales/project_identity_choices.json'

def choices():
    return json.loads(CHOICES.read_text(encoding='utf8')) if CHOICES.is_file() else {'merges':[]}

def merge_projects(connection, merges):
    """Move ACL-compatible edges, keep selected fields; caller owns transaction."""
    connection.execute('CREATE TABLE IF NOT EXISTS project_name_aliases '
                       '(case_id TEXT NOT NULL REFERENCES project_cases(case_id),name TEXT NOT NULL,'
                       'PRIMARY KEY(case_id,name))')
    tables = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
              if not r[0].startswith(('sqlite_', 'case_search'))]
    count = 0
    for old, canonical in merges:
        source = connection.execute('SELECT project_name,access_scope FROM project_cases WHERE case_id=?',(old,)).fetchone()
        target = connection.execute('SELECT project_name,access_scope FROM project_cases WHERE case_id=?',(canonical,)).fetchone()
        if not source:
            continue
        if not target or source[1] != target[1] or old == canonical:
            raise ValueError('invalid_project_identity_merge')
        connection.execute('INSERT OR IGNORE INTO project_name_aliases VALUES(?,?)',(canonical,source[0]))
        for table in tables:
            if table == 'project_cases': continue
            info = list(connection.execute(f'PRAGMA table_info("{table}")'))
            fields = [r[1] for r in info]
            if 'case_id' not in fields: continue
            if not any(r[1] == 'case_id' and r[5] for r in info):
                connection.execute(f'UPDATE "{table}" SET case_id=? WHERE case_id=?',(canonical,old))
                continue
            names = ','.join('"'+f+'"' for f in fields)
            projection = ','.join('?' if f == 'case_id' else '"'+f+'"' for f in fields)
            connection.execute(f'INSERT OR IGNORE INTO "{table}" ({names}) SELECT {projection} '
                               f'FROM "{table}" WHERE case_id=?',(canonical,old))
            connection.execute(f'DELETE FROM "{table}" WHERE case_id=?',(old,))
        connection.execute('DELETE FROM project_cases WHERE case_id=?',(old,))
        count += 1
    if 'project_case_visuals' in tables:
        connection.execute('UPDATE project_cases SET image_count=(SELECT COUNT(*) FROM project_case_visuals v '
                           'WHERE v.case_id=project_cases.case_id)')
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='case_search'").fetchone():
        connection.execute('DELETE FROM case_search')
        connection.execute('INSERT INTO case_search SELECT case_id,project_name,province,city,project_category,'
                           'product_name,source_text FROM project_cases')
    return count

def canonical_index_documents(documents):
    """Drop retired project rows without touching dense-index coordinates."""
    from backend.sales.catalog_store import CATALOG_DB_PATH, _connect
    if not CATALOG_DB_PATH.is_file(): return documents
    with _connect(CATALOG_DB_PATH) as db:
        records = {r['case_id']:dict(r) for r in db.execute('SELECT * FROM project_cases')}
    retired = {old for old, canonical in choices()['merges'] if canonical in records and old not in records}
    result = []
    for row in documents:
        if row.get('kind') != 'project_case':
            result.append(row); continue
        cid = (row.get('case') or {}).get('case_id') or row.get('id')
        if cid in retired: continue
        if cid in records:
            row = dict(row)
            row['case'] = records[cid]
            row.pop('source_case_before_catalog_choice', None)
            row['search_text'] = ' '.join(str(v) for v in records[cid].values() if v is not None)
            from backend.sales.retriever import tokenize
            row['tokens'] = tokenize(row['search_text'])
        result.append(row)
    return result
