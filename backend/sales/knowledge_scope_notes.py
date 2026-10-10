"""Reviewed scope supplements over existing evidence, never raw-source rewrites.

Stable source IDs are catalogue data, not question routing. The same enrichment
is used for every company retrieval operation; customer/Web evidence is untouched.
"""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT/'data/sales/config/knowledge_scope_notes.json'
DATABASE_PATH = ROOT/'data/sales/processed/catalog/sales_catalog.sqlite3'


def rebuild_scope_notes(c, policy=None):
    if not c.execute("SELECT 1 FROM sqlite_master WHERE name='knowledge_text_evidence'").fetchone():
        return {'notes': 0, 'targets': 0, 'skipped': 0}
    policy = policy if policy is not None else json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    for sql in (
        'CREATE TABLE IF NOT EXISTS knowledge_scope_notes(note_id TEXT PRIMARY KEY,note_text TEXT NOT NULL,basis TEXT NOT NULL)',
        '''CREATE TABLE IF NOT EXISTS knowledge_scope_note_targets(note_id TEXT REFERENCES knowledge_scope_notes,
           evidence_id TEXT REFERENCES knowledge_text_evidence,PRIMARY KEY(note_id,evidence_id))''',
        '''CREATE TABLE IF NOT EXISTS knowledge_scope_note_supports(note_id TEXT REFERENCES knowledge_scope_notes,
           evidence_id TEXT REFERENCES knowledge_text_evidence,PRIMARY KEY(note_id,evidence_id))'''):
        c.execute(sql)
    c.execute('DELETE FROM knowledge_scope_note_targets')
    c.execute('DELETE FROM knowledge_scope_note_supports')
    c.execute('DELETE FROM knowledge_scope_notes')
    known = {r[0] for r in c.execute('SELECT evidence_id FROM knowledge_text_evidence')}
    notes = policy.get('notes', [])
    seen = set(); count = target_count = skipped = 0
    for note in notes:
        nid, text = note.get('id'), note.get('text')
        targets, supports = note.get('targets'), note.get('supports')
        if (not isinstance(nid, str) or not nid or nid in seen or not isinstance(text, str)
            or not text.strip() or any(not isinstance(v, list) or not v or
              any(not isinstance(e, str) or not e for e in v) for v in (targets, supports))):
            raise ValueError('invalid_scope_note')
        seen.add(nid)
        if not set(supports) <= known:
            skipped += 1; continue
        targets = sorted(set(targets) & known)
        if not targets:
            skipped += 1; continue
        c.execute('INSERT INTO knowledge_scope_notes VALUES(?,?,?)', (nid,text.strip(),policy['basis']))
        c.executemany('INSERT INTO knowledge_scope_note_targets VALUES(?,?)', [(nid,e) for e in targets])
        c.executemany('INSERT INTO knowledge_scope_note_supports VALUES(?,?)', [(nid,e) for e in sorted(set(supports))])
        count += 1; target_count += len(targets)
    return {'notes': count, 'targets': target_count, 'skipped': skipped}


def _keys(refs):
    result = set()
    for ref in refs:
        if ref.get('source_url'):
            result.add(('url',ref['source_url']))
        elif ref.get('document_id') and ref.get('source_page') is not None:
            result.add(('page',ref['document_id'],str(ref['source_page'])))
    return result


def enrich_scope_notes(rows, database_path=DATABASE_PATH):
    """Copy evidence views, attach only public anchored notes and public support refs.

    A missing legacy table is compatible. Permission filtering is source-based,
    not a model decision; original text and index fingerprints remain unchanged.
    """
    path = Path(database_path)
    if not rows or not path.is_file():
        return rows
    with closing(sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro', uri=True)) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'knowledge_scope_notes','knowledge_scope_note_targets','knowledge_scope_note_supports',
                'knowledge_text_evidence','knowledge_evidence_sources'} <= tables:
            return rows
        public = {r[0] for r in c.execute("SELECT evidence_id FROM knowledge_text_evidence WHERE access_scope='public'")}
        refs = {}
        for eid, raw in c.execute('SELECT evidence_id,reference_json FROM knowledge_evidence_sources'):
            if eid in public:
                refs.setdefault(eid, []).append(json.loads(raw))
        notes = []
        for nid, text, basis in c.execute('SELECT note_id,note_text,basis FROM knowledge_scope_notes ORDER BY note_id'):
            targets = {r[0] for r in c.execute('SELECT evidence_id FROM knowledge_scope_note_targets WHERE note_id=?',(nid,))}
            supports = {r[0] for r in c.execute('SELECT evidence_id FROM knowledge_scope_note_supports WHERE note_id=?',(nid,))}
            # Do not expose notes inferred from inaccessible targets or supports.
            if not supports <= public or not targets:
                continue
            targets &= public
            source_keys = _keys([ref for eid in targets for ref in refs.get(eid, [])])
            support_refs = [ref for eid in sorted(supports) for ref in refs.get(eid, [])]
            notes.append((nid,text,basis,targets,source_keys,support_refs))
    output = []
    for row in rows:
        if row.get('access_scope','public') != 'public':
            output.append(row); continue
        existing = set(row.get('scope_note_ids', []))
        row_refs = list(row.get('citations') or row.get('source_refs') or [])
        keys = _keys(row_refs)
        matched = [n for n in notes if n[0] not in existing and
                   (str(row.get('chunk_id') or row.get('id') or '') in n[3] or keys & n[4])]
        if not matched:
            output.append(row); continue
        result = dict(row)
        result['text'] = str(row.get('text') or '') + '\n' + '\n'.join('已核对来源范围补充：'+n[1] for n in matched)
        result['scope_note_ids'] = sorted(existing | {n[0] for n in matched})
        merged = [*row_refs, *[ref for n in matched for ref in n[5]]]
        result['citations'] = list({json.dumps(r,ensure_ascii=False,sort_keys=True):r for r in merged}.values())
        output.append(result)
    return output
