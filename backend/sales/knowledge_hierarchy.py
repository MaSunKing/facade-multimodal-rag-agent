"""Regenerable SQL navigation over the existing company index, no new facts."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from backend.documents.hierarchy import build_hierarchy, stable_id


def rebuild_hierarchy(index_path: Path, database_path: Path) -> dict:
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    groups = {}
    for item in payload['documents']:
        # Use every original source reference, including merged provenance.
        refs = item.get('source_refs') or []
        if not refs and item.get('kind') == 'text':
            refs = [{'document_id': item.get('document_id') or item['id'],
                     'document_name': item.get('document_name') or item.get('title') or item['id']}]
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            source_id = ref.get('document_id')
            if not source_id:
                continue
            scope = str(item.get('access_scope') or item.get('visibility') or 'public')
            key = (str(source_id), scope)
            group = groups.setdefault(key, {'name': ref.get('document_name') or str(source_id), 'chunks': []})
            group['chunks'].append({'id': item['id'], 'source_refs': [ref], 'kind': item.get('kind')})
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.executescript('''
            CREATE TABLE IF NOT EXISTS knowledge_document_nodes (
              node_id TEXT PRIMARY KEY, source_document_id TEXT NOT NULL, document_name TEXT NOT NULL,
              source_origin TEXT NOT NULL, access_scope TEXT NOT NULL, index_sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS knowledge_section_nodes (
              section_id TEXT PRIMARY KEY, document_node_id TEXT NOT NULL, label TEXT NOT NULL,
              kind TEXT NOT NULL, source_refs_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS knowledge_node_evidence (
              section_id TEXT NOT NULL, evidence_id TEXT NOT NULL, PRIMARY KEY(section_id,evidence_id));
            CREATE INDEX IF NOT EXISTS knowledge_sections_parent ON knowledge_section_nodes(document_node_id);
            CREATE INDEX IF NOT EXISTS knowledge_evidence_lookup ON knowledge_node_evidence(evidence_id);
        ''')
        connection.execute('DELETE FROM knowledge_node_evidence')
        connection.execute('DELETE FROM knowledge_section_nodes')
        connection.execute('DELETE FROM knowledge_document_nodes')
        digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
        sections = links = 0
        for (document_id, scope), group in groups.items():
            node_id = stable_id(['company', document_id, scope])
            hierarchy = build_hierarchy(node_id, group['name'], group['chunks'], source_origin='company')
            connection.execute('INSERT INTO knowledge_document_nodes VALUES(?,?,?,?,?,?)',
                (node_id, document_id, group['name'], 'company', scope, digest))
            for section in hierarchy['sections']:
                connection.execute('INSERT INTO knowledge_section_nodes VALUES(?,?,?,?,?)',
                    (section['section_id'], node_id, section['label'], section['kind'],
                     json.dumps(section['source_refs'], ensure_ascii=False)))
                for key in dict.fromkeys(section['evidence_ids']):
                    connection.execute('INSERT INTO knowledge_node_evidence VALUES(?,?)', (section['section_id'], key))
                    links += 1
                sections += 1
        # Reviewed technical sections are canonical navigation nodes too. The
        # technical table remains an annotation/relationship extension, not a
        # second independent tree. Preserve its stable IDs for existing links.
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='technical_sections'").fetchone():
            indexed = {d['id']: d for d in payload['documents']}
            for sid, sha, title, category, pages, evidence, assets, provenance, scope in connection.execute(
                    'SELECT * FROM technical_sections').fetchall():
                ids = [eid for eid in json.loads(evidence) if eid in indexed]
                if not ids:
                    continue
                ref = next((r for eid in ids for r in indexed[eid].get('source_refs', [])
                            if r.get('document_id')), {})
                document_id = str(ref.get('document_id') or 'source:'+sha)
                node_id = stable_id(['company', document_id, scope])
                connection.execute('INSERT OR IGNORE INTO knowledge_document_nodes VALUES(?,?,?,?,?,?)',
                    (node_id,document_id,ref.get('document_name') or document_id,'company',scope,digest))
                for eid in ids:
                    connection.execute('DELETE FROM knowledge_node_evidence WHERE evidence_id=?', (eid,))
                connection.execute('INSERT OR REPLACE INTO knowledge_section_nodes VALUES(?,?,?,?,?)',
                    (sid,node_id,title,category,json.dumps({'pages':json.loads(pages),
                     'source_sha256':sha,'source_refs':ref},ensure_ascii=False)))
                connection.executemany('INSERT OR IGNORE INTO knowledge_node_evidence VALUES(?,?)',[(sid,eid) for eid in ids])
            connection.execute('DELETE FROM knowledge_section_nodes WHERE section_id NOT IN (SELECT section_id FROM knowledge_node_evidence)')
        sections = connection.execute('SELECT count(*) FROM knowledge_section_nodes').fetchone()[0]
        links = connection.execute('SELECT count(*) FROM knowledge_node_evidence').fetchone()[0]
        if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('hierarchy_database_integrity_failed')
    return {'documents': len(groups), 'sections': sections, 'evidence_links': links,
            'source_index_sha256': digest, 'role': 'navigation_not_generated_summary'}


def document_sections(database_path, document_id, *, allowed_access_scopes=('public',)):
    scopes = list(allowed_access_scopes)
    if not scopes or not Path(database_path).exists():
        return []
    with closing(sqlite3.connect(f'file:{Path(database_path).resolve().as_posix()}?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='knowledge_document_nodes'").fetchone():
            return []
        rows = connection.execute('SELECT s.* FROM knowledge_section_nodes s '
            'JOIN knowledge_document_nodes d ON d.node_id=s.document_node_id '
            'WHERE d.source_document_id=? AND d.access_scope IN (' + ','.join('?' for _ in scopes) + ') '
            'ORDER BY s.rowid', [document_id, *scopes]).fetchall()
        return [{**dict(row), 'evidence_ids': [r[0] for r in connection.execute(
            'SELECT evidence_id FROM knowledge_node_evidence WHERE section_id=?', (row['section_id'],))]} for row in rows]
