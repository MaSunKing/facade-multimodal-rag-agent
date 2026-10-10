"""Source-backed section navigation with public ACL and reference checks."""
import json

def canonical_evidence(c, sid, fallback):
    if c.execute("SELECT 1 FROM sqlite_master WHERE name='knowledge_node_evidence'").fetchone():
        return [r[0] for r in c.execute('SELECT evidence_id FROM knowledge_node_evidence WHERE section_id=?',(sid,))]
    return json.loads(fallback)

def related_sections(c, families):
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {'technical_sections','family_section_links'} <= tables:
        return []
    allowed_text = {r[0] for r in c.execute("SELECT evidence_id FROM knowledge_text_evidence WHERE access_scope='public'")}
    allowed_visual = {r[0] for r in c.execute("SELECT asset_id FROM visual_assets WHERE access_scope='public'")} if 'visual_assets' in tables else set()
    result = []
    for family in sorted(families):
        for row in c.execute('''SELECT s.section_id,s.title,s.evidence_ids_json,s.asset_ids_json,s.provenance_json,l.relation
          FROM family_section_links l JOIN technical_sections s USING(section_id)
          WHERE l.family=? AND s.access_scope='public' ''',(family,)):
            sid,title,eids,aids,proof,relation = row
            evidence = sorted(set(canonical_evidence(c,sid,eids)) & allowed_text)
            if not evidence: continue
            result.append(dict(section_id=sid,title=title,evidence_ids=evidence,
                asset_ids=sorted(set(json.loads(aids)) & allowed_visual),provenance=json.loads(proof),relation=relation,family=family))
    return result

def related_project_sections(c, case_ids):
    if not c.execute("SELECT 1 FROM sqlite_master WHERE name='project_section_links'").fetchone(): return []
    public = {r[0] for r in c.execute("SELECT evidence_id FROM knowledge_text_evidence WHERE access_scope='public'")}
    visuals = {r[0] for r in c.execute("SELECT asset_id FROM visual_assets WHERE access_scope='public'")}
    result = []
    for cid in case_ids:
        for row in c.execute('''SELECT s.section_id,s.evidence_ids_json,s.asset_ids_json,l.provenance_json
          FROM project_section_links l JOIN technical_sections s USING(section_id)
          WHERE l.case_id=? AND s.access_scope='public' ''',(cid,)):
            sid,eids,aids,proof = row
            evidence = sorted(set(canonical_evidence(c,sid,eids)) & public)
            if evidence: result.append(dict(section_id=sid,evidence_ids=evidence,
                asset_ids=sorted(set(json.loads(aids)) & visuals),provenance=json.loads(proof),case_id=cid))
    return result

def restore_section_edges(c):
    """Reattach canonical IDs after the directory synchronizer's refresh."""
    if not c.execute("SELECT 1 FROM sqlite_master WHERE name='technical_sections'").fetchone(): return
    texts = {r[0] for r in c.execute('SELECT evidence_id FROM knowledge_text_evidence')}
    assets = {r[0] for r in c.execute('SELECT asset_id FROM visual_assets')}
    for sid,eids,aids,proof in c.execute('SELECT section_id,evidence_ids_json,asset_ids_json,provenance_json FROM technical_sections').fetchall():
        for aid in set(json.loads(aids)) & assets:
            for eid in set(json.loads(eids)) & texts:
                c.execute('INSERT OR IGNORE INTO visual_evidence VALUES(?,?,?,?)',
                          (aid,eid,'reviewed_section_context',proof))
