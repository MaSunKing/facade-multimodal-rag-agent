"""Source-backed navigation, never a certificate of engineering applicability."""
from __future__ import annotations
import json
import re
from pathlib import Path

POLICY = Path(__file__).resolve().parents[2] / 'data/sales/technical_relation_policy.json'
STANDARD = re.compile(r'(?<![A-Za-z0-9])(?:GB|JGJ|JG|DB\d{2}|JC)\s*/?\s*T?\s*\d{2,6}\s*[-－—]\s*\d{4}(?!\d)', re.I)


def codes(text):
    return {re.sub(r'[\s/]', '', x.upper()).replace('－', '-').replace('—', '-')
            for x in STANDARD.findall(text)}


def rebuild_technical_relations(c, policy=None):
    if not c.execute("SELECT 1 FROM sqlite_master WHERE name='product_families'").fetchone():
        return {}
    policy = json.loads(POLICY.read_text(encoding='utf-8')) if policy is None else policy
    c.execute('''CREATE TABLE IF NOT EXISTS family_document_links (
      family TEXT NOT NULL REFERENCES product_families(family),
      document_key TEXT NOT NULL REFERENCES knowledge_documents(document_key),
      relation TEXT NOT NULL, provenance_json TEXT NOT NULL, generator TEXT NOT NULL,
      PRIMARY KEY(family,document_key,relation))''')
    c.execute('''CREATE TABLE IF NOT EXISTS document_reference_links (
      from_document TEXT NOT NULL REFERENCES knowledge_documents(document_key),
      to_document TEXT NOT NULL REFERENCES knowledge_documents(document_key),
      evidence_id TEXT NOT NULL REFERENCES knowledge_text_evidence(evidence_id),
      reference_code TEXT NOT NULL, provenance_json TEXT NOT NULL, generator TEXT NOT NULL,
      PRIMARY KEY(from_document,to_document,evidence_id,reference_code))''')
    generator = 'source_technical_navigation_v1'
    for table in ('family_document_links', 'document_reference_links'):
        c.execute(f'DELETE FROM {table} WHERE generator=?', (generator,))
    documents = c.execute('SELECT document_key,document_name FROM knowledge_documents').fetchall()
    families = {r[0] for r in c.execute('SELECT family FROM product_families')}
    standards = {}
    declared_codes = {}
    source_rows = c.execute('''SELECT s.document_key,s.evidence_id,e.text,e.metadata_json,
      s.reference_json FROM knowledge_evidence_sources s
      JOIN knowledge_text_evidence e USING(evidence_id)''').fetchall()
    for key, name in documents:
        for rule in policy.get('links', []):
            if name != rule['document_name'] or rule['family'] not in families:
                continue
            # Require literal proof in this document, not merely a matching filename.
            proofs = [r for r in source_rows if r[0] == key and rule['scope_quote'] in r[2]]
            if not proofs:
                continue
            proof = proofs[0]
            c.execute('INSERT OR IGNORE INTO family_document_links VALUES(?,?,?,?,?)',
              (rule['family'], key, rule['relation'], json.dumps({
                'scope_quote': rule['scope_quote'], 'evidence_id': proof[1],
                'source': json.loads(proof[4]), 'boundary': rule['boundary'],
                'engineering_applicability_verified': False}, ensure_ascii=False), generator))
        # Identify standards from their title / explicit metadata heading only.
        for code in codes(name or ''):
            standards.setdefault(code, set()).add(key)
            declared_codes.setdefault(key, set()).add(code)
    for key, eid, text, metadata, ref in source_rows:
        if '01_standard_specification' in json.loads(metadata).get('knowledge_domains', []):
            for code in codes((text[:300].split('适用边界')[0]) if text.startswith('资料名称：') else ''):
                standards.setdefault(code, set()).add(key)
                declared_codes.setdefault(key, set()).add(code)
    for key, eid, text, metadata, ref in source_rows:
        for code in codes(text):
            if code in declared_codes.get(key, set()):
                continue  # Another copy of the same standard is not a citation.
            for target in standards.get(code, ()):
                if key == target:
                    continue
                c.execute('INSERT OR IGNORE INTO document_reference_links VALUES(?,?,?,?,?,?)',
                  (key, target, eid, code, json.dumps({'source': json.loads(ref),
                    'boundary': 'literal_reference_not_compliance_or_current_applicability'},
                    ensure_ascii=False), generator))
    return {t: c.execute('SELECT count(*) FROM '+t).fetchone()[0]
            for t in ('family_document_links','document_reference_links')}


def related_technical_documents(c, product_id, allowed_access_scopes=('public',)):
    """Family navigation with ACL filtering; callers must still select evidence."""
    product = c.execute('SELECT family,access_scope FROM products WHERE product_id=?', (product_id,)).fetchone()
    if not product or product[1] not in allowed_access_scopes:
        return []
    result = []
    for key, name, relation, provenance in c.execute('''SELECT l.document_key,d.document_name,
      l.relation,l.provenance_json FROM family_document_links l
      JOIN knowledge_documents d USING(document_key) WHERE l.family=?''', (product[0],)):
        evidence = [eid for eid, scope in c.execute('''SELECT s.evidence_id,e.access_scope
          FROM knowledge_evidence_sources s JOIN knowledge_text_evidence e USING(evidence_id)
          WHERE s.document_key=?''', (key,)) if scope in allowed_access_scopes]
        if evidence:
            result.append({'document_key': key, 'document_name': name, 'relation': relation,
                           'evidence_ids': sorted(set(evidence)), 'provenance': json.loads(provenance)})
    return result
