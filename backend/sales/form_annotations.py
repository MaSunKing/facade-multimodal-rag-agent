"""Provenance-bound retrieval annotations; original source/vector text is retained."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / 'data/sales/processed/catalog/sales_catalog.sqlite3'
FAMILY_FORMS = {'真岩®无机仿石装饰板': '单板', '真岩®保温装饰一体板': '保温一体板'}


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def annotate_text(text, form):
    # Longest compound names first; never annotate a decorative layer inside
    # an insulated assembly as if the complete assembly were a single panel.
    if form != '单板' or re.search(r'保温|一体板', text):
        return text
    pattern = r'(人造无机石材装饰板(?:[（(]真岩石[）)])?|真岩®?无机仿石装饰板)(?![（(]单板[）)])'
    return re.sub(pattern, r'\1（单板）', text)


def build_annotations(documents, sections):
    """Use only unambiguous reviewed section membership / same-source record ID."""
    memberships, anchors, ambiguous_ids = {}, [], set()
    for section in sections:
        forms = {FAMILY_FORMS[f] for f in section['families'] if f in FAMILY_FORMS}
        if len(forms) != 1:
            if len(forms)>1: ambiguous_ids.update(section['evidence_ids'])
            continue
        form = next(iter(forms))
        # Cross-category navigation alone is not product-form proof.
        if form == '单板' and '单板' not in section['title']:
            continue
        if form == '保温一体板' and not re.search(r'保温|一体板', section['title']):
            continue
        for eid in section['evidence_ids']:
            memberships.setdefault(eid, []).append((form, section['section_id']))
        codes = re.findall(r'\b[A-Z][A-Z0-9-]{7,}\b', section['title'])
        for code in codes:
            anchors.append((code, form, section['section_id'], section['source_sha256'][:16]))
    rows = []
    for doc in documents:
        if doc.get('kind') != 'text':
            continue
        text = doc.get('text', '')
        if doc['id'] in ambiguous_ids: continue
        provenance = list(memberships.get(doc['id'], []))
        # Explicit catalogue declarations are registrations, not guesses based
        # on thickness, colour, installation method or a generic material name.
        declared = {form for family,form in FAMILY_FORMS.items()
                    if '产品体系：'+family in text or '产品体系: '+family in text}
        if len(declared)==1 and not re.search(r'保温|一体板',text):
            provenance.extend((form,'explicit_product_family_declaration') for form in declared)
        source_ids = {str(r.get('document_id', '')) for r in doc.get('source_refs', [])}
        for code, form, section_id, sha in anchors:
            if re.search(r'(?<![A-Z0-9])'+re.escape(code)+r'(?![A-Z0-9])', text) and any(sha in sid for sid in source_ids):
                provenance.append((form, section_id))
        forms = {form for form, _ in provenance}
        if len(forms) != 1:
            continue
        form = next(iter(forms))
        annotated = annotate_text(text, form)
        if annotated == text and form == '单板' and '单板' not in text:
            annotated += '\n板型标注：单板（依据已核对的来源关系；不扩展具体型号或试验适用范围）。'
        rows.append({'evidence_id':doc['id'], 'source_text_hash':text_hash(text), 'product_form':form,
                     'annotated_text':annotated, 'section_ids':sorted({s for _, s in provenance})})
    return rows


def apply_annotations(documents, tokenize, database_path=DB):
    path = Path(database_path)
    if not path.is_file(): return documents
    with sqlite3.connect('file:'+path.as_posix()+'?mode=ro', uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='evidence_form_annotations'").fetchone():
            return documents
        rows = {eid:(digest,form,text,prov) for eid,digest,form,text,prov in db.execute(
            'SELECT evidence_id,source_text_hash,product_form,annotated_text,provenance_json FROM evidence_form_annotations')}
    result = []
    for doc in documents:
        row = rows.get(doc.get('id'))
        if not row or text_hash(doc.get('text','')) != row[0]:
            result.append(doc); continue
        original = doc.get('text','')
        result.append({**doc, 'text':row[2], 'source_text':original,
                       'product_form':row[1], 'form_annotation_provenance':json.loads(row[3]),
                       'tokens':tokenize(row[2]+' '+row[1])})
    return result
