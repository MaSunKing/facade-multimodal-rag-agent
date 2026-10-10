"""Additive SQL directory. Source context/mentions are not semantic proof."""
from __future__ import annotations
import json
import sqlite3
import re
from collections import Counter, defaultdict

CATEGORIES = {'product_sample': 'product', 'project_case': 'project',
              'node': 'node', 'process': 'process', 'article': 'document_illustration',
              'reference': 'reference'}


def classify_directory(connection):
    previous = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        return _classify(connection)
    finally:
        connection.row_factory = previous


def _classify(c):
    # Do not rebuild existing catalogues or erase curated records.
    for ddl in (
        '''CREATE TABLE IF NOT EXISTS visual_directory (
           asset_id TEXT PRIMARY KEY REFERENCES visual_assets(asset_id),
           category TEXT NOT NULL, text_label TEXT NOT NULL CHECK(length(trim(text_label))>0),
           classification_status TEXT NOT NULL, label_basis TEXT NOT NULL,
           claim_boundary TEXT NOT NULL, source_category TEXT)''',
        '''CREATE TABLE IF NOT EXISTS visual_evidence (
           asset_id TEXT NOT NULL REFERENCES visual_assets(asset_id),
           evidence_id TEXT NOT NULL REFERENCES knowledge_text_evidence(evidence_id),
           relation TEXT NOT NULL, provenance_json TEXT NOT NULL,
           PRIMARY KEY(asset_id,evidence_id,relation))''',
        '''CREATE TABLE IF NOT EXISTS visual_document_links (
           asset_id TEXT NOT NULL REFERENCES visual_assets(asset_id),
           document_key TEXT NOT NULL REFERENCES knowledge_documents(document_key),
           relation TEXT NOT NULL, PRIMARY KEY(asset_id,document_key,relation))''',
        '''CREATE TABLE IF NOT EXISTS product_evidence (
           product_id TEXT NOT NULL REFERENCES products(product_id),
           evidence_id TEXT NOT NULL REFERENCES knowledge_text_evidence(evidence_id),
           relation TEXT NOT NULL, review_status TEXT NOT NULL, provenance_json TEXT NOT NULL,
           PRIMARY KEY(product_id,evidence_id,relation))''',
        '''CREATE TABLE IF NOT EXISTS project_evidence (
           case_id TEXT NOT NULL REFERENCES project_cases(case_id),
           evidence_id TEXT NOT NULL REFERENCES knowledge_text_evidence(evidence_id),
           relation TEXT NOT NULL, review_status TEXT NOT NULL, provenance_json TEXT NOT NULL,
           PRIMARY KEY(case_id,evidence_id,relation))''',
        'CREATE INDEX IF NOT EXISTS visual_directory_category ON visual_directory(category)',
        'CREATE INDEX IF NOT EXISTS visual_evidence_reverse ON visual_evidence(evidence_id)',
    ):
        c.execute(ddl)
    sources = defaultdict(list)
    documents = defaultdict(list)
    identities = defaultdict(set)
    def path_key(value):
        return re.sub(r'[/\\]+','/',str(value or '')).casefold().rstrip('/')
    for d in c.execute('SELECT * FROM knowledge_documents'):
        d = dict(d)
        documents[d['document_name']].append(d['document_key'])
        metadata = json.loads(d.get('metadata_json') or '{}')
        if metadata.get('original_document_id'):
            identities[('id',metadata['original_document_id'])].add(d['document_key'])
        if d.get('source_path'):
            identities[('path',path_key(d['source_path']))].add(d['document_key'])
    for s in c.execute('SELECT * FROM knowledge_evidence_sources'):
        sources[(s['document_key'], s['source_page'])].append(dict(s))
    categories = Counter()
    for row in c.execute('SELECT * FROM visual_assets').fetchall():
        a = dict(row)
        m = json.loads(a['metadata_json'] or '{}')
        catalog = m.get('catalog') or {}
        category = (m.get('human_review') or {}).get('directory_category') or CATEGORIES.get(a['category'], 'unclassified')
        occurrences = m.get('technical_occurrences') or []
        section_categories = {x.get('category') for x in occurrences}
        if category in {'reference','unclassified'} and len(section_categories)==1:
            section_category = next(iter(section_categories))
            if section_category in {'node','process'}:
                category = section_category
        status = 'source_classified' if category != 'unclassified' else 'needs_review'
        title = a['title'] or catalog.get('customer_caption') or m.get('caption')
        if not title or title == a['asset_id']:
            title = f"{a['document_name'] or '来源待确认'}，{('第'+str(a['source_page'])+'页') if a['source_page'] is not None else '位置待确认'}图片"
            status = 'needs_review'
        description = (m.get('search_metadata') or {}).get('visual_description')
        label = str(title).strip()
        if occurrences:
            label += '；来源章节：' + '；'.join(dict.fromkeys(x['title'] for x in occurrences))
        if description and description != title:
            label += '；已有视觉描述：' + str(description).strip()
        c.execute('''INSERT INTO visual_directory VALUES(?,?,?,?,?,?,?)
          ON CONFLICT(asset_id) DO UPDATE SET category=excluded.category,
          text_label=excluded.text_label,classification_status=excluded.classification_status,
          label_basis=excluded.label_basis,claim_boundary=excluded.claim_boundary,
          source_category=excluded.source_category''',
          (a['asset_id'], category, label, status, 'existing_source_metadata_no_new_pixel_review',
           'index_and_visible_description_only_not_material_or_performance_proof', a['category']))
        categories[category] += 1
        # Document names alone must be unique; ambiguous names are not guessed.
        # Same physical source can have several ingestion identities. Prefer
        # stable identity/path, then unambiguous name; never fuzzy-name join.
        keys = identities.get(('path',path_key(m.get('source_pdf') or m.get('source_path') or a.get('source_url'))),set())
        if not keys:
            keys = identities.get(('id',m.get('document_id')),set())
        if not keys and len(documents.get(a['document_name'],[])) == 1:
            keys = documents[a['document_name']]
        for document_key in keys:
            c.execute('INSERT OR IGNORE INTO visual_document_links VALUES(?,?,?)',
                      (a['asset_id'], document_key, 'source_document'))
            if a['source_page'] is not None:
                for s in sources[(document_key, a['source_page'])]:
                    c.execute('INSERT OR IGNORE INTO visual_evidence VALUES(?,?,?,?)',
                              (a['asset_id'], s['evidence_id'], 'same_page_context',
                               json.dumps({'document_key': document_key, 'source_page': a['source_page'],
                                           'source_key': s['source_key'], 'semantic_verified': False})))
            elif str(a.get('source_url') or '').startswith(('https://','http://')):
                for (key, page), source_rows in sources.items():
                    if key != document_key:
                        continue
                    for s in source_rows:
                        c.execute('INSERT OR IGNORE INTO visual_evidence VALUES(?,?,?,?)',
                            (a['asset_id'],s['evidence_id'],'source_article_context',json.dumps({
                             'document_key':document_key,'source_url':a['source_url'],
                             'semantic_verified':False})))
            elif str(a['source_url'] or '').startswith(('https://','http://')):
                for (key, page), source_rows in sources.items():
                    if key != document_key:
                        continue
                    for s in source_rows:
                        c.execute('INSERT OR IGNORE INTO visual_evidence VALUES(?,?,?,?)',
                            (a['asset_id'],s['evidence_id'],'source_article_context',json.dumps({
                             'document_key':document_key,'source_url':a['source_url'],
                             'semantic_verified':False})))
    # Explicit catalogue joins identify the entity of a source image. Transfer
    # only the contextual relationship, never promote co-page text to facts.
    for entity_table, visual_table, key, target in (
        ('products', 'product_visuals', 'product_id', 'product_evidence'),
        ('project_cases', 'project_case_visuals', 'case_id', 'project_evidence')):
        for r in c.execute(f'''SELECT DISTINCT v.{key},e.evidence_id FROM {visual_table} v
                              JOIN visual_evidence e ON e.asset_id=v.asset_id
                              WHERE e.relation='same_page_context' ''').fetchall():
            c.execute(f'INSERT OR IGNORE INTO {target} VALUES(?,?,?,?,?)',
                      (r[key],r['evidence_id'],'linked_image_source_context','context_only',
                       json.dumps({'basis': visual_table+' -> visual_evidence', 'semantic_verified': False})))
        name_col = 'name' if key == 'product_id' else 'project_name'
        # Full-name mentions are navigation candidates, NOT confirmed identity.
        for entity in c.execute(f'SELECT {key},{name_col} FROM {entity_table}').fetchall():
            name = str(entity[name_col] or '').strip()
            if len(name) < 4:
                continue
            for r in c.execute('SELECT evidence_id FROM knowledge_text_evidence WHERE instr(text,?)>0', (name,)).fetchall():
                c.execute(f'INSERT OR IGNORE INTO {target} VALUES(?,?,?,?,?)',
                          (entity[key],r['evidence_id'],'exact_name_mention','needs_review',
                           json.dumps({'basis':'literal_full_name_in_source_text', 'name':name,
                                       'semantic_verified':False}, ensure_ascii=False)))
    from backend.sales.technical_relations import rebuild_technical_relations
    technical = rebuild_technical_relations(c)
    from backend.sales.node_relations import rebuild_node_relations
    node_relations = rebuild_node_relations(c)
    return {'node_relations':node_relations,'technical_relations': technical, 'categories':dict(categories), 'assets':sum(categories.values()),
            'labels_missing':c.execute('SELECT count(*) FROM visual_directory WHERE trim(text_label)=\'\'').fetchone()[0],
            **{t:c.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in
               ('visual_document_links','visual_evidence','product_evidence','project_evidence')},
            'classification_pending':c.execute("SELECT count(*) FROM visual_directory WHERE classification_status='needs_review'").fetchone()[0],
            'products_with_text_candidates':c.execute('SELECT count(DISTINCT product_id) FROM product_evidence').fetchone()[0],
            'projects_with_text_candidates':c.execute('SELECT count(DISTINCT case_id) FROM project_evidence').fetchone()[0],
            'projects_without_text_candidates':[dict(r) for r in c.execute('SELECT case_id,project_name FROM project_cases WHERE case_id NOT IN (SELECT case_id FROM project_evidence)')],
            'policy':'source_derived_classification_no_semantic_certification'}
