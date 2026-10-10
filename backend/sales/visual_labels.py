"""Source-derived image labels; no pixel inference or automatic certification."""
from __future__ import annotations
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL_PATH = ROOT / 'data/sales/processed/catalog/visual_labels.json'
DATABASE_PATH = ROOT / 'data/sales/processed/catalog/sales_catalog.sqlite3'
FAMILY_KEYS = {'真岩®无机仿石装饰板':'decorative_panel', '真岩®保温装饰一体板':'insulated_panel'}


def company_label_view(labels):
    """Compact source fields, not guessed pixel observations or performance."""
    fields=('title','category','visual_role','product_name','product_family','product_form',
            'color_name','variant_or_code','surface_finish','insulation_type','project_names',
            'cities','provinces','caption','section_heading','node_system_references')
    result={k:labels[k] for k in fields if labels.get(k) not in (None,'',[],{},'unknown')}
    # Deduplicate repeated captions/titles without discarding distinct labels.
    if result.get('caption') in {result.get('title'),result.get('product_name')}:
        result.pop('caption',None)
    if isinstance(result.get('node_system_references'),list):
        result['node_system_references']=result['node_system_references'][:6]
    result['label_basis']=labels.get('label_basis','existing_source_metadata_and_explicit_catalogue_joins')
    return result


@lru_cache(maxsize=2)
def _read_labels(path, mtime):
    return json.loads(Path(path).read_text(encoding='utf8')).get('assets', {})


@lru_cache(maxsize=2)
def _read_sql_labels(path, mtime, wal_mtime):
    with sqlite3.connect('file:'+Path(path).as_posix()+'?mode=ro',uri=True) as connection:
        exists=connection.execute("SELECT 1 FROM sqlite_master WHERE name='visual_asset_labels'").fetchone()
        if not exists:
            return None
        return {key:json.loads(value) for key,value in connection.execute('SELECT asset_id,labels_json FROM visual_asset_labels')}


def labels_for_asset(asset_id):
    if DATABASE_PATH.is_file():
        wal=Path(str(DATABASE_PATH)+'-wal')
        labels=_read_sql_labels(str(DATABASE_PATH),DATABASE_PATH.stat().st_mtime_ns,
                                wal.stat().st_mtime_ns if wal.exists() else 0)
        if labels is not None:
            return labels.get(str(asset_id),{})
    if not LABEL_PATH.is_file():
        return {}
    return _read_labels(str(LABEL_PATH), LABEL_PATH.stat().st_mtime_ns).get(str(asset_id), {})


def build_labels(connection):
    """Use catalogue identity and explicit SQL joins, not nearby unrelated text."""
    previous_row_factory = connection.row_factory
    connection.row_factory = sqlite3.Row
    output = {}
    from backend.sales.catalog_structure import read_product_attributes
    product_attributes={r['product_id']:r for r in read_product_attributes(connection,
        [r[0] for r in connection.execute("SELECT product_id FROM products WHERE access_scope='public'")])}
    for row in connection.execute('SELECT * FROM visual_assets'):
        row = dict(row)
        metadata = json.loads(row['metadata_json'])
        catalog = metadata.get('catalog') or {}
        products = [dict(x) for x in connection.execute(
            'SELECT p.product_id,p.name AS product_name,p.family,p.surface_finish,p.insulation_type FROM product_visuals v JOIN products p ON p.product_id=v.product_id WHERE v.asset_id=?',
            (row['asset_id'],))]
        projects = [dict(x) for x in connection.execute(
            'SELECT c.case_id,c.project_name,c.city,c.province FROM project_case_visuals v JOIN project_cases c ON c.case_id=v.case_id WHERE v.asset_id=?',
            (row['asset_id'],))]
        issues = []
        citation = {'document_name':row['document_name'], 'source_page':row['source_page'],
                    'source_url':row['source_url'], 'bbox':json.loads(row['bbox_json'] or 'null')}
        title = row['title']
        if not title or title == row['asset_id']:
            issues.append('missing_descriptive_title')
        provenance = bool(row['source_url'] or (row['document_name'] and row['source_page'] is not None))
        if not provenance:
            issues.append('missing_source_location')
        if len(products) > 1:
            issues.append('multiple_product_bindings')
        if len(projects) > 1:
            issues.append('multiple_project_bindings')
        if row['review_status'] == 'needs_review':
            issues.append('source_labels_need_review')
        variant = catalog.get('variant_or_code')
        role = catalog.get('visual_role')
        if row['category'] == 'product_sample' and role not in {'product_overview','component','accessory'} and not (variant or products):
            issues.append('missing_variant_identity')
        # A substrate/anchor has a role, not an invented finish/SKU.
        linked_product = products[0] if len(products) == 1 else {}
        attr=product_attributes.get(linked_product.get('product_id'),{})
        family = linked_product.get('family')
        source_tags = list(dict.fromkeys(str(t) for t in [row['category'], role, family,
            linked_product.get('surface_finish'), linked_product.get('insulation_type'),
            variant, *[p['project_name'] for p in projects], *[p['city'] for p in projects]] if t))
        # A source's project area/date disagreement is not resolvable by looking
        # at facade pixels. Keep it separate from image-label ambiguity.
        fact_conflicts = [dict(x) for project in projects for x in connection.execute(
            'SELECT field,values_json FROM case_conflicts WHERE case_id=?', (project['case_id'],))]
        output[row['asset_id']] = {
            'schema_version':'source_visual_labels_v1', 'asset_id':row['asset_id'],
            'title':title, 'category':row['category'], 'visual_role':role or row['category'],
            'product_name':products[0]['product_name'] if len(products)==1 else row['product_name'],
            'variant_or_code':variant, 'product_ids':[p['product_id'] for p in products],
            'product_family':family, 'surface_finish':linked_product.get('surface_finish'),
            'product_family_key':FAMILY_KEYS.get(family, 'unknown'),
            'insulation_type':linked_product.get('insulation_type'),
            'product_form':attr.get('product_form'), 'color_name':attr.get('color_name'),
            'product_gallery_eligible':bool(catalog.get('product_gallery_eligible')),
            'source_tags':source_tags, 'section_heading':metadata.get('section_heading'),
            'specific_role_known':bool(role),
            'retrieval_hints':(metadata.get('search_metadata') or {}).get('recommended_search_terms') or [],
            'retrieval_hints_are_facts':False,
            'label_provenance':{'visual_role':('visual_assets.metadata_json.catalog.visual_role' if role else 'visual_assets.category (coarse role only)'),
                'variant_or_code':'visual_assets.metadata_json.catalog.variant_or_code',
                'product_details':'product_visuals -> products (explicit asset_id join)',
                'project_details':'project_case_visuals -> project_cases (explicit asset_id join)'},
            'project_names':[p['project_name'] for p in projects] or ([row['project_name']] if row['project_name'] else []),
            'case_ids':[p['case_id'] for p in projects],
            'cities':list(dict.fromkeys(p['city'] for p in projects if p['city'])),
            'provinces':list(dict.fromkeys(p['province'] for p in projects if p['province'])),
            'caption':catalog.get('customer_caption') or metadata.get('caption'),
            'citation':citation, 'sha256':row['sha256'], 'access_scope':row['access_scope'],
            'label_basis':'existing_source_metadata_and_explicit_catalogue_joins',
            'pixel_contents_verified':False, 'semantic_verified':False,
            'label_issues':issues, 'display_ready':provenance and bool(title) and not issues and metadata.get('retrieval_eligible',True),
            'retrieval_eligible':metadata.get('retrieval_eligible',True),
            'fact_conflicts':fact_conflicts,
            'claim_boundary':'Labels identify source assets; they do not establish engineering performance or compliance.'}
        literal=metadata.get('literal_text_review') or {}
        if metadata.get('technical_occurrences'):
            output[row['asset_id']]['technical_occurrences'] = metadata['technical_occurrences']
            output[row['asset_id']]['label_provenance']['technical_occurrences'] = 'reviewed_slide_section_and_source_page'
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='node_asset_relations'").fetchone():
            refs=[{'product_form':form,'installation_method':method or None,'node_kind':kind}
                  for form,method,kind in connection.execute('SELECT product_form,installation_method,node_kind FROM node_asset_relations WHERE asset_id=?',(row['asset_id'],))]
            if refs:
                output[row['asset_id']]['node_system_references']=refs
                output[row['asset_id']]['label_provenance']['node_system_references']='existing_labels_and_source_scope_not_project_usage_proof'
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='node_asset_relations'").fetchone():
            refs=[{'product_form':form,'installation_method':method or None,'node_kind':kind}
                  for form,method,kind in connection.execute('SELECT product_form,installation_method,node_kind FROM node_asset_relations WHERE asset_id=?',(row['asset_id'],))]
            if refs:
                output[row['asset_id']]['node_system_references']=refs
                output[row['asset_id']]['label_provenance']['node_system_references']='existing_labels_and_source_scope_not_project_usage_proof'
        if literal.get('review_status') == 'verified_excerpt_only':
            output[row['asset_id']]['literal_text_review'] = literal
            output[row['asset_id']]['label_provenance']['literal_text_review'] = 'visually_checked_source_image_excerpts'
        if projects and connection.execute("SELECT 1 FROM sqlite_master WHERE name='project_product_links'").fetchone():
            from backend.sales.project_product_links import read_links
            associations=[dict(link,case_id=project['case_id']) for project in projects for link in read_links(connection,project['case_id'])]
            output[row['asset_id']]['project_product_associations'] = associations
            output[row['asset_id']]['project_product_associations_are_pixel_facts'] = False
        if projects and connection.execute("SELECT 1 FROM sqlite_master WHERE name='project_product_links'").fetchone():
            from backend.sales.project_product_links import read_links
            associations=[dict(link,case_id=project['case_id']) for project in projects for link in read_links(connection,project['case_id'])]
            output[row['asset_id']]['project_product_associations'] = associations
            output[row['asset_id']]['project_product_associations_are_pixel_facts'] = False
        if metadata.get('human_review'):
            review=metadata['human_review']
            output[row['asset_id']]['human_review'] = review
            output[row['asset_id']]['original_title_before_review'] = metadata.get('original_title_before_review')
            output[row['asset_id']]['label_provenance']['human_confirmed_fields'] = 'reviewed_visual_links'
            output[row['asset_id']]['caption'] = title
            output[row['asset_id']]['directory_category'] = review.get('directory_category')
            output[row['asset_id']]['product_form'] = review.get('product_form','unknown')
            if review.get('product_family_key'):
                output[row['asset_id']]['product_family_key'] = review['product_family_key']
        label=output[row['asset_id']]
        label['description_fields']=company_label_view(label)
        label['label_provenance']['description_fields']='existing_source_metadata_and_explicit_catalogue_joins'
    connection.row_factory = previous_row_factory
    return output


def persist_labels(connection, labels):
    connection.execute('CREATE TABLE IF NOT EXISTS visual_asset_labels(asset_id TEXT PRIMARY KEY REFERENCES visual_assets(asset_id), labels_json TEXT NOT NULL)')
    connection.executemany('INSERT OR REPLACE INTO visual_asset_labels VALUES(?,?)',
                          [(key,json.dumps(value,ensure_ascii=False)) for key,value in labels.items()])
