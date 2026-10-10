"""Demo catalogue associations, with explicit source-vs-inference boundaries."""
from __future__ import annotations
import json, re, sqlite3
from pathlib import Path
import numpy as np
from PIL import Image

GENERATOR='demo_project_product_link_v1'

def descriptor(path):
    """Cheap colour reference, not a material/finish/engineering classifier."""
    with Image.open(path) as im:
        w,h=im.size
        im=im.convert('RGB').crop((int(w*.15),int(h*.15),int(w*.85),int(h*.85)))
        im.thumbnail((128,128))
        a=np.asarray(im,dtype=np.float32).reshape(-1,3)/255
        return np.concatenate((np.mean(a,axis=0),np.std(a,axis=0),np.quantile(a,[.25,.5,.75],axis=0).reshape(-1)))

def text_candidates(text, products, aliases, category_terms=None):
    """Select named variants/categories; inferred finish is never source truth."""
    text=re.sub(r'\s+','',text or '')
    result=[]
    named_bases=[p['name'].replace(p.get('surface_finish') or '\0','') for p in products]
    for product in products:
        name=product['name']; variants=[name]
        if product.get('surface_finish'):variants.append(name.replace(product['surface_finish'],''))
        for alias,target in aliases.items():
            if target in name and not any(alias in base and len(base)>len(alias) and base in text for base in named_bases):variants.append(alias)
        if product.get('insulation_type') and '一体板' in text:
            variants.extend((category_terms or {}).get(name,[]))
        matched=[v for v in variants if v and v in text]
        if matched:
            result.append((product,max(matched,key=len)))
    # If the source explicitly names a core material, prefer the insulation
    # category rather than guessing a decorative SKU from its facade colour.
    insulation=[x for x in result if x[0].get('insulation_type')]
    if insulation:return insulation
    return result

def populate_links(con, policy, root):
    old=con.row_factory;con.row_factory=sqlite3.Row
    try:return _populate(con,policy,root)
    finally:con.row_factory=old

def _populate(con,policy,root):
    existing=con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='project_product_links'").fetchone()
    migrate=bool(existing and 'demo_inferred' in existing[0])
    if migrate:
        con.execute('ALTER TABLE project_product_links RENAME TO project_product_links_previous')
    con.execute('''CREATE TABLE IF NOT EXISTS project_product_links(
        case_id TEXT NOT NULL REFERENCES project_cases(case_id),
        product_id TEXT NOT NULL REFERENCES products(product_id),
        relation_status TEXT NOT NULL CHECK(relation_status IN ('source_category_match','catalog_association')),
        association_basis TEXT NOT NULL, source_product_text TEXT NOT NULL,
        inferred_fields_json TEXT NOT NULL, provenance_json TEXT NOT NULL,
        generator TEXT NOT NULL, PRIMARY KEY(case_id,product_id))''')
    if migrate:
        con.execute('''INSERT INTO project_product_links SELECT case_id,product_id,
            CASE WHEN relation_status='source_category_match' THEN relation_status ELSE 'catalog_association' END,
            association_basis,source_product_text,inferred_fields_json,provenance_json,generator
            FROM project_product_links_previous''')
        con.execute('DROP TABLE project_product_links_previous')
    con.execute('DELETE FROM project_product_links WHERE generator=?',(GENERATOR,))
    products=[dict(x) for x in con.execute('SELECT * FROM products WHERE access_scope=?',('public',))]
    if not products:raise ValueError('no_public_products_for_demo_linking')
    vectors={};paths={}
    for p in products:
        f=con.execute('SELECT f.image_path FROM visual_assets a JOIN visual_files f ON a.sha256=f.sha256 WHERE a.asset_id=?',(p['main_visual_asset_id'],)).fetchone()
        if f:
            path=Path(f[0]);path=path if path.is_absolute() else root/path
            try:vectors[p['product_id']]=descriptor(path);paths[p['product_id']]=str(path)
            except (OSError,ValueError):pass
    counts={'source_category_projects':0,'demo_inferred_projects':0,'image_inferred_projects':0,'fallback_projects':0}
    for case in con.execute('SELECT * FROM project_cases WHERE access_scope=?',('public',)).fetchall():
        text=case['product_name'] or ''
        options=text_candidates(text,products,policy.get('demo_aliases',{}),policy.get('source_category_terms',{}))
        image_refs=[];case_vectors=[]
        for v in con.execute('''SELECT a.asset_id,f.image_path FROM project_case_visuals v
            JOIN visual_assets a ON a.asset_id=v.asset_id JOIN visual_files f ON f.sha256=a.sha256
            WHERE v.case_id=? AND a.access_scope=? ORDER BY v.display_order,a.asset_id LIMIT ?''',
            (case['case_id'],'public',int(policy.get('max_reference_images',8)))):
            path=Path(v['image_path']);path=path if path.is_absolute() else root/path
            try:case_vectors.append(descriptor(path));image_refs.append(v['asset_id'])
            except (OSError,ValueError):pass
        def distance(p):
            ref=vectors.get(p['product_id'])
            if ref is None or not case_vectors:return float('inf')
            return float(np.median([np.linalg.norm(v-ref) for v in case_vectors]))
        basis='source_named_variant_with_demo_finish_selection';fallback=False
        if options:
            # Preserve separate named variants for a multi-product source.
            groups={}
            for p,term in options:groups.setdefault(term,[]).append(p)
            chosen=[min(ps,key=lambda p:(distance(p),p['name'])) for ps in groups.values()]
            # Alias overlap must not create duplicate product links.
            chosen=list({p['product_id']:p for p in chosen}.values())
        else:
            candidates=[p for p in products if not p.get('insulation_type')] or products
            chosen=[min(candidates,key=lambda p:(distance(p),p['name']))]
            if distance(chosen[0])!=float('inf'):
                basis='demo_image_colour_similarity';counts['image_inferred_projects']+=1
            else:
                basis='demo_configured_default';fallback=True;counts['fallback_projects']+=1
                name=policy.get('fallback_product_name')
                chosen=[next((p for p in candidates if p['name']==name),chosen[0])]
        source_only=True
        for p in chosen:
            core_terms=policy.get('source_category_terms',{}).get(p['name'],[])
            supported=bool(p['name'] in text or (p.get('insulation_type') and '一体板' in text and any(t in text for t in core_terms)))
            source_only=source_only and supported
            status='source_category_match' if supported else 'catalog_association'
            inferred=[] if supported else ['product_id']+[k for k in ('surface_finish','insulation_type') if p.get(k)]
            provenance=dict(source_url=case['source_url'],image_asset_ids=image_refs,
                selected_product_main_image=p['main_visual_asset_id'],colour_distance=(round(distance(p),5) if distance(p)!=float('inf') else None),
                claim_boundary='Demo navigation association only. Colour similarity does not establish material, finish, core, dimensions, compliance or performance.',
                user_authorized_demo_inference=True)
            con.execute('INSERT OR IGNORE INTO project_product_links VALUES(?,?,?,?,?,?,?,?)',
                (case['case_id'],p['product_id'],status,'source_named_category' if supported else basis,text,json.dumps(inferred),json.dumps(provenance,ensure_ascii=False),GENERATOR))
        counts['source_category_projects' if source_only else 'demo_inferred_projects']+=1
    counts.update(projects_linked=con.execute('SELECT count(DISTINCT case_id) FROM project_product_links').fetchone()[0],links=con.execute('SELECT count(*) FROM project_product_links').fetchone()[0])
    if con.execute('PRAGMA foreign_key_check').fetchall():raise ValueError('project_product_foreign_key_error')
    return counts

def read_links(con,case_id):
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name='project_product_links'").fetchone():return []
    prior=con.row_factory;con.row_factory=sqlite3.Row
    try:
        return [dict(x) for x in con.execute('''SELECT l.product_id,p.name AS product_name,
            l.relation_status,l.association_basis,l.inferred_fields_json,l.provenance_json
            FROM project_product_links l JOIN products p ON p.product_id=l.product_id
            WHERE l.case_id=? AND p.access_scope=? ORDER BY p.name''',(case_id,'public'))]
    finally:con.row_factory=prior
