"""User-authorized demo completion, isolated from original source evidence."""
import json
import sqlite3


def complete_fields(product, method, colour=None):
    product = (product or '').strip()
    original_method = (method or '').strip()
    method = original_method or '干挂＋粘锚'
    has_form = any(term in product for term in ('单板','装饰板','一体板','保温'))
    if not has_form:
        form = '保温装饰一体板' if '粘锚' in original_method else '单板＋保温装饰一体板'
        product = (product + '；' if product else '') + '产品形态：' + form
    if colour:
        product += '；色型：' + colour
    return product, method


def apply_project_defaults(c):
    c.execute('''CREATE TABLE IF NOT EXISTS project_field_supplements(
      case_id TEXT PRIMARY KEY REFERENCES project_cases(case_id),
      original_product_name TEXT,original_installation_method TEXT,
      supplied_product_name TEXT NOT NULL,supplied_installation_method TEXT NOT NULL,
      provenance_json TEXT NOT NULL)''')
    old = c.row_factory
    c.row_factory = sqlite3.Row
    try:
        products = [dict(r) for r in c.execute('SELECT name,surface_finish,insulation_type FROM products')]
        bases = {p['name'].replace(p['surface_finish'] or '\0','') for p in products if not p['insulation_type']}
        bases.add('金麻')
        rows = c.execute("SELECT * FROM project_cases WHERE access_scope='public'").fetchall()
        counts = {'projects':len(rows),'methods_filled':0,'forms_filled':0,'colours_filled':0,'image_colour_references':0}
        for row in rows:
            prior = c.execute('SELECT * FROM project_field_supplements WHERE case_id=?',(row['case_id'],)).fetchone()
            # Reapply without appending duplicate defaults; preserve later manual edits.
            original_product = prior['original_product_name'] if prior and row['product_name']==prior['supplied_product_name'] else row['product_name']
            original_method = prior['original_installation_method'] if prior and row['installation_method']==prior['supplied_installation_method'] else row['installation_method']
            source_name = original_product or ''
            colour = None
            basis = 'source_colour_retained'
            references = []
            if not any(base in source_name for base in bases):
                colour = '多种混合'
                basis = 'user_default_unspecified_colour'
                # Existing image-based associations provide a colour reference,
                # not finish/core/performance verification. No configured default.
                if c.execute("SELECT 1 FROM sqlite_master WHERE name='project_product_links'").fetchone():
                    links = c.execute('''SELECT p.name,p.surface_finish,l.provenance_json,l.association_basis
                      FROM project_product_links l JOIN products p USING(product_id)
                      WHERE l.case_id=? AND p.insulation_type IS NULL''',(row['case_id'],)).fetchall()
                    candidates = []
                    for link in links:
                        proof = json.loads(link['provenance_json'])
                        if link['association_basis']=='demo_image_colour_similarity' and proof.get('image_asset_ids') and proof.get('colour_distance') is not None:
                            candidates.append(link['name'].replace(link['surface_finish'] or '\0',''))
                            references.append(proof)
                    if len(set(candidates))==1:
                        colour = candidates[0]
                        basis = 'existing_image_colour_reference_not_material_proof'
                        counts['image_colour_references'] += 1
            product, method = complete_fields(original_product,original_method,colour)
            counts['methods_filled'] += not bool((original_method or '').strip())
            counts['forms_filled'] += not any(t in source_name for t in ('单板','装饰板','一体板','保温'))
            counts['colours_filled'] += colour is not None
            provenance = {'user_authorized_demo_defaults':True,'colour_basis':basis,'image_references':references,
              'boundary':'Supplied demo fields, not original source facts. No surface finish inferred.'}
            c.execute('INSERT OR REPLACE INTO project_field_supplements VALUES(?,?,?,?,?,?)',
              (row['case_id'],original_product,original_method,product,method,json.dumps(provenance,ensure_ascii=False)))
            c.execute('UPDATE project_cases SET product_name=?,installation_method=? WHERE case_id=?',(product,method,row['case_id']))
            c.execute('UPDATE case_search SET product_name=? WHERE case_id=?',(product,row['case_id']))
        return counts
    finally:
        c.row_factory = old
