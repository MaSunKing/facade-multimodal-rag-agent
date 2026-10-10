"""Non-destructive, source-derived product dimensions over legacy catalog IDs.

These are directory attributes, not new SKUs or performance evidence. Unknown
attributes stay NULL; dimensions are never expanded as a Cartesian product.
"""
import json

from backend.sales.node_relations import installation_defaults


def product_form(family):
    if '结构' in family and '一体' in family:
        return 'structural_integrated'
    if '保温' in family and '板' in family:
        return 'insulated_panel'
    if '线条' in family:
        return 'decorative_line'
    if '装饰板' in family or '单板' in family:
        return 'decorative_panel'
    return 'unspecified'


def attributes(product):
    name = (product.get('name') or '').strip()
    family = (product.get('family') or '').strip()
    finish = (product.get('surface_finish') or '').strip() or None
    core = (product.get('insulation_type') or '').strip() or None
    # Only an explicit finish suffix establishes a color/finish catalog variant.
    # A core-type entry supplies no color, even if the same family has a palette.
    color = name[:-len(finish)].strip() if finish and name.endswith(finish) and not core else None
    kind = 'core_variant' if core else 'color_finish_variant' if color else 'unspecified'
    return dict(product_id=product['product_id'], family=family,
                product_form=product_form(family), entry_kind=kind,
                color_name=color or None, surface_finish=finish, insulation_core=core,
                provenance_json=json.dumps({'basis': 'existing_catalog_fields',
                    'fields': ['name', 'family', 'surface_finish', 'insulation_type'],
                    'boundary': 'Directory classification only; no new product combination.'}, ensure_ascii=False))


def rebuild_catalog_structure(c):
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'products' not in tables:
        return {'product_entries': 0}
    schema = '''
        CREATE TABLE IF NOT EXISTS catalog_product_attributes(
          product_id TEXT PRIMARY KEY REFERENCES products(product_id),
          family TEXT NOT NULL, product_form TEXT NOT NULL, entry_kind TEXT NOT NULL,
          color_name TEXT, surface_finish TEXT, insulation_core TEXT,
          provenance_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS catalog_form_navigation(
          family TEXT PRIMARY KEY, product_form TEXT NOT NULL,
          default_installation_method TEXT, method_basis TEXT NOT NULL);
        CREATE VIEW IF NOT EXISTS catalog_color_variants AS
          SELECT a.*,p.name,p.access_scope FROM catalog_product_attributes a
          JOIN products p USING(product_id) WHERE a.entry_kind='color_finish_variant';
        CREATE VIEW IF NOT EXISTS catalog_core_variants AS
          SELECT a.*,p.name,p.access_scope FROM catalog_product_attributes a
          JOIN products p USING(product_id) WHERE a.entry_kind='core_variant';
    '''
    # execute(), unlike executescript(), preserves the caller's transaction.
    for statement in schema.split(';'):
        if statement.strip():
            c.execute(statement)
    c.execute('DELETE FROM catalog_product_attributes')
    c.execute('DELETE FROM catalog_form_navigation')
    cursor = c.execute('SELECT * FROM products')
    keys = [d[0] for d in cursor.description]
    products = [dict(zip(keys, r)) for r in cursor.fetchall()]
    columns = ('product_id', 'family', 'product_form', 'entry_kind', 'color_name',
               'surface_finish', 'insulation_core', 'provenance_json')
    for product in products:
        row = attributes(product)
        c.execute('INSERT INTO catalog_product_attributes VALUES(?,?,?,?,?,?,?,?)',
                  tuple(row[k] for k in columns))
    families = {p.get('family') or '' for p in products}
    if 'product_families' in tables:
        families.update(r[0] for r in c.execute('SELECT family FROM product_families'))
    defaults = installation_defaults()
    for family in sorted(families):
        form = product_form(family)
        method = defaults.get(form)
        c.execute('INSERT INTO catalog_form_navigation VALUES(?,?,?,?)',
                  (family, form, method, 'user_confirmed_catalog_default' if method else 'unspecified'))
    from backend.sales.finish_options import rebuild_finish_options
    finish_report = rebuild_finish_options(c)
    from backend.sales.knowledge_scope_notes import rebuild_scope_notes
    scope_report = rebuild_scope_notes(c)
    from backend.sales.product_source_profiles import rebuild_product_profiles
    profile_report = rebuild_product_profiles(c)
    return {'product_entries': len(products),
            'finish_options': finish_report,
            'scope_notes': scope_report,
            'source_profiles': profile_report,
            'entry_kinds': dict(c.execute('SELECT entry_kind,count(*) FROM catalog_product_attributes GROUP BY entry_kind')),
            'distinct_colors': c.execute('SELECT count(DISTINCT color_name) FROM catalog_product_attributes').fetchone()[0],
            'distinct_finishes': c.execute('SELECT count(DISTINCT surface_finish) FROM catalog_product_attributes').fetchone()[0],
            'distinct_cores': c.execute('SELECT count(DISTINCT insulation_core) FROM catalog_product_attributes').fetchone()[0]}


def read_product_attributes(c, product_ids):
    if not c.execute("SELECT 1 FROM sqlite_master WHERE name='catalog_product_attributes'").fetchone():
        return []
    result = []
    for pid in product_ids:
        cursor = c.execute('''SELECT a.product_id,a.family,a.product_form,a.entry_kind,
          a.color_name,a.surface_finish,a.insulation_core FROM catalog_product_attributes a
          JOIN products p USING(product_id) WHERE a.product_id=? AND p.access_scope='public' ''', (pid,))
        keys = [d[0] for d in cursor.description]
        from backend.sales.finish_options import read_finish_options
        for r in cursor.fetchall():
            row = dict(zip(keys, r))
            row['available_surface_finishes'] = read_finish_options(c, pid)
            row['finish_options_basis'] = 'user_confirmed_product_options' if row['available_surface_finishes'] else None
            result.append(row)
    return result
