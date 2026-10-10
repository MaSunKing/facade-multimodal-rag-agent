"""Explicit human-reviewed associations; never infer finish from image pixels."""
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2]/'data/sales/reviewed_visual_links.json'


def apply_reviewed_links(c, path=DEFAULT_PATH):
    path=Path(path)
    if not path.is_file():
        return 0
    payload=json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 'reviewed_visual_links_v1':
        raise ValueError('invalid_reviewed_links_schema')
    c.execute('''CREATE TABLE IF NOT EXISTS reviewed_visual_links(
      asset_id TEXT NOT NULL REFERENCES visual_assets(asset_id),
      product_id TEXT NOT NULL REFERENCES products(product_id),
      review_json TEXT NOT NULL, PRIMARY KEY(asset_id,product_id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS reviewed_asset_classification(
      asset_id TEXT PRIMARY KEY REFERENCES visual_assets(asset_id), review_json TEXT NOT NULL)''')
    for link in payload['links']:
        row=c.execute('SELECT title,metadata_json FROM visual_assets WHERE asset_id=?',(link['asset_id'],)).fetchone()
        product_id=link.get('product_id')
        product=c.execute('SELECT product_id FROM products WHERE product_id=?',(product_id,)).fetchone() if product_id else None
        if not row or (product_id and not product) or not str(link.get('review_basis') or '').strip():
            raise ValueError('reviewed_link_missing_identity_or_provenance')
        if link.get('directory_category') not in {None,'product','project','node','process','reference','document_illustration','material_component','material_sample','installation_accessory'}:
            raise ValueError('invalid_reviewed_category')
        if link.get('product_family_key') not in {None,'unknown','decorative_panel','insulated_panel'}:
            raise ValueError('invalid_reviewed_product_family')
        metadata=json.loads(row[1] or '{}')
        metadata.setdefault('original_title_before_review',row[0])
        metadata.setdefault('original_catalog_before_review',dict(metadata.get('catalog') or {}))
        metadata['human_review']=link
        if 'visual_role' in link or 'product_gallery_eligible' in link:
            catalog=metadata.setdefault('catalog',{})
            for field in ('visual_role','product_gallery_eligible'):
                if field in link:
                    catalog[field]=link[field]
        c.execute('UPDATE visual_assets SET title=?,metadata_json=? WHERE asset_id=?',
                  (link.get('title') or row[0],json.dumps(metadata,ensure_ascii=False),link['asset_id']))
        c.execute('INSERT OR REPLACE INTO reviewed_asset_classification VALUES(?,?)',
                  (link['asset_id'],json.dumps(link,ensure_ascii=False)))
        if not product_id:
            continue
        if not c.execute('SELECT 1 FROM product_visuals WHERE product_id=? AND asset_id=?',
                         (link['product_id'],link['asset_id'])).fetchone():
            rank=c.execute('SELECT coalesce(max(display_order),0)+1 FROM product_visuals WHERE product_id=?',
                           (link['product_id'],)).fetchone()[0]
            c.execute('INSERT INTO product_visuals VALUES(?,?,?,?)',(link['product_id'],link['asset_id'],'sample',rank))
        c.execute('INSERT OR REPLACE INTO reviewed_visual_links VALUES(?,?,?)',
                  (link['asset_id'],link['product_id'],json.dumps(link,ensure_ascii=False)))
    return len(payload['links'])
