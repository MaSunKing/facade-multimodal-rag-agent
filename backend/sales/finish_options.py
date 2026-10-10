"""Explicit catalog options, separate from a SKU's registered finish and media."""
import json
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[2] / 'data/sales/config/product_finish_options.json'


def rebuild_finish_options(connection, policy=None):
    policy = policy if policy is not None else json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    finishes = policy.get('finishes', [])
    if not isinstance(finishes, list) or not finishes or any(not isinstance(f, str) or not f.strip() for f in finishes):
        raise ValueError('invalid_product_finish_options')
    finishes = list(dict.fromkeys(f.strip() for f in finishes))
    connection.execute('''CREATE TABLE IF NOT EXISTS catalog_product_finish_options(
        product_id TEXT NOT NULL REFERENCES products(product_id),
        finish TEXT NOT NULL, display_order INTEGER NOT NULL, provenance_json TEXT NOT NULL,
        PRIMARY KEY(product_id,finish))''')
    connection.execute('DELETE FROM catalog_product_finish_options')
    proof = json.dumps({'basis': policy['basis'], 'boundary': policy['boundary'],
                       'source': 'product_finish_options.json'}, ensure_ascii=False)
    products = list(connection.execute('SELECT product_id FROM products'))
    for (pid,) in products:
        connection.executemany('INSERT INTO catalog_product_finish_options VALUES(?,?,?,?)',
            [(pid, f, i, proof) for i, f in enumerate(finishes)])
    return {'products': len(products), 'option_links': len(products)*len(finishes)}


def read_finish_options(connection, product_id):
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='catalog_product_finish_options'").fetchone():
        return []
    return [r[0] for r in connection.execute('''SELECT o.finish
        FROM catalog_product_finish_options o JOIN products p USING(product_id)
        WHERE o.product_id=? AND p.access_scope='public' ORDER BY o.display_order''', (product_id,))]


def finish_option_text(connection, product_id):
    options = read_finish_options(connection, product_id)
    from backend.sales.product_source_profiles import read_product_profile
    profile = read_product_profile(connection, product_id)
    owner = profile.get('color_name') or '产品体系'
    return (owner+'层可选饰面（不是本条目型号的登记饰面；用户确认文字登记）：' + '、'.join(options) +
            '；可选饰面不表示已新增独立型号、样板图或检测报告') if options else ''
