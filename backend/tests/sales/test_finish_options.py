import sqlite3
import unittest
from backend.sales.finish_options import rebuild_finish_options, read_finish_options, finish_option_text


class FinishOptionsTests(unittest.TestCase):
    def test_options_are_additive_idempotent_not_new_skus_or_media(self):
        with sqlite3.connect(':memory:') as c:
            c.execute('CREATE TABLE products(product_id TEXT PRIMARY KEY,name TEXT,surface_finish TEXT,access_scope TEXT)')
            c.executemany('INSERT INTO products VALUES(?,?,?,?)',
                [('a','某色光面','光面','public'),('b','某芯材板',None,'public'),('p','私有样品',None,'private')])
            original = c.execute('SELECT * FROM products').fetchall()
            policy={'basis':'user_confirmed_product_options','boundary':'text only',
                    'finishes':['纹面','亮面','柔光面']}
            first = rebuild_finish_options(c, policy)
            self.assertEqual(first, rebuild_finish_options(c, policy))
            self.assertEqual(c.execute('SELECT * FROM products').fetchall(),original)
            self.assertEqual(first['option_links'],9)
            self.assertEqual(read_finish_options(c,'a'),policy['finishes'])
            self.assertEqual(read_finish_options(c,'b'),policy['finishes'])
            self.assertEqual(read_finish_options(c,'p'),[])
            self.assertIn('不表示已新增独立型号',finish_option_text(c,'a'))
            self.assertEqual(c.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE '%visual%'").fetchone()[0],0)

    def test_legacy_catalog_read_remains_compatible(self):
        with sqlite3.connect(':memory:') as c:
            self.assertEqual(read_finish_options(c,'a'),[])


if __name__=='__main__': unittest.main()
