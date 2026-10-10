import sqlite3
import unittest
from backend.sales.product_source_profiles import rebuild_product_profiles, read_product_profile, product_profile_text
from backend.sales.generation_policy import CATALOG_SCOPE_POLICY
from backend.sales.draft_prompt import company_prompt, document_prompt, general_prompt
from backend.sales.answer_review import REVISION_PROMPT


class ProductProfileTests(unittest.TestCase):
    def fixture(self):
        c=sqlite3.connect(':memory:')
        c.execute('CREATE TABLE products(product_id TEXT PRIMARY KEY,access_scope TEXT)')
        c.execute('CREATE TABLE catalog_product_attributes(product_id TEXT,product_form TEXT,color_name TEXT,surface_finish TEXT)')
        c.execute('CREATE TABLE knowledge_text_evidence(evidence_id TEXT,access_scope TEXT)')
        c.executemany('INSERT INTO products VALUES(?,?)',[('a','public'),('b','public'),('c','public'),('p','internal')])
        c.executemany('INSERT INTO catalog_product_attributes VALUES(?,?,?,?)',[('a','panel','青色','光面'),('b','panel','青色','纹面'),('c','core',None,None),('p','panel','青色','光面')])
        c.executemany('INSERT INTO knowledge_text_evidence VALUES(?,?)',[('s','public'),('m','public')])
        policy={'basis':'reviewed','forms':{'panel':{'source_id':'s','colors':['青色'],'panel_thickness':'12mm'},'core':{'source_id':'s','colors':['青色'],'cores':['芯材甲']}},'models_source_id':'m','color_models':[{'color':'青色','finish':'光面','code':'X1'}]}
        return c,policy

    def test_exact_bindings_scope_idempotence_no_new_products(self):
        c,p=self.fixture()
        with c:
            before=c.execute('SELECT * FROM products').fetchall()
            self.assertEqual(rebuild_product_profiles(c,p),rebuild_product_profiles(c,p))
            self.assertEqual(before,c.execute('SELECT * FROM products').fetchall())
            self.assertEqual(read_product_profile(c,'a')['registered_color_models'][0]['code'],'X1')
            self.assertEqual(read_product_profile(c,'b')['registered_color_models'],[])
            self.assertEqual(read_product_profile(c,'c')['registered_color_models'],[])
            self.assertEqual(read_product_profile(c,'p'),{})
            self.assertIn('不是完整一体板组合编码',product_profile_text(c,'a'))
            self.assertIn('指定来源/版本或检测样品',product_profile_text(c,'a'))
        c.close()

    def test_missing_or_private_support_does_not_supply_model(self):
        c,p=self.fixture()
        c.execute("UPDATE knowledge_text_evidence SET access_scope='internal' WHERE evidence_id='m'")
        rebuild_product_profiles(c,p)
        self.assertNotIn('registered_color_models',read_product_profile(c,'a'))
        p['forms']['panel']['source_id']='missing'
        rebuild_product_profiles(c,p)
        self.assertEqual(read_product_profile(c,'a'),{})
        c.close()

    def test_legacy_and_all_prompt_paths(self):
        c=sqlite3.connect(':memory:')
        self.assertEqual(read_product_profile(c,'x'),{})
        c.close()
        for prompt in [company_prompt('identity'),document_prompt(),general_prompt(),REVISION_PROMPT]:
            self.assertIn(CATALOG_SCOPE_POLICY,prompt)
        for prompt in [company_prompt('identity'),document_prompt(),general_prompt()]:
            self.assertNotIn('和下一步直接融合',prompt)
            self.assertIn('不默认添加下一步建议',prompt)


if __name__=='__main__':unittest.main()
