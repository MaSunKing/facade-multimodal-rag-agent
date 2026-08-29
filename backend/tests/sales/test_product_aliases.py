from __future__ import annotations

import unittest

from backend.sales.retriever import expand_product_aliases, load_product_alias_groups


class ProductAliasTests(unittest.TestCase):
    def test_confirmed_inorganic_mortar_alias_expands_to_canonical_product(self) -> None:
        expanded = expand_product_aliases("无机砂浆一体板有什么特点？", load_product_alias_groups())
        self.assertIn("真岩®石保温装饰一体板", expanded)

    def test_unrelated_query_is_not_expanded(self) -> None:
        query = "女儿墙节点怎么施工？"
        self.assertEqual(expand_product_aliases(query, load_product_alias_groups()), query)

    def test_canonical_short_name_does_not_dilute_intent_terms(self) -> None:
        query = "真岩石的产能和售后保障"
        self.assertEqual(expand_product_aliases(query, load_product_alias_groups()), query)


if __name__ == "__main__":
    unittest.main()
