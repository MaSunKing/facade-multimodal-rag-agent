from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.sales.catalog_store import clean_catalog_introduction, retrieve_catalog_evidence


class CatalogStoreTests(unittest.TestCase):
    def test_public_page_noise_is_removed_without_mutating_product_description(self) -> None:
        cleaned = clean_catalog_introduction(
            "黄金麻荔枝面 QQ 真岩®无机饰面板用于建筑外装。品牌：真岩® "
            "标准规格：1220×2440 0312 - 7745966 二维码",
            product_name="黄金麻荔枝面",
        )
        self.assertEqual(cleaned, "真岩®无机饰面板用于建筑外装。")

    def test_structured_hit_cites_original_web_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            connection = sqlite3.connect(path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE products(
                        product_id TEXT, name TEXT, family TEXT, surface_finish TEXT,
                        insulation_type TEXT, introduction TEXT, standard_specification TEXT,
                        panel_thickness TEXT, weight_kg_m2 TEXT, insulation_thickness TEXT,
                        application_scope TEXT, source_url TEXT, main_visual_asset_id TEXT,
                        access_scope TEXT
                    );
                    CREATE TABLE project_cases(
                        case_id TEXT, project_name TEXT, province TEXT, city TEXT,
                        project_category TEXT, product_name TEXT, product_specificity TEXT,
                        installation_method TEXT, area_m2 TEXT, completion_year TEXT,
                        image_count INTEGER, source_url TEXT, source_text TEXT, access_scope TEXT
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO products VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "p1", "黄金麻荔枝面", "真岩装饰板", "荔枝面", None, "介绍",
                        "1220×2440", "9mm", None, None, "建筑外墙",
                        "https://example.test/product", "v1", "public",
                    ),
                )
                connection.execute(
                    "INSERT INTO project_cases VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "c1", "山东示例项目", "山东省", "济南市", "办公楼",
                        "黄金麻", "catalogue_matched", "粘锚", "10000㎡", "2025", 3,
                        "https://example.test/case", "案例正文", "public",
                    ),
                )
                connection.commit()
            finally:
                connection.close()
            result = retrieve_catalog_evidence(
                "山东黄金麻",
                target_terms=["黄金麻"],
                case_filters={"locations": ["山东"], "products": ["黄金麻"]},
                database_path=path,
            )
            self.assertEqual({item["catalog_record_type"] for item in result}, {"product", "project_case"})
            urls = {item["citations"][0]["source_url"] for item in result}
            self.assertEqual(urls, {"https://example.test/product", "https://example.test/case"})
            self.assertTrue(all(item["source_type"] == "catalog_sql" for item in result))

            product_only = retrieve_catalog_evidence(
                "黄金麻产品介绍",
                target_terms=["黄金麻"],
                include_project_cases=False,
                database_path=path,
            )
            self.assertEqual([item["catalog_record_type"] for item in product_only], ["product"])
            self.assertIn("产品介绍：介绍", product_only[0]["text"])
            self.assertNotIn("山东示例项目", product_only[0]["text"])


if __name__ == "__main__":
    unittest.main()
