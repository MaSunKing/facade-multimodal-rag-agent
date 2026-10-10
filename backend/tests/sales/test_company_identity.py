from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.sales.company_identity import load_company_identity, identity_prompt_context
from scripts.reconcile_company_knowledge import deduplicate_text, profile_records
from scripts.build_sales_catalog_db import build, consolidate_catalogue_cases
from backend.sales.catalog_store import retrieve_catalog_evidence


class CompanyIdentityTests(unittest.TestCase):
    def test_named_catalogue_variant_does_not_return_other_shared_finishes(self):
        from backend.sales.retriever import LocalRagRetriever
        retriever = LocalRagRetriever()
        result = retriever.retrieve('我想了解黄金麻荔枝面，给我对应的产品图片和简介。', wants_visuals=True, visual_scope='product', visual_k=5, product_overview_request=False)
        images = result['visual_assets']
        self.assertTrue(images)
        self.assertTrue(all('黄金麻' in v.get('customer_title', '') for v in images))

    def test_overview_keeps_approved_sections_outside_hybrid_shortlist(self):
        from unittest.mock import patch
        from backend.sales.retriever import LocalRagRetriever
        retriever = LocalRagRetriever(index_path=Path('data/sales/processed/rag_index/lexical_index.json'))
        approved = [d for d in retriever.documents if d.get('sales_playbook_use') == 'approved_product_master_profile']
        shortlist = [(1.0, d) for d in approved if not str(d['id']).startswith('companytxt_')]
        with patch.object(retriever, '_hybrid_scored', return_value=shortlist):
            result = retriever.retrieve('公司产品目录 产品体系 产品总档案', top_k=5, product_overview_request=True)
        kept = {e['chunk_id'] for e in result['text_evidence']}
        self.assertTrue({d['id'] for d in approved}.issubset(kept))

    def test_identity_is_manufacturer_not_model_or_city_case(self):
        identity = load_company_identity()
        self.assertEqual(identity["legal_name"], "河北大自然石材有限公司")
        self.assertEqual(identity["city"], "保定市")
        self.assertIn("not permission or routing", identity_prompt_context())
        self.assertEqual(len(identity["product_families"]), 3)

    def test_identity_evidence_is_cited_and_not_performance_certification(self):
        records = profile_records(load_company_identity())
        self.assertEqual(len({r["chunk_id"] for r in records}), 6)
        self.assertTrue(all(r["source_refs"][0]["source_url"].startswith("https://") for r in records))
        self.assertTrue(any("不能替代" in r["text"] for r in records))

    def test_exact_dedup_retains_provenance_and_different_values(self):
        rows = [{"chunk_id":str(i),"text":text,"index_eligible":True,"fact_eligible":True,"source_refs":[{"document_name":str(i)}]} for i,text in enumerate(["厚度20mm", "厚度20mm", "厚度18mm"])]
        merged, changes = deduplicate_text(rows)
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(merged[0]["source_refs"]), 2)
        self.assertEqual(len(changes), 1)

    def test_real_catalogue_integrity_and_identity_retrieval(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"catalog.sqlite3"
            audit = build(path,Path(folder)/"audit.json")
            self.assertEqual(audit["integrity_check"], "ok")
            self.assertEqual(audit["missing_visual_assets"], [])
            self.assertEqual(audit["product_with_main_image_count"], audit["product_count"])
            con = sqlite3.connect(path)
            self.assertEqual(con.execute("pragma foreign_key_check").fetchall(), [])
            self.assertEqual(con.execute("select count(*) from project_case_visuals where asset_id not in(select asset_id from visual_assets)").fetchone()[0],0)
            con.close()
            records = retrieve_catalog_evidence("你们是谁",database_path=path)
            self.assertEqual(records[0]["catalog_record_type"],"company_identity")
            self.assertIn("河北大自然",records[0]["text"])


    def test_case_duplicates_preserve_conflicts(self):
        records = [{"case_id":"a","project_name":"示例项目","area_m2":"200㎡","visual_asset_ids":["x"]},
                   {"case_id":"b","project_name":"示例项目","area_m2":"300㎡","visual_asset_ids":["y"]}]
        result = consolidate_catalogue_cases(records)
        self.assertEqual(len(result),1)
        self.assertIsNone(result[0]["area_m2"])
        self.assertEqual(result[0]["visual_asset_ids"],["x","y"])
        self.assertEqual(result[0]["field_disagreements"]["area_m2"],["200㎡","300㎡"])


if __name__ == "__main__":
    unittest.main()
