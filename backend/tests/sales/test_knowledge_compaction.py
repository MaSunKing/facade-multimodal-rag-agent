import unittest
import json
import tempfile
from pathlib import Path
from backend.sales.knowledge_compaction import compact_documents
from backend.sales.retriever import LocalRagRetriever


def row(key, text, scope="public", doc="d"):
    return {"id": key, "kind": "text", "text": text, "tokens": [],
            "access_scope": scope, "source_refs": [{"document_id": doc, "source_page": key}]}


class KnowledgeCompactionTests(unittest.TestCase):
    def test_duplicates_keep_all_provenance_and_original_text(self):
        text = "外墙饰面的施工安装必须保留适用条件，相关产品的数值、单位、版本和项目要求不能省略。" * 2
        original = [row("a", text.replace("外墙", "外 墙")), row("b", text)]
        kept, aliases = compact_documents(original)
        self.assertEqual(aliases, {"b": "a"})
        self.assertEqual(len(kept[0]["source_refs"]), 2)
        self.assertEqual(kept[0]["duplicate_variants"][0]["text"], text)
        self.assertNotIn("duplicate_variants", original[0])
        self.assertFalse(compact_documents(kept)[1])

    def test_numbers_conditions_versions_entities_and_acl_not_merged(self):
        cases = [("厚度18mm", "厚度20mm"), ("可以浸水", "不可以浸水"),
                 ("GB 2020", "GB 2021"), ("白麻饰面", "黄金麻饰面"),
                 ("not applicable", "notapplicable")]
        for a, b in cases:
            self.assertEqual(len(compact_documents([row("a", a), row("b", b)])[0]), 2)
        self.assertEqual(len(compact_documents([row("a", "厚度20mm"), row("b", "厚度20mm", "internal")])[0]), 2)
        self.assertEqual(len(compact_documents([row("a", "尺寸50mm"), row("b", "尺寸50mm", doc="other")])[0]), 2)

    def test_authority_and_image_ids_not_collapsed(self):
        a, b = row("a", "厚度20mm"), row("b", "厚度20mm")
        a["source_taxonomy"] = [{"source_authority": "official_standard"}]
        b["source_taxonomy"] = [{"source_authority": "manufacturer"}]
        self.assertFalse(compact_documents([a, b])[1])
        a["kind"] = b["kind"] = "visual"
        self.assertEqual(len(compact_documents([a, b])[0]), 2)

    def test_aliases_respect_retriever_acl(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lexical_index.json"
            path.write_text(json.dumps({"documents": [row("public", "公开资料"),
                row("private", "内部资料", "internal")],
                "document_aliases": {"old_public": "public", "old_private": "private"}}), encoding="utf-8")
            retriever = LocalRagRetriever(index_path=path)
            self.assertIn("old_public", retriever.document_position_by_id)
            self.assertNotIn("old_private", retriever.document_position_by_id)
