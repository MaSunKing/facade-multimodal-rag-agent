"""Regression checks for task-aware, local RAG routing.

These tests use the prepared local index only.  They never load the generation
model and therefore do not consume GPU memory.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["RAG_HYBRID_ENABLED"] = "0"

from backend.app import _fallback_question_plan
from backend.sales.retriever import LocalRagRetriever


class TaskRoutingTests(unittest.TestCase):
    def test_generic_task_fallbacks_are_grammar_based(self) -> None:
        cases = {
            "干挂的具体流程是什么？": "procedure",
            "窗洞口节点怎么做？": "node_detail",
            "有哪些山东项目？": "case_reference",
            "我的旧楼适合干挂吗？": "project_fit",
            "保温装饰一体板是什么？": "factual_lookup",
        }
        for question, expected_task_type in cases.items():
            with self.subTest(question=question):
                self.assertEqual(_fallback_question_plan(question)["task_type"], expected_task_type)

    def test_retriever_works_with_a_synthetic_public_index(self) -> None:
        document = {
            "id": "chunk_1",
            "kind": "text",
            "text": "施工流程包括基层检查、龙骨安装和板材固定。",
            "tokens": ["施工流程", "施工", "流程", "龙骨", "安装", "板材", "固定"],
            "source_refs": [{"document_name": "示例施工方案", "source_page": 3, "customer_shareable": True}],
        }
        payload = {
            "metadata": {"strategy": "test"},
            "average_document_length": len(document["tokens"]),
            "document_frequency": {token: 1 for token in document["tokens"]},
            "documents": [document],
        }
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "index.json"
            index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result = LocalRagRetriever(index_path).retrieve("龙骨安装流程", top_k=3, visual_k=0)

        self.assertEqual(result["text_evidence"][0]["chunk_id"], "chunk_1")
        self.assertIn("龙骨安装", result["text_evidence"][0]["text"])


if __name__ == "__main__":
    unittest.main()
