from __future__ import annotations

import json
import unittest

from backend.app import compact_grounded_payload_for_generation, recover_missing_rag_aspects
from backend.sales.context_engine import (
    choose_context_budget,
    optimise_evidence_context,
    validate_packed_evidence,
)
from backend.sales.tool_planner import ToolPlan


class _CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(str(item.get("content") or "") for item in messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


class ContextEngineTests(unittest.TestCase):
    def test_missing_planner_aspect_gets_one_bounded_subquery(self) -> None:
        class _Retriever:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def retrieve(self, query, **kwargs):
                self.queries.append(query)
                return {
                    "text_evidence": [
                        {"id": "fire", "text": "保温装饰一体板防火要求应按项目条件核验。"}
                    ]
                }

        retriever = _Retriever()
        result, queries = recover_missing_rag_aspects(
            retriever,
            {"text_evidence": [{"id": "base", "text": "旧楼外立面采用保温装饰一体板前应先检查基层。"}]},
            target_terms=["旧楼外立面", "保温装饰一体板", "防火"],
            retrieval_mode="project_fit",
        )
        self.assertIn("保温装饰一体板 防火", queries)
        self.assertEqual(len(result["text_evidence"]), 2)

    def test_recovered_aspects_are_reserved_before_generic_evidence(self) -> None:
        ranked, audit = optimise_evidence_context(
            "旧楼改造的锚固和防火怎么核验？",
            [
                {"evidence_id": "T1", "text": "旧楼改造应检查基层并按设计施工。"},
                {
                    "evidence_id": "T2",
                    "text": "外墙保温锚栓的数量和规格应结合基层与设计确定。",
                    "retrieval_aspect": "锚固",
                },
                {
                    "evidence_id": "T3",
                    "text": "建筑防火须结合建筑类型、高度、构造和设计文件判断。",
                    "retrieval_aspect": "防火",
                },
            ],
            target_terms=["旧楼改造", "锚固", "防火"],
        )
        self.assertEqual({item["evidence_id"] for item in ranked[:2]}, {"T2", "T3"})
        self.assertEqual(set(audit["reserved_retrieval_aspects"]), {"防火", "锚固"})

    def test_dynamic_budget_uses_source_demand_not_file_size(self) -> None:
        simple = choose_context_budget(
            ToolPlan(tools=["company_rag"]),
            has_documents=False,
            has_image=False,
        )
        cross_document = choose_context_budget(
            ToolPlan(
                tools=["customer_documents", "company_rag", "public_web_search"],
                document_scope="cross_document",
                target_terms=["黄金麻", "山东", "办公楼"],
                wants_visuals=True,
            ),
            has_documents=True,
            has_image=False,
        )
        self.assertGreater(cross_document.candidate_text_tokens, simple.candidate_text_tokens)
        self.assertGreaterEqual(cross_document.max_images, 2)
        self.assertNotIn("file_size", cross_document.components)

    def test_deduplicates_and_protects_table_condition(self) -> None:
        source = {
            "evidence_id": "U1",
            "text": "[TABLE sheet=参数] [COLUMNS] 产品|厚度 [ROW] 黄金麻厚度：8mm；雨天不得施工。",
            "source_group": "参数",
            "original_chunk_id": "table_1",
        }
        ranked, audit = optimise_evidence_context(
            "黄金麻厚度是多少，雨天能施工吗？",
            [source, {**source, "evidence_id": "T1"}],
            target_terms=["黄金麻"],
        )
        self.assertEqual(len(ranked), 1)
        relations = set(ranked[0]["protected_relation_types"])
        self.assertTrue({"entity_value", "table_row", "condition"}.issubset(relations))
        self.assertEqual(audit["removed_duplicate_evidence_ids"], ["T1"])
        self.assertTrue(audit["coverage_sufficient_before_generation"])

    def test_cross_source_raw_scores_are_not_compared(self) -> None:
        ranked, audit = optimise_evidence_context(
            "黄金麻规格",
            [
                {"evidence_id": "T1", "text": "黄金麻规格为1220×2440mm", "score": 0.1},
                {"evidence_id": "W1", "text": "其他产品宣传页面", "score": 9999},
            ],
            target_terms=["黄金麻"],
        )
        self.assertEqual(ranked[0]["evidence_id"], "T1")
        self.assertFalse(audit["raw_scores_compared_across_sources"])

    def test_conflicting_values_are_one_packing_group(self) -> None:
        ranked, audit = optimise_evidence_context(
            "黄金麻厚度是多少？",
            [
                {"evidence_id": "U1", "text": "黄金麻厚度：8mm"},
                {"evidence_id": "T1", "text": "黄金麻厚度：10mm"},
            ],
            target_terms=["黄金麻"],
        )
        self.assertEqual(len(audit["conflict_groups"]), 1)
        self.assertEqual(ranked[0]["packing_group_id"], ranked[1]["packing_group_id"])
        invalid = validate_packed_evidence(ranked, [ranked[0]])
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["violations"][0]["type"], "partial_conflict_group")

    def test_prompt_compaction_removes_complete_conflict_group(self) -> None:
        evidence, _ = optimise_evidence_context(
            "厚度",
            [
                {"evidence_id": "U0", "text": "简短说明"},
                {"evidence_id": "U1", "text": "黄金麻厚度：8mm" + "甲" * 180},
                {"evidence_id": "T1", "text": "黄金麻厚度：10mm" + "乙" * 180},
            ],
        )
        payload_text, audit = compact_grounded_payload_for_generation(
            {"evidence": evidence},
            _CharacterTokenizer(),
            max_prompt_tokens=600,
            system_prompt="",
        )
        kept = {item["evidence_id"] for item in json.loads(payload_text)["evidence"]}
        self.assertFalse(({"U1", "T1"} & kept) in ({"U1"}, {"T1"}))
        self.assertTrue(audit["semantic_group_packing_enabled"])
        self.assertTrue(audit["integrity_check"]["valid"])

    def test_company_rag_evidence_uses_the_same_exact_prompt_budget(self) -> None:
        payload_text, audit = compact_grounded_payload_for_generation(
            {
                "customer_question": "旧楼改造要注意什么？",
                "retrieved_text_evidence": [
                    {"evidence_id": "T1", "text": "选材应结合设计文件。" + "甲" * 400},
                    {"evidence_id": "T2", "text": "锚固和验收应按项目条件核验。" + "乙" * 400},
                ],
            },
            _CharacterTokenizer(),
            max_prompt_tokens=520,
            system_prompt="短规则",
        )
        packed = json.loads(payload_text)["retrieved_text_evidence"]
        self.assertTrue(packed)
        self.assertLessEqual(audit["actual_prompt_tokens"], 520)
        self.assertEqual(
            {item["evidence_id"] for item in packed},
            set(audit["kept_evidence_ids"]),
        )

    def test_prompt_budget_compacts_history_before_evidence(self) -> None:
        payload_text, audit = compact_grounded_payload_for_generation(
            {
                "customer_question": "请根据证据回答",
                "conversation_context": ["历史" * 900, "补充" * 900, "最新" * 900],
                "retrieved_text_evidence": [
                    {"evidence_id": "T1", "text": "防火与节能必须分别按相应标准核验。"},
                    {"evidence_id": "T2", "text": "锚栓规格和数量须结合基层与设计确定。"},
                ],
            },
            _CharacterTokenizer(),
            max_prompt_tokens=420,
            system_prompt="短规则",
        )
        packed = json.loads(payload_text)
        self.assertTrue(audit["budget_satisfied"])
        self.assertEqual(
            {item["evidence_id"] for item in packed["retrieved_text_evidence"]},
            {"T1", "T2"},
        )
        self.assertTrue(audit["non_evidence_compaction_stages"])


if __name__ == "__main__":
    unittest.main()
