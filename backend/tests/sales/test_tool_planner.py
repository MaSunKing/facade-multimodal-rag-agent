from __future__ import annotations

import json
import unittest

from backend.sales.tool_planner import guard_plan
from backend.app import (
    DraftRequest,
    VISUAL_NUMERIC_REVIEW_WARNING,
    apply_visual_observation_caveat,
    compact_grounded_payload_for_generation,
    evidence_support_audit,
    has_visual_grounding,
    is_safe_grounded_answer,
    normalise_nonfactual_output_fields,
    parse_json,
    recover_truncated_grounded_json,
    repair_uploaded_attachment_availability_claims,
    remove_unsupported_numeric_sentences,
)


class ToolPlannerPolicyTests(unittest.TestCase):
    def test_parsed_attachment_is_not_described_as_missing(self) -> None:
        result = {
            "answerable": False,
            "customer_reply": "当前知识库中未找到可分析的文件或图片内容。请重新上传有效文件或图片后再试。",
            "missing_information": ["未提供可分析的文件或图片"],
            "next_action": "请重新上传文件或图片。",
            "key_points": [],
            "citations": [],
        }
        document_result = {
            "documents": [{"file_name": "测试资料.xlsx"}],
            "input_snapshot": {
                "selected_document_index_window_count": 1,
                "selected_content_window_count": 0,
            },
        }

        repaired = repair_uploaded_attachment_availability_claims(result, document_result)

        self.assertIn("附件已上传并完成解析", repaired["customer_reply"])
        self.assertIn("测试资料.xlsx", repaired["customer_reply"])
        self.assertIn("无需重新上传", repaired["next_action"])
        self.assertNotIn("未提供可分析", "".join(repaired["missing_information"]))

    def test_true_missing_session_language_is_not_repaired(self) -> None:
        result = {
            "answerable": False,
            "customer_reply": "请重新上传文件或图片。",
            "missing_information": [],
            "next_action": "请重新上传。",
        }

        repaired = repair_uploaded_attachment_availability_claims(result, {"documents": []})

        self.assertEqual(repaired["customer_reply"], "请重新上传文件或图片。")

    def test_unsupported_numeric_appendix_is_removed_without_changing_supported_summary(self) -> None:
        result = {
            "customer_reply": "工作簿包含159个工作表。部分表格显示12个月截止数据。",
            "key_points": [],
            "risk_warnings": [],
        }

        filtered = remove_unsupported_numeric_sentences(result, ["12"])

        self.assertEqual(filtered["customer_reply"], "工作簿包含159个工作表。")
        self.assertIn("已省略", filtered["risk_warnings"][0])

    def test_numeric_filter_keeps_supported_totals_before_unsupported_ratio(self) -> None:
        result = {
            "customer_reply": (
                "2. 费用：行政费用795,968.46元，而业务费用212,604.39元，"
                "行政费用占比79.5%。"
            ),
            "key_points": [],
            "risk_warnings": [],
        }

        filtered = remove_unsupported_numeric_sentences(result, ["79.5%"])

        self.assertIn("行政费用795,968.46元", filtered["customer_reply"])
        self.assertIn("业务费用212,604.39元", filtered["customer_reply"])
        self.assertNotIn("79.5%", filtered["customer_reply"])
        self.assertNotIn("2. 费用", filtered["customer_reply"])

    def test_grounded_payload_compaction_uses_real_prompt_budget_without_mutating_source(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                return "\n".join(str(message["content"]) for message in messages)

            def __call__(self, text, **_kwargs):
                return {"input_ids": list(range(text.count("x") + 100))}

        payload = {
            "customer_question": "test",
            "evidence": [
                {"evidence_id": f"E{index}", "text": "x" * 100}
                for index in range(1, 6)
            ],
        }
        original_count = len(payload["evidence"])

        _payload_text, audit = compact_grounded_payload_for_generation(
            payload,
            FakeTokenizer(),
            max_prompt_tokens=350,
        )

        self.assertEqual(len(payload["evidence"]), original_count)
        self.assertLess(audit["kept_evidence_count"], original_count)
        self.assertLessEqual(audit["actual_prompt_tokens"], 350)

    def test_grounded_payload_compaction_keeps_table_tail_before_structure_index(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                return "\n".join(str(message["content"]) for message in messages)

            def __call__(self, text, **_kwargs):
                weighted_length = sum(
                    text.count(marker) for marker in ("XMARK", "YMARK", "ZMARK")
                ) + 100
                return {"input_ids": list(range(weighted_length))}

        payload = {
            "customer_question": "analyse",
            "evidence": [
                {
                    "evidence_id": "U1",
                    "evidence_scope": "document_index",
                    "original_chunk_id": "index",
                    "source_refs": [{"large": "metadata"}],
                    "text": "structure" * 20,
                },
                {
                    "evidence_id": "U2",
                    "evidence_scope": "content",
                    "original_chunk_id": "table-a",
                    "text": "[ROW source=excel;sheet=S;section=T] head=1 " + "XMARK" * 80,
                },
                {
                    "evidence_id": "U3",
                    "evidence_scope": "content",
                    "original_chunk_id": "table-b",
                    "text": "other" + "ZMARK" * 150,
                },
                {
                    "evidence_id": "U4",
                    "evidence_scope": "content",
                    "original_chunk_id": "table-a",
                    "text": "[ROW source=excel;sheet=S;section=T] total=99 " + "YMARK" * 30,
                },
            ],
        }

        payload_text, audit = compact_grounded_payload_for_generation(
            payload,
            FakeTokenizer(),
            max_prompt_tokens=250,
        )

        compacted = json.loads(payload_text)
        self.assertEqual(audit["kept_evidence_ids"][:2], ["U2", "U4"])
        self.assertEqual([item["evidence_id"] for item in compacted["evidence"]], ["U2", "U4"])
        self.assertNotIn("source_refs", compacted["evidence"][0])
        self.assertNotIn("source=excel", compacted["evidence"][0]["text"])

    def test_whole_document_compaction_reserves_one_structure_index(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                return "\n".join(str(message["content"]) for message in messages)

            def __call__(self, text, **_kwargs):
                return {"input_ids": list(range(text.count("TOKEN") + 10))}

        payload = {
            "attachment_context": {"global_document_question": True},
            "evidence": [
                {"evidence_id": "U1", "evidence_scope": "document_index", "text": "TOKEN"},
                {"evidence_id": "U2", "evidence_scope": "content", "text": "TOKEN"},
            ],
        }

        payload_text, _audit = compact_grounded_payload_for_generation(
            payload,
            FakeTokenizer(),
            max_prompt_tokens=100,
        )

        compacted = json.loads(payload_text)
        self.assertEqual(
            [item["evidence_id"] for item in compacted["evidence"]],
            ["U1", "U2"],
        )

    def test_json_parser_uses_first_complete_object_when_model_adds_commentary(self) -> None:
        parsed = parse_json(
            'Here is the plan: {"tools":["general_chat"],"reason":"ordinary question"}'
            ' Extra example: {"tools":["public_web_search"]}'
        )
        self.assertEqual(parsed["tools"], ["general_chat"])
        self.assertEqual(parsed["reason"], "ordinary question")

    def test_truncated_grounded_json_recovery_requires_complete_reply_and_citations(self) -> None:
        raw = (
            '{"intent":"document_qa","normalized_terms":[],"answerable":true,'
            '"customer_reply":"有依据的分析。","key_points":[],'
            '"citations":[{"evidence_id":"U1"}],"missing_information":[],'
            '"risk_warnings":"'
        )

        recovered = recover_truncated_grounded_json(raw)

        self.assertEqual(recovered["customer_reply"], "有依据的分析。")
        self.assertEqual(recovered["citations"], [{"evidence_id": "U1"}])
        self.assertEqual(recovered["risk_warnings"], [])
        self.assertIsInstance(recovered["next_action"], str)

    def test_harmless_empty_next_action_list_is_normalised_to_string(self) -> None:
        value = normalise_nonfactual_output_fields(
            {"normalized_terms": ["黄金麻"], "next_action": [], "image_observations": []}
        )
        self.assertIsInstance(value["next_action"], str)
        self.assertTrue(value["next_action"])
        self.assertEqual(
            value["normalized_terms"],
            [{"term": "黄金麻", "normalized": "黄金麻"}],
        )

    def test_explicit_missing_information_can_fill_empty_refusal_reply(self) -> None:
        value = normalise_nonfactual_output_fields(
            {
                "answerable": False,
                "customer_reply": "",
                "normalized_terms": [],
                "missing_information": ["当前知识库未找到已审核且匹配的图片。"],
                "next_action": "可补充具体型号。",
                "image_observations": [],
            }
        )
        self.assertEqual(value["customer_reply"], "当前知识库未找到已审核且匹配的图片。")

    def test_guard_removes_unavailable_and_unapproved_tools(self) -> None:
        plan = guard_plan(
            {"tools": ["public_web_search", "customer_documents", "company_rag"], "reason": "test"},
            has_documents=False,
            has_image=False,
            facade_related=True,
            web_allowed=False,
        )
        self.assertEqual(plan.tools, ["company_rag"])

    def test_uploaded_documents_are_not_silently_ignored(self) -> None:
        plan = guard_plan(
            {"tools": ["general_chat"], "reason": "bad proposal"},
            has_documents=True,
            has_image=False,
            facade_related=False,
            web_allowed=False,
        )
        self.assertEqual(plan.tools[0], "customer_documents")

    def test_web_permission_alone_does_not_select_search(self) -> None:
        plan = guard_plan(
            {
                "tools": ["general_chat", "public_web_search"],
                "requires_public_web": False,
                "reason": "casual greeting",
            },
            has_documents=False,
            has_image=False,
            facade_related=False,
            web_allowed=True,
        )
        self.assertEqual(plan.tools, ["general_chat"])
        self.assertFalse(plan.requires_public_web)

    def test_semantically_required_and_permitted_web_search_is_retained(self) -> None:
        plan = guard_plan(
            {
                "tools": ["public_web_search"],
                "requires_public_web": True,
                "web_source_profile": "public_project",
                "reason": "current named public project",
            },
            has_documents=False,
            has_image=False,
            facade_related=False,
            web_allowed=True,
        )
        self.assertEqual(plan.tools, ["public_web_search"])
        self.assertTrue(plan.requires_public_web)
        self.assertEqual(plan.web_source_profile, "public_project")

    def test_model_can_select_local_company_rag_without_keyword_gate(self) -> None:
        plan = guard_plan(
            {
                "tools": ["company_rag"],
                "requires_public_web": False,
                "reason": "the user refers to our products",
            },
            has_documents=False,
            has_image=False,
            # "Introduce your products" contains no facade-specific keyword.
            facade_related=False,
            web_allowed=False,
        )
        self.assertEqual(plan.tools, ["company_rag"])

    def test_model_business_and_visual_semantics_survive_policy_guard(self) -> None:
        plan = guard_plan(
            {
                "tools": ["company_rag"],
                "intent": "construction",
                "task_type": "node_detail",
                "retrieval_query": "窗洞口 收口 节点图",
                "target_terms": ["窗洞口"],
                "wants_visuals": True,
                "visual_scope": "node",
                "reason": "node image request",
            },
            has_documents=False,
            has_image=False,
            facade_related=False,
            web_allowed=False,
        )
        self.assertEqual(plan.task_type, "node_detail")
        self.assertEqual(plan.retrieval_query, "窗洞口 收口 节点图")
        self.assertTrue(plan.wants_visuals)
        self.assertEqual(plan.visual_scope, "node")

    def test_product_overview_contract_adds_required_local_source(self) -> None:
        plan = guard_plan(
            {
                "tools": ["general_chat"],
                "intent": "product_parameter",
                "task_type": "factual_lookup",
                "product_overview": True,
                "reason": "company catalogue",
            },
            has_documents=False,
            has_image=False,
            facade_related=False,
            web_allowed=False,
        )
        self.assertIn("company_rag", plan.tools)
        self.assertTrue(plan.product_overview)

    def test_support_audit_flags_numbers_absent_from_cited_evidence(self) -> None:
        audit = evidence_support_audit(
            {
                "customer_reply": "建议使用123个锚固件。",
                "key_points": ["锚固件数量为123"],
                "citations": [{"evidence_id": "U1"}],
            },
            {"U1": {"text": "施工说明要求使用100个锚固件。"}},
        )
        self.assertEqual(audit["unsupported_numeric_claims"], ["123"])
        self.assertFalse(audit["passed"])

    def test_support_audit_accepts_thousands_separator_and_source_rounding_only(self) -> None:
        audit = evidence_support_audit(
            {
                "customer_reply": (
                    "收入为426,632.23元，利润为-581,940.62元；费用占比79.5%。"
                ),
                "key_points": [],
                "citations": [{"evidence_id": "U1"}],
            },
            {
                "U1": {
                    "text": "收入426632.23，利润-581940.616，行政费用795968.456。"
                }
            },
        )

        self.assertEqual(audit["unsupported_numeric_claims"], ["79.5%"])
        self.assertFalse(audit["passed"])

    def test_support_audit_understands_period_ranges_and_loss_magnitudes(self) -> None:
        audit = evidence_support_audit(
            {
                "customer_reply": "2025年1-12月合计亏损581,940.62元。",
                "key_points": [],
                "citations": [{"evidence_id": "U1"}],
            },
            {
                "U1": {
                    "text": "2025.1-2025.12；1月；12月；合计利润-581940.616。"
                }
            },
        )

        self.assertEqual(audit["unsupported_numeric_claims"], [])
        self.assertTrue(audit["passed"])

    def test_visual_numeric_transcription_is_reviewable_not_rejected(self) -> None:
        result = {
            "customer_reply": "图中公式的结果写为1/4。",
            "key_points": [],
            "citations": [{"evidence_id": "V1"}],
            "image_observations": ["页面中直接显示概率结果1/4。"],
            "risk_warnings": [],
        }
        audit = evidence_support_audit(
            result,
            {
                "V1": {
                    "text": "客户上传的数学题解答页面",
                    "visual_direct_observation": True,
                }
            },
            allow_visual_observation=True,
        )
        self.assertEqual(audit["unsupported_numeric_claims"], [])
        self.assertEqual(
            audit["visual_numeric_claims_requiring_review"], ["1", "4"]
        )
        self.assertTrue(audit["visual_observation_grounded"])
        self.assertTrue(audit["passed"])
        repaired = apply_visual_observation_caveat(result, audit)
        self.assertIn(VISUAL_NUMERIC_REVIEW_WARNING, repaired["risk_warnings"])

    def test_document_qa_image_only_answer_is_schema_valid(self) -> None:
        result = {
            "intent": "document_qa",
            "normalized_terms": [],
            "answerable": True,
            "customer_reply": "这是一页英文数学解答。",
            "key_points": [],
            "citations": [],
            "missing_information": [],
            "risk_warnings": [],
            "next_action": "如需可继续转录公式。",
            "image_observations": ["页面包含英文说明和数学公式。"],
        }
        self.assertTrue(
            is_safe_grounded_answer(result, {"V1"}, allow_image_only=True)
        )

    def test_session_visual_counts_as_visual_grounding_without_resending_base64(self) -> None:
        request = DraftRequest(customer_question="请说明已上传图片的内容")

        self.assertTrue(
            has_visual_grounding(
                request,
                [{"evidence_id": "image:document:1", "document_name": "sample.png"}],
            )
        )
        self.assertFalse(has_visual_grounding(request, []))


if __name__ == "__main__":
    unittest.main()
