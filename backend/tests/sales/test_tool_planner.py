from __future__ import annotations

import json
import unittest

from backend.sales.tool_planner import fallback_plan, guard_plan
from backend.sales.context_engine import ContextBudget
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
    repair_session_visual_citations,
    repair_uploaded_attachment_availability_claims,
    remove_unsupported_numeric_sentences,
    sanitize_customer_document_presentation,
    _select_customer_visual_inputs,
)


class ToolPlannerPolicyTests(unittest.TestCase):
    def test_global_multi_image_question_respects_configured_two_image_limit(self) -> None:
        budget = ContextBudget(
            candidate_text_tokens=6000,
            max_prompt_tokens=4650,
            max_images=2,
            max_output_tokens=720,
            components={},
        )
        visuals = [{"visual_id": f"image:{index}"} for index in range(1, 5)]
        selected = _select_customer_visual_inputs(
            {
                "selected_visuals": visuals,
                "input_snapshot": {"global_document_question": True},
            },
            budget,
        )
        self.assertEqual([item["visual_id"] for item in selected], ['image:1', 'image:2'])

    def test_local_visual_lookup_keeps_normal_gpu_image_budget(self) -> None:
        budget = ContextBudget(
            candidate_text_tokens=6000,
            max_prompt_tokens=4200,
            max_images=2,
            max_output_tokens=560,
            components={},
        )
        visuals = [{"visual_id": f"image:{index}"} for index in range(1, 5)]
        selected = _select_customer_visual_inputs(
            {
                "selected_visuals": visuals,
                "input_snapshot": {"global_document_question": False},
            },
            budget,
        )
        self.assertEqual(len(selected), 2)

    def test_customer_document_presentation_hides_ids_and_preserves_column_semantics(self) -> None:
        result = {
            "customer_reply": "U1为《预算表》。U9中墙面乳胶漆单价119.8元/m²，公式未执行。",
            "key_points": ["U10显示乳胶漆单价119.8元/m²。"],
            "risk_warnings": [],
        }
        evidence = {
            "U9": {
                "document_name": "预算.xlsx",
                "text": (
                    "[COLUMNS] A=序号 | B=项目名称 | C=单位 | D=工程量 | "
                    "E=单价（元） | F=合计（元） | G=备注\n"
                    "[ROW] A10[序号]='1' | B10[项目名称]='乳胶漆' | C10[单位]='m²' | "
                    "D10[工程量]='119.8' | E10[单价（元）]='' | F10[合计（元）]='0'"
                )
            }
        }

        sanitized = sanitize_customer_document_presentation(result, evidence)

        self.assertNotRegex(sanitized["customer_reply"], r"\bU\d+\b")
        self.assertNotIn("119.8元/m²", sanitized["customer_reply"])
        self.assertIn("工程量为119.8m²（单价未填写）", sanitized["customer_reply"])
        self.assertNotIn("公式未执行", sanitized["customer_reply"])
        self.assertIn("工程量与单价", sanitized["risk_warnings"][-1])

    def test_customer_document_presentation_repairs_sheet_count_and_signed_pair(self) -> None:
        result = {
            "customer_reply": "第三份为行政费用表。3月、4月利润为正（6978.88元、-14140.71元）。",
            "key_points": [],
            "risk_warnings": [],
        }
        evidence = {
            "U1": {"document_name": "财务.xlsx", "text": "利润表"},
            "U2": {"document_name": "预算.xlsx", "text": "预算表"},
        }

        sanitized = sanitize_customer_document_presentation(result, evidence)

        self.assertIn("另一个工作表为行政费用表", sanitized["customer_reply"])
        self.assertIn("3月利润为正（6978.88元）", sanitized["customer_reply"])
        self.assertIn("4月利润为负（-14140.71元）", sanitized["customer_reply"])

    def test_customer_document_presentation_neutralizes_unbenchmarked_judgements(self) -> None:
        result = {
            "customer_reply": "行政费用占比过高，建议拆分高成本项目。",
            "key_points": ["行政费用支出占比高"],
            "risk_warnings": ["大额工资报销，利润持续亏损"],
        }
        sanitized = sanitize_customer_document_presentation(
            result, {"U1": {"document_name": "经营表.xlsx", "text": "行政费用 工资报销 利润"}}
        )

        combined = "\n".join([
            sanitized["customer_reply"],
            *sanitized["key_points"],
            *sanitized["risk_warnings"],
        ])
        self.assertNotIn("占比过高", combined)
        self.assertNotIn("高成本项目", combined)
        self.assertNotIn("大额工资报销", combined)
        self.assertNotIn("持续亏损", combined)

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

    def test_numeric_filter_removes_complete_numbered_item_without_leaving_fragments(self) -> None:
        result = {
            "customer_reply": (
                "初步分析：1）收入下降；2）预算缺少单价；"
                "3）行政费用约11.88万元，建议优化成本。"
                "优化建议：1）复核亏损月份；2）补充市场报价；"
                "3）某项占比13.22%，建议核验。"
            ),
            "key_points": [],
            "risk_warnings": [],
        }

        filtered = remove_unsupported_numeric_sentences(result, ["11.88", "13.22%"])

        self.assertIn("收入下降", filtered["customer_reply"])
        self.assertIn("预算缺少单价", filtered["customer_reply"])
        self.assertIn("复核亏损月份", filtered["customer_reply"])
        self.assertIn("补充市场报价", filtered["customer_reply"])
        self.assertNotRegex(filtered["customer_reply"], r"(?:^|[；;。:：])\s*[23][）.]" )
        self.assertNotIn("优化成本", filtered["customer_reply"])
        self.assertNotIn("13.22%", filtered["customer_reply"])

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

    def test_cross_document_compaction_keeps_content_from_each_file(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                return "\n".join(str(message["content"]) for message in messages)

            def __call__(self, text, **_kwargs):
                return {"input_ids": list(range(text.count("TOKEN") + 20))}

        payload = {
            "attachment_context": {"global_document_question": True},
            "evidence": [
                {
                    "evidence_id": "A1",
                    "document_name": "A.xlsx",
                    "evidence_scope": "document_index",
                    "text": "A index " + "TOKEN" * 20,
                },
                {
                    "evidence_id": "A2",
                    "document_name": "A.xlsx",
                    "evidence_scope": "content",
                    "source_group": "A-sheet-1",
                    "text": "A values " + "TOKEN" * 40,
                },
                {
                    "evidence_id": "A3",
                    "document_name": "A.xlsx",
                    "evidence_scope": "content",
                    "source_group": "A-sheet-2",
                    "text": "A more values " + "TOKEN" * 40,
                },
                {
                    "evidence_id": "B1",
                    "document_name": "B.xlsx",
                    "evidence_scope": "document_index",
                    "text": "B index " + "TOKEN" * 20,
                },
                {
                    "evidence_id": "B2",
                    "document_name": "B.xlsx",
                    "evidence_scope": "content",
                    "source_group": "B-sheet-1",
                    "text": "B values " + "TOKEN" * 40,
                },
            ],
        }

        payload_text, audit = compact_grounded_payload_for_generation(
            payload,
            FakeTokenizer(),
            max_prompt_tokens=105,
        )

        compacted = json.loads(payload_text)
        kept_content_documents = {
            item["document_name"]
            for item in compacted["evidence"]
            if item.get("evidence_scope") == "content"
        }
        self.assertEqual(kept_content_documents, {"A.xlsx", "B.xlsx"})
        self.assertEqual(audit["kept_document_count"], 2)

    def test_cross_document_compaction_interleaves_sheet_representatives_before_indexes(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                return "\n".join(str(message["content"]) for message in messages)

            def __call__(self, text, **_kwargs):
                return {"input_ids": list(range(text.count("TOKEN") + 10))}

        payload = {
            "attachment_context": {"global_document_question": True},
            "evidence": [
                {"evidence_id": "A0", "document_name": "A.xlsx", "evidence_scope": "document_index", "text": "TOKEN"},
                {"evidence_id": "A1", "document_name": "A.xlsx", "evidence_scope": "content", "source_group": "费用", "text": "TOKEN"},
                {"evidence_id": "A2", "document_name": "A.xlsx", "evidence_scope": "content", "source_group": "利润", "text": "TOKEN"},
                {"evidence_id": "B0", "document_name": "B.xlsx", "evidence_scope": "document_index", "text": "TOKEN"},
                {"evidence_id": "B1", "document_name": "B.xlsx", "evidence_scope": "content", "source_group": "预算", "text": "TOKEN"},
            ],
        }

        payload_text, _audit = compact_grounded_payload_for_generation(
            payload, FakeTokenizer(), max_prompt_tokens=100
        )
        ids = [item["evidence_id"] for item in json.loads(payload_text)["evidence"]]
        self.assertEqual(ids[:3], ["A2", "B1", "A1"])

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

    def test_omitted_presentation_only_lists_are_safely_added(self) -> None:
        value = normalise_nonfactual_output_fields(
            {
                "intent": "application_condition",
                "answerable": True,
                "citations": [{"evidence_id": "T1"}],
                "customer_reply": "旧楼改造需先核验基层。",
                "key_points": [],
                "missing_information": [],
                "risk_warnings": [],
                "next_action": "补充基层检测资料。",
            }
        )

        self.assertEqual(value["normalized_terms"], [])
        self.assertEqual(value["image_observations"], [])
        self.assertTrue(is_safe_grounded_answer(value, {"T1"}))

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

    def test_attachment_availability_does_not_override_semantic_plan(self) -> None:
        plan = guard_plan(
            {"tools": ["general_chat"], "reason": "unrelated current question"},
            has_documents=True,
            has_image=False,
            facade_related=False,
            web_allowed=False,
        )
        self.assertEqual(plan.tools, ["general_chat"])

    def test_fallback_does_not_treat_an_old_document_session_as_relevant(self) -> None:
        plan = fallback_plan(
            has_documents=True,
            has_image=False,
            facade_related=False,
            web_requested=False,
        )
        self.assertEqual(plan.tools, ["general_chat"])

    def test_fallback_uses_document_when_semantic_scope_is_known(self) -> None:
        plan = fallback_plan(
            has_documents=True,
            has_image=False,
            facade_related=False,
            web_requested=False,
            document_scope="local_lookup",
        )
        self.assertEqual(plan.tools, ["customer_documents"])

    def test_explicit_document_scope_repairs_missing_document_tool(self) -> None:
        plan = guard_plan(
            {
                "tools": ["general_chat"],
                "document_scope": "whole_document",
                "retrieval_query": "summarise the uploaded report",
                "reason": "uploaded report analysis",
            },
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

    def test_support_audit_rejects_uncited_promotional_superlative(self) -> None:
        audit = evidence_support_audit(
            {
                "customer_reply": "该方案维护成本更低，是市场最佳选择。",
                "key_points": [],
                "citations": [{"evidence_id": "T1"}],
            },
            {"T1": {"text": "该系统可用于外墙装饰，具体选型应结合项目条件。"}},
        )
        self.assertEqual(
            audit["unsupported_promotional_claims"],
            ["维护成本更低", "最佳"],
        )
        self.assertFalse(audit["passed"])

    def test_support_audit_rejects_fire_energy_standard_conflation(self) -> None:
        audit = evidence_support_audit(
            {
                "customer_reply": "防火依据建筑节能工程施工质量验收标准执行。",
                "key_points": [],
                "citations": [{"evidence_id": "T1"}, {"evidence_id": "T2"}],
            },
            {
                "T1": {"text": "GB 50411-2019 是建筑节能工程施工质量验收标准。"},
                "T2": {"text": "GB 55037-2022 是建筑防火通用规范。"},
            },
        )
        self.assertEqual(
            audit["incompatible_standard_scope_claims"],
            ["fire_claim_bound_to_energy_standard"],
        )
        self.assertFalse(audit["passed"])

    def test_support_audit_ignores_numbered_recommendation_markers(self) -> None:
        audit = evidence_support_audit(
            {
                "customer_reply": "建议：1）核验预算；2）复核合同；3. 建立月度台账。",
                "key_points": [],
                "citations": [{"evidence_id": "U1"}],
            },
            {"U1": {"text": "预算 合同 月度台账"}},
        )

        self.assertEqual(audit["unsupported_numeric_claims"], [])

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

    def test_multi_image_answer_can_keep_visually_grounded_key_points(self) -> None:
        result = {
            "intent": "document_qa",
            "normalized_terms": [],
            "answerable": True,
            "customer_reply": "四张截图均为概率论材料，分别涉及集合、样本空间和概率计算。",
            "key_points": ["材料主题为概率论", "包含集合与概率计算"],
            "citations": [],
            "missing_information": [],
            "risk_warnings": [],
            "next_action": "可继续逐题讲解。",
            "image_observations": ["四张图片中均可见英文数学文字与公式。"],
        }
        self.assertTrue(
            is_safe_grounded_answer(result, {"V1", "V2", "V3", "V4"}, allow_image_only=True)
        )
        self.assertFalse(
            is_safe_grounded_answer(result, {"V1", "V2", "V3", "V4"}, allow_image_only=False)
        )

    def test_dropped_document_wrapper_citation_maps_to_same_shown_visual(self) -> None:
        result = {
            "citations": [
                {"evidence_id": "U1"},
                {"evidence_id": "U3"},
                {"evidence_id": "UNKNOWN"},
            ]
        }
        repaired = repair_session_visual_citations(
            result,
            model_visible_evidence_ids={"U1", "V1", "V2"},
            evidence_by_id={
                "U1": {"document_name": "第一页.png"},
                "U3": {"document_name": "第二页.png"},
            },
            selected_visuals=[
                {"document_name": "第一页.png"},
                {"document_name": "第二页.png"},
            ],
        )
        self.assertEqual(
            repaired["citations"],
            [
                {"evidence_id": "U1"},
                {"evidence_id": "V2"},
                {"evidence_id": "UNKNOWN"},
            ],
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
