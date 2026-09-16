"""Regression checks for task-aware, local RAG routing.

These tests use the prepared local index only.  They never load the generation
model and therefore do not consume GPU memory.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ["RAG_HYBRID_ENABLED"] = "0"

from backend.app import (
    DraftRequest,
    _fallback_question_plan,
    _retrieval_with_catalog_evidence,
    _scope_evidence_for_plan,
    _semantic_complete_excerpt,
    _names_specific_procedure,
    _single_evidence_answer,
    apply_case_filters,
    catalogue_case_answer,
    customer_visible_retrieval,
    direct_authoritative_fact_evidence,
    direct_numeric_evidence,
    dynamic_private_business_data_kind,
    is_facade_domain_request,
    load_local_runtime_environment,
    question_visual_scope,
    question_wants_visuals,
    resolve_current_public_query,
    resolve_visual_request,
    sanitize_direct_visual_output,
)
from backend.sales.retriever import LocalRagRetriever, is_product_overview_query


@unittest.skipUnless(os.getenv("RUN_ENTERPRISE_DATA_TESTS") == "1", "Requires separately authorised enterprise index; public core checks run without business data")
class TaskRoutingTests(unittest.TestCase):
    def test_clear_topic_switch_does_not_inherit_facade_domain_from_history(self) -> None:
        request = DraftRequest(
            customer_question="今天青岛天气如何",
            conversation_context=[
                {"role": "user", "content": "介绍一下黄金麻产品"},
                {"role": "assistant", "content": "黄金麻属于外墙装饰产品。"},
            ],
        )
        self.assertFalse(is_facade_domain_request(request))

    def test_local_environment_loader_preserves_explicit_process_value(self) -> None:
        variable = "FACADE_TEST_LOCAL_ENV_VALUE"
        previous = os.environ.get(variable)
        try:
            os.environ[variable] = "process"
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "facade.local.env"
                path.write_text(f"{variable}=file\n", encoding="utf-8")
                load_local_runtime_environment(path)
            self.assertEqual(os.environ[variable], "process")
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous

    def test_non_case_plan_excludes_case_labelled_evidence(self) -> None:
        evidence = [
            {"text": "黄金麻产品介绍", "content_labels": ["product_information"]},
            {
                "text": "项目类型：办公楼 text 使用产品：黄金麻",
                "content_labels": ["product_information", "project_case_reference"],
            },
        ]
        scoped = _scope_evidence_for_plan(evidence, include_project_cases=False)
        self.assertEqual([item["text"] for item in scoped], ["黄金麻产品介绍"])

    def test_case_titled_web_evidence_is_excluded_from_company_profile_question(self) -> None:
        evidence = [
            {
                "text": "北京祥瑞生产基地项目",
                "citations": [
                    {
                        "document_name": "北京-祥瑞生产基地项目-真岩案例",
                        "section_heading": "项目详情",
                    }
                ],
            },
            {"text": "5万平方米自有生产基地", "citations": []},
        ]
        scoped = _scope_evidence_for_plan(evidence, include_project_cases=False)
        self.assertEqual([item["text"] for item in scoped], ["5万平方米自有生产基地"])

    def test_relative_web_query_is_bound_to_current_china_date(self) -> None:
        resolved = resolve_current_public_query("今天青岛天气如何")
        self.assertIn("当前日期", resolved)
        self.assertRegex(resolved, r"\d{4}-\d{2}-\d{2}$")
        self.assertEqual(resolve_current_public_query("青岛历史气候"), "青岛历史气候")

    def test_fallback_excerpt_never_slices_a_complete_relation(self) -> None:
        long_clause = "产品黄金麻适用外墙装饰" * 120 + "。"
        excerpt = _semantic_complete_excerpt(long_clause, preferred_chars=80)
        self.assertEqual(excerpt, long_clause)
        self.assertTrue(excerpt.endswith("。"))

    def test_structured_catalogue_record_precedes_raw_retrieval_for_fallback(self) -> None:
        combined = _retrieval_with_catalog_evidence(
            {"text_evidence": [{"evidence_id": "T1", "text": "网页原文"}]},
            [{"evidence_id": "S1", "text": "结构化产品资料", "catalog_record_type": "product"}],
        )
        self.assertEqual(combined["text_evidence"][0]["evidence_id"], "S1")

    @staticmethod
    def _mixed_retrieval_fixture() -> dict:
        return {
            "text_evidence": [
                {
                    "text": "实际用于回答的施工条款。",
                    "citations": [
                        {
                            "document_name": "施工方案",
                            "source_page": 8,
                            "section_heading": "施工要求",
                        }
                    ],
                },
                {
                    "text": "未被本次回答引用的其他条款。",
                    "citations": [
                        {
                            "document_name": "通用规范",
                            "source_page": 20,
                            "section_heading": "一般规定",
                        }
                    ],
                },
            ],
            "project_cases": [
                {
                    "project_name": "无关案例",
                    "project_type": "办公楼",
                    "product": "示例产品",
                    "citation": {"document_name": "项目画册", "source_page": 30},
                }
            ],
            "visual_assets": [],
            "meta": {"strategy": "test_mixed_retrieval"},
        }

    def test_visible_retrieval_follows_actual_text_citation_not_project_candidates(self) -> None:
        visible = customer_visible_retrieval(
            self._mixed_retrieval_fixture(),
            citation_ids=["T1"],
        )
        self.assertEqual(visible["result_count"], 1)
        self.assertEqual(visible["supporting_results"][0]["result_id"], "T1")
        self.assertEqual(visible["supporting_results"][0]["document_name"], "施工方案")
        self.assertNotIn("无关案例", visible["supporting_results"][0]["excerpt"])

    def test_visible_retrieval_with_no_citations_exposes_no_supporting_rows(self) -> None:
        visible = customer_visible_retrieval(
            self._mixed_retrieval_fixture(),
            citation_ids=[],
        )
        self.assertEqual(visible["result_count"], 0)
        self.assertEqual(visible["supporting_results"], [])

    def test_catalogue_case_track_keeps_case_ids_and_rows_aligned(self) -> None:
        retrieval = self._mixed_retrieval_fixture()
        response = catalogue_case_answer(
            DraftRequest(customer_question="有哪些项目案例？"),
            retrieval,
        )
        citation_ids = [item["evidence_id"] for item in response["citations"]]
        result_ids = [item["result_id"] for item in response["retrieval"]["supporting_results"]]
        self.assertEqual(citation_ids, ["C1"])
        self.assertEqual(result_ids, citation_ids)
        self.assertEqual(
            response["retrieval"]["supporting_results"][0]["section_heading"],
            "无关案例",
        )

    def test_dynamic_private_business_values_are_guarded_without_product_hardcoding(self) -> None:
        cases = {
            "今天仓库里某型号还有多少平方米现货？": "inventory_availability",
            "现在库存中可供发货的板材还有几块？": "inventory_availability",
            "某客户上个月还有多少应收账款未付？": "customer_balance",
            "请查询这个客户当前的未回款余额。": "customer_balance",
        }
        for question, expected_kind in cases.items():
            with self.subTest(question=question):
                self.assertEqual(
                    dynamic_private_business_data_kind(DraftRequest(customer_question=question)),
                    expected_kind,
                )

    def test_static_rules_and_public_project_queries_are_not_dynamic_private_data(self) -> None:
        questions = (
            "板材在仓库中应如何存放并保持通风？",
            "库存管理规范有哪些要求？",
            "应收账款是什么意思？",
            "应收账款余额如何计算？",
            "山东最近有什么旧楼改造项目？",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNone(
                    dynamic_private_business_data_kind(DraftRequest(customer_question=question))
                )

    def test_short_value_follow_up_can_inherit_inventory_subject(self) -> None:
        request = DraftRequest(
            customer_question="那现在还有多少？",
            conversation_context=[
                {"role": "user", "content": "我想查询某型号的库存。"},
                {"role": "assistant", "content": "请说明要查询的具体内容。"},
            ],
        )
        self.assertEqual(dynamic_private_business_data_kind(request), "inventory_availability")

    def test_visual_intent_has_general_scopes(self) -> None:
        cases = {
            "展示几个办公楼案例图片": "case",
            "你们的产品案例，给我图片加解释": "case",
            "给我看窗洞口节点图": "node",
            "有干挂施工流程图吗": "process",
            "发一下某型号板材的产品照片": "product",
            "有没有相关图片": "mixed",
        }
        for question, expected_scope in cases.items():
            with self.subTest(question=question):
                self.assertTrue(question_wants_visuals(question))
                self.assertEqual(question_visual_scope(question), expected_scope)

    def test_short_visual_follow_up_inherits_only_immediate_customer_case_scope(self) -> None:
        inherited = DraftRequest(
            customer_question="石榴红有图片吗？",
            conversation_context=[
                {"role": "user", "content": "你们有哪些学校项目案例？"},
                {"role": "assistant", "content": "已展示相关案例。"},
            ],
        )
        self.assertEqual(resolve_visual_request(inherited), (True, "case"))

        unrelated_latest_turn = DraftRequest(
            customer_question="石榴红有图片吗？",
            conversation_context=[
                {"role": "user", "content": "你们有哪些学校项目案例？"},
                {"role": "assistant", "content": "已展示相关案例。"},
                {"role": "user", "content": "报价需要哪些条件？"},
                {"role": "assistant", "content": "需要面积和规格。"},
            ],
        )
        self.assertEqual(resolve_visual_request(unrelated_latest_turn), (True, "product"))

    def test_current_case_visual_subject_filters_ranked_cases_without_named_hardcoding(self) -> None:
        retrieval = {
            "project_cases": [
                {"case_id": "c1", "project_name": "甲办公楼项目", "product": "真岩®石海棠红"},
                {"case_id": "c2", "project_name": "乙学校项目", "product": "真岩®石石榴红"},
            ],
            "visual_assets": [],
            "meta": {},
        }
        selected = apply_case_filters(
            retrieval,
            {"locations": [], "project_types": [], "installation_methods": [], "products": []},
            current_question="石榴红有图片吗？",
        )
        self.assertEqual([case["case_id"] for case in selected["project_cases"]], ["c2"])

    def test_case_gallery_uses_one_linked_primary_image_per_case_in_case_order(self) -> None:
        retriever = LocalRagRetriever.__new__(LocalRagRetriever)
        retriever.visual_by_asset_id = {
            "a1": {
                "asset_id": "a1",
                "effective_image_kind": "project_photo",
                "asset_type": "original_pdf_image",
                "customer_title": "甲项目实景",
            },
            "a1b": {
                "asset_id": "a1b",
                "effective_image_kind": "project_photo",
                "asset_type": "original_pdf_image",
                "customer_title": "甲项目第二张",
            },
            "a2": {
                "asset_id": "a2",
                "effective_image_kind": "unclassified_original_visual",
                "asset_type": "original_pdf_page_render",
                "customer_title": "乙项目原页",
            },
            "unrelated": {
                "asset_id": "unrelated",
                "effective_image_kind": "project_photo",
                "asset_type": "original_pdf_image",
            },
        }
        cases = [
            {
                "case_id": "case-a",
                "project_name": "甲项目",
                "product": "产品甲",
                "installation_method": "工艺甲",
                "area_m2": "1000",
                "completion_year": "2024",
                "score": 9,
                "visual_asset_ids": ["a1", "a1b"],
            },
            {
                "case_id": "case-b",
                "project_name": "乙项目",
                "product": "产品乙",
                "installation_method": "工艺乙",
                "area_m2": "2000",
                "completion_year": "2023",
                "score": 8,
                "visual_asset_ids": ["a2"],
            },
        ]
        visuals = retriever.visuals_for_project_cases(cases, visual_k=5)
        self.assertEqual([item["asset_id"] for item in visuals], ["a1", "a2"])
        self.assertEqual([item["related_case_id"] for item in visuals], ["case-a", "case-b"])
        self.assertTrue(all(item["gallery_type"] == "case" for item in visuals))
        self.assertEqual(visuals[0]["installation_method"], "工艺甲")
        self.assertIn("甲项目", visuals[0]["explanation"])

    def test_case_gallery_never_substitutes_an_unrelated_visual(self) -> None:
        retriever = LocalRagRetriever.__new__(LocalRagRetriever)
        retriever.visual_by_asset_id = {
            "unrelated": {
                "asset_id": "unrelated",
                "effective_image_kind": "project_photo",
                "asset_type": "original_pdf_image",
            }
        }
        visuals = retriever.visuals_for_project_cases(
            [{"case_id": "case-missing", "project_name": "无图案例", "visual_asset_ids": ["missing"]}],
            visual_k=5,
        )
        self.assertEqual(visuals, [])

    def test_product_visuals_use_only_catalog_approved_exact_photos(self) -> None:
        retriever = LocalRagRetriever.__new__(LocalRagRetriever)
        candidates = [
            (
                100.0,
                {
                    "asset_id": "page",
                    "kind": "visual",
                    "customer_title": "产品目录原页",
                    "asset_type": "original_pdf_page_render",
                    "effective_image_kind": "unclassified_original_visual",
                    "search_text": "GHM2517 产品原页",
                    "tokens": ["产品"],
                },
            ),
            (
                20.0,
                {
                    "asset_id": "component",
                    "kind": "visual",
                    "customer_title": "相关板材构件",
                    "asset_type": "original_pdf_image",
                    "effective_image_kind": "component_photo",
                    "search_text": "相关板材构件",
                    "tokens": ["板材"],
                },
            ),
            (
                10.0,
                {
                    "asset_id": "exact-photo",
                    "kind": "visual",
                    "customer_title": "GHM2517 饰面样板",
                    "asset_type": "original_pdf_image",
                    "effective_image_kind": "product_photo",
                    "gallery_type": "product_sample",
                    "product_gallery_eligible": True,
                    "visual_role": "product_variant",
                    "review_status": "auto_source_linked",
                    "product_name": "真岩®石",
                    "variant_or_code": "黄金麻 / GHM2517",
                    "display_priority": 100,
                    "search_text": "GHM2517 饰面样板",
                    "tokens": ["饰面", "样板"],
                },
            ),
        ]
        ranked = retriever._prioritise_product_visuals(
            "请展示GHM2517产品图片",
            candidates,
            product_overview_request=False,
        )
        self.assertEqual(ranked[0][1]["asset_id"], "exact-photo")
        self.assertNotIn("page", [item[1]["asset_id"] for item in ranked])

        fallback_only = retriever._prioritise_product_visuals(
            "请展示GHM2517产品图片",
            [candidates[0]],
            product_overview_request=False,
        )
        self.assertEqual(fallback_only, [])

    def test_product_overview_recognises_inventory_and_contextual_follow_up(self) -> None:
        self.assertTrue(is_product_overview_query("你们有什么产品"))
        self.assertTrue(is_product_overview_query("你们有什么产品\n详细介绍一下"))
        self.assertFalse(is_product_overview_query("详细介绍一下窗洞口节点"))

    def test_product_overview_prioritises_reviewed_master_profile(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve("你们有什么产品", top_k=5, visual_k=0)
        evidence = result["text_evidence"]
        self.assertEqual(len(evidence), 5)
        self.assertTrue(result["meta"]["knowledge_domain_routing"]["product_overview_priority"])
        self.assertTrue(all(item["text"].startswith("真岩产品总档案。") for item in evidence))
        self.assertTrue(all(item["sales_playbook_use"] == "approved_product_master_profile" for item in evidence))
        self.assertIn("核心产品线", evidence[0]["text"])
        self.assertIn("产品体系总览", evidence[0]["citations"][0]["section_heading"])

    def test_product_overview_direct_return_uses_hierarchy_not_finish_samples(self) -> None:
        finish_row = {
            "text": "产品样式包括甲、乙、丙。",
            "sales_playbook_use": "approved_product_master_profile",
            "citations": [{"section_heading": "饰面类型与产品样式"}],
        }
        hierarchy_row = {
            "text": "核心产品线及交付形态。",
            "sales_playbook_use": "approved_product_master_profile",
            "citations": [{"section_heading": "产品体系总览"}],
        }
        selected = direct_authoritative_fact_evidence(
            "你们有哪些产品？",
            {"text_evidence": [finish_row, hierarchy_row]},
            "factual_lookup",
            product_overview=True,
        )
        self.assertIs(selected, hierarchy_row)

    def test_specific_product_question_does_not_force_master_profile(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve("真岩石防火性能如何", top_k=3, visual_k=0)
        self.assertFalse(result["meta"]["knowledge_domain_routing"]["product_overview_priority"])
        self.assertIn("防火性能", result["text_evidence"][0]["text"])

    def test_generic_task_fallbacks_are_grammar_based(self) -> None:
        cases = {
            "干挂的具体流程是什么？": "procedure",
            "窗洞口节点怎么做？": "node_detail",
            "有哪些山东项目？": "case_reference",
            "我的旧楼适合干挂吗？": "project_fit",
            "真岩石是什么？": "factual_lookup",
        }
        for question, expected_task_type in cases.items():
            with self.subTest(question=question):
                self.assertEqual(_fallback_question_plan(question)["task_type"], expected_task_type)

    def test_procedure_mode_keeps_one_document_and_source_order(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "干挂的具体流程是什么？",
            top_k=8,
            visual_k=3,
            retrieval_mode="procedure",
        )
        evidence = result["text_evidence"]
        self.assertGreaterEqual(len(evidence), 4)
        source_names = {item["citations"][0]["document_name"] for item in evidence}
        self.assertEqual(source_names, {"真岩®无机仿石材穿透法施工方案2026"})
        joined = "\n".join(item["text"] for item in evidence)
        self.assertIn("龙骨安装", joined)
        self.assertIn("板材", joined)
        self.assertEqual(
            result["meta"]["knowledge_domain_routing"]["retrieval_mode"],
            "procedure",
        )

    def test_named_enterprise_scheme_beats_conflicting_generic_standard(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "岩棉一体板锚固件进入混凝土和砌体基层的有效锚固深度分别是多少？",
            top_k=5,
            visual_k=0,
        )
        self.assertEqual(
            result["text_evidence"][0]["citations"][0]["document_name"],
            "真岩®无机仿石材岩棉保温装饰一体板施工方案",
        )
        self.assertIn("25", result["text_evidence"][0]["text"])
        self.assertIn("50", result["text_evidence"][0]["text"])

    def test_named_procedure_anchors_on_substantive_step_list(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "穿透法方案列出的板材安装主要工序顺序是什么？",
            top_k=8,
            visual_k=0,
            retrieval_mode="procedure",
        )
        first = result["text_evidence"][0]
        self.assertEqual(
            first["citations"][0]["document_name"],
            "真岩®无机仿石材穿透法施工方案2026",
        )
        self.assertIn("预打孔", first["text"])
        self.assertIn("自攻丝", first["text"])

    def test_standard_request_prefers_normative_clause_over_commentary(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "人造板材幕墙规范对幕墙材料的耐候性提出了什么原则？",
            top_k=5,
            visual_k=0,
        )
        first = result["text_evidence"][0]["text"]
        self.assertIn("适应幕墙所在地的气候、环境", first)
        self.assertIn("设计使用年限", first)

    def test_reviewed_product_composition_profile_is_prioritised(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "真岩石饰面板的主要原材料和成型方式是什么？",
            top_k=5,
            visual_k=0,
        )
        self.assertEqual(
            result["text_evidence"][0]["sales_playbook_use"],
            "approved_product_master_profile",
        )
        self.assertIn("约3毫米", result["text_evidence"][0]["text"])

    def test_long_question_prioritises_final_requested_condition(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "岩棉保温装饰一体板施工前，基层抹灰层至少需要养护多久？",
            top_k=5,
            visual_k=0,
        )
        self.assertIn("养护期必须大于15天", result["text_evidence"][0]["text"])

    def test_direct_numeric_evidence_requires_matching_dimension_and_metric(self) -> None:
        wrong_length = {"text": "枪嘴伸入缝隙约4mm，胶高出板面3～5mm。"}
        wrong_time_context = {"text": "修补料待面料静养24小时后进行打磨。"}
        matching_time = {"text": "基层墙体抹灰层的养护期必须大于15天。"}
        selected = direct_numeric_evidence(
            "岩棉保温装饰一体板施工前，基层抹灰层至少需要养护多久？",
            {"text_evidence": [wrong_length, wrong_time_context, matching_time]},
        )
        self.assertIs(selected, matching_time)

    def test_direct_numeric_evidence_supports_distinct_measurement_dimensions(self) -> None:
        cases = (
            ("板缝宽度是多少？", "板缝宽度应为6～8mm。"),
            ("外窗台流水坡度是多少？", "外窗台流水坡度应保持3%～5%。"),
            ("每平方米需要多少个锚栓？", "锚栓数量不应少于8个/m²。"),
            ("施工环境温度最低是多少？", "施工时环境温度不宜低于5℃。"),
        )
        for question, text in cases:
            with self.subTest(question=question):
                evidence = {"text": text}
                self.assertIs(
                    direct_numeric_evidence(question, {"text_evidence": [evidence]}),
                    evidence,
                )

    def test_direct_numeric_evidence_fails_closed_for_ambiguous_quantity(self) -> None:
        evidence = {"text": "文档中同时出现3mm、24小时和5%。"}
        self.assertIsNone(
            direct_numeric_evidence("这个数值是多少？", {"text_evidence": [evidence]})
        )

    def test_named_method_condition_is_not_forced_into_full_procedure_mode(self) -> None:
        self.assertFalse(
            _names_specific_procedure("穿透法方案对雨天板材防护和雨后处理有什么要求？")
        )
        self.assertTrue(
            _names_specific_procedure("穿透法方案列出的板材安装主要工序顺序是什么？")
        )

    def test_weather_requirement_retrieval_keeps_the_direct_condition(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "穿透法方案对雨天板材防护和雨后处理有什么要求？",
            top_k=5,
            visual_k=0,
            retrieval_mode="factual_lookup",
        )
        passages = [item["text"] for item in result["text_evidence"]]
        self.assertIn("雨停后必须掀开遮雨器", passages[0])
        self.assertTrue(any("雨停后必须掀开遮雨器" in passage for passage in passages))

    def test_pre_action_question_prioritises_ordered_preparation_evidence(self) -> None:
        question = "岩棉一体板板缝在注入密封胶前应如何处理？"
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            question,
            top_k=5,
            visual_k=0,
            retrieval_mode="factual_lookup",
        )
        self.assertIn("胶粘剂干燥后", result["text_evidence"][0]["text"])
        self.assertIn("粘贴美纹纸", result["text_evidence"][0]["text"])
        selected = direct_authoritative_fact_evidence(
            question,
            result,
            "factual_lookup",
        )
        self.assertIsNotNone(selected)
        self.assertIn("嵌入聚苯乙烯泡沫条", selected["text"])

    def test_pre_action_direct_return_rejects_unordered_definition(self) -> None:
        question = "在涂刷涂料前应如何处理？"
        definition = {
            "text": "涂料是用于表面装饰的材料。",
            "source_taxonomy": [{"document_category": "enterprise_construction_method"}],
        }
        ordered_steps = {
            "text": "待基层干燥后，先清理灰尘，然后再涂刷涂料。",
            "source_taxonomy": [{"document_category": "enterprise_construction_method"}],
        }
        selected = direct_authoritative_fact_evidence(
            question,
            {"text_evidence": [definition, ordered_steps]},
            "factual_lookup",
        )
        self.assertIs(selected, ordered_steps)

    def test_requirement_question_returns_complete_trusted_top_evidence(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "岩棉一体板方案对外窗台流水坡度有什么要求？",
            top_k=5,
            visual_k=0,
            retrieval_mode="node_detail",
        )
        selected = direct_authoritative_fact_evidence(
            "岩棉一体板方案对外窗台流水坡度有什么要求？",
            result,
            "node_detail",
        )
        self.assertIsNotNone(selected)
        self.assertIn("3%-5%", selected["text"])
        self.assertIn("所有的板缝全部密封", selected["text"])

    def test_standard_handling_question_keeps_every_parallel_obligation(self) -> None:
        retriever = LocalRagRetriever()
        result = retriever.retrieve(
            "幕墙性能检测未达设计要求时，安装缺陷与设计或材料原因分别应如何处理？",
            top_k=5,
            visual_k=0,
        )
        selected = direct_authoritative_fact_evidence(
            "幕墙性能检测未达设计要求时，安装缺陷与设计或材料原因分别应如何处理？",
            result,
            "comparison",
        )
        self.assertIsNotNone(selected)
        self.assertIn("检测报告应记载所做的修改或工艺改进", selected["text"])
        self.assertIn("重新制作试件", selected["text"])

    def test_source_typo_is_explained_only_in_customer_display(self) -> None:
        evidence = {
            "text": "施工单位同时提供出场合格证和产品检验报告。",
            "citations": [],
        }
        response = _single_evidence_answer(
            None,
            {"text_evidence": [evidence], "visual_assets": [], "meta": {}},
            evidence,
            intent="construction",
            mode="test",
            prefix="依据：",
        )
        self.assertEqual(evidence["text"], "施工单位同时提供出场合格证和产品检验报告。")
        self.assertIn("出厂合格证", response["customer_reply"])
        self.assertEqual(
            response["normalized_terms"],
            [{"term": "出场合格证", "normalized": "出厂合格证"}],
        )
        self.assertIn("疑似", response["customer_reply"])
        self.assertIn("正式使用前请核对", response["customer_reply"])

    def test_approved_comparison_exposes_unquantified_superlative_conflict(self) -> None:
        evidence = {
            "text": "甲产品仿真度最高；乙产品与石材相似度最高，且质感无差异。",
            "citations": [{"document_name": "公司批准比较资料"}],
        }
        response = _single_evidence_answer(
            None,
            {"text_evidence": [evidence], "visual_assets": [], "meta": {}},
            evidence,
            intent="comparison",
            mode="approved_comparison_evidence",
            prefix="根据公司批准的产品比较口径：",
        )
        self.assertIn("未给出统一量化指标", response["customer_reply"])
        self.assertIn(
            "source_contains_unquantified_comparison_superlatives",
            response["risk_warnings"],
        )

    def test_choice_condition_affinity_prefers_both_choices_with_thresholds(self) -> None:
        query = "甲工艺还是乙工艺，应依据什么条件选择？"
        concrete_rule = {
            "text": "基层平整度大于5mm时采用甲工艺，小于5mm时采用乙工艺。"
        }
        generic_summary = {"text": "甲工艺或乙工艺应结合项目条件选择。"}
        self.assertGreater(
            LocalRagRetriever._choice_condition_affinity(query, concrete_rule),
            LocalRagRetriever._choice_condition_affinity(query, generic_summary),
        )

    def test_planner_expansion_cannot_hide_specific_choice_condition_clause(self) -> None:
        retriever = LocalRagRetriever()
        customer_question = "岩棉保温装饰一体板采用点粘法还是条粘法，方案依据什么条件选择？"
        expanded_query = f"{customer_question}\n岩棉保温装饰一体板 施工方案 选择条件"
        result = retriever.retrieve(expanded_query, top_k=5, visual_k=0)
        first = result["text_evidence"][0]
        self.assertIn("点粘法", first["text"])
        self.assertIn("条粘法", first["text"])
        self.assertIn("5mm/2m", first["text"].replace(" ", ""))
        self.assertIn("80%", first["text"])

        selected = direct_authoritative_fact_evidence(
            customer_question,
            result,
            "factual_lookup",
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["chunk_id"], first["chunk_id"])

    def test_direct_visual_action_keeps_a_safe_visible_action_summary(self) -> None:
        reply, observations = sanitize_direct_visual_output(
            "采石场图片中，挖掘机正在进行什么作业？",
            "图中挖掘机正在向卡车装载石料。",
            ["一台挖掘机", "一辆卡车", "车斗内可见石料"],
        )
        self.assertIn("向卡车装载石料", reply)
        self.assertIn("图中挖掘机正在向卡车装载石料", observations)


if __name__ == "__main__":
    unittest.main()
