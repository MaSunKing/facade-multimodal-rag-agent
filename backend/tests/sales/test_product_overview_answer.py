"""Regression tests for complete, deterministic product overviews.

These checks use only the local lexical index.  They never load the answer
model or consume GPU memory.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ["RAG_HYBRID_ENABLED"] = "0"

from backend.app import (
    DraftRequest,
    _run_facade_rag_answer,
    product_overview_answer,
    product_overview_safe_retrieval_query,
    resolve_product_overview_request,
)
from backend.sales.retriever import LocalRagRetriever, is_product_overview_query


@unittest.skipUnless(os.getenv("RUN_ENTERPRISE_DATA_TESTS") == "1", "Requires separately authorised enterprise index; not included in public fixtures")
class ProductOverviewAnswerTests(unittest.TestCase):
    def test_company_product_discovery_phrasing_is_an_overview(self) -> None:
        questions = (
            "我想了解一下你们的产品 给我图片",
            "介绍一下你们公司的产品体系",
            "我想看看公司的产品",
            "请介绍企业产品",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(is_product_overview_query(question))

    def test_specific_model_questions_are_not_product_overviews(self) -> None:
        questions = (
            "我想了解一下你们的GHM2517产品",
            "介绍一下GHB2516这个型号",
            "你们有哪些产品里包含GHM2517？",
            "详细介绍一下你们的黄金麻产品",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertFalse(is_product_overview_query(question))

    def test_named_follow_up_does_not_inherit_a_previous_product_overview(self) -> None:
        for current_question in ("黄金麻有图片吗", "详细介绍黄金麻"):
            with self.subTest(current_question=current_question):
                request = DraftRequest(
                    customer_question=current_question,
                    conversation_context=[
                        {"role": "user", "content": "我想了解一下你们的产品"},
                        {"role": "assistant", "content": "已介绍产品体系。"},
                    ],
                )
                self.assertFalse(resolve_product_overview_request(request))
                combined = f"我想了解一下你们的产品\n{current_question}"
                safe_query = product_overview_safe_retrieval_query(
                    request,
                    combined,
                    current_question,
                    product_overview_request=False,
                )
                self.assertEqual(safe_query, current_question)
                self.assertFalse(is_product_overview_query(safe_query))
                routed = LocalRagRetriever().retrieve(safe_query, top_k=1, visual_k=0)
                self.assertFalse(
                    routed["meta"]["knowledge_domain_routing"]["product_overview_priority"]
                )

    def test_subjectless_detail_follow_up_can_inherit_the_immediate_overview(self) -> None:
        request = DraftRequest(
            customer_question="详细介绍一下",
            conversation_context=[
                {"role": "user", "content": "我想了解一下你们的产品"},
                {"role": "assistant", "content": "已简要介绍产品体系。"},
            ],
        )
        self.assertTrue(resolve_product_overview_request(request))

        unrelated_latest = DraftRequest(
            customer_question="详细介绍一下",
            conversation_context=[
                {"role": "user", "content": "我想了解一下你们的产品"},
                {"role": "assistant", "content": "已简要介绍产品体系。"},
                {"role": "user", "content": "报价需要哪些条件"},
                {"role": "assistant", "content": "需要面积和规格。"},
            ],
        )
        self.assertFalse(resolve_product_overview_request(unrelated_latest))

    def test_direct_compatibility_call_keeps_deterministic_overview_renderer(self) -> None:
        requests = (
            DraftRequest(customer_question="我想了解一下你们的产品 给我图片"),
            DraftRequest(
                customer_question="详细介绍一下",
                conversation_context=[
                    {"role": "user", "content": "我想了解一下你们的产品"},
                    {"role": "assistant", "content": "已简要介绍产品体系。"},
                ],
            ),
        )
        for request in requests:
            with self.subTest(question=request.customer_question):
                with patch("backend.app.understand_customer_question") as planner:
                    response = _run_facade_rag_answer(request, allow_public_web=False)
                planner.assert_not_called()
                self.assertTrue(response.answerable)
                self.assertFalse(response.meta["model_used"])
                self.assertEqual(response.meta["mode"], "deterministic_product_master_overview")
                self.assertEqual(response.meta["profile_section_count"], 5)

    def test_overview_retrieval_keeps_every_approved_profile_section(self) -> None:
        result = LocalRagRetriever().retrieve(
            "我想了解一下你们的产品 给我图片",
            top_k=3,
            visual_k=0,
        )
        approved = [
            item
            for item in result["text_evidence"]
            if item.get("sales_playbook_use") == "approved_product_master_profile"
        ]
        headings = {
            item["citations"][0]["section_heading"]
            for item in approved
        }
        self.assertEqual(
            headings,
            {
                "产品体系总览",
                "饰面类型与产品样式",
                "材料构成与形成方式",
                "产品特点与维护",
                "应用范围与配套施工资料",
            },
        )
        self.assertEqual(len(approved), 5)
        self.assertTrue(result["meta"]["knowledge_domain_routing"]["product_overview_priority"])

    def test_generic_heading_cannot_consume_an_approved_overview_slot(self) -> None:
        retriever = LocalRagRetriever()
        headings = (
            "产品体系总览",
            "饰面类型与产品样式",
            "材料构成与形成方式",
            "产品特点与维护",
            "应用范围与配套施工资料",
        )
        approved = [
            {
                "id": f"approved-{index}",
                "kind": "text",
                "text": f"真岩产品总档案。{heading}的审核内容",
                "tokens": ["产品", "真岩"],
                "sales_playbook_use": "approved_product_master_profile",
                "source_refs": [
                    {
                        "document_name": "真岩产品总档案（公司资料汇总版）",
                        "section_heading": heading,
                    }
                ],
            }
            for index, heading in enumerate(headings)
        ]
        generic = {
            "id": "generic-heading",
            "kind": "text",
            "text": "产品介绍",
            "tokens": ["产品", "介绍"],
            "source_refs": [{"document_name": "普通目录", "section_heading": "产品介绍"}],
        }
        retriever.documents = [generic, *approved]
        ranked = [(10.0, generic), *[(9.0 - index, item) for index, item in enumerate(approved)]]

        with patch.object(retriever, "_hybrid_scored", return_value=ranked), patch.object(
            retriever, "_rank_with_rules", return_value=ranked
        ):
            result = retriever.retrieve("介绍一下你们的产品", top_k=3, visual_k=0)

        self.assertEqual(
            [item["chunk_id"] for item in result["text_evidence"]],
            [item["id"] for item in approved],
        )
        self.assertTrue(
            all(
                item.get("sales_playbook_use") == "approved_product_master_profile"
                for item in result["text_evidence"]
            )
        )

    def test_deterministic_overview_is_complete_cited_and_keeps_visuals(self) -> None:
        question = "我想了解一下你们的产品 给我图片"
        retriever = LocalRagRetriever()
        retrieval = retriever.retrieve(
            question,
            top_k=3,
            visual_k=5,
            wants_visuals=True,
            visual_scope="product",
            visual_query=question,
        )
        response = product_overview_answer(
            DraftRequest(customer_question=question),
            retrieval,
        )

        self.assertTrue(response["answerable"])
        self.assertFalse(response["meta"]["model_used"])
        self.assertEqual(response["meta"]["mode"], "deterministic_product_master_overview")
        self.assertEqual(response["meta"]["profile_section_count"], 5)
        self.assertTrue(response["meta"]["wants_visuals"])
        self.assertEqual(response["meta"]["visual_scope"], "product")
        self.assertEqual(response["visual_assets"], retrieval["visual_assets"])

        reply = response["customer_reply"]
        for expected in (
            "核心产品线",
            "外墙无机饰面层仿石装饰板",
            "保温装饰一体板",
            "荔枝面",
            "黄金麻",
            "无机石材颗粒",
            "水泥纤维基板",
            "基层、锚固条件、保温要求与节点条件",
        ):
            self.assertIn(expected, reply)

        cited_headings = {
            citation.get("section_heading")
            for citation in response["citations"]
        }
        self.assertTrue(
            {
                "产品体系总览",
                "饰面类型与产品样式",
                "材料构成与形成方式",
                "产品特点与维护",
                "应用范围与配套施工资料",
            }.issubset(cited_headings)
        )
        self.assertEqual(response["retrieval"]["result_count"], 5)


if __name__ == "__main__":
    unittest.main()
