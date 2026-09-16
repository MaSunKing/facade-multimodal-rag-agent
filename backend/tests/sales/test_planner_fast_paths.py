from __future__ import annotations

import unittest
from unittest.mock import patch

import backend.app as facade_app


class PlannerFastPathTests(unittest.TestCase):
    def setUp(self) -> None:
        with facade_app._planner_cache_lock:
            facade_app._planner_cache.clear()

    def test_explicit_product_overview_uses_reviewed_catalogue_fast_path(self) -> None:
        request = facade_app.DraftRequest(customer_question="我想了解一下你们的产品")
        with patch.object(facade_app, "load_model", side_effect=AssertionError("planner model loaded")) as planner:
            plan = facade_app.plan_customer_tools(request)
        planner.assert_not_called()
        self.assertEqual(plan.tools, ["company_rag"])
        self.assertTrue(plan.product_overview)
        self.assertEqual(plan.task_type, "factual_lookup")
        self.assertEqual(plan.reason, "reviewed_product_catalogue_fast_path")

    def test_product_overview_fast_path_preserves_explicit_image_request(self) -> None:
        request = facade_app.DraftRequest(
            customer_question="我想了解一下你们的产品，给我看几张产品图片"
        )
        with patch.object(
            facade_app,
            "load_model",
            side_effect=AssertionError("planner model loaded"),
        ):
            plan = facade_app.plan_customer_tools(request)
        self.assertTrue(plan.product_overview)
        self.assertTrue(plan.wants_visuals)
        self.assertEqual(plan.visual_scope, "product")

    def test_constrained_overview_phrase_still_uses_semantic_planner(self) -> None:
        request = facade_app.DraftRequest(
            customer_question="你们有什么产品不适合旧楼外立面改造？"
        )
        with patch.object(
            facade_app,
            "load_model",
            side_effect=RuntimeError("expected semantic planner"),
        ) as planner:
            plan = facade_app.plan_customer_tools(request)
        planner.assert_called_once()
        self.assertFalse(plan.product_overview)
        self.assertEqual(plan.task_type, "project_fit")

    def test_standalone_greeting_skips_model_planner_and_web(self) -> None:
        request = facade_app.DraftRequest(customer_question="你好！", use_online_search=True)
        with patch.object(facade_app, "load_model", side_effect=AssertionError("planner model loaded")):
            plan = facade_app.plan_customer_tools(request)
        self.assertEqual(plan.tools, ["general_chat"])
        self.assertFalse(plan.requires_public_web)

    def test_standalone_greeting_skips_answer_model_too(self) -> None:
        request = facade_app.DraftRequest(customer_question="你好！", use_online_search=True)
        plan = facade_app.plan_customer_tools(request)
        with patch.object(facade_app, "load_model", side_effect=AssertionError("answer model loaded")):
            response = facade_app._CustomerAnswerWorkflowCallbacks.answer_planned(request, plan)
        self.assertEqual(response.meta["mode"], "bounded_social_fast_path")
        self.assertIn("有什么想要了解", response.customer_reply)
        self.assertEqual(response.online_sources, [])

    def test_greeting_plus_domain_question_still_uses_semantic_planner(self) -> None:
        request = facade_app.DraftRequest(customer_question="你好，请介绍一下干挂施工节点")
        self.assertFalse(facade_app.is_bounded_social_turn(request.customer_question))
        with patch.object(facade_app, "load_model", side_effect=RuntimeError("expected planner")) as mocked:
            plan = facade_app.plan_customer_tools(request)
        mocked.assert_called_once()
        self.assertIn("company_rag", plan.tools)

    def test_dynamic_private_value_keeps_source_free_guard(self) -> None:
        request = facade_app.DraftRequest(customer_question="现在黄金麻还有多少库存？")
        with patch.object(facade_app, "load_model", side_effect=AssertionError("planner model loaded")):
            plan = facade_app.plan_customer_tools(request)
        self.assertEqual(plan.tools, [])
        self.assertTrue(plan.reason.startswith("dynamic_private_business_data_guard:"))

    def test_structured_cases_are_limited_to_case_tasks(self) -> None:
        retrieval = {"project_cases": [{"case_id": "case-1"}]}
        for task_type in ("factual_lookup", "procedure", "node_detail", "comparison", "commercial"):
            self.assertEqual(facade_app.structured_cases_for_task(retrieval, task_type), [], task_type)
        for task_type in ("case_reference", "project_fit"):
            self.assertEqual(
                facade_app.structured_cases_for_task(retrieval, task_type),
                retrieval["project_cases"],
                task_type,
            )

    def test_model_task_semantics_are_not_reclassified_by_question_keywords(self) -> None:
        plan = facade_app.ToolPlan(
            tools=["company_rag"],
            intent="product_parameter",
            task_type="factual_lookup",
            retrieval_query="案例材料 参数",
            case_reference=False,
            reason="the question asks for a material fact mentioned in a case",
        )
        resolved = facade_app.question_plan_from_tool_plan(
            plan,
            "这个案例里用的材料参数是什么？",
        )
        self.assertEqual(resolved["task_type"], "factual_lookup")
        self.assertFalse(resolved["case_reference"])

    def test_planner_cache_is_exact_not_semantic(self) -> None:
        first_input = {
            "question": "介绍工厂",
            "conversation_context": [],
            "permissions": {"public_web_search": False},
        }
        similar_but_different = {
            **first_input,
            "question": "介绍生产基地",
        }
        plan = facade_app.ToolPlan(
            tools=["company_rag"],
            retrieval_query="工厂 生产基地",
            reason="model plan",
        )
        first_key = facade_app._planner_cache_key(first_input)
        facade_app._cache_planner_plan(first_key, plan)
        self.assertIsNotNone(facade_app._get_cached_planner_plan(first_key))
        self.assertIsNone(
            facade_app._get_cached_planner_plan(
                facade_app._planner_cache_key(similar_but_different)
            )
        )


if __name__ == "__main__":
    unittest.main()
