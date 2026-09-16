import threading
import unittest
from unittest.mock import patch

from backend import app
from backend.documents import customer_sessions as sessions
from backend.sales.context_engine import choose_context_budget
from backend.sales.task_memory import accepted_updates
from backend.sales.tool_planner import ToolPlan, guard_plan


class FunctionalHardeningTests(unittest.TestCase):
    def test_invalid_memory_output_does_not_break_answer(self):
        self.assertEqual(accepted_updates("预算80万元", [None, {"key": "x"}]), [])
        request = app.DraftRequest(customer_question="预算80万元", memory_enabled=True)
        app.capture_task_memory_updates(request, {"memory_updates": [{"key": "a.budget", "value": "80万元",
            "quote": "预算80万元", "mode": "asserted"}]})
        self.assertEqual(request._memory_updates[0].value, "80万元")

    def test_model_selected_memory_basis_does_not_search_company(self):
        plan = guard_plan({"tools": ["company_rag"], "answer_basis": "conversation"},
                          has_documents=False, has_image=False, facade_related=True, web_allowed=True)
        self.assertEqual(plan.tools, ["general_chat"])
        self.assertEqual(plan.answer_basis, "conversation")

    def test_regular_product_query_keeps_rag(self):
        plan = guard_plan({"tools": ["company_rag"]},
                          has_documents=False, has_image=False, facade_related=True, web_allowed=True)
        self.assertEqual(plan.tools, ["company_rag"])

    def test_partial_json_never_repairs_a_factual_string(self):
        complete = '{"intent":"product_parameter","answerable":true,"citations":[],"customer_reply":"已提供图片","key_points":[],"next_action":"'
        recovered = app.recover_truncated_grounded_json(complete)
        self.assertEqual(recovered["customer_reply"], "已提供图片")
        with self.assertRaises(ValueError):
            app.recover_truncated_grounded_json('{"intent":"product_parameter","answerable":true,"customer_reply":"厚度为')

    def test_direct_image_is_not_attachment_v1(self):
        manifest = [{"image_number": 1, "origin": "direct_upload", "source_candidates": []},
                    {"image_number": 2, "source_candidates": [{"evidence_id": "V1"}]}]
        result = {"citations": [{"evidence_id": "V1"}],
                  "image_observations": ["[image2|V1] visible object"]}
        audit = app.visual_input_coverage_audit(result, [{}], manifest)
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["missing_image_observations"], [1])
        result["image_observations"].append("[image1|direct_upload] another object")
        self.assertTrue(app.visual_input_coverage_audit(result, [{}], manifest)["complete"])

    def test_larger_output_only_for_multi_document_scope(self):
        plan = ToolPlan(tools=["customer_documents"], document_scope="cross_document")
        small = choose_context_budget(plan, has_documents=True, has_image=False, source_document_count=2)
        large = choose_context_budget(plan, has_documents=True, has_image=False, source_document_count=4)
        self.assertGreater(large.max_output_tokens, small.max_output_tokens)
        self.assertLessEqual(large.max_output_tokens, 800)
        self.assertEqual(large.max_prompt_tokens, small.max_prompt_tokens)

    def test_attachment_capacity_does_not_evict_existing_session(self):
        with patch.object(sessions, "_sessions", {}):
            first = sessions.add_files([("a.txt", b"first evidence")], owner_id="owner")
            with patch.object(sessions, "MAX_RESIDENT_BYTES", 1):
                with self.assertRaises(sessions.AttachmentCapacityError):
                    sessions.add_files([("b.txt", b"second evidence")], owner_id="owner")
            self.assertIsNotNone(sessions.get_session(first["session_id"], owner_id="owner"))
            self.assertEqual(len(sessions._sessions), 1)

    def test_parser_admission_and_release(self):
        slot = threading.BoundedSemaphore(1)
        slot.acquire()
        with patch.object(sessions, "_parse_slot", slot):
            with self.assertRaises(sessions.AttachmentCapacityError):
                sessions.add_files([("a.txt", b"a")])
            slot.release()
            with self.assertRaises(ValueError):
                sessions.add_files([])
            self.assertTrue(slot.acquire(blocking=False))

    def test_full_session_count_rejects_before_parse(self):
        with patch.object(sessions, "_sessions", {}), patch.object(sessions, "MAX_RESIDENT_SESSIONS", 0), \
             patch.object(sessions, "ingest_uploaded_file") as parse:
            with self.assertRaises(sessions.AttachmentCapacityError):
                sessions.add_files([("a.txt", b"a")])
            parse.assert_not_called()
