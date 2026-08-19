from __future__ import annotations

import unittest

from backend.sales.tool_planner import guard_plan
from backend.app import evidence_support_audit


class ToolPlannerPolicyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
