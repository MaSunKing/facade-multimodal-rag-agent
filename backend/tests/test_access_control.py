from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend import access_control
from backend.sales.retriever import LocalRagRetriever


class AccessControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.old_db = access_control.STATE_DB_PATH
        self.old_token = access_control.BOOTSTRAP_TOKEN_PATH
        self.old_ready = access_control._SCHEMA_READY
        access_control.STATE_DB_PATH = root / "state.sqlite3"
        access_control.BOOTSTRAP_TOKEN_PATH = root / "bootstrap.txt"
        access_control._SCHEMA_READY = False
        access_control.ensure_schema()

    def tearDown(self) -> None:
        access_control.STATE_DB_PATH = self.old_db
        access_control.BOOTSTRAP_TOKEN_PATH = self.old_token
        access_control._SCHEMA_READY = self.old_ready
        self.temp.cleanup()

    def test_first_account_is_admin_and_can_grant_internal_access(self) -> None:
        setup_token = access_control.BOOTSTRAP_TOKEN_PATH.read_text(encoding="utf-8").strip()
        admin = access_control.create_initial_admin(
            setup_token=setup_token,
            username="owner",
            password="correct-horse-battery",
            display_name="负责人",
        )
        self.assertEqual(admin.role, "admin")
        self.assertTrue(admin.can_access_internal)
        self.assertFalse(access_control.BOOTSTRAP_TOKEN_PATH.exists())

        token, authenticated, _ = access_control.login("owner", "correct-horse-battery")
        self.assertEqual(access_control.principal_for_token(token), authenticated)
        visual_ticket = access_control.issue_visual_ticket(authenticated, "asset-1")
        self.assertEqual(
            access_control.principal_for_visual_ticket(visual_ticket, "asset-1"), authenticated
        )
        self.assertIsNone(access_control.principal_for_visual_ticket(visual_ticket, "asset-2"))
        member = access_control.create_user(
            username="sales01",
            password="sales-password-01",
            display_name="销售一组",
            role="member",
            can_access_internal=False,
        )
        self.assertFalse(bool(member["can_access_internal"]))
        updated = access_control.update_user(
            str(member["user_id"]), role=None, can_access_internal=True, active=None
        )
        self.assertTrue(bool(updated["can_access_internal"]))

    def test_only_visible_scopes_enter_retriever(self) -> None:
        index = Path(self.temp.name) / "index.json"
        documents = [
            {"id": "pub", "kind": "text", "text": "公开产品", "tokens": ["公开"], "access_scope": "public"},
            {"id": "int", "kind": "text", "text": "内部报价", "tokens": ["内部"], "access_scope": "internal"},
        ]
        index.write_text(
            json.dumps(
                {
                    "metadata": {},
                    "documents": documents,
                    "document_frequency": {},
                    "average_document_length": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        public = LocalRagRetriever(index, allowed_access_scopes={"public"})
        internal = LocalRagRetriever(index, allowed_access_scopes={"public", "internal"})
        self.assertEqual([item["id"] for item in public.documents], ["pub"])
        self.assertEqual({item["id"] for item in internal.documents}, {"pub", "int"})


if __name__ == "__main__":
    unittest.main()
