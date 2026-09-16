from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import access_control


class AccessControlWebAuditTests(unittest.TestCase):
    def test_failed_web_tool_is_persisted_as_failed_with_error_metadata(self) -> None:
        # sqlite3 context managers commit but do not explicitly close on every
        # supported Python build; Windows can briefly retain the file handle.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            state_db = root / "runtime" / "facade_state.sqlite3"
            with patch.object(access_control, "ROOT", root), patch.object(
                access_control, "STATE_DB_PATH", state_db
            ):
                request_id, started = access_control.begin_agent_request(
                    principal=None,
                    attachment_session_id=None,
                    query_length=8,
                )
                response = {
                    "meta": {
                        "orchestration": {
                            "tools": ["public_web_search"],
                            "workflow": "bounded_tool_agent_v2",
                        },
                        "online_search": {
                            "status": "failed",
                            "trigger": "customer_selected",
                            "error_code": "network_error",
                            "retryable": True,
                            "attempts": 2,
                            "quota": {"used": 0, "remaining": 45},
                        },
                    },
                    "retrieval": {"result_count": 0, "supporting_results": []},
                    "online_sources": [],
                    "citations": [],
                    "visual_assets": [],
                }
                access_control.finish_agent_request(
                    request_id,
                    started,
                    response=response,
                )

                with sqlite3.connect(state_db) as connection:
                    tool = connection.execute(
                        "SELECT status, error_type, summary_json FROM tool_runs WHERE request_id=?",
                        (request_id,),
                    ).fetchone()
                self.assertEqual(tool[0], "failed")
                self.assertEqual(tool[1], "network_error")
                self.assertEqual(json.loads(tool[2])["online_search"]["attempts"], 2)

                snapshot_path = next((root / "runtime" / "request_snapshots").rglob(f"{request_id}.json"))
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                self.assertEqual(snapshot["online_search"]["error_code"], "network_error")


if __name__ == "__main__":
    unittest.main()
