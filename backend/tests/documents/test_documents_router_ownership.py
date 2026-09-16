from __future__ import annotations

import unittest
from io import BytesIO

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend.documents.customer_sessions import (
    bind_session_owner,
    delete_session,
    get_session,
    issue_customer_visual_ticket,
)
from backend.documents.ownership import attachment_owner_id, bind_attachment_request_owner
from backend.documents.router import router


class CustomerDocumentRouterOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        application = FastAPI()
        application.include_router(router)

        @application.get("/indirect-session-read/{session_id}")
        def indirect_session_read(
            session_id: str,
            _owner: str | None = Depends(bind_attachment_request_owner),
        ) -> dict[str, bool]:
            # Mirrors app.py/LangGraph helpers, which intentionally do not
            # accept an owner from the customer-controlled request body.
            return {"visible": get_session(session_id) is not None}

        self.client = TestClient(application)
        self.client_a = "web_" + "a" * 32
        self.client_b = "web_" + "b" * 32
        self.owner_a = attachment_owner_id(None, self.client_a, required=True)
        self.session_ids: list[str] = []

    def tearDown(self) -> None:
        for session_id in self.session_ids:
            delete_session(session_id, owner_id=self.owner_a)
        self.client.close()

    def _upload(self, *, client_id: str | None = None, file_name: str = "private.txt", content: bytes = b"private"):
        headers = {"X-Facade-Client-ID": client_id} if client_id else {}
        return self.client.post(
            "/api/copilot/documents",
            headers=headers,
            files=[("files", (file_name, content, "application/octet-stream"))],
        )

    def test_status_append_and_delete_require_same_browser_owner(self) -> None:
        missing_identity = self._upload()
        self.assertEqual(missing_identity.status_code, 400)

        created = self._upload(client_id=self.client_a)
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session_id"]
        self.session_ids.append(session_id)

        self.assertEqual(
            self.client.get(
                f"/api/copilot/documents/{session_id}",
                headers={"X-Facade-Client-ID": self.client_a},
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(
                f"/api/copilot/documents/{session_id}",
                headers={"X-Facade-Client-ID": self.client_b},
            ).status_code,
            404,
        )
        self.assertTrue(
            self.client.get(
                f"/indirect-session-read/{session_id}",
                headers={"X-Facade-Client-ID": self.client_a},
            ).json()["visible"]
        )
        self.assertFalse(
            self.client.get(
                f"/indirect-session-read/{session_id}",
                headers={"X-Facade-Client-ID": self.client_b},
            ).json()["visible"]
        )
        foreign_append = self.client.post(
            "/api/copilot/documents",
            headers={"X-Facade-Client-ID": self.client_b},
            data={"session_id": session_id},
            files=[("files", ("foreign.txt", b"foreign", "text/plain"))],
        )
        self.assertEqual(foreign_append.status_code, 404)
        self.assertEqual(
            self.client.delete(
                f"/api/copilot/documents/{session_id}",
                headers={"X-Facade-Client-ID": self.client_b},
            ).status_code,
            404,
        )
        removed = self.client.delete(
            f"/api/copilot/documents/{session_id}",
            headers={"X-Facade-Client-ID": self.client_a},
        )
        self.assertEqual(removed.status_code, 200)
        self.session_ids = []

    def test_visual_requires_owner_header_or_scoped_ticket(self) -> None:
        from PIL import Image

        stream = BytesIO()
        Image.new("RGB", (96, 64), color=(220, 225, 230)).save(stream, format="PNG")
        created = self._upload(client_id=self.client_a, file_name="private.png", content=stream.getvalue())
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session_id"]
        self.session_ids.append(session_id)
        session = get_session(session_id, owner_id=self.owner_a)
        self.assertIsNotNone(session)
        assert session is not None
        document = session.documents[0]
        visual = document.visuals[0]
        endpoint = f"/api/copilot/documents/{session_id}/visual/{document.document_id}/{visual.visual_id}"

        self.assertEqual(
            self.client.get(endpoint, headers={"X-Facade-Client-ID": self.client_b}).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(endpoint, headers={"X-Facade-Client-ID": self.client_a}).status_code,
            200,
        )
        with bind_session_owner(self.owner_a):
            ticket = issue_customer_visual_ticket(session_id, document.document_id, visual.visual_id)
        self.assertTrue(ticket)
        ticketed = self.client.get(endpoint, params={"ticket": ticket})
        self.assertEqual(ticketed.status_code, 200)
        self.assertEqual(ticketed.headers["cache-control"], "private, no-store, max-age=0")

    def test_expired_browser_session_id_is_replaced_not_reused(self) -> None:
        stale_id = "upload_" + "e" * 32
        response = self.client.post(
            "/api/copilot/documents",
            headers={"X-Facade-Client-ID": self.client_a},
            data={"session_id": stale_id},
            files=[("files", ("fresh.txt", b"fresh", "text/plain"))],
        )
        self.assertEqual(response.status_code, 200, response.text)
        fresh_id = response.json()["session_id"]
        self.assertNotEqual(fresh_id, stale_id)
        self.session_ids.append(fresh_id)


if __name__ == "__main__":
    unittest.main()
