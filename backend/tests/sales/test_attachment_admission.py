import unittest
from unittest.mock import patch
from fastapi import HTTPException
from backend import app


class AttachmentAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_session_stops_before_planner_and_queue(self):
        request=app.DraftRequest(customer_question='介绍附件',document_session_id='foreign-or-expired')
        with patch('backend.app.get_session',return_value=None) as session, \
             patch('backend.app.grounded_answer',side_effect=AssertionError('no model workflow')), \
             patch('backend.app._answer_admission') as admission:
            with self.assertRaises(HTTPException) as caught:
                await app.answer_http(request, http_request=None, principal=None, attachment_owner='browser:own')
        self.assertEqual(caught.exception.status_code,404)
        self.assertEqual(caught.exception.code,'ATTACHMENT_SESSION_UNAVAILABLE')
        session.assert_called_once_with('foreign-or-expired',owner_id='browser:own')
        admission.acquire.assert_not_called()


if __name__=='__main__': unittest.main()
