from __future__ import annotations

import unittest
from io import BytesIO

from backend.documents.customer_sessions import add_files, delete_session, get_visual_asset, retrieve, temporary_visual_files


class CustomerDocumentSessionTests(unittest.TestCase):
    def tearDown(self) -> None:
        for session_id in getattr(self, "sessions", []):
            delete_session(session_id)

    def test_two_text_files_are_retrievable_with_provenance(self) -> None:
        result = add_files(
            [
                ("施工说明.txt", "窗洞口采用锚固安装并设置防水收口。".encode("utf-8")),
                ("项目备注.txt", "项目位于济南，外墙采用保温装饰一体板。".encode("utf-8")),
            ]
        )
        self.sessions = [result["session_id"]]
        selected = retrieve(result["session_id"], "窗洞口怎么安装？")
        self.assertEqual(selected["status"], "ok")
        self.assertEqual(len(selected["documents"]), 2)
        self.assertTrue(any(item["document_name"] == "施工说明.txt" for item in selected["evidence"]))
        self.assertTrue(selected["input_snapshot"]["canonical_evidence_preserved"])

    def test_more_than_four_files_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum_four_files"):
            add_files([(f"{index}.txt", b"test") for index in range(5)])

    def test_excel_evidence_returns_sheet_and_row_range(self) -> None:
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "施工方案"
        sheet.append(["项目", "安装方式"])
        sheet.append(["办公楼", "锚固安装"])
        stream = BytesIO()
        workbook.save(stream)
        result = add_files([("方案.xlsx", stream.getvalue())])
        self.sessions = [result["session_id"]]
        selected = retrieve(result["session_id"], "办公楼采用什么安装方式？")
        precise = [
            citation
            for item in selected["evidence"]
            for citation in item["citations"]
            if citation.get("sheet_name") == "施工方案"
        ]
        self.assertTrue(precise)
        self.assertTrue(any(citation.get("source_range") for citation in precise))

    def test_image_asset_is_retained_and_selected_for_local_vlm(self) -> None:
        from PIL import Image

        stream = BytesIO()
        Image.new("RGB", (320, 180), color=(210, 215, 220)).save(stream, format="PNG")
        result = add_files([("节点图.png", stream.getvalue())])
        self.sessions = [result["session_id"]]
        selected = retrieve(result["session_id"], "请识别节点图中的构造")
        self.assertEqual(len(selected["selected_visuals"]), 1)
        self.assertEqual(selected["input_snapshot"]["selected_visual_ids"], ["image:document:1"])
        selected_visual = selected["selected_visuals"][0]
        stored = get_visual_asset(result["session_id"], selected_visual["document_id"], selected_visual["visual_id"])
        self.assertIsNotNone(stored)
        self.assertEqual(stored.image_bytes, stream.getvalue())
        self.assertIsNone(get_visual_asset(result["session_id"], "another-document", selected_visual["visual_id"]))
        with temporary_visual_files(selected["selected_visuals"]) as paths:
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
