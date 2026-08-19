from __future__ import annotations

from io import BytesIO
import unittest

from openpyxl import Workbook

from backend.documents.evidence_v2 import convert_intermediate_to_v2
from backend.document_parsing.ingestion import (
    IntermediateDocument,
    SourcePointer,
    evidence_block,
    finalize_intermediate_evidence,
    ingest_excel,
    ingest_text,
)


class EvidenceV2Tests(unittest.TestCase):
    def test_text_source_becomes_typed_text_span_without_mutating_v1(self) -> None:
        result = ingest_text("2025年营业收入1200万元")
        original_version = result.intermediate.evidence_schema_version

        first = convert_intermediate_to_v2(result.intermediate)
        second = convert_intermediate_to_v2(result.intermediate)

        self.assertEqual(original_version, "evidence-document-v1")
        self.assertEqual(result.intermediate.evidence_schema_version, original_version)
        self.assertEqual(first.evidence_schema_version, "evidence-document-v2")
        self.assertEqual(first.revision_id, second.revision_id)
        self.assertEqual(first.evidence_snapshot_hash, second.evidence_snapshot_hash)
        block = next(item for item in first.elements if item.element_type == "block")
        self.assertEqual(block.source.location.location_type, "text_span")
        self.assertEqual(block.source.location.start, 0)

    def test_excel_table_and_cells_receive_stable_global_ids_and_range_location(self) -> None:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "收入汇总"
        worksheet.append(["项目", "2025年"])
        worksheet.append(["营业收入", 118_000_000])
        worksheet["C2"] = "=B2/100000000"
        stream = BytesIO()
        workbook.save(stream)
        workbook.close()

        result = ingest_excel("report.xlsx", stream.getvalue())
        evidence = convert_intermediate_to_v2(result.intermediate)
        table = next(item for item in evidence.elements if item.element_type == "table")

        self.assertTrue(table.element_id.startswith(f"{evidence.document_id}/"))
        self.assertEqual(table.source.location.location_type, "excel_range")
        self.assertEqual(table.source.location.sheet_name, "收入汇总")
        self.assertEqual(table.source.location.range, "A1:C2")
        formula_cell = next(cell for row in table.rows for cell in row if cell.coordinate == "C2")
        self.assertEqual(formula_cell.formula, "=B2/100000000")
        self.assertEqual(formula_cell.cell_id, f"{table.element_id}/cell/C2")

    def test_file_hash_and_snapshot_hash_are_separate(self) -> None:
        result = ingest_text("2025年营业收入1200万元")
        evidence = convert_intermediate_to_v2(result.intermediate)

        self.assertEqual(evidence.file.file_sha256, result.intermediate.file_sha256)
        self.assertEqual(len(evidence.evidence_snapshot_hash), 64)
        self.assertNotEqual(evidence.file.file_sha256, evidence.evidence_snapshot_hash)

    def test_pdf_page_location_preserves_raw_bbox_without_claiming_normalization(self) -> None:
        source = SourcePointer(
            source_type="pdf",
            file_name="report.pdf",
            page_number=12,
            section_id="pdf:page:12",
            parser="fixture",
            bounding_box=(120.0, 200.0, 900.0, 700.0),
        )
        document = finalize_intermediate_evidence(
            IntermediateDocument(
                source_type="pdf",
                file_name="report.pdf",
                parser="fixture",
                tables=[],
                blocks=[
                    evidence_block(
                        block_id="pdf:page:12",
                        kind="page_text",
                        source=source,
                        text="营业收入118亿元",
                    )
                ],
            ),
            b"fixture-pdf",
        )

        evidence = convert_intermediate_to_v2(document)
        block = next(item for item in evidence.elements if item.element_type == "block")

        self.assertEqual(block.source.location.location_type, "page_region")
        self.assertEqual(block.source.location.page_number, 12)
        self.assertEqual(block.source.location.raw_bbox, (120.0, 200.0, 900.0, 700.0))
        self.assertIsNone(block.source.location.normalized_bbox)
        self.assertEqual(block.source.location.coordinate_system.bbox_format, "xyxy_source")


if __name__ == "__main__":
    unittest.main()
