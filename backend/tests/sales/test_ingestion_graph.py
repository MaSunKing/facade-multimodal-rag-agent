"""Tests for the review-first local knowledge-intake graph."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.sales.ingestion_graph import IntakeOptions, LocalIntakeOperations, build_knowledge_intake_graph


class FakeIntakeOperations:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def validate_source(self, options):
        self.calls.append("validate")
        return {"run_id": "run_1", "run_dir": "C:/tmp/run_1", "source_path": "C:/tmp/a.pdf", "source_format": "pdf"}

    def classify_document(self, options, run):
        self.calls.append("classify")
        return {"knowledge_domain": "02_product_information", "review_required": True}

    def extract_local_assets(self, options, run, classification):
        self.calls.append("extract")
        return {"status": "complete", "text_record_count": 8, "visual_asset_count": 3}

    def annotate_local_visuals(self, options, extraction):
        self.calls.append("annotate")
        return {"status": "queued_for_local_annotation", "asset_count": 3}

    def build_review_package(self, options, run, classification, extraction, visual_annotation):
        self.calls.append("review")
        return {"status": "needs_human_approval", "promotion_status": "not_promoted", "review_manifest": "C:/tmp/review.json"}


class KnowledgeIntakeGraphTests(unittest.TestCase):
    def test_graph_finishes_as_review_package_without_promotion(self) -> None:
        fake = FakeIntakeOperations()
        graph = build_knowledge_intake_graph(fake)
        state = graph.invoke({"options": IntakeOptions(source_path=Path("C:/tmp/a.pdf"), dry_run=True)})

        self.assertEqual(fake.calls, ["validate", "classify", "extract", "annotate", "review"])
        self.assertEqual(state["review"]["status"], "needs_human_approval")
        self.assertEqual(state["review"]["promotion_status"], "not_promoted")
        self.assertEqual([item["node"] for item in state["events"]], [
            "validate_source",
            "classify_document",
            "extract_local_assets",
            "annotate_visual_assets",
            "build_review_package",
        ])

    def test_document_classification_uses_a_public_fixture_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "外墙保温装饰板粘锚施工工艺.pdf"
            source.write_bytes(b"placeholder")
            taxonomy_path = Path(temporary) / "taxonomy.json"
            taxonomy_path.write_text(
                '{"documents":{"外墙保温装饰板粘锚施工工艺":{'
                '"knowledge_domain":"03_construction_method",'
                '"document_category":"enterprise_construction_method",'
                '"source_authority":"enterprise_document",'
                '"customer_fact_policy":"construction_reference"}}}',
                encoding="utf-8",
            )
            options = IntakeOptions(source_path=source, taxonomy_path=taxonomy_path)
            run = LocalIntakeOperations().validate_source(options)
            classification = LocalIntakeOperations().classify_document(options, run)

        self.assertEqual(classification["knowledge_domain"], "03_construction_method")
        self.assertEqual(classification["classification_source"], "existing_taxonomy")
        self.assertTrue(classification["review_required"])
        self.assertFalse(classification["customer_shareable"])
