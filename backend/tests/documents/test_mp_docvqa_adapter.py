from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from pypdf import PdfWriter

from backend.documents.evidence_v2 import EvidenceDocumentV2
from backend.documents.training_sample import TrainingSample
import importlib.util
if importlib.util.find_spec("training") is None:
    raise unittest.SkipTest("Optional MP-DocVQA training adapter is not part of this functional RAG release")

from training.document_qa.adapters.mp_docvqa import (
    MPDocVQAAdapter,
    MPDocVQARecord,
    load_official_records,
    validate_document_split_isolation,
)


def _blank_pdf(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    from io import BytesIO

    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _record(
    question_id: int,
    *,
    answer_page_idx: int | None = 2,
    split: str = "train",
) -> MPDocVQARecord:
    return MPDocVQARecord.model_validate(
        {
            "questionId": question_id,
            "question": "What is the invoice total?",
            "doc_id": "doc_fixture",
            "page_ids": ["page_a", "page_b", "page_c"],
            "answers": ["$1,250", "1250 dollars"],
            "answer_page_idx": answer_page_idx,
            "data_split": split,
        }
    )


def _window_record(
    question_id: int,
    page_ids: list[str],
    answer_page_idx: int,
) -> MPDocVQARecord:
    return MPDocVQARecord.model_validate(
        {
            "questionId": question_id,
            "question": f"Question {question_id}?",
            "doc_id": "doc_variable_window",
            "page_ids": page_ids,
            "answers": [f"answer-{question_id}"],
            "answer_page_idx": answer_page_idx,
            "data_split": "train",
        }
    )


class MPDocVQAAdapterTests(unittest.TestCase):
    def test_official_zero_based_index_two_maps_to_evidence_page_three(self) -> None:
        with TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "fixture.pdf"
            pdf_path.write_bytes(_blank_pdf(3))
            evidence = MPDocVQAAdapter.parse_pdf(pdf_path)
            sample, audit = MPDocVQAAdapter().build_sample(_record(101), evidence)

        self.assertEqual(sample.annotation_level, "page")
        self.assertEqual(sample.acceptable_answers, ["$1,250", "1250 dollars"])
        self.assertEqual(len(sample.selected_evidence_ids), 1)
        self.assertEqual(sample.evidence_ids, sample.selected_evidence_ids)
        selected = next(
            item for item in evidence.elements if item.element_id == sample.selected_evidence_ids[0]
        )
        self.assertEqual(selected.source.location.page_number, 3)
        self.assertEqual(audit.official_answer_page_index, 2)
        self.assertEqual(audit.resolved_page_number, 3)
        self.assertEqual(audit.official_page_id, "page_c")

    def test_out_of_range_page_is_written_as_explicit_failed_mapping(self) -> None:
        with TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "fixture.pdf"
            pdf_path.write_bytes(_blank_pdf(3))
            evidence = MPDocVQAAdapter.parse_pdf(pdf_path)
            sample, audit = MPDocVQAAdapter().build_sample(
                _record(102, answer_page_idx=3), evidence
            )

        self.assertEqual(sample.adapter_mapping.status, "failed")
        self.assertEqual(sample.selected_evidence_ids, [])
        self.assertEqual(sample.adapter_mapping.failure_reason, "answer_page_index_out_of_range")
        self.assertEqual(audit.failure_reason, "answer_page_index_out_of_range")

    def test_same_document_cannot_cross_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "Document split leakage"):
            validate_document_split_isolation(
                [_record(103, split="train"), _record(104, split="test")]
            )

    def test_one_evidence_file_supports_multiple_schema_valid_samples(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "documents"
            output = root / "output"
            documents.mkdir()
            (documents / "doc_fixture.pdf").write_bytes(_blank_pdf(3))
            records = [_record(105, answer_page_idx=0), _record(106, answer_page_idx=2)]

            summary = MPDocVQAAdapter().convert(
                records,
                documents_dir=documents,
                output_dir=output,
            )

            evidence_files = list((output / "evidence").glob("*.json"))
            sample_lines = (output / "samples" / "mpdocvqa_train.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            evidence = EvidenceDocumentV2.model_validate_json(
                evidence_files[0].read_text(encoding="utf-8")
            )
            samples = [TrainingSample.model_validate_json(line) for line in sample_lines]
            audit_lines = (output / "mapping_audit.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(evidence_files), 1)
        self.assertEqual(len(samples), 2)
        self.assertEqual({sample.document_id for sample in samples}, {evidence.document_id})
        self.assertEqual(summary.schema_valid_rate, 1.0)
        self.assertEqual(summary.samples_page_mapped, 2)
        self.assertEqual(len(audit_lines), 2)

    def test_evidence_identity_is_stable_for_same_pdf(self) -> None:
        with TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "fixture.pdf"
            pdf_path.write_bytes(_blank_pdf(3))
            first = MPDocVQAAdapter.parse_pdf(pdf_path)
            second = MPDocVQAAdapter.parse_pdf(pdf_path)

        self.assertEqual(first.file.file_sha256, second.file.file_sha256)
        self.assertEqual(first.evidence_snapshot_hash, second.evidence_snapshot_hash)
        self.assertEqual(first.revision_id, second.revision_id)
        self.assertEqual(
            [item.element_id for item in first.elements],
            [item.element_id for item in second.elements],
        )

    def test_official_json_wrapper_is_supported(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.json"
            payload = {"dataset_name": "MP-DocVQA", "data": [_record(107).model_dump()]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_official_records(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(str(loaded[0].questionId), "107")

    def test_official_page_images_create_stable_multimodal_asset_reference(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            output = root / "output"
            images.mkdir()
            for name, color in [
                ("page_c", "blue"),
                ("page_a", "red"),
                ("page_b", "green"),
            ]:
                Image.new("RGB", (24, 16), color=color).save(images / f"{name}.png")

            summary = MPDocVQAAdapter().convert(
                [_record(108, answer_page_idx=2)],
                documents_dir=None,
                page_images_dir=images,
                output_dir=output,
                refusal_candidate_ratio=1.0,
            )
            lines = (output / "samples" / "mpdocvqa_train.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            samples = [TrainingSample.model_validate_json(line) for line in lines]
            sample = next(item for item in samples if item.task_type == "grounded_document_qa")
            selection = next(item for item in samples if item.task_type == "evidence_selection")
            asset = sample.input_context.visual_assets[0]
            stored_asset = output / asset.storage_ref
            evidence_path = next((output / "evidence").glob("*.json"))
            evidence = EvidenceDocumentV2.model_validate_json(
                evidence_path.read_text(encoding="utf-8")
            )
            stored_asset_exists = stored_asset.is_file()
            refusal = json.loads(
                (output / "review" / "mpdocvqa_refusal_candidates.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            output_second = root / "output_second"
            MPDocVQAAdapter().convert(
                [_record(108, answer_page_idx=2)],
                documents_dir=None,
                page_images_dir=images,
                output_dir=output_second,
                refusal_candidate_ratio=1.0,
            )
            second_samples = [
                TrainingSample.model_validate_json(line)
                for line in (output_second / "samples" / "mpdocvqa_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            second_selection = next(
                item for item in second_samples if item.task_type == "evidence_selection"
            )

        self.assertEqual(summary.samples_page_mapped, 2)
        self.assertEqual(summary.grounded_qa_samples, 1)
        self.assertEqual(summary.evidence_selection_samples, 1)
        self.assertEqual(summary.refusal_review_candidates, 1)
        self.assertEqual(sample.input_context.input_modalities, ["image", "text"])
        self.assertEqual(sample.input_context.image_refs, [asset.storage_ref])
        self.assertEqual(asset.page_number, 3)
        self.assertTrue(asset.storage_ref.endswith("0003_page_c.png"))
        self.assertTrue(asset.visual_asset_id.startswith(f"{evidence.document_id}/"))
        self.assertTrue(stored_asset_exists)
        self.assertEqual(len(asset.sha256), 64)
        selected = next(
            item for item in evidence.elements if item.element_id == sample.selected_evidence_ids[0]
        )
        self.assertEqual(selected.source.location.location_type, "image_region")
        self.assertEqual(selected.source.location.page_number, 3)
        self.assertEqual(len(selection.candidate_evidence_ids), 3)
        self.assertIn(selection.selected_evidence_ids[0], selection.candidate_evidence_ids)
        self.assertEqual(
            selection.candidate_evidence_ids,
            [item.evidence_id for item in selection.input_context.visual_assets],
        )
        self.assertEqual(
            selection.input_context.image_refs,
            [item.storage_ref for item in selection.input_context.visual_assets],
        )
        self.assertEqual(len(selection.candidate_sampling.negative_evidence_ids), 2)
        self.assertEqual(selection.candidate_sampling.strategy, "adjacent_then_stable_hash")
        self.assertEqual(
            selection.candidate_evidence_ids,
            second_selection.candidate_evidence_ids,
        )
        self.assertEqual(
            selection.candidate_sampling.deterministic_seed_sha256,
            second_selection.candidate_sampling.deterministic_seed_sha256,
        )
        self.assertFalse(refusal["training_eligible"])
        self.assertEqual(refusal["validation_status"], "needs_verification")
        self.assertNotIn(selection.selected_evidence_ids[0], refusal["candidate_evidence_ids"])

    def test_missing_official_page_image_is_audited_not_silently_skipped(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            output = root / "output"
            images.mkdir()
            Image.new("RGB", (8, 8), color="white").save(images / "page_a.png")

            summary = MPDocVQAAdapter().convert(
                [_record(109)],
                documents_dir=None,
                page_images_dir=images,
                output_dir=output,
            )
            audit = json.loads((output / "mapping_audit.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(summary.samples_written, 0)
        self.assertEqual(summary.samples_failed, 1)
        self.assertEqual(audit["mapping_status"], "failed")
        self.assertEqual(audit["failure_reason"], "source_page_image_not_found:page_b")

    def test_max_questions_limits_official_questions_not_generated_rows(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            output = root / "output"
            images.mkdir()
            for page_id in ("page_a", "page_b", "page_c"):
                Image.new("RGB", (8, 8), color="white").save(images / f"{page_id}.png")

            summary = MPDocVQAAdapter().convert(
                [_record(110), _record(111)],
                documents_dir=None,
                page_images_dir=images,
                output_dir=output,
                max_questions=1,
                refusal_candidate_ratio=0.0,
            )
            lines = (output / "samples" / "mpdocvqa_train.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(summary.questions_converted, 1)
        self.assertEqual(summary.samples_written, 2)
        self.assertEqual(len(lines), 2)

    def test_same_document_supports_different_overlapping_page_windows(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            output = root / "output"
            images.mkdir()
            for page_id in ("page_a", "page_b", "page_c", "page_d", "page_e"):
                Image.new("RGB", (8, 8), color="white").save(images / f"{page_id}.png")
            records = [
                _window_record(201, ["page_a", "page_b", "page_c"], 2),
                _window_record(202, ["page_b", "page_c", "page_d", "page_e"], 2),
            ]

            summary = MPDocVQAAdapter().convert(
                records,
                documents_dir=None,
                page_images_dir=images,
                output_dir=output,
                max_negative_pages=3,
                refusal_candidate_ratio=0.0,
            )
            evidence = EvidenceDocumentV2.model_validate_json(
                next((output / "evidence").glob("*.json")).read_text(encoding="utf-8")
            )
            samples = [
                TrainingSample.model_validate_json(line)
                for line in (output / "samples" / "mpdocvqa_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            audits = [
                json.loads(line)
                for line in (output / "mapping_audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(summary.documents_converted, 1)
        self.assertEqual(summary.questions_converted, 2)
        self.assertEqual(summary.samples_failed, 0)
        page_elements = [
            item
            for item in evidence.elements
            if item.element_type == "block" and item.block_kind == "page_text"
        ]
        self.assertEqual(len(page_elements), 5)
        grounded_201 = next(item for item in samples if item.sample_id == "mpdocvqa_201")
        grounded_202 = next(item for item in samples if item.sample_id == "mpdocvqa_202")
        page_201 = next(
            item.source.location.page_number
            for item in evidence.elements
            if item.element_id == grounded_201.selected_evidence_ids[0]
        )
        page_202 = next(
            item.source.location.page_number
            for item in evidence.elements
            if item.element_id == grounded_202.selected_evidence_ids[0]
        )
        self.assertEqual(page_201, 3)  # official page_c
        self.assertEqual(page_202, 4)  # local index 2 is official page_d
        audit_202 = next(item for item in audits if item["sample_id"] == "mpdocvqa_202")
        self.assertEqual(audit_202["official_page_id"], "page_d")
        self.assertEqual(audit_202["resolved_page_number"], 4)

        selection_202 = next(
            item for item in samples if item.sample_id == "mpdocvqa_202_selection"
        )
        candidate_pages = {
            asset.page_number for asset in selection_202.input_context.visual_assets
        }
        self.assertEqual(candidate_pages, {2, 3, 4, 5})
        self.assertNotIn(1, candidate_pages)  # page_a is outside question 202's window

    def test_contradictory_page_window_order_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            for page_id in ("page_a", "page_b"):
                Image.new("RGB", (8, 8), color="white").save(images / f"{page_id}.png")
            records = [
                _window_record(203, ["page_a", "page_b"], 0),
                _window_record(204, ["page_b", "page_a"], 0),
            ]

            with self.assertRaisesRegex(ValueError, "Contradictory page_ids ordering"):
                MPDocVQAAdapter().convert(
                    records,
                    documents_dir=None,
                    page_images_dir=images,
                    output_dir=root / "output",
                )

    def test_disconnected_windows_become_separate_document_views(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            output = root / "output"
            images.mkdir()
            for page_id in ("p1", "p2", "p3", "p8", "p9", "p10"):
                Image.new("RGB", (8, 8), color="white").save(images / f"{page_id}.png")
            records = [
                _window_record(205, ["p1", "p2", "p3"], 1),
                _window_record(206, ["p8", "p9", "p10"], 0),
            ]

            summary = MPDocVQAAdapter().convert(
                records,
                documents_dir=None,
                page_images_dir=images,
                output_dir=output,
                refusal_candidate_ratio=0.0,
            )
            evidence_files = list((output / "evidence").glob("*.json"))
            samples = [
                TrainingSample.model_validate_json(line)
                for line in (output / "samples" / "mpdocvqa_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            audits = [
                json.loads(line)
                for line in (output / "mapping_audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(summary.documents_seen, 1)
        self.assertEqual(summary.documents_converted, 1)
        self.assertEqual(summary.document_views_converted, 2)
        self.assertEqual(len(evidence_files), 2)
        grounded = [item for item in samples if item.task_type == "grounded_document_qa"]
        self.assertEqual(len({item.document_id for item in grounded}), 2)
        self.assertTrue(all(item.adapter_mapping.order_status == "disconnected_component" for item in grounded))
        self.assertTrue(all(item.adapter_mapping.canonical_page_number is None for item in grounded))
        self.assertEqual(
            {item.adapter_mapping.local_page_index for item in grounded},
            {0, 1},
        )
        self.assertEqual(
            {item.adapter_mapping.official_page_id for item in grounded},
            {"p2", "p8"},
        )
        self.assertEqual(len({item.adapter_mapping.document_view_id for item in grounded}), 2)
        self.assertTrue(all(item["canonical_page_number"] is None for item in audits))
        self.assertTrue(all(item["order_status"] == "disconnected_component" for item in audits))


if __name__ == "__main__":
    unittest.main()
