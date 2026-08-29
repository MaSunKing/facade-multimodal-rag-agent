import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.sales.retriever import (
    LocalRagRetriever,
    RAG_INDEX_PATH,
    reciprocal_rank_fusion,
    select_rerank_candidates,
)


class HybridFusionTests(unittest.TestCase):
    def test_rrf_keeps_dense_only_text_in_the_rerank_budget(self) -> None:
        documents = {
            **{
                f"lexical-{index:02d}": {
                    "id": f"lexical-{index:02d}",
                    "kind": "text",
                    "text": f"lexical {index}",
                }
                for index in range(1, 49)
            },
            "dense-only": {"id": "dense-only", "kind": "text", "text": "semantic result"},
        }
        lexical = [
            (49.0 - index, documents[f"lexical-{index:02d}"])
            for index in range(1, 49)
        ]
        fused = reciprocal_rank_fusion(
            lexical,
            ["dense-only", *[f"lexical-{index:02d}" for index in range(48, 0, -1)]],
            documents,
        )
        candidates = select_rerank_candidates(
            fused,
            [f"lexical-{index:02d}" for index in range(1, 49)],
            ["dense-only", *[f"lexical-{index:02d}" for index in range(48, 0, -1)]],
            documents,
        )
        first_24_text_ids = [document["id"] for _, document in candidates[:24]]
        self.assertIn("dense-only", first_24_text_ids)

    def test_rrf_keeps_lexical_top_candidate_missing_from_dense(self) -> None:
        documents = {
            **{
                f"lexical-{index:02d}": {
                    "id": f"lexical-{index:02d}",
                    "kind": "text",
                    "text": f"lexical {index}",
                }
                for index in range(1, 49)
            },
            **{
                f"dense-only-{index:02d}": {
                    "id": f"dense-only-{index:02d}",
                    "kind": "text",
                    "text": f"dense {index}",
                }
                for index in range(1, 3)
            },
        }
        lexical_ids = [f"lexical-{index:02d}" for index in range(1, 49)]
        # Almost all lower-ranked lexical results also occur in the dense list,
        # so their two RRF contributions crowd out lexical ranks 1 and 2.  The
        # two best lexical hits are deliberately absent from dense retrieval.
        dense_ids = [
            "dense-only-01",
            "dense-only-02",
            *[f"lexical-{index:02d}" for index in range(3, 49)],
        ]
        lexical = [
            (49.0 - index, documents[document_id])
            for index, document_id in enumerate(lexical_ids)
        ]
        fused = reciprocal_rank_fusion(lexical, dense_ids, documents)

        candidates = select_rerank_candidates(
            fused,
            lexical_ids,
            dense_ids,
            documents,
            non_visual_limit=24,
        )

        candidate_ids = [document["id"] for _, document in candidates[:24]]
        self.assertIn("lexical-01", candidate_ids)
        self.assertIn("lexical-02", candidate_ids)
        self.assertTrue(any(document_id.startswith("dense-only-") for document_id in candidate_ids))

    def test_dense_validation_requires_fingerprint_and_exact_id_set(self) -> None:
        previous = os.environ.get("RAG_HYBRID_ENABLED")
        os.environ["RAG_HYBRID_ENABLED"] = "1"
        try:
            retriever = LocalRagRetriever.__new__(LocalRagRetriever)
            retriever.payload = {"metadata": {"index_fingerprint": "lexical-v1"}}
            retriever.documents = [
                {"id": "a", "text": "A"},
                {"id": "b", "text": "B"},
            ]
            retriever._dense_ids = []
            retriever._dense_vectors = None
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dense_path = root / "dense.npz"
                metadata_path = root / "dense.json"
                np.savez(
                    dense_path,
                    ids=np.asarray(["a", "b"]),
                    vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                )
                metadata_path.write_text(
                    json.dumps({"source_index_fingerprint": "lexical-v1"}),
                    encoding="utf-8",
                )
                valid = retriever._validate_dense_index(dense_path, metadata_path)
                self.assertTrue(valid["ready"])
                self.assertEqual(valid["reason"], None)

                metadata_path.write_text(
                    json.dumps({"source_index_fingerprint": "older-index"}),
                    encoding="utf-8",
                )
                stale = retriever._validate_dense_index(dense_path, metadata_path)
                self.assertFalse(stale["ready"])
                self.assertEqual(stale["reason"], "fingerprint_mismatch")

                metadata_path.write_text(
                    json.dumps({"source_index_fingerprint": "lexical-v1"}),
                    encoding="utf-8",
                )
                np.savez(
                    dense_path,
                    ids=np.asarray(["a"]),
                    vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
                )
                incomplete = retriever._validate_dense_index(dense_path, metadata_path)
                self.assertFalse(incomplete["ready"])
                self.assertEqual(incomplete["reason"], "document_id_mismatch")
                self.assertEqual(incomplete["missing_document_count"], 1)
        finally:
            if previous is None:
                os.environ.pop("RAG_HYBRID_ENABLED", None)
            else:
                os.environ["RAG_HYBRID_ENABLED"] = previous

    def test_hybrid_rule_bonus_is_bounded_below_strong_reranker_signal(self) -> None:
        retriever = LocalRagRetriever.__new__(LocalRagRetriever)
        documents = [
            {
                "id": "strong-model",
                "kind": "text",
                "text": "strong",
                "tokens": ["问题"],
                "_retrieval_score_mode": "hybrid",
            },
            {
                "id": "large-rule",
                "kind": "text",
                "text": "rule",
                "tokens": ["问题"],
                "_retrieval_score_mode": "hybrid",
            },
        ]
        zero_methods = (
            "_source_name_affinity",
            "_rare_query_affinity",
            "_rare_query_character_coverage",
            "_exact_query_phrase_affinity",
            "_question_subject_affinity",
            "temporal_precondition_affinity",
        )
        patches = [patch.object(retriever, name, return_value=0.0) for name in zero_methods]
        for active_patch in patches:
            active_patch.start()
        try:
            with patch.object(
                retriever,
                "_choice_condition_affinity",
                side_effect=lambda _query, document: 360.0 if document["id"] == "large-rule" else 0.0,
            ):
                ranked = retriever._rank_with_rules(
                    "问题",
                    [(0.9, documents[0]), (0.2, documents[1])],
                    node_atlas_request=False,
                    standard_request=False,
                    procedure_request=False,
                    product_overview_request=False,
                    reviewed_product_fact_request=False,
                )
            self.assertEqual(ranked[0][1]["id"], "strong-model")
            self.assertAlmostEqual(ranked[1][0], 0.6)
        finally:
            for active_patch in reversed(patches):
                active_patch.stop()

    def test_final_sort_preserves_requested_domain_priority(self) -> None:
        retriever = LocalRagRetriever.__new__(LocalRagRetriever)
        requested = {
            "id": "standard",
            "kind": "text",
            "text": "standard",
            "knowledge_domains": ["01_standard_specification"],
            "_retrieval_score_mode": "hybrid",
        }
        generic = {
            "id": "generic",
            "kind": "text",
            "text": "generic",
            "_retrieval_score_mode": "hybrid",
        }
        affinity_methods = (
            "_source_name_affinity",
            "_rare_query_affinity",
            "_rare_query_character_coverage",
            "_exact_query_phrase_affinity",
            "_question_subject_affinity",
            "_choice_condition_affinity",
            "temporal_precondition_affinity",
        )
        patches = [patch.object(retriever, name, return_value=0.0) for name in affinity_methods]
        for active_patch in patches:
            active_patch.start()
        try:
            ranked = retriever._rank_with_rules(
                "规范",
                [(0.8, requested), (0.99, generic)],
                node_atlas_request=False,
                standard_request=True,
                procedure_request=False,
                product_overview_request=False,
                reviewed_product_fact_request=False,
            )
            self.assertEqual(ranked[0][1]["id"], "standard")
        finally:
            for active_patch in reversed(patches):
                active_patch.stop()

    def test_weak_overlap_rejects_single_chinese_character_only(self) -> None:
        self.assertFalse(
            LocalRagRetriever._has_substantive_query_overlap(
                ["图", "施工方案"], {"tokens": ["图", "标准"]}
            )
        )
        self.assertTrue(
            LocalRagRetriever._has_substantive_query_overlap(
                ["图", "施工方案"], {"tokens": ["图", "施工方案"]}
            )
        )

    def test_answer_form_affinity_prefers_measurement_action_over_requirement(self) -> None:
        query = "如何检查基层表面平整度"
        actionable = {
            "text": "基层墙体表面平整度用2m靠尺测量，缝间隙应小于5mm。"
        }
        requirement_only = {"text": "基层墙体表面平整度应符合相关标准要求。"}
        self.assertEqual(
            LocalRagRetriever._answer_form_affinity(query, actionable), 0.18
        )
        self.assertEqual(
            LocalRagRetriever._answer_form_affinity(query, requirement_only), 0.0
        )
        self.assertEqual(LocalRagRetriever._rerank_text_limit(query), 32)

    def test_answer_form_affinity_prefers_actual_composition_list_over_heading(self) -> None:
        query = "系统由哪些主要材料组成"
        direct_list = {
            "text": "由保温装饰板、粘结砂浆、锚固件、嵌缝材料和密封胶组成。"
        }
        descriptive = {
            "text": "由保温材料、装饰面板以及胶粘剂、连接件复合而成。"
        }
        heading = {"text": "系统组成材料、主要原材料发生变化时应重新检验。"}
        self.assertGreater(
            LocalRagRetriever._answer_form_affinity(query, direct_list),
            LocalRagRetriever._answer_form_affinity(query, descriptive),
        )
        self.assertEqual(LocalRagRetriever._answer_form_affinity(query, heading), 0.0)
        self.assertEqual(LocalRagRetriever._rerank_text_limit(query), 32)

    def test_real_predicate_questions_keep_gold_evidence_in_lexical_top_five(self) -> None:
        if not RAG_INDEX_PATH.exists():
            self.skipTest("public release intentionally excludes the private RAG index")
        previous = os.environ.get("RAG_HYBRID_ENABLED")
        os.environ["RAG_HYBRID_ENABLED"] = "0"
        try:
            retriever = LocalRagRetriever()
            cases = (
                ("如何检查基层表面平整度", "txt_782b8e5c5569410a"),
                ("系统由哪些主要材料组成", "txt_f082a9f29e1d659a"),
            )
            for query, gold_id in cases:
                result = retriever.retrieve(query, top_k=5, visual_k=0, case_k=0)
                returned_ids = [item["chunk_id"] for item in result["text_evidence"]]
                self.assertIn(gold_id, returned_ids, msg=f"{query}: {returned_ids}")
        finally:
            if previous is None:
                os.environ.pop("RAG_HYBRID_ENABLED", None)
            else:
                os.environ["RAG_HYBRID_ENABLED"] = previous


if __name__ == "__main__":
    unittest.main()
