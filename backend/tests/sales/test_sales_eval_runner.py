"""Regression tests for the local sales benchmark result contract."""

from __future__ import annotations

import unittest

from scripts.run_sales_rag_eval import prediction_text, result_record


class SalesEvalRunnerTests(unittest.TestCase):
    def test_prediction_uses_customer_reply_without_repeating_visual_observations(self) -> None:
        response = {
            "customer_reply": "图中可见挖掘机正在向卡车装载石料。",
            "key_points": [],
            "image_observations": ["挖掘机正在向卡车装载石料"],
        }
        self.assertEqual(
            prediction_text(response),
            "图中可见挖掘机正在向卡车装载石料。",
        )

    def test_direct_visual_input_is_not_scored_as_rag_visual_retrieval(self) -> None:
        sample = {
            "sample_id": "visual_fixture",
            "task_type": "grounded_visual_qa",
            "capability": "visual_scene_understanding",
            "question": "图中有什么？",
            "answerable": True,
            "expected_answer": "一辆卡车",
            "expected_points": ["卡车"],
            "gold_evidence": [
                {"evidence_type": "visual", "evidence_id": "asset_gold"}
            ],
        }
        response = {
            "answerable": True,
            "customer_reply": "图中可见一辆卡车。",
            "key_points": [],
            "citations": [],
            "visual_assets": [],
            "image_observations": ["一辆卡车"],
            "meta": {"customer_image_processed_locally": True},
        }
        record = result_record(sample, response, 1.0)
        self.assertEqual(record["input_visual_ids"], ["asset_gold"])
        self.assertTrue(record["visual_input_processed"])
        self.assertIsNone(record["visual_evidence_hit"])
        self.assertEqual(record["retrieved_visual_ids"], [])


if __name__ == "__main__":
    unittest.main()
