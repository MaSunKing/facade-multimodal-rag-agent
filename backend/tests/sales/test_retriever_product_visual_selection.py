from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.sales.retriever import LocalRagRetriever


def product_visual(
    asset_id: str,
    *,
    role: str,
    title: str,
    variant: str | None = None,
    priority: int = 100,
    eligible: bool = True,
    review_status: str = "auto_source_linked",
) -> dict:
    return {
        "id": f"visual:{asset_id}",
        "kind": "visual",
        "asset_id": asset_id,
        "customer_title": title,
        "customer_caption": title,
        "asset_type": "original_pdf_image",
        "effective_image_kind": "product_photo",
        "gallery_type": "product_sample",
        "product_name": "真岩®石",
        "canonical_product": "真岩®石",
        "variant_or_code": variant,
        "visual_role": role,
        "product_gallery_eligible": eligible,
        "review_status": review_status,
        "display_priority": priority,
        "citation": {"document_name": "产品目录", "source_page": 1},
    }


class ProductVisualSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = LocalRagRetriever.__new__(LocalRagRetriever)
        self.overview = product_visual(
            "overview", role="product_overview", title="真岩®石仿石装饰板产品展示", priority=110
        )
        self.variant = product_visual(
            "variant",
            role="product_variant",
            title="黄金麻 GHM2517 饰面板",
            variant="黄金麻 / GHM2517",
        )
        self.accessory = product_visual(
            "anchor", role="accessory", title="外墙锚钉产品展示", eligible=False, priority=20
        )

    def test_broad_overview_returns_only_overview_and_explicit_variant(self) -> None:
        ranked = self.retriever._prioritise_product_visuals(
            "我想了解一下你们的产品 给我图片",
            [(20.0, self.variant), (200.0, self.accessory), (10.0, self.overview)],
            product_overview_request=True,
        )
        self.assertEqual([item[1]["asset_id"] for item in ranked], ["overview", "variant"])

    def test_specific_unknown_variant_returns_no_substitute(self) -> None:
        ranked = self.retriever._prioritise_product_visuals(
            "给我看白麻产品图片",
            [(20.0, self.variant), (10.0, self.overview)],
            product_overview_request=False,
        )
        self.assertEqual(ranked, [])

    def test_specific_code_returns_matching_variant_and_stable_payload(self) -> None:
        ranked = self.retriever._prioritise_product_visuals(
            "请展示 GHM2517 产品图片",
            [(10.0, self.overview), (20.0, self.variant)],
            product_overview_request=False,
        )
        self.assertEqual([item[1]["asset_id"] for item in ranked], ["variant"])
        payload = self.retriever._visual_payload(
            ranked[0][1],
            score=ranked[0][0],
            gallery_type="product",
            matched_variant_or_code="GHM2517",
        )
        self.assertEqual(payload["product_name"], "真岩®石")
        self.assertEqual(payload["variant_or_code"], "黄金麻 / GHM2517")
        self.assertEqual(payload["visual_role"], "product_variant")
        self.assertIn("真岩®石 黄金麻 / GHM2517样板图", payload["explanation"])
        self.assertIn("来源页与位置可追溯", payload["explanation"])

        overview_payload = self.retriever._visual_payload(
            self.overview,
            score=10.0,
            gallery_type="product",
        )
        self.assertIn("真岩®石产品样板总览", overview_payload["explanation"])
        self.assertIn("来源页与位置可追溯", overview_payload["explanation"])

    def test_review_required_or_project_visual_is_not_product_gallery_eligible(self) -> None:
        needs_review = product_visual(
            "review", role="product_variant", title="白麻样板", variant="白麻", review_status="needs_review"
        )
        project = {
            **product_visual("case", role="application_effect", title="项目实景", eligible=False),
            "gallery_type": "project_case",
        }
        self.assertFalse(self.retriever._is_product_gallery_visual(needs_review))
        self.assertFalse(self.retriever._is_product_gallery_visual(project))

    def _retriever_with_global_pool_missing_gallery(self) -> LocalRagRetriever:
        """Build the smallest complete retriever needed for an end-to-end call."""

        retriever = LocalRagRetriever.__new__(LocalRagRetriever)
        text = {
            "id": "text:generic",
            "kind": "text",
            "text": "公司产品资料",
            "tokens": ["公司", "产品", "资料"],
            "source_refs": [],
        }
        retriever.documents = [text, self.overview, self.variant, self.accessory]
        retriever.document_count = len(retriever.documents)
        retriever.document_frequency = {}
        retriever.avg_doc_length = 4.0
        retriever.product_alias_groups = []
        retriever.payload = {"metadata": {}}
        retriever._hybrid_validation = {
            "requested": False,
            "ready": False,
            "reason": "test_fixture",
        }
        retriever.document_position_by_id = {
            str(document.get("id")): index
            for index, document in enumerate(retriever.documents)
        }
        retriever.visual_by_asset_id = {
            str(document.get("asset_id")): document
            for document in retriever.documents
            if document.get("asset_id")
        }
        return retriever

    def test_model_product_overview_uses_full_reviewed_gallery_pool(self) -> None:
        retriever = self._retriever_with_global_pool_missing_gallery()
        # Simulate a global hybrid rerank pool containing no visual at all.  A
        # model-selected product overview must still read the canonical gallery.
        with patch.object(
            retriever,
            "_hybrid_scored",
            return_value=[(10.0, retriever.documents[0])],
        ):
            result = retriever.retrieve(
                "公司产品目录 产品体系",
                top_k=1,
                visual_k=5,
                case_k=0,
                wants_visuals=True,
                visual_scope="product",
                visual_query="我想了解一下你们的产品，给我图片",
                product_overview_request=True,
            )

        self.assertEqual(
            [item["asset_id"] for item in result["visual_assets"]],
            ["overview", "variant"],
        )

    def test_model_specific_unknown_product_does_not_get_gallery_substitute(self) -> None:
        retriever = self._retriever_with_global_pool_missing_gallery()
        with patch.object(
            retriever,
            "_hybrid_scored",
            return_value=[(10.0, retriever.documents[0])],
        ):
            result = retriever.retrieve(
                "白麻 产品图片",
                top_k=1,
                visual_k=5,
                case_k=0,
                wants_visuals=True,
                visual_scope="product",
                visual_query="请给我白麻产品图片",
                product_overview_request=False,
            )

        self.assertEqual(result["visual_assets"], [])


if __name__ == "__main__":
    unittest.main()

