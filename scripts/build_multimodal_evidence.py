"""Build traceable text-image evidence bundles for the local RAG system.

The system keeps text and original images as independent source assets, but a
customer-facing image must never be detached from its local textual context.
This script therefore creates a relation layer rather than merging image pixels
and prose into an untraceable blob.  Every bundle points to the original image,
the source page, its crop coordinates, and the most relevant same-page text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READY_DIR = ROOT / "data" / "sales" / "processed" / "rag_ready"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").lower().replace("®", ""))


def valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def spatial_distance(first: tuple[float, float, float, float] | None, second: tuple[float, float, float, float] | None) -> float:
    """Distance between bounding boxes; zero means overlap or touch."""

    if first is None or second is None:
        return 900.0
    x0, y0, x1, y1 = first
    a0, b0, a1, b1 = second
    horizontal = max(a0 - x1, x0 - a1, 0.0)
    vertical = max(b0 - y1, y0 - b1, 0.0)
    return math.hypot(horizontal, vertical)


def text_entries(records: Iterable[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for source in record.get("source_refs") or []:
            if not isinstance(source, dict):
                continue
            document_name = str(source.get("document_name") or "")
            source_page = source.get("source_page")
            if not document_name or not isinstance(source_page, int):
                continue
            by_page[(document_name, source_page)].append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "text": str(record.get("text") or ""),
                    "facts_eligible": bool(record.get("fact_eligible")),
                    "bbox": source.get("bbox"),
                    "section_heading": source.get("section_heading"),
                    "knowledge_domains": record.get("knowledge_domains") or [],
                    "content_labels": record.get("content_labels") or [],
                    "citation": {
                        "document_name": document_name,
                        "source_page": source_page,
                        "section_heading": source.get("section_heading"),
                        "bbox": source.get("bbox"),
                    },
                }
            )
    return by_page


def link_score(asset: dict[str, Any], entry: dict[str, Any]) -> tuple[float, list[str]]:
    """Rank only same-page textual context; never create an unsupported link."""

    score = 1.0  # Same document and page are required before this function runs.
    reasons = ["same_document_page"]
    asset_heading = compact(asset.get("section_heading"))
    entry_heading = compact(entry.get("section_heading"))
    if asset_heading and entry_heading and asset_heading == entry_heading:
        score += 4.0
        reasons.append("same_section_heading")

    nearby = compact(asset.get("nearby_text"))
    text = compact(entry.get("text"))
    if text and nearby and (text in nearby or nearby in text):
        score += 6.0
        reasons.append("nearby_text_match")

    caption = compact(asset.get("caption"))
    if caption and text and (caption in text or text in caption):
        score += 4.0
        reasons.append("caption_text_match")

    distance = spatial_distance(valid_bbox(asset.get("bbox")), valid_bbox(entry.get("bbox")))
    if distance <= 80:
        score += 3.0
        reasons.append("spatially_adjacent")
    elif distance <= 260:
        score += 1.5
        reasons.append("same_page_nearby")
    elif distance <= 650:
        score += 0.5
        reasons.append("same_page_context")

    if any(marker in entry.get("text", "") for marker in ("如下图", "见图", "如图", "下图", "上图", "图")):
        score += 1.0
        reasons.append("figure_reference_text")
    return score, reasons


def bundle_for_asset(asset: dict[str, Any], page_entries: list[dict[str, Any]]) -> dict[str, Any]:
    ranked: list[tuple[float, dict[str, Any], list[str]]] = []
    for entry in page_entries:
        score, reasons = link_score(asset, entry)
        ranked.append((score, entry, reasons))
    ranked.sort(key=lambda item: item[0], reverse=True)

    # Context is used to find the right original crop.  Technical support is
    # stricter: only text blocks that already passed the native-PDF validation
    # can be returned as customer-facing evidence alongside an image.
    selected_context: list[dict[str, Any]] = []
    for score, entry, reasons in ranked[:4]:
        selected_context.append(
            {
                "chunk_id": entry["chunk_id"],
                "text": entry["text"],
                "facts_eligible": entry["facts_eligible"],
                "citation": entry["citation"],
                "link_score": round(score, 3),
                "link_reasons": reasons,
            }
        )
    selected_evidence = [item for item in selected_context if item["facts_eligible"]]

    searchable_context = "\n".join(
        value
        for value in (
            str(asset.get("customer_title") or ""),
            str(asset.get("caption") or ""),
            str(asset.get("section_heading") or ""),
            str(asset.get("nearby_text") or ""),
            *[item["text"] for item in selected_context],
        )
        if value.strip()
    )
    domains = unique(
        domain
        for values in ([asset.get("knowledge_domains") or []] + [item.get("knowledge_domains") or [] for item in page_entries])
        for domain in values
    )
    return {
        "record_type": "multimodal_evidence_bundle",
        "bundle_id": f"bundle_{asset.get('asset_id')}",
        "asset_id": asset.get("asset_id"),
        "document_name": asset.get("document_name"),
        "source_pdf": asset.get("source_pdf"),
        "source_page": asset.get("source_page"),
        "bbox": asset.get("bbox"),
        "image_path": asset.get("image_path"),
        "customer_title": asset.get("customer_title"),
        "asset_type": asset.get("asset_type"),
        "effective_image_kind": asset.get("effective_image_kind"),
        "knowledge_domains": domains,
        "visual_labels": asset.get("visual_labels") or [],
        "section_heading": asset.get("section_heading"),
        "caption": asset.get("caption"),
        "nearby_text": asset.get("nearby_text"),
        "supporting_text": selected_evidence,
        "context_text": selected_context,
        "retrieval_text": searchable_context[:6000],
        "retrieval_eligible": bool(asset.get("retrieval_eligible")) and bool(selected_context),
        "customer_shareable": bool(asset.get("customer_shareable")),
        "facts_eligible": False,
        "technical_text_evidence_available": bool(selected_evidence),
        "visual_evidence_role": "supplemental_original_visual_with_linked_text_context",
        "citation_rule": "Return the original image only with its linked same-page text evidence and source page. Image understanding alone is never a technical fact source.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build linked image-text evidence bundles from tagged local RAG evidence.")
    parser.add_argument("--ready-dir", type=Path, default=DEFAULT_READY_DIR)
    args = parser.parse_args()
    ready_dir = args.ready_dir.resolve()
    text_path = ready_dir / "text_evidence_tagged.jsonl"
    visual_path = ready_dir / "visual_assets_tagged.jsonl"
    if not text_path.exists() or not visual_path.exists():
        raise FileNotFoundError("Missing tagged evidence. Run apply_knowledge_taxonomy.py first.")

    by_page = text_entries(read_jsonl(text_path))
    bundles: list[dict[str, Any]] = []
    for asset in read_jsonl(visual_path):
        if not asset.get("customer_shareable") or not asset.get("image_path"):
            continue
        key = (str(asset.get("document_name") or ""), asset.get("source_page"))
        page_entries = by_page.get(key, []) if isinstance(key[1], int) else []
        bundles.append(bundle_for_asset(asset, page_entries))

    output = ready_dir / "multimodal_evidence_bundles.jsonl"
    write_jsonl(output, bundles)
    summary = {
        "bundle_count": len(bundles),
        "retrieval_eligible_bundle_count": sum(bool(bundle["retrieval_eligible"]) for bundle in bundles),
        "bundles_with_linked_text": sum(bool(bundle["supporting_text"]) for bundle in bundles),
        "bundles_by_domain": dict(Counter(domain for bundle in bundles for domain in bundle["knowledge_domains"])),
    }
    (ready_dir / "multimodal_link_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
