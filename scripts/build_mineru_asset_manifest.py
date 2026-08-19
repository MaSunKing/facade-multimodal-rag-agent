"""Convert local MinerU output into traceable text and image assets.

The script does not call a model or upload files.  It keeps each extracted
image tied to its original document page, crop bounding box and nearby text so
the RAG layer can return the original source asset to a customer later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def clean_text(value: str) -> str:
    return " ".join(value.replace("\u3000", " ").split())


def collect_text(value: Any) -> list[str]:
    """Recursively collect MinerU's text leaf nodes without serialising dicts."""

    if value is None:
        return []
    if isinstance(value, str):
        text = clean_text(value)
        return [text] if text else []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(collect_text(item))
        return parts
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return collect_text(value["content"])
        parts: list[str] = []
        for key, item in value.items():
            if key in {"path", "image_path", "bbox"}:
                continue
            parts.extend(collect_text(item))
        return parts
    return []


def block_text(block: dict[str, Any]) -> str:
    return " ".join(collect_text(block.get("content")))


def text_neighborhood(items: list[dict[str, Any]], item_index: int) -> str:
    """Return nearby textual context for an image/table asset on the same page."""

    nearby: list[str] = []
    for index in range(max(0, item_index - 3), min(len(items), item_index + 3)):
        if index == item_index:
            continue
        candidate = items[index]
        if candidate.get("type") in {"paragraph", "title", "text"}:
            text = block_text(candidate)
            if text:
                nearby.append(text)
    return "\n".join(nearby)


def locate_content_list(source_dir: Path) -> Path:
    matches = sorted(source_dir.glob("*_content_list_v2.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_content_list_v2.json in {source_dir}; found {len(matches)}."
        )
    return matches[0]


def locate_origin_pdf(source_dir: Path) -> Path | None:
    matches = sorted(source_dir.glob("*_origin.pdf"))
    return matches[0] if matches else None


def stable_document_id(document_name: str) -> str:
    digest = hashlib.sha1(document_name.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_assets(source_dir: Path, output_dir: Path, customer_shareable: bool) -> tuple[int, int]:
    content_list_path = locate_content_list(source_dir)
    origin_pdf = locate_origin_pdf(source_dir)
    pages = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(pages, list):
        raise ValueError("MinerU content list must be a page array.")

    document_name = origin_pdf.stem.removesuffix("_origin") if origin_pdf else source_dir.parent.name
    original_source = ROOT / "data" / "sales" / "raw" / "product_pdfs" / f"{document_name}.pdf"
    source_pdf = original_source if original_source.exists() else origin_pdf
    document_id = stable_document_id(document_name)
    text_records: list[dict[str, Any]] = []
    asset_records: list[dict[str, Any]] = []

    for page_index, raw_page in enumerate(pages, start=1):
        if not isinstance(raw_page, list):
            continue
        page_items = [item for item in raw_page if isinstance(item, dict)]
        active_heading: str | None = None

        for item_index, item in enumerate(page_items):
            item_type = str(item.get("type") or "other")
            text = block_text(item)
            bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else None

            if item_type == "title" and text:
                active_heading = text

            if item_type in {"title", "paragraph", "text", "list"} and text:
                text_records.append(
                    {
                        "record_type": "text_evidence",
                        "document_id": document_id,
                        "document_name": document_name,
                        "source_pdf": str(source_pdf) if source_pdf else None,
                        "source_page": page_index,
                        "bbox": bbox,
                        "section_heading": active_heading,
                        "text": text,
                        "review_status": "unreviewed",
                        "customer_shareable": customer_shareable,
                    }
                )

            if item_type not in {"image", "table"}:
                continue

            content = item.get("content") if isinstance(item.get("content"), dict) else {}
            image_source = content.get("image_source") if isinstance(content.get("image_source"), dict) else {}
            relative_image = image_source.get("path") if isinstance(image_source.get("path"), str) else None
            image_path = source_dir / relative_image if relative_image else None
            caption = " ".join(collect_text(content.get("image_caption")))
            footnote = " ".join(collect_text(content.get("image_footnote")))
            label = clean_text(" ".join(part for part in (caption, footnote) if part))
            asset_number = len(asset_records) + 1

            asset_records.append(
                {
                    "record_type": "visual_asset",
                    "asset_id": f"{document_id}_p{page_index:03d}_{item_type}_{asset_number:03d}",
                    "document_id": document_id,
                    "document_name": document_name,
                    "source_pdf": str(source_pdf) if source_pdf else None,
                    "source_page": page_index,
                    "bbox": bbox,
                    "asset_type": "original_pdf_image" if item_type == "image" else "original_pdf_table",
                    "image_path": str(image_path) if image_path and image_path.exists() else None,
                    "caption": label or None,
                    "nearby_text": text_neighborhood(page_items, item_index) or None,
                    "section_heading": active_heading,
                    "customer_shareable": customer_shareable,
                    "review_status": "needs_visual_review",
                    "semantic_annotation_status": "not_started",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "text_evidence.jsonl", text_records)
    write_jsonl(output_dir / "visual_assets.jsonl", asset_records)
    metadata = {
        "document_id": document_id,
        "document_name": document_name,
        "source_dir": str(source_dir),
        "source_pdf": str(source_pdf) if source_pdf else None,
        "page_count": len(pages),
        "text_record_count": len(text_records),
        "visual_asset_count": len(asset_records),
        "customer_shareable_default": customer_shareable,
    }
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(text_records), len(asset_records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build text and original-image manifests from local MinerU output.")
    parser.add_argument("source_dir", type=Path, help="MinerU 'auto' output directory.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for RAG-ready JSONL manifests.")
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Mark every emitted record as not customer-shareable. The current prototype defaults to shareable.",
    )
    args = parser.parse_args()
    text_count, asset_count = build_assets(
        args.source_dir.resolve(), args.output.resolve(), customer_shareable=not args.internal_only
    )
    print(f"Created {text_count} text records and {asset_count} original visual assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
