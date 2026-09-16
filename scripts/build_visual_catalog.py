"""Build a persistent, source-backed gallery catalogue without copying images.

The RAG index historically linked project cases to visuals only while building
the in-memory retrieval documents.  That was enough for serving, but it left
``project_cases_tagged.jsonl`` with empty ``visual_asset_ids`` and made the
relationship difficult to audit or curate.  This script makes the relationship
durable and creates a small gallery manifest whose records only *reference* the
original extracted image files.

Automatic case/image links are deliberately conservative: the document name
and physical source page must match.  The provenance and confidence of every
link are recorded so a future human-curated override can be distinguished from
this deterministic fallback.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READY_DIR = ROOT / "data" / "sales" / "processed" / "rag_ready"
DEFAULT_OUTPUT = DEFAULT_READY_DIR / "visual_catalog.jsonl"
DEFAULT_REPORT = DEFAULT_READY_DIR / "visual_catalog_report.json"

CATALOG_VERSION = "1.1"
LINK_GENERATOR = "scripts/build_visual_catalog.py"
VISUAL_ROLES = {
    "product_overview",
    "product_variant",
    "application_effect",
    "component",
    "comparison_reference",
    "accessory",
}
PRODUCT_VARIANTS = (
    "葡萄牙米黄",
    "卡拉麦里金",
    "定制白麻",
    "黄金麻",
    "灰砂岩",
    "白麻",
    "金麻",
    "深咖",
    "咖色",
)
PRODUCT_CODE_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,6}[-]?\d{2,6}(?:-\d+)?(?![A-Za-z0-9])", re.I)


def visual_source_text(record: dict[str, Any]) -> tuple[str, str]:
    """Return high-confidence labels and broader visual descriptions separately.

    A catalogue page can mention several neighbouring products.  Role rules
    therefore prefer the crop's own title/caption/section and use nearby text
    only for supporting context.  This keeps the classifier source-grounded
    without tying it to a page number or asset ID.
    """

    metadata = record.get("search_metadata") if isinstance(record.get("search_metadata"), dict) else {}
    primary = "\n".join(
        str(value).strip()
        for value in (
            record.get("customer_title"),
            record.get("caption"),
            record.get("section_heading"),
        )
        if isinstance(value, str) and value.strip()
    )
    supporting_values: list[str] = []
    for value in (
        record.get("nearby_text"),
        metadata.get("visual_description"),
        *(metadata.get("visible_components") or []),
        *(metadata.get("recommended_search_terms") or []),
    ):
        if isinstance(value, str) and value.strip():
            supporting_values.append(value.strip())
    return primary, "\n".join(supporting_values)


def classify_visual_role(record: dict[str, Any], *, gallery_type: str = "product_sample") -> str:
    """Classify a shareable crop by source-visible semantics, not appearance IDs.

    The order is intentionally fail-closed for a product gallery: installation
    accessories, components and comparison references are removed before the
    generic product/variant rules are considered.  Unknown product photos fall
    back to ``application_effect`` and therefore cannot silently become a
    customer-facing product sample.
    """

    if gallery_type == "project_case":
        return "application_effect"

    primary, supporting = visual_source_text(record)
    primary_compact = re.sub(r"\s+", "", primary).lower()
    all_compact = re.sub(r"\s+", "", f"{primary}\n{supporting}").lower()

    accessory_markers = (
        "锚钉", "锚栓", "紧固件", "连接件", "挂件", "扣件", "托架", "膨胀螺栓", "辅材",
    )
    component_markers = (
        "生产工艺--基板", "生产工艺—基板", "基板堆叠", "纤维水泥板", "木质纤维水泥板",
        "结构截面", "截面图", "截面结构", "保温芯材", "保温层构造", "材料构成",
    )
    comparison_markers = (
        "质感对比", "效果对比", "产品对比", "对比样板", "对比参考", "天然石材样品",
    )
    application_markers = (
        "项目案例", "工程案例", "应用案例", "应用效果", "安装效果", "上墙效果", "建筑外墙安装",
        "外立面实景", "建筑立面", "施工完成", "完工效果",
    )
    overview_markers = (
        "产品展示", "产品总览", "产品系列", "仿石装饰板展示", "装饰板产品展示", "样板叠放",
    )

    if any(marker in all_compact for marker in accessory_markers):
        return "accessory"
    if any(marker in primary_compact for marker in component_markers) or any(
        marker in all_compact for marker in ("结构截面", "截面图", "保温芯材")
    ):
        return "component"
    if any(marker in all_compact for marker in comparison_markers):
        return "comparison_reference"
    if any(marker in all_compact for marker in application_markers):
        return "application_effect"

    _, variant_or_code = grounded_product_fields(f"{primary}\n{supporting}")
    if variant_or_code:
        return "product_variant"
    if any(marker in all_compact for marker in overview_markers):
        return "product_overview"
    return "application_effect"


def is_product_gallery_eligible(
    record: dict[str, Any],
    *,
    visual_role: str,
    review_state: str,
) -> bool:
    """Only audited overview/variant crops may enter the broad product gallery."""

    return bool(
        visual_role in {"product_overview", "product_variant"}
        and review_state != "needs_review"
        and str(record.get("asset_type") or "") != "original_pdf_page_render"
        and str(record.get("effective_image_kind") or "") == "product_photo"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_source_pdf(record: dict[str, Any], project_root: Path) -> tuple[str | None, bool]:
    """Resolve legacy ``data/raw`` paths into the current ``data/sales/raw`` tree."""

    raw_value = str(record.get("source_pdf") or "").strip()
    if raw_value:
        raw_path = Path(raw_value)
        if raw_path.is_file():
            return str(raw_path.resolve()), False

        legacy_root = project_root / "data" / "raw"
        try:
            relative = raw_path.relative_to(legacy_root)
        except ValueError:
            relative = None
        if relative is not None:
            migrated = project_root / "data" / "sales" / "raw" / relative
            if migrated.is_file():
                return str(migrated.resolve()), True

    document_name = str(record.get("source_document") or record.get("document_name") or "").strip()
    if document_name:
        for extension in (".pdf", ".PDF"):
            candidate = project_root / "data" / "sales" / "raw" / "product_pdfs" / f"{document_name}{extension}"
            if candidate.is_file():
                return str(candidate.resolve()), raw_value != str(candidate.resolve())
    return raw_value or None, False


def eligible_visual(record: dict[str, Any]) -> bool:
    return bool(
        record.get("asset_id")
        and record.get("retrieval_eligible")
        and record.get("customer_shareable")
        and record.get("image_path")
    )


def case_visual_score(record: dict[str, Any]) -> int:
    """Prefer a focused project crop; retain a full page only as fallback."""

    score = 0
    image_kind = str(record.get("effective_image_kind") or "")
    labels = set(record.get("visual_labels") or [])
    asset_type = str(record.get("asset_type") or "")
    if image_kind == "project_photo":
        score += 120
    if "project_case_image" in labels:
        score += 80
    if asset_type == "original_pdf_image":
        score += 30
    if image_kind == "unclassified_original_visual" or asset_type == "original_pdf_page_render":
        score -= 30
    if record.get("source_context_consistency") == "consistent":
        score += 10
    if record.get("review_required"):
        score -= 60
    return score


def link_confidence(record: dict[str, Any]) -> str:
    image_kind = str(record.get("effective_image_kind") or "")
    labels = set(record.get("visual_labels") or [])
    if image_kind == "project_photo" and "project_case_image" in labels and not record.get("review_required"):
        return "high"
    if image_kind == "project_photo" or "project_case_image" in labels:
        return "medium"
    return "fallback"


def review_status(record: dict[str, Any]) -> str:
    if record.get("review_required"):
        return "needs_review"
    if str(record.get("asset_type") or "") == "original_pdf_page_render":
        return "source_page_fallback"
    if record.get("source_context_consistency") == "consistent":
        return "auto_source_linked"
    return "auto_linked_review_pending"


def grounded_product_fields(text: str, explicit_product: str | None = None) -> tuple[str | None, str | None]:
    """Extract only source-visible product/variant strings; never infer by appearance."""

    searchable = " ".join(part for part in (explicit_product, text) if isinstance(part, str) and part.strip())
    compact = searchable.replace(" ", "")
    canonical: str | None = None
    if explicit_product and explicit_product.strip():
        canonical = explicit_product.strip()
    elif "真岩®石" in searchable or "真岩石" in compact:
        canonical = "真岩®石"

    variants: list[str] = []
    for value in PRODUCT_VARIANTS:
        if value not in searchable:
            continue
        if value == "金麻" and "黄金麻" in searchable:
            continue
        if value == "白麻" and "定制白麻" in searchable:
            continue
        variants.append(value)
    variants.extend(match.upper() for match in PRODUCT_CODE_PATTERN.findall(searchable))
    variant_or_code = " / ".join(dict.fromkeys(variants)) or None
    return canonical, variant_or_code


def case_customer_caption(case: dict[str, Any]) -> str:
    details = [
        f"使用产品：{case['product']}" if case.get("product") else "",
        f"施工工艺：{case['installation_method']}" if case.get("installation_method") else "",
        f"使用面积：{case['area_m2']}" if case.get("area_m2") else "",
        f"建成时间：{case['completion_year']}" if case.get("completion_year") else "",
    ]
    suffix = "；".join(value for value in details if value)
    return f"{case.get('project_name') or '项目案例'}｜{suffix}" if suffix else str(case.get("project_name") or "项目案例")


def build_visual_catalog(
    ready_dir: Path = DEFAULT_READY_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    *,
    project_root: Path = ROOT,
    persist_clean_cases: bool = True,
) -> dict[str, Any]:
    visual_path = ready_dir / "visual_assets_tagged.jsonl"
    case_path = ready_dir / "project_cases_tagged.jsonl"
    clean_case_path = ready_dir / "project_cases.jsonl"
    if not visual_path.is_file() or not case_path.is_file():
        raise FileNotFoundError("visual_assets_tagged.jsonl and project_cases_tagged.jsonl are required")

    visuals = read_jsonl(visual_path)
    visual_by_id = {
        str(record["asset_id"]): record
        for record in visuals
        if eligible_visual(record)
    }
    visuals_by_document_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in visual_by_id.values():
        page = record.get("source_page")
        if isinstance(page, int):
            visuals_by_document_page[(str(record.get("document_name") or ""), page)].append(record)
    for records in visuals_by_document_page.values():
        records.sort(key=lambda item: (-case_visual_score(item), str(item.get("asset_id") or "")))

    cases = read_jsonl(case_path)
    updated_cases: list[dict[str, Any]] = []
    source_paths_repaired = 0
    unlinked_case_ids: list[str] = []
    for case in cases:
        document_name = str(case.get("source_document") or "")
        source_pages = case.get("source_pages") or [case.get("source_page")]
        linked_records: list[dict[str, Any]] = []
        matched_pages: dict[str, int] = {}
        for page in source_pages:
            if not isinstance(page, int):
                continue
            for visual in visuals_by_document_page.get((document_name, page), []):
                asset_id = str(visual["asset_id"])
                if asset_id not in matched_pages:
                    matched_pages[asset_id] = page
                    linked_records.append(visual)
        linked_records.sort(key=lambda item: (-case_visual_score(item), int(matched_pages[str(item["asset_id"])]), str(item["asset_id"])))

        linkages = [
            {
                "asset_id": str(visual["asset_id"]),
                "method": "same_document_same_source_page",
                "matched_source_page": matched_pages[str(visual["asset_id"])],
                "confidence": link_confidence(visual),
                "provenance": LINK_GENERATOR,
            }
            for visual in linked_records
        ]
        source_pdf, repaired = resolve_source_pdf(case, project_root)
        source_paths_repaired += int(repaired)
        visual_asset_ids = [item["asset_id"] for item in linkages]
        updated = {
            **case,
            "source_pdf": source_pdf,
            "visual_asset_ids": visual_asset_ids,
            "hero_visual_asset_id": visual_asset_ids[0] if visual_asset_ids else None,
            "visual_linkages": linkages,
            "visual_linkage_version": CATALOG_VERSION,
        }
        updated_cases.append(updated)
        if not visual_asset_ids:
            unlinked_case_ids.append(str(case.get("case_id") or ""))

    write_jsonl_atomic(case_path, updated_cases)

    # Keep the clean case file path-safe and link-ready too, so a subsequent
    # taxonomy refresh does not temporarily reintroduce empty relationships.
    if persist_clean_cases and clean_case_path.is_file():
        updated_by_id = {str(case.get("case_id")): case for case in updated_cases}
        refreshed_clean: list[dict[str, Any]] = []
        for clean_case in read_jsonl(clean_case_path):
            linked = updated_by_id.get(str(clean_case.get("case_id")))
            if linked is None:
                refreshed_clean.append(clean_case)
                continue
            refreshed_clean.append(
                {
                    **clean_case,
                    "source_pdf": linked.get("source_pdf"),
                    "visual_asset_ids": linked.get("visual_asset_ids") or [],
                    "hero_visual_asset_id": linked.get("hero_visual_asset_id"),
                    "visual_linkages": linked.get("visual_linkages") or [],
                    "visual_linkage_version": CATALOG_VERSION,
                }
            )
        write_jsonl_atomic(clean_case_path, refreshed_clean)

    catalog_records: list[dict[str, Any]] = []
    linked_case_asset_ids: set[str] = set()
    for case in updated_cases:
        canonical_product, variant_or_code = grounded_product_fields(
            str(case.get("source_text") or ""),
            str(case.get("product") or "") or None,
        )
        for linkage in case.get("visual_linkages") or []:
            asset_id = str(linkage["asset_id"])
            visual = visual_by_id.get(asset_id)
            if visual is None:
                continue
            linked_case_asset_ids.add(asset_id)
            is_hero = asset_id == case.get("hero_visual_asset_id")
            source_pdf, _ = resolve_source_pdf(visual, project_root)
            visual_role = classify_visual_role(visual, gallery_type="project_case")
            catalog_records.append(
                {
                    "record_type": "visual_catalog_entry",
                    "catalog_version": CATALOG_VERSION,
                    "gallery_type": "project_case",
                    "asset_id": asset_id,
                    "canonical_product": canonical_product,
                    "product_name": canonical_product,
                    "variant_or_code": variant_or_code,
                    "visual_role": visual_role,
                    "product_gallery_eligible": False,
                    "case_id": case.get("case_id"),
                    "project_name": case.get("project_name"),
                    "hero": is_hero,
                    "display_priority": 100 if is_hero else max(40, 80 + case_visual_score(visual) // 20),
                    "review_status": review_status(visual),
                    "customer_caption": case_customer_caption(case),
                    "source_document": visual.get("document_name"),
                    "source_pdf": source_pdf,
                    "source_page": visual.get("source_page"),
                    "bbox": visual.get("bbox"),
                    "image_path": visual.get("image_path"),
                    "effective_image_kind": visual.get("effective_image_kind"),
                    "visual_labels": visual.get("visual_labels") or [],
                    "link_confidence": linkage.get("confidence"),
                    "provenance": {
                        "method": linkage.get("method"),
                        "matched_source_page": linkage.get("matched_source_page"),
                        "generator": LINK_GENERATOR,
                        "original_image_copied": False,
                    },
                }
            )

    for visual in visual_by_id.values():
        if visual.get("effective_image_kind") != "product_photo":
            continue
        context = "\n".join(
            str(value or "")
            for value in (
                visual.get("customer_title"),
                visual.get("caption"),
                visual.get("nearby_text"),
                visual.get("section_heading"),
                # The source-document title is itself audited metadata.  It
                # can establish the product family without guessing from the
                # pixels when a crop only names a finish/code.
                visual.get("document_name"),
            )
        )
        canonical_product, variant_or_code = grounded_product_fields(context)
        source_pdf, _ = resolve_source_pdf(visual, project_root)
        explicit_variant = bool(variant_or_code)
        visual_role = classify_visual_role(visual, gallery_type="product_sample")
        review_state = review_status(visual)
        product_gallery_eligible = is_product_gallery_eligible(
            visual,
            visual_role=visual_role,
            review_state=review_state,
        )
        role_priority = {
            "product_overview": 110,
            "product_variant": 100,
            "application_effect": 60,
            "component": 40,
            "comparison_reference": 30,
            "accessory": 20,
        }[visual_role]
        catalog_records.append(
            {
                "record_type": "visual_catalog_entry",
                "catalog_version": CATALOG_VERSION,
                "gallery_type": "product_sample",
                "asset_id": str(visual["asset_id"]),
                "canonical_product": canonical_product,
                "product_name": canonical_product,
                "variant_or_code": variant_or_code,
                "visual_role": visual_role,
                "product_gallery_eligible": product_gallery_eligible,
                "case_id": None,
                "project_name": None,
                "hero": explicit_variant,
                "display_priority": role_priority,
                "review_status": review_state,
                "customer_caption": str(visual.get("customer_title") or "产品样品原图"),
                "source_document": visual.get("document_name"),
                "source_pdf": source_pdf,
                "source_page": visual.get("source_page"),
                "bbox": visual.get("bbox"),
                "image_path": visual.get("image_path"),
                "effective_image_kind": visual.get("effective_image_kind"),
                "visual_labels": visual.get("visual_labels") or [],
                "link_confidence": "high" if explicit_variant else "medium",
                "provenance": {
                    "method": "source_visible_product_label",
                    "generator": LINK_GENERATOR,
                    "original_image_copied": False,
                },
            }
        )

    catalog_records.sort(
        key=lambda item: (
            0 if item["gallery_type"] == "product_sample" else 1,
            0 if item.get("product_gallery_eligible") else 1,
            str(item.get("case_id") or ""),
            -int(item.get("display_priority") or 0),
            str(item.get("asset_id") or ""),
        )
    )
    write_jsonl_atomic(output_path, catalog_records)

    gallery_counts = Counter(str(item["gallery_type"]) for item in catalog_records)
    role_counts = Counter(str(item["visual_role"]) for item in catalog_records)
    valid_case_source_pdfs = sum(
        bool(case.get("source_pdf")) and Path(str(case["source_pdf"])).is_file()
        for case in updated_cases
    )
    legacy_case_source_paths = sum(
        "\\data\\raw\\" in str(case.get("source_pdf") or "").lower()
        for case in updated_cases
    )
    missing_catalog_images = [
        str(item.get("image_path") or "")
        for item in catalog_records
        if not item.get("image_path") or not Path(str(item["image_path"])).is_file()
    ]
    report = {
        "generator": LINK_GENERATOR,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_version": CATALOG_VERSION,
        "source_visual_asset_count": len(visuals),
        "eligible_visual_asset_count": len(visual_by_id),
        "project_case_count": len(updated_cases),
        "linked_project_case_count": len(updated_cases) - len(unlinked_case_ids),
        "unlinked_project_case_count": len(unlinked_case_ids),
        "unlinked_case_ids": unlinked_case_ids,
        "source_pdf_paths_repaired": source_paths_repaired,
        "valid_project_case_source_pdf_count": valid_case_source_pdfs,
        "legacy_project_case_source_path_count": legacy_case_source_paths,
        "catalog_entry_count": len(catalog_records),
        "catalog_entries_by_gallery_type": dict(gallery_counts),
        "catalog_entries_by_visual_role": dict(role_counts),
        "product_gallery_eligible_count": sum(
            bool(item.get("product_gallery_eligible")) for item in catalog_records
        ),
        "unique_catalogued_asset_count": len({str(item["asset_id"]) for item in catalog_records}),
        "project_case_asset_reference_count": sum(
            len(case.get("visual_asset_ids") or []) for case in updated_cases
        ),
        "missing_catalog_image_path_count": len(missing_catalog_images),
        "missing_catalog_image_paths": missing_catalog_images,
        "original_images_copied": 0,
    }
    write_json_atomic(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Persist project-case visual links and build a source-backed gallery manifest.")
    parser.add_argument("--ready-dir", type=Path, default=DEFAULT_READY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-persist-clean-cases", action="store_true")
    args = parser.parse_args()
    report = build_visual_catalog(
        args.ready_dir.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        persist_clean_cases=not args.no_persist_clean_cases,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
