"""Prepare traceable, customer-safe RAG evidence from local ingestion assets.

This script deliberately keeps the original MinerU manifests untouched.  It
creates a separate ``data/sales/processed/rag_ready`` dataset where:

* extraction-format artifacts are repaired conservatively;
* repaired parameter strings are checked against the native text of the
  original PDF page when possible;
* boilerplate is retained for auditability but removed from retrieval;
* original image crops remain the only customer-returnable visual source;
* vision-model descriptions are search metadata only, never technical facts.

No file is uploaded and no model is called by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - friendly runtime failure
    raise SystemExit("PyMuPDF is required. Install it in the local ai6130 environment.") from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_ROOT = ROOT / "data" / "sales" / "processed" / "rag_assets"
DEFAULT_VISUAL_SEMANTICS = DEFAULT_ASSETS_ROOT / "all_visual_semantics_qwen3vl.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "sales" / "processed" / "rag_ready"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if isinstance(value, dict):
                yield value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]}"


def compact_number(value: str) -> str:
    """Remove extraction spaces inside a numeric token, preserving decimals."""

    return re.sub(r"\s+", "", value)


def clean_extraction_text(raw_text: str) -> tuple[str, list[str]]:
    """Apply only known, reversible layout-to-text repairs.

    We do not invent missing content.  The numeric repair handles MinerU's
    LaTex-style ``5 0 ^ { * } 3 0`` artefact by restoring multiplication
    separators, then native-PDF validation decides whether it is fact eligible.
    """

    text = raw_text.replace("\u3000", " ").strip()
    actions: list[str] = []

    if "text_list unordered text" in text:
        text = text.replace("text_list unordered text", "")
        actions.append("removed_mineru_list_marker")
    if "注 text 意" in text:
        text = text.replace("注 text 意", "注意")
        actions.append("repaired_split_word_注意")

    number_with_multiply = re.compile(
        r"(?P<number>\d(?:\s*\d)*(?:\s*\.\s*\d(?:\s*\d)*)?)\s*\^\s*\{\s*\*\s*\}"
    )
    if number_with_multiply.search(text):
        text = number_with_multiply.sub(lambda match: compact_number(match["number"]) + "×", text)
        actions.append("repaired_latex_multiplication")

    # Examples handled: ``\\ \mathrm { m m }`` and ``\\mathrm { { m m } }``.
    unit_pattern = re.compile(r"\\\s*\\?mathrm\s*\{\s*(?:\{\s*)?m\s*m\s*\}?\s*\}")
    if unit_pattern.search(text):
        text = unit_pattern.sub("mm", text)
        actions.append("repaired_latex_mm_unit")

    # After the LaTex tokens above are removed, MinerU can still leave spaces
    # within a dimensional expression, e.g. ``50× 30× 3 . 0 mm``.  Tighten only
    # complete dimensions; ordinary prose spacing is deliberately untouched.
    dimension_pattern = re.compile(
        r"(?<!\d)"
        r"\d(?:\s*\d)*(?:\s*\.\s*\d(?:\s*\d)*)?"
        r"(?:\s*[×*]\s*\d(?:\s*\d)*(?:\s*\.\s*\d(?:\s*\d)*)?)+"
        r"\s*(?:mm|㎜)",
        flags=re.IGNORECASE,
    )
    if dimension_pattern.search(text):
        text = dimension_pattern.sub(lambda match: re.sub(r"\s+", "", match.group(0)), text)
        actions.append("normalised_dimension_spacing")

    text = re.sub(r"\s+", " ", text).strip()
    return text, actions


def has_unresolved_extraction_artifact(text: str) -> bool:
    patterns = ("text_list unordered text", "^ {", "\\mathrm", "注 text 意")
    return any(pattern in text for pattern in patterns)


def normalise_for_match(value: str) -> str:
    value = value.lower().replace("㎜", "mm").replace("×", "*")
    value = re.sub(r"\s+", "", value)
    return value


PARAMETER_PATTERN = re.compile(
    r"\d+(?:[×*]\d+)+(?:\.\d+)?(?:mm|㎜)?|\d+(?:\.\d+)?(?:mm|㎜)",
    flags=re.IGNORECASE,
)


def parameter_tokens(value: str) -> list[str]:
    return [normalise_for_match(token) for token in PARAMETER_PATTERN.findall(value)]


class NativePdfText:
    """Small page-text cache used only to validate a repaired source fragment."""

    def __init__(self) -> None:
        self._pages: dict[tuple[str, int], str | None] = {}

    def page_text(self, source_pdf: str | None, source_page: int | None) -> str | None:
        if not source_pdf or not source_page:
            return None
        key = (source_pdf, int(source_page))
        if key in self._pages:
            return self._pages[key]
        path = Path(source_pdf)
        result: str | None = None
        try:
            if path.exists():
                with fitz.open(path) as document:
                    page_index = int(source_page) - 1
                    if 0 <= page_index < document.page_count:
                        result = document.load_page(page_index).get_text("text")
        except Exception:  # A missing/corrupt source becomes a review signal, not a crash.
            result = None
        self._pages[key] = result
        return result


def validate_parameter_repair(clean_text: str, native_page_text: str | None) -> tuple[bool | None, list[str]]:
    """Check repaired parameter tokens against the native source page.

    ``None`` means no parameter repair validation was needed.  A true result is
    intentionally narrow: every dimensional token present in the cleaned block
    must also appear in the native page after harmless unit/spacing normalisation.
    """

    tokens = parameter_tokens(clean_text)
    if not tokens:
        return None, []
    if not native_page_text:
        return False, tokens
    normal_native = normalise_for_match(native_page_text)
    return all(token in normal_native for token in tokens), tokens


def is_boilerplate(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if compact in {"客户签字", "日期", "清理完成"}:
        return True
    return "河北大自然石材有限公司" in compact


def source_ref(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": record.get("document_id"),
        "document_name": record.get("document_name"),
        "source_pdf": record.get("source_pdf"),
        "source_page": record.get("source_page"),
        "bbox": record.get("bbox"),
        "section_heading": record.get("section_heading"),
    }


def build_clean_text_records(assets_root: Path, native_pdf: NativePdfText) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    raw_records: list[dict[str, Any]] = []
    for path in sorted(assets_root.rglob("text_evidence.jsonl")):
        raw_records.extend(read_jsonl(path))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repair_examples: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for raw in raw_records:
        raw_text = str(raw.get("text") or "")
        clean_text, actions = clean_extraction_text(raw_text)
        unresolved = has_unresolved_extraction_artifact(clean_text)
        needs_native_validation = "repaired_latex_multiplication" in actions or "repaired_latex_mm_unit" in actions
        native_page = native_pdf.page_text(raw.get("source_pdf"), raw.get("source_page")) if needs_native_validation else None
        native_validated, tokens = validate_parameter_repair(clean_text, native_page) if needs_native_validation else (None, [])
        boilerplate = is_boilerplate(clean_text)

        if not clean_text:
            counters["empty_after_cleaning"] += 1
            continue
        if actions:
            counters["records_normalised"] += 1
            for action in actions:
                counters[f"action_{action}"] += 1
        if unresolved:
            counters["unresolved_artifact_records"] += 1
        if native_validated is False:
            counters["native_validation_failed"] += 1
        if native_validated is True:
            counters["native_validation_passed"] += 1
        if boilerplate:
            counters["boilerplate_records"] += 1

        prepared = {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "normalization_actions": actions,
            "unresolved_extraction_artifact": unresolved,
            "native_pdf_validation": native_validated,
            "validated_parameter_tokens": tokens,
            "index_eligible": not boilerplate and not unresolved and native_validated is not False,
            "fact_eligible": not boilerplate and not unresolved and native_validated is not False,
            "exclusion_reason": "boilerplate" if boilerplate else None,
            "source_ref": source_ref(raw),
        }
        grouped[clean_text].append(prepared)
        if actions or native_validated is False:
            repair_examples.append(
                {
                    "source_ref": prepared["source_ref"],
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    "normalization_actions": actions,
                    "native_pdf_validation": native_validated,
                    "validated_parameter_tokens": tokens,
                }
            )

    output: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []
    for clean_text, members in grouped.items():
        first = members[0]
        refs = [member["source_ref"] for member in members]
        eligible = all(member["index_eligible"] for member in members)
        fact_eligible = all(member["fact_eligible"] for member in members)
        record = {
            "record_type": "clean_text_evidence",
            "chunk_id": stable_id("txt", clean_text),
            "text": clean_text,
            "raw_text_variants": list(dict.fromkeys(member["raw_text"] for member in members)),
            "normalization_actions": sorted({action for member in members for action in member["normalization_actions"]}),
            "native_pdf_validation": [member["native_pdf_validation"] for member in members],
            "validated_parameter_tokens": sorted({token for member in members for token in member["validated_parameter_tokens"]}),
            "index_eligible": eligible,
            "fact_eligible": fact_eligible,
            "customer_shareable": True,
            "source_refs": refs,
        }
        output.append(record)
        if not eligible and first["exclusion_reason"] != "boilerplate":
            reason = first["exclusion_reason"] or "unresolved_extraction_artifact_or_native_validation"
            manual_review.append(
                {
                    "record_type": "manual_review_text",
                    "item_id": record["chunk_id"],
                    "reason": reason,
                    "text": clean_text,
                    "source_refs": refs,
                }
            )

    counters["raw_text_records"] = len(raw_records)
    counters["unique_clean_text_chunks"] = len(output)
    counters["exact_duplicates_collapsed"] = len(raw_records) - len(output)
    return output, manual_review, dict(counters), repair_examples


def first_nonempty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_raw_visual_assets(assets_root: Path) -> dict[str, dict[str, Any]]:
    """Load every original PDF crop, including assets not yet VLM-labelled.

    A local vision label is useful retrieval metadata, but it must never decide
    whether the original page crop is retained.  This keeps drawings, tables
    and photographs available through their page text even while a larger VLM
    annotation job is pending.
    """

    output: dict[str, dict[str, Any]] = {}
    for path in sorted(assets_root.rglob("visual_assets.jsonl")):
        for record in read_jsonl(path):
            asset_id = str(record.get("asset_id") or "")
            if asset_id:
                output[asset_id] = record
    return output


def build_clean_visual_records(
    assets_root: Path, visual_semantics_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    semantic_rows = list(read_jsonl(visual_semantics_path))
    semantics_by_asset_id = {
        str(record.get("asset_id")): record
        for record in semantic_rows
        if record.get("asset_id")
    }
    raw_assets_by_id = load_raw_visual_assets(assets_root)
    asset_ids = list(dict.fromkeys([*semantics_by_asset_id, *raw_assets_by_id]))
    output: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for asset_id in asset_ids:
        raw_asset = raw_assets_by_id.get(asset_id, {})
        semantic_row = semantics_by_asset_id.get(asset_id, {})
        # A VLM annotation manifest normally repeats the source fields.  Merge
        # it over the original asset so a partially-written annotation can
        # never erase the original crop path, page or surrounding text.
        raw = {**raw_asset, **semantic_row}
        semantic = raw.get("visual_semantics") if isinstance(raw.get("visual_semantics"), dict) else {}
        has_local_visual_annotation = bool(semantic_row)
        asset_type = raw.get("asset_type")
        consistency = (
            semantic.get("source_context_consistency") or "unknown"
            if has_local_visual_annotation
            else "not_model_annotated"
        )
        deterministic_table = asset_type == "original_pdf_table"
        image_kind = (
            "table_or_parameter_sheet"
            if deterministic_table
            else semantic.get("image_kind") or "unclassified_original_visual"
        )
        customer_title = first_nonempty(
            raw.get("caption"), semantic.get("customer_image_caption"), semantic.get("visual_description")
        )
        text_context_present = bool(
            first_nonempty(raw.get("caption"), raw.get("nearby_text"), raw.get("section_heading"))
        )
        if not customer_title and text_context_present:
            customer_title = first_nonempty(raw.get("section_heading"), raw.get("nearby_text"))
        if not customer_title:
            customer_title = f"原始文档图示（第{raw.get('source_page') or '?'}页）"

        # If a vision label exists, use its consistency check.  Otherwise the
        # crop remains searchable through its source-page text and is clearly
        # marked as pending visual annotation rather than silently discarded.
        retrieval_eligible = bool(raw.get("image_path")) and (
            (consistency == "consistent" and bool(customer_title))
            if has_local_visual_annotation
            else text_context_present
        )
        review_reasons: list[str] = []
        if has_local_visual_annotation and consistency != "consistent":
            review_reasons.append("visual_context_not_consistent")
        if not raw.get("image_path"):
            review_reasons.append("missing_original_image_path")
        if not text_context_present:
            review_reasons.append("missing_page_text_context")
        if not customer_title:
            review_reasons.append("missing_customer_title")

        record = {
            "record_type": "clean_visual_asset",
            "asset_id": raw.get("asset_id"),
            "document_id": raw.get("document_id"),
            "document_name": raw.get("document_name"),
            "source_pdf": raw.get("source_pdf"),
            "source_page": raw.get("source_page"),
            "bbox": raw.get("bbox"),
            "asset_type": asset_type,
            "effective_image_kind": image_kind,
            "image_path": raw.get("image_path"),
            "customer_title": customer_title,
            "caption": raw.get("caption"),
            "nearby_text": raw.get("nearby_text"),
            "section_heading": raw.get("section_heading"),
            "search_metadata": {
                "visual_description": semantic.get("visual_description"),
                "visible_components": semantic.get("visible_components") or [],
                "recommended_search_terms": semantic.get("recommended_search_terms") or [],
            },
            "visual_annotation_status": "locally_annotated" if has_local_visual_annotation else "pending_local_annotation",
            "retrieval_eligible": retrieval_eligible,
            "facts_eligible": False,
            "customer_shareable": True,
            "source_context_consistency": consistency,
            "visual_annotation_confidence": semantic.get("confidence") or "unknown",
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "citation_rule": "Return only the original image_path with source_pdf, source_page and bbox. Vision metadata is retrieval-only and must not be stated as a technical fact.",
        }
        output.append(record)
        counters[f"effective_kind_{image_kind}"] += 1
        counters[f"consistency_{consistency}"] += 1
        counters["locally_annotated" if has_local_visual_annotation else "pending_local_annotation"] += 1
        if retrieval_eligible:
            counters["retrieval_eligible"] += 1
        else:
            counters["retrieval_excluded"] += 1
        if deterministic_table:
            counters["table_type_overrides"] += 1
        if review_reasons:
            manual_review.append(
                {
                    "record_type": "manual_review_visual",
                    "item_id": record["asset_id"],
                    "reason": ",".join(review_reasons),
                    "customer_title": customer_title,
                    "source_pdf": raw.get("source_pdf"),
                    "source_page": raw.get("source_page"),
                    "bbox": raw.get("bbox"),
                }
            )

    counters["raw_visual_records"] = len(raw_assets_by_id)
    counters["local_visual_annotation_records"] = len(semantics_by_asset_id)
    return output, manual_review, dict(counters)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare locally-ingested construction evidence for RAG indexing.")
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--visual-semantics", type=Path, default=DEFAULT_VISUAL_SEMANTICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    assets_root = args.assets_root.resolve()
    visual_semantics = args.visual_semantics.resolve()
    output_dir = args.output.resolve()
    if not assets_root.exists():
        raise SystemExit(f"Asset root not found: {assets_root}")
    if not visual_semantics.exists():
        raise SystemExit(f"Visual semantic manifest not found: {visual_semantics}")

    output_dir.mkdir(parents=True, exist_ok=True)
    native_pdf = NativePdfText()
    text_records, text_review, text_counts, examples = build_clean_text_records(assets_root, native_pdf)
    visual_records, visual_review, visual_counts = build_clean_visual_records(assets_root, visual_semantics)
    review_queue = text_review + visual_review

    write_jsonl(output_dir / "text_evidence_clean.jsonl", text_records)
    write_jsonl(output_dir / "visual_assets_clean.jsonl", visual_records)
    write_jsonl(output_dir / "manual_review_queue.jsonl", review_queue)
    report = {
        "generator": "scripts/prepare_rag_evidence.py",
        "input_assets_root": str(assets_root),
        "input_visual_semantics": str(visual_semantics),
        "outputs": {
            "text_evidence_clean": str(output_dir / "text_evidence_clean.jsonl"),
            "visual_assets_clean": str(output_dir / "visual_assets_clean.jsonl"),
            "manual_review_queue": str(output_dir / "manual_review_queue.jsonl"),
        },
        "text_quality": text_counts,
        "visual_quality": visual_counts,
        "manual_review_count": len(review_queue),
        "repair_examples": examples[:20],
        "safety_rules": {
            "vision_metadata_is_technical_fact_source": False,
            "visual_facts_eligible": False,
            "customer_visual_must_use_original_crop": True,
            "every_retrieved_item_keeps_source_page": True,
            "customer_shareable_default": True,
        },
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Created {len(text_records)} clean text chunks and {len(visual_records)} clean visual assets.")
    print(f"Manual review queue: {len(review_queue)} item(s).")
    print(f"Quality report: {output_dir / 'quality_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
