"""Build a small, local BM25 index from RAG-cleaned evidence.

The index contains only eligible text evidence and eligible original visual
assets.  It is an explicit first retrieval layer, not a cloud vector database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.sales.retriever import tokenize


DEFAULT_READY_DIR = ROOT / "data" / "sales" / "processed" / "rag_ready"
DEFAULT_OUTPUT = ROOT / "data" / "sales" / "processed" / "rag_index" / "lexical_index.json"


def document_text(document: dict[str, Any]) -> str:
    """Return the exact customer-searchable content represented by one index row."""

    return str(document.get("text") or document.get("search_text") or "").strip()


def index_fingerprint(documents: list[dict[str, Any]]) -> str:
    """Create a stable content fingerprint shared with the dense-index builder.

    A lexical rebuild can add or change evidence while leaving an older dense
    file on disk.  Persisting this digest lets the online retriever fail closed
    instead of silently presenting a partial dense index as a full hybrid one.
    """

    digest = hashlib.sha256()
    for document in documents:
        digest.update(str(document.get("id") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(document_text(document).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def visual_catalog_search_text(catalog_entry: dict[str, Any] | None) -> str:
    if not catalog_entry:
        return ""
    fields = [
        catalog_entry.get("customer_caption"),
        catalog_entry.get("product_name") or catalog_entry.get("canonical_product"),
        catalog_entry.get("variant_or_code"),
        catalog_entry.get("project_name"),
        catalog_entry.get("visual_role"),
    ]
    if catalog_entry.get("product_gallery_eligible"):
        fields.append("产品 产品图 产品图片 产品样板 产品展示")
    return "\n".join(str(field).strip() for field in fields if isinstance(field, str) and field.strip())


def visual_search_text(
    record: dict[str, Any],
    catalog_entry: dict[str, Any] | None = None,
) -> str:
    metadata = record.get("search_metadata") if isinstance(record.get("search_metadata"), dict) else {}
    source_caption = record.get("caption")
    # An original PDF caption is stronger than any VLM description.  Using the
    # VLM description alongside a caption can broaden a precise label (for
    # example, “阴角图”) into a wrong neighbouring concept (“阳角”).
    if isinstance(source_caption, str) and source_caption.strip():
        source_text = source_caption.strip()
    else:
        terms = metadata.get("recommended_search_terms") or []
        components = metadata.get("visible_components") or []
        fields = [
            record.get("customer_title"),
            metadata.get("visual_description"),
            " ".join(str(term) for term in terms),
            " ".join(str(component) for component in components),
        ]
        source_text = "\n".join(
            str(field).strip() for field in fields if isinstance(field, str) and field.strip()
        )
    return "\n".join(
        value for value in (source_text, visual_catalog_search_text(catalog_entry)) if value
    )


def project_case_search_text(record: dict[str, Any]) -> str:
    """Keep every catalogue field searchable without treating it as a technical claim."""

    fields = (
        record.get("project_name"),
        record.get("region"),
        record.get("project_type"),
        record.get("application_area"),
        record.get("product"),
        record.get("installation_method"),
        record.get("area_m2"),
        record.get("completion_year"),
        "应用案例 项目案例 真岩石",
    )
    return "\n".join(str(field).strip() for field in fields if isinstance(field, str) and field.strip())


def build_index(ready_dir: Path, output_path: Path) -> dict[str, Any]:
    # Tagged evidence is generated from the clean evidence without mutating it.
    # Prefer it when available so every retrieval item carries a knowledge domain.
    text_path = ready_dir / "text_evidence_tagged.jsonl"
    visual_path = ready_dir / "visual_assets_tagged.jsonl"
    case_path = ready_dir / "project_cases_tagged.jsonl"
    if not text_path.exists():
        text_path = ready_dir / "text_evidence_clean.jsonl"
    if not visual_path.exists():
        visual_path = ready_dir / "visual_assets_clean.jsonl"
    if not case_path.exists():
        case_path = ready_dir / "project_cases.jsonl"
    bundle_path = ready_dir / "multimodal_evidence_bundles.jsonl"
    catalog_path = ready_dir / "visual_catalog.jsonl"
    if not text_path.exists() or not visual_path.exists():
        raise FileNotFoundError("缺少 RAG 清洗结果。请先运行 scripts\\prepare_rag_evidence.ps1。")

    documents: list[dict[str, Any]] = []
    for record in read_jsonl(text_path):
        access_scope = str(record.get("access_scope") or record.get("visibility") or "public")
        if access_scope not in {"public", "internal"}:
            continue
        if not record.get("index_eligible") or not record.get("fact_eligible"):
            continue
        if access_scope == "public" and not record.get("customer_shareable", True):
            continue
        text = str(record.get("text") or "").strip()
        tokens = tokenize(text)
        if tokens:
            documents.append(
                {
                    "id": record["chunk_id"],
                    "kind": "text",
                    "text": text,
                    "tokens": tokens,
                    "source_refs": record.get("source_refs") or [],
                    "knowledge_domains": record.get("knowledge_domains") or [],
                    "document_categories": record.get("document_categories") or [],
                    "content_labels": record.get("content_labels") or [],
                    "source_taxonomy": record.get("source_taxonomy") or [],
                    "sales_playbook_use": record.get("sales_playbook_use"),
                    "access_scope": access_scope,
                }
            )

    visual_records = list(read_jsonl(visual_path))
    catalog_entries_by_asset_id: dict[str, list[dict[str, Any]]] = {}
    if catalog_path.exists():
        for entry in read_jsonl(catalog_path):
            asset_id = str(entry.get("asset_id") or "")
            if asset_id:
                catalog_entries_by_asset_id.setdefault(asset_id, []).append(entry)

    def primary_catalog_entry(asset_id: str) -> dict[str, Any] | None:
        entries = catalog_entries_by_asset_id.get(asset_id) or []
        if not entries:
            return None
        return sorted(
            entries,
            key=lambda item: (
                1 if item.get("gallery_type") == "product_sample" else 0,
                1 if item.get("hero") else 0,
                int(item.get("display_priority") or 0),
            ),
            reverse=True,
        )[0]
    bundles_by_asset_id: dict[str, dict[str, Any]] = {}
    if bundle_path.exists():
        bundles_by_asset_id = {
            str(bundle.get("asset_id")): bundle
            for bundle in read_jsonl(bundle_path)
            if bundle.get("asset_id")
        }
    visual_ids_by_document_page: dict[tuple[str, int], list[str]] = {}
    for record in visual_records:
        access_scope = str(record.get("access_scope") or record.get("visibility") or "public")
        if access_scope not in {"public", "internal"} or not record.get("retrieval_eligible"):
            continue
        if access_scope == "public" and not record.get("customer_shareable"):
            continue
        catalog_entry = primary_catalog_entry(str(record.get("asset_id") or ""))
        bundle = bundles_by_asset_id.get(str(record.get("asset_id")))
        linked_context = str(bundle.get("retrieval_text") or "") if bundle else ""
        search_text = "\n".join(
            value for value in (visual_search_text(record, catalog_entry), linked_context) if value
        )
        tokens = tokenize(search_text)
        if tokens:
            document_name = str(record.get("document_name") or "")
            source_page = record.get("source_page")
            if isinstance(source_page, int):
                visual_ids_by_document_page.setdefault((document_name, source_page), []).append(str(record["asset_id"]))
            documents.append(
                {
                    "id": f"visual:{record['asset_id']}",
                    "kind": "visual",
                    "asset_id": record["asset_id"],
                    "customer_title": record.get("customer_title"),
                    "asset_type": record.get("asset_type"),
                    "effective_image_kind": record.get("effective_image_kind"),
                    "gallery_type": catalog_entry.get("gallery_type") if catalog_entry else None,
                    "canonical_product": (
                        catalog_entry.get("canonical_product") if catalog_entry else None
                    ),
                    "product_name": (
                        catalog_entry.get("product_name")
                        or catalog_entry.get("canonical_product")
                        if catalog_entry
                        else None
                    ),
                    "variant_or_code": catalog_entry.get("variant_or_code") if catalog_entry else None,
                    "visual_role": catalog_entry.get("visual_role") if catalog_entry else None,
                    "product_gallery_eligible": bool(
                        catalog_entry and catalog_entry.get("product_gallery_eligible")
                    ),
                    "review_status": catalog_entry.get("review_status") if catalog_entry else None,
                    "display_priority": int(catalog_entry.get("display_priority") or 0) if catalog_entry else 0,
                    "hero": bool(catalog_entry and catalog_entry.get("hero")),
                    "customer_caption": catalog_entry.get("customer_caption") if catalog_entry else None,
                    "case_id": catalog_entry.get("case_id") if catalog_entry else None,
                    "project_name": catalog_entry.get("project_name") if catalog_entry else None,
                    "visual_catalog_version": catalog_entry.get("catalog_version") if catalog_entry else None,
                    "knowledge_domains": record.get("knowledge_domains") or [],
                    "document_categories": record.get("document_categories") or [],
                    "visual_labels": record.get("visual_labels") or [],
                    "linked_text_evidence": bundle.get("supporting_text") if bundle else [],
                    "multimodal_bundle_id": bundle.get("bundle_id") if bundle else None,
                    "search_text": search_text,
                    "tokens": tokens,
                    "citation": {
                        "document_name": record.get("document_name"),
                        "source_page": record.get("source_page"),
                        "bbox": record.get("bbox"),
                    },
                    "image_path": record.get("image_path"),
                    "access_scope": access_scope,
                }
            )

    if case_path.exists():
        for record in read_jsonl(case_path):
            access_scope = str(record.get("access_scope") or record.get("visibility") or "public")
            if access_scope not in {"public", "internal"} or not record.get("case_id"):
                continue
            if access_scope == "public" and not record.get("customer_shareable"):
                continue
            search_text = project_case_search_text(record)
            tokens = tokenize(search_text)
            if not tokens:
                continue
            document_name = str(record.get("source_document") or "")
            source_pages = record.get("source_pages") or [record.get("source_page")]
            # The visual-catalogue build persists reviewed/deterministic links
            # in the case record.  Respect those ordered IDs (hero first), and
            # retain the former same-page inference only for older indexes.
            visual_asset_ids = [
                str(asset_id)
                for asset_id in (record.get("visual_asset_ids") or [])
                if str(asset_id)
            ]
            if not visual_asset_ids:
                for source_page in source_pages:
                    if isinstance(source_page, int):
                        visual_asset_ids.extend(visual_ids_by_document_page.get((document_name, source_page), []))
            record = {**record, "visual_asset_ids": list(dict.fromkeys(visual_asset_ids))}
            documents.append(
                {
                    "id": record["case_id"],
                    "kind": "project_case",
                    "search_text": search_text,
                    "tokens": tokens,
                    "case": record,
                    "access_scope": access_scope,
                }
            )

    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document["tokens"]))
    average_document_length = sum(len(document["tokens"]) for document in documents) / max(len(documents), 1)
    taxonomy_version = None
    taxonomy_report_path = ready_dir / "knowledge_taxonomy_report.json"
    if taxonomy_report_path.exists():
        taxonomy_report = json.loads(taxonomy_report_path.read_text(encoding="utf-8"))
        taxonomy_version = taxonomy_report.get("taxonomy_version")
    payload = {
        "metadata": {
            "generator": "scripts/build_rag_index.py",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "strategy": "local_bm25_lexical",
            "text_document_count": sum(document["kind"] == "text" for document in documents),
            "visual_document_count": sum(document["kind"] == "visual" for document in documents),
            "project_case_count": sum(document["kind"] == "project_case" for document in documents),
            "multimodal_bundle_count": len(bundles_by_asset_id),
            "visual_catalog_entry_count": sum(
                len(entries) for entries in catalog_entries_by_asset_id.values()
            ),
            "taxonomy_version": taxonomy_version,
            "tagged_evidence": text_path.name.endswith("_tagged.jsonl"),
            "index_fingerprint": index_fingerprint(documents),
            "privacy": "local_index_no_cloud_upload",
            "access_control_version": "knowledge_access_v1",
            "access_scope_counts": {
                "public": sum(document.get("access_scope") == "public" for document in documents),
                "internal": sum(document.get("access_scope") == "internal" for document in documents),
            },
        },
        "average_document_length": average_document_length,
        "document_frequency": dict(document_frequency),
        "documents": documents,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local lexical RAG index from clean evidence.")
    parser.add_argument("--ready-dir", type=Path, default=DEFAULT_READY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_index(args.ready_dir.resolve(), args.output.resolve())
    meta = payload["metadata"]
    print(
        f"Created local BM25 index with {meta['text_document_count']} text records and "
        f"{meta['visual_document_count']} visual records: {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
