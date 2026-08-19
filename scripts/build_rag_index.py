"""Build a small, local BM25 index from RAG-cleaned evidence.

The index contains only eligible text evidence and eligible original visual
assets.  It is an explicit first retrieval layer, not a cloud vector database.
"""

from __future__ import annotations

import argparse
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


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def visual_search_text(record: dict[str, Any]) -> str:
    metadata = record.get("search_metadata") if isinstance(record.get("search_metadata"), dict) else {}
    source_caption = record.get("caption")
    # An original PDF caption is stronger than any VLM description.  Using the
    # VLM description alongside a caption can broaden a precise label (for
    # example, “阴角图”) into a wrong neighbouring concept (“阳角”).
    if isinstance(source_caption, str) and source_caption.strip():
        return source_caption.strip()
    terms = metadata.get("recommended_search_terms") or []
    components = metadata.get("visible_components") or []
    fields = [
        record.get("customer_title"),
        metadata.get("visual_description"),
        " ".join(str(term) for term in terms),
        " ".join(str(component) for component in components),
    ]
    return "\n".join(str(field).strip() for field in fields if isinstance(field, str) and field.strip())


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
        "应用案例 项目案例 保温装饰一体板",
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
    if not text_path.exists() or not visual_path.exists():
        raise FileNotFoundError("缺少 RAG 清洗结果。请先运行 scripts\\prepare_rag_evidence.ps1。")

    documents: list[dict[str, Any]] = []
    for record in read_jsonl(text_path):
        if not record.get("index_eligible") or not record.get("fact_eligible") or not record.get("customer_shareable", True):
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
                }
            )

    visual_records = list(read_jsonl(visual_path))
    bundles_by_asset_id: dict[str, dict[str, Any]] = {}
    if bundle_path.exists():
        bundles_by_asset_id = {
            str(bundle.get("asset_id")): bundle
            for bundle in read_jsonl(bundle_path)
            if bundle.get("asset_id")
        }
    visual_ids_by_document_page: dict[tuple[str, int], list[str]] = {}
    for record in visual_records:
        if not record.get("retrieval_eligible") or not record.get("customer_shareable"):
            continue
        bundle = bundles_by_asset_id.get(str(record.get("asset_id")))
        linked_context = str(bundle.get("retrieval_text") or "") if bundle else ""
        search_text = "\n".join(value for value in (visual_search_text(record), linked_context) if value)
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
                }
            )

    if case_path.exists():
        for record in read_jsonl(case_path):
            if not record.get("customer_shareable") or not record.get("case_id"):
                continue
            search_text = project_case_search_text(record)
            tokens = tokenize(search_text)
            if not tokens:
                continue
            document_name = str(record.get("source_document") or "")
            source_pages = record.get("source_pages") or [record.get("source_page")]
            visual_asset_ids: list[str] = []
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
                }
            )

    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document["tokens"]))
    average_document_length = sum(len(document["tokens"]) for document in documents) / max(len(documents), 1)
    payload = {
        "metadata": {
            "generator": "scripts/build_rag_index.py",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "strategy": "local_bm25_lexical",
            "text_document_count": sum(document["kind"] == "text" for document in documents),
            "visual_document_count": sum(document["kind"] == "visual" for document in documents),
            "project_case_count": sum(document["kind"] == "project_case" for document in documents),
            "multimodal_bundle_count": len(bundles_by_asset_id),
            "taxonomy_version": "1.0" if (ready_dir / "knowledge_taxonomy_report.json").exists() else None,
            "tagged_evidence": text_path.name.endswith("_tagged.jsonl"),
            "privacy": "local_index_no_cloud_upload",
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
