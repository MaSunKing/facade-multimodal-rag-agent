"""Build a local visual-reference index for conservative façade appearance matching.

This script does not train or classify a product.  It embeds only customer-shareable
internal product/project images with Qwen3-VL's local vision encoder, then stores
unit-normalized vectors for nearest-reference lookup.  The runtime treats every
result as an *appearance candidate*, never product authentication.

Run while the local FastAPI service is stopped, because it loads Qwen3-VL-8B
in the same 4-bit configuration and needs most of the 16 GB GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_ASSETS = ROOT / "data" / "sales" / "processed" / "rag_ready" / "visual_assets_tagged.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "sales" / "processed" / "visual_identity_index"
REFERENCE_KINDS = {"product_photo", "project_photo"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_reference_candidate(record: dict) -> bool:
    image_path = Path(str(record.get("image_path") or ""))
    return bool(
        record.get("retrieval_eligible")
        and record.get("customer_shareable")
        and record.get("effective_image_kind") in REFERENCE_KINDS
        and record.get("source_context_consistency") != "inconsistent"
        and not record.get("review_required")
        and image_path.is_file()
    )


def compact_manifest_record(record: dict) -> dict:
    return {
        "asset_id": record["asset_id"],
        "customer_title": record.get("customer_title") or "内部外观参考图",
        "reference_kind": record.get("effective_image_kind"),
        "citation": {
            "document_name": record.get("document_name"),
            "source_page": record.get("source_page"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.assets.is_file():
        raise FileNotFoundError(f"找不到视觉资产清单：{args.assets}")
    index_path = args.output_dir / "appearance_index.npz"
    manifest_path = args.output_dir / "appearance_manifest.json"
    if (index_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError("外观索引已存在；确认要重建时加入 --force。")

    candidates = [record for record in read_jsonl(args.assets) if is_reference_candidate(record)]
    if not candidates:
        raise RuntimeError("没有找到可用于外观匹配的内部参考图片。")

    # Import only after we know that work is required; this loads no model until
    # image_appearance_embedding is first called.
    from backend.app import image_appearance_embedding

    vectors: list[np.ndarray] = []
    manifest: list[dict] = []
    for index, record in enumerate(candidates, start=1):
        image_path = Path(str(record["image_path"]))
        print(f"[{index}/{len(candidates)}] {record['asset_id']}: {image_path.name}", flush=True)
        try:
            vector = image_appearance_embedding(image_path)
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__}: {exc}", flush=True)
            continue
        vectors.append(vector)
        manifest.append(compact_manifest_record(record))

    if len(vectors) < 3:
        raise RuntimeError("成功嵌入的参考图不足 3 张，未生成索引。")

    matrix = np.stack(vectors).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(index_path, vectors=matrix)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "reference_count": len(manifest),
        "embedding_dimension": int(matrix.shape[1]),
        "reference_kinds": sorted(REFERENCE_KINDS),
        "index_path": str(index_path),
        "manifest_path": str(manifest_path),
        "warning": "Scores are for internal appearance-reference retrieval only; they do not authenticate product brand, truth, quality or engineering suitability.",
    }
    (args.output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
