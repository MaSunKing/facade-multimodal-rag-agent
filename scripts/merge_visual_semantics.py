"""Merge locally generated per-document visual annotations into one manifest.

The combined file is only a convenience input for RAG evidence preparation.
Each source row remains tied to its local original image crop and page number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_ROOT = ROOT / "data" / "sales" / "processed" / "rag_assets"
DEFAULT_OUTPUT = DEFAULT_ASSETS_ROOT / "all_visual_semantics_qwen3vl.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge local Qwen3-VL annotation manifests.")
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--existing-base",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Existing combined manifest for legacy documents that have no per-document output yet.",
    )
    args = parser.parse_args()

    assets_root = args.assets_root.resolve()
    output = args.output.resolve()
    sources: list[Path] = []
    existing_base = args.existing_base.resolve()
    if existing_base.exists() and existing_base != output:
        sources.append(existing_base)
    elif existing_base.exists():
        # The pre-existing global manifest contains the three already processed
        # construction plans.  Read it before overwriting the same path.
        sources.append(existing_base)
    sources.extend(
        path
        for path in sorted(assets_root.glob("*/visual_semantics_qwen3vl.jsonl"))
        if path.resolve() != output
    )

    by_asset_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        for row in read_jsonl(source):
            asset_id = str(row.get("asset_id") or "")
            if asset_id:
                by_asset_id[asset_id] = row
    if not by_asset_id:
        raise SystemExit("No visual annotation records were found to merge.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for asset_id in sorted(by_asset_id):
            handle.write(json.dumps(by_asset_id[asset_id], ensure_ascii=False) + "\n")
    print(f"Merged {len(by_asset_id)} local visual annotation records: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
