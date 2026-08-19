"""Run the review-first, local-only LangGraph knowledge-intake pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.sales.ingestion_graph import DEFAULT_RUN_ROOT, DOMAIN_DEFAULTS, IntakeOptions, build_knowledge_intake_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Local-only, review-first knowledge-base intake workflow.")
    parser.add_argument("source", type=Path, help="Local PDF or DOCX file to process. Keep source files under data/sales/raw/intake when possible.")
    parser.add_argument("--knowledge-domain", choices=sorted(DOMAIN_DEFAULTS), default=None)
    parser.add_argument("--document-category", default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--annotate-visuals", action="store_true", help="Run local Qwen3-VL candidate labels for extracted PDF images.")
    parser.add_argument("--visual-limit", type=int, default=0, help="Annotate at most this many visual assets; 0 means all.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse a completed local extraction in the same run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and classify only; do not parse or write source-derived outputs.")
    args = parser.parse_args()

    options = IntakeOptions(
        source_path=args.source,
        run_root=args.run_root,
        knowledge_domain=args.knowledge_domain,
        document_category=args.document_category,
        annotate_visuals=args.annotate_visuals,
        visual_limit=max(0, args.visual_limit),
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    state = build_knowledge_intake_graph().invoke({"options": options})
    review = state["review"]
    print(
        json.dumps(
            {
                "status": review["status"],
                "review_manifest": review["review_manifest"],
                "run_id": state["run"]["run_id"],
                "knowledge_domain": state["classification"]["knowledge_domain"],
                "text_record_count": state["extraction"].get("text_record_count"),
                "visual_asset_count": state["extraction"].get("visual_asset_count"),
                "visual_annotation_status": state["visual_annotation"]["status"],
                "promotion_status": review["promotion_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
