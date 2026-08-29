"""Build a local dense vector index from the already-audited lexical index."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.sales.dense_retrieval import EMBEDDING_MODEL_DIR, LocalQwenEmbedding


DEFAULT_LEXICAL_INDEX = ROOT / "data" / "sales" / "processed" / "rag_index" / "lexical_index.json"
DEFAULT_OUTPUT = ROOT / "data" / "sales" / "processed" / "rag_index" / "dense_index.npz"
DEFAULT_METADATA = ROOT / "data" / "sales" / "processed" / "rag_index" / "dense_index_metadata.json"


def document_text(document: dict[str, Any]) -> str:
    return str(document.get("text") or document.get("search_text") or "").strip()


def index_fingerprint(documents: list[dict[str, Any]]) -> str:
    """Match ``build_rag_index.index_fingerprint`` without importing the builder."""

    digest = hashlib.sha256()
    for document in documents:
        digest.update(str(document.get("id") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(document_text(document).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local Qwen dense RAG index.")
    parser.add_argument("--lexical-index", type=Path, default=DEFAULT_LEXICAL_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    args = parser.parse_args()

    lexical = json.loads(args.lexical_index.read_text(encoding="utf-8"))
    documents = [document for document in lexical.get("documents", []) if document_text(document)]
    lexical_fingerprint = str((lexical.get("metadata") or {}).get("index_fingerprint") or "")
    computed_fingerprint = index_fingerprint(documents)
    if lexical_fingerprint and lexical_fingerprint != computed_fingerprint:
        raise ValueError("lexical index fingerprint does not match its document contents")
    lexical_fingerprint = lexical_fingerprint or computed_fingerprint
    model = LocalQwenEmbedding(device=args.device)
    vectors = model.encode(
        [document_text(document) for document in documents],
        query=False,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f"{args.output.name}.tmp.npz")
    temporary_metadata = args.metadata.with_name(f"{args.metadata.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    temporary_metadata.unlink(missing_ok=True)
    np.savez_compressed(
        temporary_output,
        ids=np.array([str(document["id"]) for document in documents]),
        vectors=vectors,
    )
    meta = {
        "generator": "scripts/build_dense_index.py",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(EMBEDDING_MODEL_DIR),
        "device": model.device,
        "document_count": len(documents),
        "embedding_dimension": int(vectors.shape[1]) if len(vectors) else 0,
        "normalised": True,
        "source_lexical_index": str(args.lexical_index),
        "source_index_fingerprint": lexical_fingerprint,
        "privacy": "local_index_no_cloud_upload",
    }
    temporary_metadata.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    os.replace(temporary_metadata, args.metadata)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
