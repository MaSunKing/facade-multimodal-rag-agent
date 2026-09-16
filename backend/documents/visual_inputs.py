"""Bounded multi-source visual input with content deduplication."""
import hashlib
from pathlib import Path


def merge_visual_paths(direct: Path | None, evidence: list[Path], limit: int = 4):
    selected, seen, records = [], set(), []
    candidates = ([(direct, "direct_upload")] if direct else []) + [(p, "retrieved_asset") for p in evidence]
    for path, origin in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        status = "duplicate" if digest in seen else "budget_excluded" if len(selected) >= limit else "selected"
        records.append({"origin": origin, "sha256": digest, "status": status})
        if status == "selected":
            seen.add(digest)
            selected.append(path)
    return selected, {"candidates": records, "selected_count": len(selected), "coverage_complete": not any(r["status"] == "budget_excluded" for r in records)}
