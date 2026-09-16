"""Reproducible CPU source checks, with explicit skips and failure reporting."""
from __future__ import annotations

import compileall
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["CUSTOMER_DOCUMENT_OCR_ENABLED"] = "0"
os.environ["CUSTOMER_ATTACHMENT_SEMANTIC_RERANK"] = "0"
os.environ["RAG_HYBRID_ENABLED"] = "0"


def main():
    started = time.perf_counter()
    syntax = all(compileall.compile_dir(str(ROOT / p), quiet=1) for p in ("backend", "scripts"))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "backend/tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {"syntax_valid": syntax, "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
        "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "failures": [test.id() for test, _ in result.failures], "errors": [test.id() for test, _ in result.errors],
        "successful": syntax and result.wasSuccessful(), "model_generation_tested": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3)}
    folder = ROOT / "runtime/public_smoke"; folder.mkdir(parents=True, exist_ok=True)
    (folder / "cpu_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "skipped"}, ensure_ascii=False))
    print(f"Explicitly skipped {len(result.skipped)} tests; reasons are in the local JSON report.")
    raise SystemExit(0 if report["successful"] else 1)


if __name__ == "__main__":
    main()
