"""Three fixed local HTTP generation demos; not an accuracy benchmark.

Raw responses are private runtime output. Review them before publishing:
they may reference business index contents or signed/session URLs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx
from public_smoke import fixtures

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    dict(id="joint_sources", files=["sample_checklist.docx"],
         question="请总结上传检查表的核对要求，再结合企业真岩石产品资料介绍能确认的产品信息和仍需核对的项目条件。DemoPanel-X是虚构产品，不要当作真岩石，也不要声称已符合工程标准。"),
    dict(id="source_difference", files=["sample_product.pdf", "sample_spec.xlsx"],
         question="比较上传的PDF和Excel中DemoPanel-X的厚度记录：有哪些数值、分别在哪里、版本是否相同？请同时引用两个文件，不擅自认定哪个值正确。"),
    dict(id="visual_and_text", files=["sample_drawing.png", "sample_checklist.docx"],
         question="请根据上传示意图描述Panel与Support的可见关系，并根据上传Word列出施工前需要核对的事项。这是虚构示意图，只描述可见内容，不推断承载能力或合规性。"),
]

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Source checkout of the already-running service, for provenance only.")
    parser.add_argument("--fixtures-dir", type=Path,
                        help="Reuse frozen fixture bytes for a paired retest instead of regenerating containers.")
    args = parser.parse_args()
    if urlparse(args.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Only local HTTP services are allowed; web search is disabled.")
    out = args.output.resolve()
    if out.exists():
        parser.error("Use a new immutable output directory.")
    out.mkdir(parents=True)
    names = sorted({n for c in CASES for n in c["files"]})
    files = ({n:(args.fixtures_dir/n).read_bytes() for n in names}
             if args.fixtures_dir else fixtures())
    headers = {"x-facade-client-id": "public_real_agent_demo_20260917"}
    report = {"fixture_provenance": "project_authored_synthetic", "cases": [],
              "web_search_authorized": False, "question_variants": False,
              "source_sha256": {n: hashlib.sha256(b).hexdigest() for n,b in files.items()}}
    fingerprint_paths = ["backend/app.py", "backend/sales/answer_graph.py",
        "backend/sales/tool_planner.py", "backend/sales/context_engine.py",
        "backend/sales/fact_normalization.py", "backend/documents/customer_sessions.py",
        "backend/documents/table_context.py"]
    fingerprints = {n: hashlib.sha256((args.source_root/n).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
                    for n in fingerprint_paths}
    save(out/"protocol.json", {**report, "questions": CASES,
        "execution_provenance": {"started_at":datetime.now(timezone.utc).isoformat(),
            "service_scope":"Caller-specified running source checkout; no inference of process version from HTTP200.",
            "source_is_public_checkout":args.source_root.resolve()==ROOT,
            "fingerprint_normalization":"CRLF_to_LF"}, "code_fingerprints":fingerprints})
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=180, trust_env=False) as client:
        report["health_before"] = client.get("/health/ready").json()
        save(out/"raw_report.json", report)
        for case in CASES:
            sid = None
            result = {"id": case["id"], "question": case["question"], "files": case["files"]}
            start = time.perf_counter()
            try:
                uploaded = client.post("/api/copilot/documents", files=[("files",(n,files[n])) for n in case["files"]])
                result["upload_status"] = uploaded.status_code
                uploaded.raise_for_status()
                result["upload_summary"] = uploaded.json()
                sid = result["upload_summary"]["session_id"]
                response = client.post("/api/copilot/answer", json={"customer_question":case["question"],
                    "document_session_id":sid,"use_online_search":False,"memory_enabled":False})
                result["answer_status"] = response.status_code
                result["response"] = response.json()
            except Exception as exc:
                result["exception_type"] = type(exc).__name__
                result["error"] = str(exc)
            finally:
                result["latency_seconds"] = round(time.perf_counter()-start,3)
                if sid:
                    try:
                        result["session_deleted"] = client.delete(f"/api/copilot/documents/{sid}").status_code == 200
                    except Exception:
                        result["session_deleted"] = False
            report["cases"].append(result)
            save(out/f"{case['id']}.raw.json", result)
            save(out/"raw_report.json", report)
            print(json.dumps({k:result.get(k) for k in ["id","upload_status","answer_status","latency_seconds","exception_type"]}), flush=True)
        report["health_after"] = client.get("/health/ready").json()
        save(out/"raw_report.json", report)

if __name__ == "__main__":
    main()
