"""Run the reviewed 100-question sales RAG benchmark against the local API.

The runner is deliberately sequential because the local generation model and
hybrid retriever share a 16 GB GPU.  Every completed case is appended to disk,
so an interrupted run can resume without repeating successful inference.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import mimetypes
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "data/sales/evaluation/rag_eval_v1_draft/questions.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/sales_rag_eval_v03_fixed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: str) -> str:
    return re.sub(r"[\s，。；：、,.!?！？（）()《》\[\]【】'\"“”‘’\-—]+", "", value).casefold()


def point_exactly_covered(point: str, answer: str) -> bool:
    """Conservative lexical diagnostic; the LLM judge handles paraphrases."""

    normalized_point = normalize_text(point)
    normalized_answer = normalize_text(answer)
    if not normalized_point:
        return True
    return normalized_point in normalized_answer


def request_answer(endpoint: str, sample: dict[str, Any], timeout: float) -> dict[str, Any]:
    payload = {
        "customer_question": sample["question"],
        "conversation_context": sample.get("conversation_context") or [],
        "project_context": {},
        "use_online_search": False,
    }
    if sample.get("task_type") == "grounded_visual_qa":
        visual = next(
            (
                item
                for item in sample.get("gold_evidence") or []
                if item.get("evidence_type") == "visual" and item.get("image_path")
            ),
            None,
        )
        if visual is None:
            raise ValueError(f"Visual sample {sample['sample_id']} has no image_path")
        image_path = Path(str(visual["image_path"]))
        if not image_path.is_file():
            raise ValueError(f"Visual sample image is missing: {image_path}")
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload["image_data_url"] = f"data:{media_type};base64,{encoded}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json; charset=utf-8"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def prediction_text(response: dict[str, Any]) -> str:
    """Return the customer-visible answer without duplicating diagnostic fields."""

    reply = str(response.get("customer_reply") or "").strip()
    parts = [reply] if reply else []
    rendered = normalize_text(reply)
    for value in response.get("key_points") or []:
        point = str(value).strip()
        if point and normalize_text(point) not in rendered:
            parts.append(point)
            rendered += normalize_text(point)
    # Direct-image safety handling already builds customer_reply from the
    # cleaned observations.  image_observations remains structured audit data,
    # not a second answer body.  Use it only as a last-resort fallback.
    if not parts:
        parts.extend(
            str(value).strip()
            for value in response.get("image_observations") or []
            if str(value).strip()
        )
    return "\n".join(parts)


def result_record(sample: dict[str, Any], response: dict[str, Any], latency: float) -> dict[str, Any]:
    answer = prediction_text(response)
    expected_points = [str(value) for value in sample.get("expected_points") or []]
    covered = [point_exactly_covered(point, answer) for point in expected_points]
    retrieved_visual_ids = [
        str(item.get("asset_id") or "") for item in response.get("visual_assets") or [] if item.get("asset_id")
    ]
    input_visual_ids = [
        str(item["evidence_id"])
        for item in sample.get("gold_evidence") or []
        if item.get("evidence_type") == "visual"
    ]
    direct_visual_input = sample.get("task_type") == "grounded_visual_qa"
    visual_input_processed = bool(
        direct_visual_input
        and (response.get("meta") or {}).get("customer_image_processed_locally")
    )
    predicted_answerable = bool(response.get("answerable"))
    return {
        "sample_id": sample["sample_id"],
        "task_type": sample["task_type"],
        "capability": sample["capability"],
        "question": sample["question"],
        "gold_answerable": bool(sample["answerable"]),
        "predicted_answerable": predicted_answerable,
        "answer": answer,
        "expected_answer": sample.get("expected_answer"),
        "expected_points": expected_points,
        "lexical_point_hits": covered,
        "lexical_keypoint_coverage": sum(covered) / len(covered) if covered else None,
        "citation_count": len(response.get("citations") or []),
        "citations": response.get("citations") or [],
        # A directly uploaded image and a visual retrieved from the company RAG
        # are different evidence channels.  Preserve the legacy keys for old
        # readers, but never report a false retrieval miss for direct-image QA.
        "returned_visual_ids": retrieved_visual_ids,
        "retrieved_visual_ids": retrieved_visual_ids,
        "gold_visual_ids": input_visual_ids,
        "input_visual_ids": input_visual_ids,
        "visual_input_processed": visual_input_processed,
        "visual_evidence_hit": None if direct_visual_input else (
            bool(set(retrieved_visual_ids) & set(input_visual_ids)) if input_visual_ids else None
        ),
        "risk_warnings": response.get("risk_warnings") or [],
        "missing_information": response.get("missing_information") or [],
        "image_observations": response.get("image_observations") or [],
        "visual_input_attached": direct_visual_input,
        "retrieval": response.get("retrieval") or {},
        "meta": response.get("meta") or {},
        "latency_seconds": round(latency, 3),
        "success": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def compute_metrics(rows: list[dict[str, Any]], expected_total: int) -> dict[str, Any]:
    successful = [row for row in rows if row.get("success")]
    answerable = [row for row in successful if row["gold_answerable"]]
    refusals = [row for row in successful if not row["gold_answerable"]]
    true_positive_refusals = sum(not row["predicted_answerable"] for row in refusals)
    predicted_refusals = sum(not row["predicted_answerable"] for row in successful)
    false_refusals = sum(not row["predicted_answerable"] for row in answerable)
    refusal_precision = true_positive_refusals / predicted_refusals if predicted_refusals else 0.0
    refusal_recall = true_positive_refusals / len(refusals) if refusals else 0.0
    refusal_f1 = (
        2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
        if refusal_precision + refusal_recall
        else 0.0
    )
    lexical_scores = [row["lexical_keypoint_coverage"] for row in answerable if row["lexical_keypoint_coverage"] is not None]
    visual_rows = [row for row in successful if row.get("task_type") == "grounded_visual_qa"]
    latencies = sorted(float(row["latency_seconds"]) for row in successful)

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, round((len(values) - 1) * fraction))
        return round(values[index], 3)

    return {
        "dataset": "sales_rag_eval_v1_draft_v0.3-runner",
        "expected_cases": expected_total,
        "completed_cases": len(successful),
        "failed_cases": len(rows) - len(successful),
        "run_complete": len(successful) == expected_total,
        "answerability_accuracy": (
            sum(row["gold_answerable"] == row["predicted_answerable"] for row in successful) / len(successful)
            if successful
            else 0.0
        ),
        "false_refusal_rate_on_answerable": false_refusals / len(answerable) if answerable else 0.0,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "citation_presence_rate_on_answered": (
            sum(row["citation_count"] > 0 for row in answerable if row["predicted_answerable"])
            / max(1, sum(row["predicted_answerable"] for row in answerable))
        ),
        "lexical_keypoint_coverage_diagnostic": sum(lexical_scores) / len(lexical_scores) if lexical_scores else 0.0,
        # These cases test direct customer-image understanding.  Knowledge-base
        # visual retrieval is a different track and must not be inferred from
        # whether the already supplied gold image appears in returned assets.
        "visual_input_attached_rate": (
            sum(bool(row.get("visual_input_attached")) for row in visual_rows) / len(visual_rows)
            if visual_rows
            else None
        ),
        "visual_input_processed_rate": (
            sum(bool(row.get("visual_input_processed")) for row in visual_rows) / len(visual_rows)
            if visual_rows
            else None
        ),
        "visual_evidence_recall": None,
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "median": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "by_task_type": dict(Counter(row["task_type"] for row in successful)),
        "scoring_note": (
            "Lexical keypoint coverage is a conservative diagnostic only and is not the final answer score. "
            "No external LLM judge is used. Reviewed deterministic assertions and citation-support checks "
            "must be used for the formal score."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local sales RAG agent sequentially")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/api/copilot/answer")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--inter-case-delay",
        type=float,
        default=16.0,
        help="Seconds between sequential requests; 16s stays below the public demo limit of 4 requests/minute.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Optional comma-separated sample IDs; applied before --limit.",
    )
    args = parser.parse_args()

    samples = read_jsonl(args.questions)
    if args.sample_ids.strip():
        requested_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
        sample_by_id = {str(item["sample_id"]): item for item in samples}
        missing_ids = [item for item in requested_ids if item not in sample_by_id]
        if missing_ids:
            raise ValueError(f"Unknown sample IDs: {missing_ids}")
        samples = [sample_by_id[item] for item in requested_ids]
    if args.limit > 0:
        samples = samples[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "predictions.jsonl"
    existing = {row["sample_id"]: row for row in read_jsonl(raw_path)} if raw_path.exists() else {}

    for sample_index, sample in enumerate(samples):
        if sample["sample_id"] in existing and existing[sample["sample_id"]].get("success"):
            continue
        last_error = ""
        for attempt in range(args.retries + 1):
            started = time.perf_counter()
            try:
                response = request_answer(args.endpoint, sample, args.timeout)
                record = result_record(sample, response, time.perf_counter() - started)
                append_jsonl(raw_path, record)
                existing[sample["sample_id"]] = record
                print(json.dumps({"sample_id": sample["sample_id"], "status": "ok", "latency": record["latency_seconds"]}, ensure_ascii=False), flush=True)
                break
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    if isinstance(exc, HTTPError) and exc.code == 429:
                        # The backend intentionally protects the 16 GB GPU with
                        # a rolling per-IP limit.  Wait out that window instead
                        # of misclassifying admission control as a model error.
                        time.sleep(20.0 * (attempt + 1))
                    else:
                        time.sleep(3.0 * (attempt + 1))
        else:
            record = {
                "sample_id": sample["sample_id"],
                "task_type": sample["task_type"],
                "capability": sample["capability"],
                "gold_answerable": bool(sample["answerable"]),
                "success": False,
                "error": last_error,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(raw_path, record)
            existing[sample["sample_id"]] = record
            print(json.dumps({"sample_id": sample["sample_id"], "status": "failed", "error": last_error}, ensure_ascii=False), flush=True)

        if sample_index < len(samples) - 1 and args.inter_case_delay > 0:
            time.sleep(args.inter_case_delay)

    ordered = [existing[sample["sample_id"]] for sample in samples if sample["sample_id"] in existing]
    metrics = compute_metrics(ordered, len(samples))
    (args.output_dir / "metrics_deterministic.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "questions": str(args.questions),
                "endpoint": args.endpoint,
                "sequential": True,
                "online_search": False,
                "inter_case_delay_seconds": args.inter_case_delay,
                "case_total": len(samples),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
