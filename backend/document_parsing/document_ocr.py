"""Optional deterministic OCR and layout recovery for document-page assets.

PP-StructureV3 is deliberately optional: a missing OCR runtime must never
block the existing Qwen review path.  When present it returns literal text,
coordinates-derived table structure and recognition confidence before a
vision-language model is asked to inspect only weak or incomplete pages.
"""

from __future__ import annotations

import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import Any, Callable

from backend.document_parsing.ingestion import IntakeResult, ValidationIssue, VisionDocumentCandidate, VisionTableCandidate, VisualAsset


MAX_DOCUMENT_OCR_ASSETS = 4
OCR_CONFIDENCE_FOR_QWEN_FALLBACK = 0.86
_STRUCTURED_NUMERIC_LABEL = re.compile(
    r"\b(?:total|subtotal|tax|amount|price|quantity|qty|balance)\b|合计|小计|总计|金额|税额|单价|数量",
    flags=re.IGNORECASE,
)
_NUMERIC_SURFACE = re.compile(r"(?:[$€£¥]|\b\d{1,3}(?:[,，.]\d{3})+(?:[.,]\d+)?\b|\b\d+(?:[.,]\d+)?%\b)")
_pipeline: Any | None = None
_text_pipeline: Any | None = None
_text_pipeline_error: str | None = None
# PaddleX defaults to the user's home directory, which is not a reliable
# writable location for a desktop service.  Its native Windows runtime also
# expects an ASCII-safe model path, so use the OS temp root by default.  A
# stable ASCII-only path can be supplied via FINANCE_OCR_CACHE_HOME.
PADDLE_CACHE_HOME = Path(os.getenv("FINANCE_OCR_CACHE_HOME", Path(gettempdir()) / "finance-paddlex-cache"))
TEXT_INDEX_CACHE_HOME = Path(os.getenv("FINANCE_OCR_TEXT_CACHE_HOME", Path(gettempdir()) / "finance-text-index-cache"))
TEXT_INDEX_MODEL_NAMES = (
    os.getenv('FINANCE_OCR_TEXT_DETECTION_MODEL', 'PP-OCRv5_mobile_det'),
    os.getenv('FINANCE_OCR_TEXT_RECOGNITION_MODEL', 'PP-OCRv5_mobile_rec'),
)


class OcrRuntimeUnavailable(RuntimeError):
    """The optional local PP-Structure runtime is not installed or configured."""


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self.header_rows: set[int] = set()
        self._header_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            self._header_cell = tag == "th"

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                if self._header_cell:
                    self.header_rows.add(len(self.rows))
                self.rows.append(self._row)
            self._row = None
            self._header_cell = False


def _html_table(value: str) -> VisionTableCandidate | None:
    if "<table" not in value.lower():
        return None
    parser = _HtmlTableParser()
    try:
        parser.feed(value)
    except Exception:
        return None
    if not parser.rows:
        return None
    headers = parser.rows[0] if 0 in parser.header_rows else []
    rows = parser.rows[1:] if headers else parser.rows
    return VisionTableCandidate(headers=headers[:16], rows=[row[:16] for row in rows[:80]])


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    for attribute in ("to_dict", "json", "to_json"):
        member = getattr(value, attribute, None)
        if member is None:
            continue
        try:
            converted = member() if callable(member) else member
        except Exception:
            continue
        if isinstance(converted, str):
            try:
                converted = json.loads(converted)
            except json.JSONDecodeError:
                continue
        if isinstance(converted, dict):
            return converted
    return None


def _strings_for_keys(value: Any, keys: set[str], *, maximum: int = 24) -> list[str]:
    found: list[str] = []

    def visit(item: Any, key: str | None = None) -> None:
        if len(found) >= maximum:
            return
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key).lower())
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif key in keys and isinstance(item, (str, int, float)):
            text = str(item).strip()
            if text and text not in found:
                found.append(text[:800])

    visit(value)
    return found


def _scores(value: Any) -> list[float]:
    values: list[float] = []

    def visit(item: Any, key: str | None = None) -> None:
        if len(values) >= 500:
            return
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key).lower())
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif key in {"rec_score", "rec_scores", "score", "scores", "confidence"}:
            try:
                number = float(item)
            except (TypeError, ValueError):
                return
            if 0.0 <= number <= 1.0:
                values.append(number)

    visit(value)
    return values


def _table_candidates(value: Any) -> list[VisionTableCandidate]:
    found: list[VisionTableCandidate] = []

    def visit(item: Any, key: str | None = None) -> None:
        if len(found) >= 4:
            return
        if isinstance(item, dict):
            raw_headers = item.get("headers")
            raw_rows = item.get("rows")
            if isinstance(raw_headers, list) or isinstance(raw_rows, list):
                headers = [str(cell).strip()[:240] for cell in raw_headers[:16]] if isinstance(raw_headers, list) else []
                rows = [
                    [str(cell).strip()[:240] for cell in row[:16]]
                    for row in raw_rows[:80]
                    if isinstance(row, list) and any(str(cell).strip() for cell in row)
                ] if isinstance(raw_rows, list) else []
                if headers or rows:
                    found.append(VisionTableCandidate(title=str(item.get("title") or "").strip()[:160] or None, headers=headers, rows=rows))
            for child_key, child in item.items():
                child_key_text = str(child_key).lower()
                if isinstance(child, str) and ("html" in child_key_text or "table" in child_key_text):
                    table = _html_table(child)
                    if table is not None:
                        found.append(table)
                else:
                    visit(child, child_key_text)
        elif isinstance(item, list):
            for child in item:
                visit(child, key)

    visit(value)
    return found[:4]


def _normalise_ppstructure_output(raw_results: list[Any]) -> dict[str, Any]:
    payloads = [_as_mapping(result) for result in raw_results]
    payloads = [payload for payload in payloads if payload is not None]
    text_blocks: list[str] = []
    tables: list[VisionTableCandidate] = []
    scores: list[float] = []
    for payload in payloads:
        for text in _strings_for_keys(payload, {"rec_text", "rec_texts", "text", "texts", "content", "ocr_text"}):
            if text not in text_blocks:
                text_blocks.append(text)
        tables.extend(_table_candidates(payload))
        scores.extend(_scores(payload))
    confidence = sum(scores) / len(scores) if scores else 0.0
    return {
        "document_type": "table" if tables else "report" if text_blocks else None,
        "text_blocks": text_blocks[:24],
        "tables": tables[:4],
        "confidence": min(0.98, max(0.0, confidence)),
    }


def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    PADDLE_CACHE_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLE_CACHE_HOME))
    try:
        from paddleocr import PPStructureV3
    except ImportError as exc:
        raise OcrRuntimeUnavailable("PP-StructureV3 is not installed in the local OCR runtime.") from exc
    device = os.getenv("FINANCE_OCR_DEVICE", "cpu")
    # Paddle 3.3's oneDNN CPU path currently fails on part of PP-StructureV3
    # on Windows.  Keep it off unless the deployment explicitly opts in.
    enable_mkldnn = os.getenv("FINANCE_OCR_ENABLE_MKLDNN", "false").lower() in {"1", "true", "yes"}
    try:
        _pipeline = PPStructureV3(device=device, enable_mkldnn=enable_mkldnn)
    except TypeError:
        _pipeline = PPStructureV3()
    return _pipeline


def run_pp_structure_v3(asset: VisualAsset) -> dict[str, Any]:
    """Use PP-StructureV3 for literal OCR, layout and table reconstruction."""

    if not asset.image_bytes:
        raise OcrRuntimeUnavailable("The document page has no image bytes for OCR.")
    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }.get(asset.media_type)
    if suffix is None:
        raise OcrRuntimeUnavailable("The page image format is not supported by the local OCR adapter.")
    path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="finance-ppstructure-", suffix=suffix, delete=False) as file:
            file.write(asset.image_bytes)
            path = Path(file.name)
        pipeline = _get_pipeline()
        results = list(pipeline.predict(str(path)))
        return _normalise_ppstructure_output(results)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _get_text_pipeline() -> Any:
    """CPU-only cached text index; no layout/table/LLM inference here."""
    global _text_pipeline, _text_pipeline_error
    if _text_pipeline is not None:
        return _text_pipeline
    if _text_pipeline_error is not None:
        raise OcrRuntimeUnavailable(_text_pipeline_error)
    TEXT_INDEX_CACHE_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(TEXT_INDEX_CACHE_HOME))
    try:
        from paddleocr import PaddleOCR
        options = dict(device="cpu", enable_mkldnn=False,
                       use_doc_orientation_classify=False,
                       use_doc_unwarping=False, use_textline_orientation=False)
        # Reuse the already downloaded local models, without an implicit
        # network download during customer upload.
        for role, name in zip(("detection", "recognition"), TEXT_INDEX_MODEL_NAMES):
            directory = TEXT_INDEX_CACHE_HOME / "official_models" / name
            if not (directory / 'inference.yml').is_file():
                raise OcrRuntimeUnavailable("Local page text-index model cache is incomplete.")
            options[f"text_{role}_model_dir"] = str(directory)
            options[f"text_{role}_model_name"] = name
        _text_pipeline = PaddleOCR(**options)
    except Exception as exc:
        _text_pipeline_error = "Local CPU page text index could not initialize."
        raise OcrRuntimeUnavailable(_text_pipeline_error) from exc
    return _text_pipeline


def pack_ocr_index_lines(lines: list[str], *, maximum: int = 24,
                         block_characters: int = 800) -> tuple[list[str], bool]:
    """Group literal lines, retaining line boundaries; disclose overflow."""
    blocks: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        # Oversized lines are split, not silently prefix-truncated.
        for start in range(0, len(line), block_characters):
            piece = line[start:start + block_characters]
            if blocks and len(blocks[-1]) + len(piece) + 1 <= block_characters:
                blocks[-1] += "\n" + piece
            else:
                blocks.append(piece)
    return blocks[:maximum], len(blocks) > maximum


def run_page_text_index(asset: VisualAsset) -> dict[str, Any]:
    """Literal search hints only; selected pages still require visual reading."""
    if not asset.image_bytes:
        raise OcrRuntimeUnavailable("No page image bytes are available.")
    path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="page-text-index-", suffix=".png", delete=False) as file:
            file.write(asset.image_bytes)
            path = Path(file.name)
        payloads = [_as_mapping(item) for item in _get_text_pipeline().predict(str(path))]
        lines: list[str] = []
        scores: list[float] = []
        for payload in payloads:
            if payload is None:
                continue
            nested = payload.get("res", payload)
            lines.extend(str(line) for line in nested.get("rec_texts", []) if str(line).strip())
            scores.extend(_scores(nested))
        blocks, truncated = pack_ocr_index_lines(lines)
        return {"text_blocks": blocks, "tables": [], "document_type": "report",
                "confidence": sum(scores) / len(scores) if scores else 0.0,
                "index_truncated": truncated,
                "recognizer": "paddle_text_index"}
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _candidate_from_ocr(asset: VisualAsset, payload: dict[str, Any]) -> VisionDocumentCandidate:
    raw_tables = payload.get("tables")
    tables = raw_tables if isinstance(raw_tables, list) else []
    confidence = payload.get("confidence", 0.0)
    try:
        numeric_confidence = min(0.98, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        numeric_confidence = 0.0
    return VisionDocumentCandidate(
        visual_id=asset.visual_id,
        status="candidate_ready",
        document_type=str(payload.get("document_type") or "").strip()[:80] or None,
        title=str(payload.get("title") or "").strip()[:240] or None,
        text_blocks=[str(value).strip()[:800] for value in payload.get("text_blocks", [])[:24] if str(value).strip()],
        tables=[table for table in tables[:4] if isinstance(table, VisionTableCandidate)],
        confidence=numeric_confidence,
        recognizer=str(payload.get("recognizer") or "pp_structure_v3"),
        message=("Literal text-index candidate; not verified layout or a training label."
                 + (" Text-index overflow: some lines are pending." if payload.get("index_truncated") else ""))
                if payload.get("recognizer") == "paddle_text_index"
                else "PP-StructureV3 OCR/layout/table candidate; requires customer confirmation.",
    )


def augment_with_document_ocr_candidates(
    result: IntakeResult,
    infer_document: Callable[[VisualAsset], dict[str, Any]] = run_pp_structure_v3,
    *, deadline: float | None = None,
) -> IntakeResult:
    """Attach bounded PP-Structure candidates while keeping all output review-only."""

    assets = [asset for asset in result.intermediate.visual_assets if asset.kind == "document_page"]
    recognizer = "paddle_text_index" if infer_document is run_page_text_index else "pp_structure_v3"
    if recognizer == 'paddle_text_index' and hasattr(result, 'pdf'):
        # Reliable native text is already indexed. OCR only unavailable or
        # suspect text layers, not every photograph/chart in a native PDF.
        required = {page.page_number for page in result.pdf.page_inspections
                    if page.parse_route != 'direct_text'}
        assets = [asset for asset in assets if asset.source.page_number in required]
    candidates: list[VisionDocumentCandidate] = []
    issues: list[ValidationIssue] = []
    for asset in assets[:MAX_DOCUMENT_OCR_ASSETS]:
        if deadline is not None and time.monotonic() >= deadline:
            candidates.append(VisionDocumentCandidate(visual_id=asset.visual_id,
                status='unavailable',recognizer=recognizer,
                message='OCR indexing time budget exhausted; original page retained, indexing pending.'))
            continue
        if asset.delivery_status != "ready_for_vision" or not asset.image_bytes:
            candidates.append(VisionDocumentCandidate(visual_id=asset.visual_id, status="unavailable", recognizer=recognizer, message="This page could not be prepared for local OCR."))
            continue
        try:
            candidates.append(_candidate_from_ocr(asset, infer_document(asset)))
        except OcrRuntimeUnavailable as exc:
            candidates.append(VisionDocumentCandidate(visual_id=asset.visual_id, status="unavailable", recognizer=recognizer, message=str(exc)[:300]))
        except Exception:
            candidates.append(VisionDocumentCandidate(visual_id=asset.visual_id, status="failed", recognizer=recognizer, message="Local OCR did not return a usable candidate."))
    for asset in assets[MAX_DOCUMENT_OCR_ASSETS:]:
        candidates.append(VisionDocumentCandidate(visual_id=asset.visual_id, status="unavailable", recognizer="pp_structure_v3", message=f"One request processes at most {MAX_DOCUMENT_OCR_ASSETS} document pages with PP-StructureV3."))

    for candidate in candidates:
        if candidate.status != "candidate_ready":
            code = ('document_ocr_index_budget_exhausted' if 'budget exhausted' in str(candidate.message)
                    else 'document_ocr_prediction_failed' if candidate.status == 'failed'
                    else 'document_ocr_unavailable')
            issues.append(ValidationIssue(severity="warning", code=code, message="专用 OCR 未完成该页识别，已保留页面并交由视觉模型回退处理。", source=next((asset.source for asset in assets if asset.visual_id == candidate.visual_id), None)))
    return result.model_copy(update={"vision_document_candidates": [*result.vision_document_candidates, *candidates], "validation": [*result.validation, *issues]})


def visual_fallback_asset_ids(result: IntakeResult) -> set[str]:
    """Return pages whose OCR result is weak enough to warrant Qwen review."""

    ocr_by_asset = {candidate.visual_id: candidate for candidate in result.vision_document_candidates if candidate.recognizer == "pp_structure_v3"}
    fallback: set[str] = set()
    for asset in result.intermediate.visual_assets:
        if asset.kind != "document_page":
            continue
        candidate = ocr_by_asset.get(asset.visual_id)
        if candidate is None or candidate.status != "candidate_ready" or candidate.confidence < OCR_CONFIDENCE_FOR_QWEN_FALLBACK or not candidate.text_blocks:
            fallback.add(asset.visual_id)
            continue
        # A high character-recognition score does not prove that document
        # structure was recovered.  Receipts, invoices and other numeric
        # layouts frequently arrive as loose OCR lines even though the user
        # needs line items.  Escalate only when general structural and numeric
        # signals coexist but PP-Structure returned no table; plain narrative
        # pages do not take this expensive route.
        joined_text = "\n".join(candidate.text_blocks)
        numeric_lines = sum(bool(_NUMERIC_SURFACE.search(text)) for text in candidate.text_blocks)
        if not candidate.tables and len(candidate.text_blocks) >= 5 and numeric_lines >= 2 and _STRUCTURED_NUMERIC_LABEL.search(joined_text):
            fallback.add(asset.visual_id)
    return fallback
