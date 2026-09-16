"""Generic, review-only visual transcription for business documents.

This module is intentionally domain-neutral.  It can describe text, tables,
forms and lists from a customer image or a scanned PDF page, but it does not
decide whether a value is financial, sales, HR or project data.  A downstream
agent may use the reviewed candidate only after the customer confirms it.
"""

from __future__ import annotations

import json
from typing import Callable

from backend.document_parsing.ingestion import (
    IntakeResult,
    ValidationIssue,
    VisionDocumentCandidate,
    VisionTableCandidate,
    VisualAsset,
    render_intermediate_for_model,
)


MAX_DOCUMENT_VISION_ASSETS = 4
MAX_TEXT_BLOCKS = 24
MAX_TABLES = 4
MAX_TABLE_COLUMNS = 16
MAX_TABLE_ROWS = 80
MAX_CELL_CHARACTERS = 240


def _json_object(raw: str) -> dict[str, object] | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    candidates = [cleaned]
    if 0 <= start < end:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _texts(value: object, *, maximum: int, character_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    texts = [str(item).strip()[:character_limit] for item in value[:maximum]]
    return [item for item in texts if item]


def _table_candidates(value: object) -> list[VisionTableCandidate]:
    if not isinstance(value, list):
        return []
    tables: list[VisionTableCandidate] = []
    for raw_table in value[:MAX_TABLES]:
        if not isinstance(raw_table, dict):
            continue
        headers = _texts(raw_table.get("headers"), maximum=MAX_TABLE_COLUMNS, character_limit=MAX_CELL_CHARACTERS)
        raw_rows = raw_table.get("rows")
        rows: list[list[str]] = []
        if isinstance(raw_rows, list):
            for raw_row in raw_rows[:MAX_TABLE_ROWS]:
                if not isinstance(raw_row, list):
                    continue
                row = [str(cell).strip()[:MAX_CELL_CHARACTERS] for cell in raw_row[:MAX_TABLE_COLUMNS]]
                if any(row):
                    rows.append(row)
        title = str(raw_table.get("title") or "").strip()[:160] or None
        if title or headers or rows:
            tables.append(VisionTableCandidate(title=title, headers=headers, rows=rows))
    return tables


def build_document_vision_prompt(result: IntakeResult, asset: VisualAsset) -> str:
    """Prompt for a literal, domain-neutral document-layout transcription."""

    payload = {
        "asset": {
            "visual_id": asset.visual_id,
            "source": asset.source.model_dump(mode="json"),
            "metadata": asset.metadata,
        },
        "evidence_json_render": render_intermediate_for_model(result.intermediate, max_characters=4_000),
    }
    return """你是本地企业文档识别器。请只根据当前图片，提取可见的原始版面内容；不要预设这是财务、销售、人事或项目资料。

规则：
1. 只抄录肉眼可见的标题、段落、字段名、表头和表格单元格。看不清就留空，不猜测、不补全、不计算、不换算单位。
2. document_type 只能概括版式，例如 table、form、contract、report、list、invoice、receipt、photo、other；不得使用业务领域分类。
3. 不输出分析结论、风险判断、指标映射或 Excel 操作；也不要把视觉识别出的数字当作已确认事实。
4. text_blocks 保留重要文字块；tables 只保留图中明显的表格。每个单元格保持原始表面文本。
5. 输出是“待客户核对的候选转录”，因此宁可少提取，也不要编造。

只输出合法 JSON，不要 Markdown：
{"document_type":"","title":"","text_blocks":[""],"tables":[{"title":"","headers":[""],"rows":[[""]]}],"confidence":0.0}

并列证据如下：%s""" % json.dumps(payload, ensure_ascii=False)


def parse_document_vision_candidate(raw: str, asset: VisualAsset) -> VisionDocumentCandidate:
    """Validate a model transcription without promoting it to canonical evidence."""

    parsed = _json_object(raw)
    if parsed is None:
        return VisionDocumentCandidate(
            visual_id=asset.visual_id,
            status="failed",
            message="本地视觉模型没有返回可校验的候选转录；未采用任何内容。",
        )
    try:
        confidence = min(0.75, max(0.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return VisionDocumentCandidate(
        visual_id=asset.visual_id,
        status="candidate_ready",
        document_type=str(parsed.get("document_type") or "").strip()[:80] or None,
        title=str(parsed.get("title") or "").strip()[:240] or None,
        text_blocks=_texts(parsed.get("text_blocks"), maximum=MAX_TEXT_BLOCKS, character_limit=800),
        tables=_table_candidates(parsed.get("tables")),
        confidence=confidence,
    )


def augment_with_document_vision_candidates(
    result: IntakeResult,
    infer_document: Callable[[VisualAsset, str], str],
    *,
    asset_ids: set[str] | None = None,
) -> IntakeResult:
    """Run bounded Qwen transcription for OCR fallbacks, never as confirmed facts."""

    document_assets = [
        asset
        for asset in result.intermediate.visual_assets
        if asset.kind == "document_page" and (asset_ids is None or asset.visual_id in asset_ids)
    ]
    ready = [asset for asset in document_assets if asset.delivery_status == "ready_for_vision" and asset.image_bytes]
    candidates: list[VisionDocumentCandidate] = []
    issues: list[ValidationIssue] = []

    for asset in document_assets:
        if asset.delivery_status != "ready_for_vision" or not asset.image_bytes:
            candidates.append(
                VisionDocumentCandidate(
                    visual_id=asset.visual_id,
                    status="unavailable",
                    message="该页面未能渲染为可供本地视觉模型读取的图片；仅保留来源和元数据。",
                )
            )

    for asset in ready[:MAX_DOCUMENT_VISION_ASSETS]:
        try:
            candidate = parse_document_vision_candidate(
                infer_document(asset, build_document_vision_prompt(result, asset)),
                asset,
            )
        except Exception:
            candidate = VisionDocumentCandidate(
                visual_id=asset.visual_id,
                status="failed",
                message="本次本地视觉识别不可用；未采用任何候选文字或表格。",
            )
        candidates.append(candidate)
        if candidate.status != "candidate_ready":
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="document_vision_unavailable",
                    message="部分图片或扫描页未完成候选转录，需改传可编辑文件或稍后重试。",
                    source=asset.source,
                )
            )

    for asset in ready[MAX_DOCUMENT_VISION_ASSETS:]:
        candidates.append(
            VisionDocumentCandidate(
                visual_id=asset.visual_id,
                status="unavailable",
                message=f"单次上传最多识别 {MAX_DOCUMENT_VISION_ASSETS} 个图片或扫描页；其余页面仍保留来源信息。",
            )
        )
    return result.model_copy(
        update={
            "vision_document_candidates": [*result.vision_document_candidates, *candidates],
            "validation": [*result.validation, *issues],
        }
    )
