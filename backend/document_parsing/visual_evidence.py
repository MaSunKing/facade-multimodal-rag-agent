"""Guarded local-vision interpretation for Excel visual evidence.

Visual observations are intentionally supplementary: a chart preview may
describe a trend or a visible label, but it must never create a financial fact
or override a number that came from an original worksheet cell.
"""

from __future__ import annotations

import json
from typing import Callable

from backend.document_parsing.ingestion import IntakeResult, ValidationIssue, VisualAsset, VisualObservation, render_intermediate_for_model


MAX_VISUAL_MODEL_ASSETS = 4


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


def build_visual_observation_prompt(result: IntakeResult, asset: VisualAsset) -> str:
    """Give Qwen3-VL a visual object beside compact canonical evidence."""

    payload = {
        "asset": {
            "visual_id": asset.visual_id,
            "kind": asset.kind,
            "source": asset.source.model_dump(mode="json"),
            "metadata": asset.metadata,
        },
        "evidence_json_render": render_intermediate_for_model(result.intermediate, max_characters=8_000),
    }
    return """你正在核对一个 Excel 中的视觉对象。图片与 Evidence JSON 同时提供：

1. 图片只可用于识别图表类型、标题、图例、明显趋势、异常方向或说明性文字。
2. 原始单元格、公式和 Evidence JSON 是金额、期间和财务事实的唯一依据；绝不可从图片像素估读或生成精确金额。
3. 不要把图片观察转换为收入、成本、利润等财务事实，也不要覆盖表格中的数据。
4. 若图表内容模糊、图例不清或与表格不一致，宁可输出空 findings 并说明需要人工确认。

只输出合法 JSON，不要 Markdown：
{"chart_type":"图表类型或空字符串","title":"标题或空字符串","findings":["最多6条非数值观察"],"confidence":0.0}

并列证据如下：
%s""" % json.dumps(payload, ensure_ascii=False)


def parse_visual_observation(raw: str, asset: VisualAsset) -> VisualObservation:
    """Validate a vision response and keep it strictly non-numeric/supplementary."""

    parsed = _json_object(raw)
    if parsed is None:
        return VisualObservation(
            visual_id=asset.visual_id,
            status="failed",
            message="视觉模型未返回可校验 JSON，未采用任何视觉结论。",
        )
    raw_findings = parsed.get("findings")
    findings = [str(item).strip()[:240] for item in raw_findings] if isinstance(raw_findings, list) else []
    findings = [item for item in findings if item][:6]
    raw_confidence = parsed.get("confidence")
    try:
        confidence = min(0.85, max(0.0, float(raw_confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    chart_type = str(parsed.get("chart_type") or "").strip()[:80] or None
    title = str(parsed.get("title") or "").strip()[:240] or None
    return VisualObservation(
        visual_id=asset.visual_id,
        status="candidate_ready",
        chart_type=chart_type,
        title=title,
        findings=findings,
        confidence=confidence,
        requires_confirmation=True,
    )


def augment_with_visual_observations(
    result: IntakeResult,
    infer_visual: Callable[[VisualAsset, str], str],
) -> IntakeResult:
    """Run a bounded local visual pass without affecting financial facts."""

    candidates = [
        asset
        for asset in result.intermediate.visual_assets
        if asset.kind in {"embedded_image", "chart_preview", "unrendered_chart"}
        and asset.delivery_status == "ready_for_vision"
        and asset.image_bytes
    ]
    candidates.sort(key=lambda asset: 0 if asset.kind == "chart_preview" else 1)
    observations: list[VisualObservation] = []
    issues: list[ValidationIssue] = []
    for asset in result.intermediate.visual_assets:
        if asset.kind not in {"embedded_image", "chart_preview", "unrendered_chart"}:
            continue
        if asset.delivery_status != "ready_for_vision" or not asset.image_bytes:
            observations.append(
                VisualObservation(
                    visual_id=asset.visual_id,
                    status="unavailable",
                    message="当前运行环境无法将此视觉对象渲染为图片；已保留其来源和结构元数据。",
                )
            )
    for asset in candidates[:MAX_VISUAL_MODEL_ASSETS]:
        try:
            observation = parse_visual_observation(infer_visual(asset, build_visual_observation_prompt(result, asset)), asset)
        except Exception:
            observation = VisualObservation(
                visual_id=asset.visual_id,
                status="failed",
                message="本次视觉分析不可用；未采用任何视觉结论。",
            )
        observations.append(observation)
        if observation.status != "candidate_ready":
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="visual_evidence_unavailable",
                    message="部分 Excel 图表未完成视觉理解；原始表格和公式仍已保留。",
                    source=asset.source,
                )
            )
    for asset in candidates[MAX_VISUAL_MODEL_ASSETS:]:
        observations.append(
            VisualObservation(
                visual_id=asset.visual_id,
                status="unavailable",
                message=f"一次上传最多分析 {MAX_VISUAL_MODEL_ASSETS} 个视觉对象；其余对象已保留元数据。",
            )
        )
    return result.model_copy(update={"visual_observations": observations, "validation": [*result.validation, *issues]})
