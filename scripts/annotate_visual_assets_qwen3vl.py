"""Add local Qwen3-VL semantic tags to MinerU-extracted original image assets.

Inputs are the JSONL assets emitted by build_mineru_asset_manifest.py.  The
model receives only an extracted image crop plus its page-local source text;
it never receives a whole PDF and never uploads an image.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen3-VL-8B-Instruct"
REPAIRED_CROP_DIR = ROOT / "data" / "sales" / "processed" / "repaired_visual_crops"


def resolve_local_image_asset(asset: dict[str, Any]) -> dict[str, Any] | None:
    """Return a usable local crop, repairing a MinerU directory placeholder.

    MinerU occasionally emits an ``images`` directory rather than an image
    filename for a table/image asset.  Rendering the source PDF's recorded
    page/bounding box keeps the repair private, deterministic and traceable.
    """

    image_path = Path(str(asset.get("image_path") or ""))
    if image_path.is_file():
        return asset

    source_pdf = Path(str(asset.get("source_pdf") or ""))
    source_page = asset.get("source_page")
    bbox = asset.get("bbox")
    if not source_pdf.is_file() or not isinstance(source_page, int) or not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        import fitz

        coordinates = [float(value) for value in bbox]
        clip = fitz.Rect(*coordinates)
        if clip.is_empty or clip.width <= 0 or clip.height <= 0:
            return None
        REPAIRED_CROP_DIR.mkdir(parents=True, exist_ok=True)
        repaired_path = REPAIRED_CROP_DIR / f"{asset.get('asset_id')}.png"
        if not repaired_path.is_file():
            with fitz.open(source_pdf) as pdf:
                page = pdf.load_page(source_page - 1)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
                pixmap.save(repaired_path)
        if not repaired_path.is_file():
            return None
        repaired = dict(asset)
        repaired["image_path"] = str(repaired_path)
        repaired["image_path_origin"] = "locally_rendered_pdf_bbox_crop"
        return repaired
    except Exception:
        return None


def parse_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
    start, end = candidate.find("{"), candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {
        "image_kind": "other",
        "visual_description": "模型未能生成可解析的结构化描述。",
        "visible_components": [],
        "visually_supported_relationships": [],
        "recommended_search_terms": [],
        "source_context_consistency": "insufficient",
        "customer_image_caption": None,
        "review_notes": ["模型输出无法解析，需要人工复核。"],
        "confidence": "low",
    }


def prompt_for_asset(asset: dict[str, Any]) -> str:
    return f"""你在为建筑外墙建材知识库标注一张从 PDF 原页自动裁出的图片。

已知来源信息（仅用于核对，不得自行补全技术参数）：
- 文件：{asset.get('document_name')}
- 页码：{asset.get('source_page')}
- 图片脚注：{asset.get('caption') or '无'}
- 所在章节：{asset.get('section_heading') or '无'}
- 临近原文：{asset.get('nearby_text') or '无'}

仅根据图片可见内容和上述原文，输出一个 JSON 对象：
{{
  "image_kind": "component_photo | product_photo | project_photo | construction_detail_drawing | process_diagram | other",
  "visual_description": "一句简洁中文描述",
  "visible_components": ["图中可见的部件或对象"],
  "visually_supported_relationships": ["只描述图中明确可见的连接、位置或层次关系"],
  "recommended_search_terms": ["适合检索的中文短语"],
  "source_context_consistency": "consistent | inconsistent | insufficient",
  "customer_image_caption": "适合在客户页面展示的图片标题；无把握则为 null",
  "review_notes": ["需要人工复核的事项"],
  "confidence": "low | medium | high"
}}

严格规则：
1. 不要编造尺寸、强度、防火等级、施工参数、项目面积或工程结论。
2. 组件照片、项目照片、单张节点图通常没有施工顺序，不要生成步骤。
3. 若原文与图片无法互相印证，标记 insufficient 或 inconsistent。
4. customer_image_caption 只能基于图中或提供的原文，不得添加宣传性表述。
5. 所有结果均为候选标注，review_notes 不能为空。"""


def load_model(model_dir: Path):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL annotation requires CUDA.")
    if not model_dir.exists():
        raise FileNotFoundError(f"Qwen3-VL model directory not found: {model_dir}")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
    )
    model.eval()
    return model, processor


def annotate_asset(model, processor, asset: dict[str, Any], max_new_tokens: int) -> tuple[dict[str, Any], str]:
    import torch
    from qwen_vl_utils import process_vision_info

    image_path = Path(str(asset["image_path"]))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path.resolve()),
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1024 * 28 * 28,
                },
                {"type": "text", "text": prompt_for_asset(asset)},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    annotation = parse_json(raw)
    annotation["review_notes"] = annotation.get("review_notes") or ["需人工复核图片与来源文字是否一致。"]
    return annotation, raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate MinerU original image assets with local Qwen3-VL.")
    parser.add_argument("assets", type=Path, help="Path to visual_assets.jsonl.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Maximum assets to process; 0 means all.")
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed rows in --output and process only asset IDs that are still missing.",
    )
    args = parser.parse_args()

    # PowerShell 5.1 may prepend a UTF-8 BOM when it creates the combined
    # JSONL manifest. utf-8-sig accepts both BOM and non-BOM files.
    raw_assets = [json.loads(line) for line in args.assets.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    assets: list[dict[str, Any]] = []
    unavailable_assets: list[str] = []
    for raw_asset in raw_assets:
        resolved = resolve_local_image_asset(raw_asset)
        if resolved is None:
            unavailable_assets.append(str(raw_asset.get("asset_id") or "unknown"))
        else:
            assets.append(resolved)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_asset_ids: set[str] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                completed_asset_ids.add(str(json.loads(line).get("asset_id") or ""))
            except json.JSONDecodeError:
                # A partially written final line must not block resuming the
                # batch. The corresponding asset will simply be regenerated.
                continue
        completed_asset_ids.discard("")
        assets = [asset for asset in assets if str(asset.get("asset_id") or "") not in completed_asset_ids]
    if args.limit > 0:
        assets = assets[: args.limit]
    if not assets:
        print("No missing image assets to annotate.")
        return 0
    if unavailable_assets:
        print(f"Skipped {len(unavailable_assets)} asset(s) without a usable local PDF crop: {', '.join(unavailable_assets)}")

    model, processor = load_model(args.model)
    try:
        mode = "a" if args.resume and args.output.exists() else "w"
        with args.output.open(mode, encoding="utf-8", newline="\n") as handle:
            for index, asset in enumerate(assets, start=1):
                annotation, raw = annotate_asset(model, processor, asset, args.max_new_tokens)
                result = {
                    **asset,
                    "semantic_annotation_status": "candidate_ready",
                    "visual_semantics": annotation,
                    "visual_model": str(args.model),
                    "raw_visual_model_response": raw,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{index}/{len(assets)}] annotated {asset['asset_id']}")
    finally:
        del model
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
