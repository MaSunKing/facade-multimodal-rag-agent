"""Apply local knowledge-domain metadata to prepared RAG evidence.

This script never changes raw PDF files, MinerU output, original image assets or
the existing cleaned evidence.  It creates tagged copies used by the retrieval
index, so every tag can be regenerated or audited independently.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READY_DIR = ROOT / "data" / "sales" / "processed" / "rag_ready"
DEFAULT_TAXONOMY = ROOT / "data" / "sales" / "config" / "knowledge_taxonomy_v1.json"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def taxonomy_for_document(document_name: Any, documents: dict[str, Any]) -> dict[str, Any]:
    name = str(document_name or "")
    value = documents.get(name)
    if isinstance(value, dict):
        return value
    return {
        "knowledge_domain": "unclassified",
        "document_category": "unclassified",
        "source_authority": "unknown",
        "customer_fact_policy": "review_before_use",
    }


def text_content_labels(text: str, heading: str, category: str) -> list[str]:
    corpus = f"{heading}\n{text}"
    labels: list[str] = []
    if category == "enterprise_product_catalogue":
        labels.append("product_information")
        if "应用案例" in corpus or "项目类型" in corpus or "建成时间" in corpus:
            labels.append("project_case_reference")
        if any(term in corpus for term in ("规格", "厚度", "尺寸", "重量", "颜色", "饰面")):
            labels.append("product_parameter_or_finish")
        if any(term in corpus for term in ("寿命", "不褪色", "不开裂", "降低", "优势")):
            labels.append("enterprise_claim")
    elif category == "enterprise_construction_method":
        labels.append("construction_method")
        if any(term in corpus for term in ("施工", "安装", "粘贴", "锚固", "挂件", "开槽", "放线", "龙骨")):
            labels.append("construction_step")
        if any(term in corpus for term in ("基层", "平整度", "条件", "适用", "温度")):
            labels.append("installation_condition")
        if any(term in corpus for term in ("验收", "检查", "允许偏差", "质量")):
            labels.append("quality_or_acceptance")
    elif category in {"engineering_standard", "material_standard"}:
        labels.append("standard_requirement")
        if any(term in corpus for term in ("材料", "性能", "指标", "等级")):
            labels.append("material_requirement")
        if any(term in corpus for term in ("设计", "构造", "计算")):
            labels.append("design_requirement")
        if any(term in corpus for term in ("验收", "检查", "检验")):
            labels.append("acceptance_requirement")
    elif category == "application_atlas":
        labels.append("node_detail")
        if any(term in corpus for term in ("门窗", "洞口", "窗口")):
            labels.append("window_opening")
        if "阴角" in corpus:
            labels.append("internal_corner")
        if "阳角" in corpus:
            labels.append("external_corner")
        if "勒脚" in corpus:
            labels.append("plinth")
        if "女儿墙" in corpus:
            labels.append("parapet")
    elif category == "internal_sales_playbook":
        labels.append("internal_sales_playbook")
        if any(term in corpus for term in ("什么是保温装饰一体板", "无机饰面", "装饰板", "保温装饰一体板", "颜色", "尺寸", "安装方式")):
            labels.append("supplementary_product_information")
        if any(term in corpus for term in ("助播", "直播间", "钩子", "小结", "总结", "客户提问", "疑问")):
            labels.append("sales_enablement_or_objection_handling")
        if any(term in corpus for term in ("对比", "区别", "天然石材", "有机仿石涂料", "陶瓷薄板")):
            labels.append("comparison_talking_point")
        if any(term in corpus for term in ("寿命", "成本", "造价", "强度", "安全性", "供货能力", "负责安装")):
            labels.append("claim_or_commercial_talking_point")
    return unique(labels) or ["general_reference"]


def sales_playbook_policy(text: str, heading: str) -> dict[str, Any]:
    """Keep sales enablement separate from customer-facing factual evidence.

    This is intentionally conservative.  A neutral product definition may
    supplement the product catalogue; performance, cost, comparison and
    commercial promises stay preserved locally but are not added to the
    customer factual index until a higher-authority source is linked.
    """

    corpus = f"{heading}\n{text}"
    internal_markers = ("助播", "直播间", "钩子", "打在公屏", "领取样品", "回答客户提问")
    claim_markers = (
        "无差别",
        "与建筑体同寿命",
        "50年",
        "60%",
        "3倍",
        "1.5倍",
        "安全性更高",
        "成本低于",
        "确保",
        "不会剥落",
        "不褪色",
        "100米",
        "供货能力",
        "负责安装",
        "抗折强度",
        "单点锚固力",
        "坠落冲击力",
        "抗冲击",
        "造价",
    )
    product_markers = ("什么是保温装饰一体板", "无机饰面", "装饰板", "保温装饰一体板", "颜色", "尺寸", "安装方式")
    quantified_claim = re.search(r"\d+(?:\.\d+)?\s*(?:%|倍|年|米|MPa|KN|J)", corpus, flags=re.IGNORECASE)

    if any(marker in corpus for marker in claim_markers) or quantified_claim:
        return {
            "sales_playbook_use": "internal_claim_needs_authoritative_evidence",
            "index_eligible": False,
            "fact_eligible": False,
            "customer_shareable": False,
        }
    if any(marker in corpus for marker in internal_markers):
        return {
            "sales_playbook_use": "internal_sales_enablement_only",
            "index_eligible": False,
            "fact_eligible": False,
            "customer_shareable": False,
        }
    if any(marker in corpus for marker in product_markers):
        return {
            "sales_playbook_use": "supplementary_product_information",
            "index_eligible": True,
            "fact_eligible": True,
            "customer_shareable": True,
        }
    return {
        "sales_playbook_use": "internal_context_requires_review",
        "index_eligible": False,
        "fact_eligible": False,
        "customer_shareable": False,
    }


def visual_labels(record: dict[str, Any], category: str) -> list[str]:
    corpus = "\n".join(
        str(record.get(key) or "")
        for key in ("customer_title", "caption", "nearby_text", "section_heading")
    )
    labels: list[str] = []
    image_kind = str(record.get("effective_image_kind") or "")
    if image_kind == "table_or_parameter_sheet":
        labels.append("table_or_parameter_sheet")
    if category == "enterprise_product_catalogue":
        labels.append("product_or_case_visual")
        if "应用案例" in corpus or "项目" in corpus:
            labels.append("project_case_image")
        if any(term in corpus for term in ("颜色", "饰面", "样板", "单板")):
            labels.append("product_finish")
    elif category == "enterprise_construction_method":
        labels.append("construction_visual")
        if any(term in corpus for term in ("步骤", "安装", "施工", "粘贴", "锚固", "挂件", "开槽")):
            labels.append("installation_step_or_detail")
    elif category == "application_atlas":
        labels.append("node_detail")
    elif category in {"engineering_standard", "material_standard"}:
        labels.append("standard_table_or_figure")
    return unique(labels) or ["unclassified_visual"]


def source_taxonomy(source_ref: dict[str, Any], documents: dict[str, Any]) -> dict[str, Any]:
    document_name = source_ref.get("document_name")
    info = taxonomy_for_document(document_name, documents)
    # Every DOCX entering this pipeline comes from data/sales/raw/sales_playbooks.
    # A future playbook may not yet have an explicit document-name entry in
    # the taxonomy.  Treat it conservatively as internal sales material until
    # someone adds an audited mapping instead of accidentally exposing it as
    # an unclassified, customer-shareable source.
    if (
        str(info.get("document_category") or "") == "unclassified"
        and str(source_ref.get("source_pdf") or "").lower().endswith(".docx")
    ):
        info = {
            "knowledge_domain": "06_sales_playbook",
            "document_category": "internal_sales_playbook",
            "source_authority": "internal_sales_enablement",
            "customer_fact_policy": "supplementary_product_reference_with_validation",
        }
    return {
        "document_name": document_name,
        "knowledge_domain": info["knowledge_domain"],
        "document_category": info["document_category"],
        "source_authority": info["source_authority"],
        "customer_fact_policy": info["customer_fact_policy"],
    }


def tag_text(records: Iterable[dict[str, Any]], documents: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        refs = record.get("source_refs") if isinstance(record.get("source_refs"), list) else []
        source_tags = [source_taxonomy(ref, documents) for ref in refs if isinstance(ref, dict)]
        labels: list[str] = []
        for ref, tag in zip(refs, source_tags):
            labels.extend(text_content_labels(
                str(record.get("text") or ""),
                str(ref.get("section_heading") or ""),
                str(tag["document_category"]),
            ))
        categories = unique(tag["document_category"] for tag in source_tags)
        # A duplicated text block may also exist in an authoritative document;
        # in that case retain the stronger source instead of applying the
        # internal-playbook restriction to the shared text.
        sales_only = categories == ["internal_sales_playbook"]
        sales_policy = (
            sales_playbook_policy(str(record.get("text") or ""), str(refs[0].get("section_heading") or ""))
            if sales_only and refs
            else {
                "sales_playbook_use": None,
                "index_eligible": bool(record.get("index_eligible")),
                "fact_eligible": bool(record.get("fact_eligible")),
                "customer_shareable": bool(record.get("customer_shareable", True)),
            }
        )
        output.append({
            **record,
            "taxonomy_version": "1.1",
            "knowledge_domains": unique(tag["knowledge_domain"] for tag in source_tags),
            "document_categories": categories,
            "content_labels": unique(labels),
            "source_taxonomy": source_tags,
            "index_eligible": bool(record.get("index_eligible")) and bool(sales_policy["index_eligible"]),
            "fact_eligible": bool(record.get("fact_eligible")) and bool(sales_policy["fact_eligible"]),
            "customer_shareable": bool(sales_policy["customer_shareable"]),
            "sales_playbook_use": sales_policy["sales_playbook_use"],
        })
    return output


def tag_visual(records: Iterable[dict[str, Any]], documents: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        tag = taxonomy_for_document(record.get("document_name"), documents)
        output.append({
            **record,
            "taxonomy_version": "1.1",
            "knowledge_domains": [tag["knowledge_domain"]],
            "document_categories": [tag["document_category"]],
            "visual_labels": visual_labels(record, str(tag["document_category"])),
            "visual_evidence_role": "supplemental_original_visual",
            "source_authority": tag["source_authority"],
        })
    return output


def tag_cases(records: Iterable[dict[str, Any]], documents: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        tag = taxonomy_for_document(record.get("source_document"), documents)
        output.append({
            **record,
            "taxonomy_version": "1.1",
            "knowledge_domain": "05_project_case",
            "knowledge_domains": ["05_project_case", tag["knowledge_domain"]],
            "document_category": tag["document_category"],
            "content_labels": ["project_case_reference"],
            "source_authority": tag["source_authority"],
            "case_evidence_role": "enterprise_catalogue_case_reference",
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Create tagged local RAG evidence without changing clean evidence.")
    parser.add_argument("--ready-dir", type=Path, default=DEFAULT_READY_DIR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    args = parser.parse_args()
    ready_dir = args.ready_dir.resolve()
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    documents = taxonomy.get("documents") if isinstance(taxonomy.get("documents"), dict) else {}

    text = tag_text(read_jsonl(ready_dir / "text_evidence_clean.jsonl"), documents)
    visual = tag_visual(read_jsonl(ready_dir / "visual_assets_clean.jsonl"), documents)
    cases = tag_cases(read_jsonl(ready_dir / "project_cases.jsonl"), documents)
    write_jsonl(ready_dir / "text_evidence_tagged.jsonl", text)
    write_jsonl(ready_dir / "visual_assets_tagged.jsonl", visual)
    write_jsonl(ready_dir / "project_cases_tagged.jsonl", cases)

    summary = {
        "taxonomy_version": taxonomy.get("version"),
        "text_records": len(text),
        "visual_records": len(visual),
        "project_case_records": len(cases),
        "text_by_domain": dict(Counter(domain for record in text for domain in record["knowledge_domains"])),
        "visual_by_domain": dict(Counter(domain for record in visual for domain in record["knowledge_domains"])),
    }
    (ready_dir / "knowledge_taxonomy_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
