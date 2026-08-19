"""Private, review-first LangGraph workflow for knowledge-base intake.

This workflow intentionally stops at a local review package.  It does not add
new documents to the customer-facing RAG index, change the taxonomy, or upload
anything to a cloud service.  Promotion is a separate, explicit human action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "data" / "sales" / "processed" / "intake_runs"
DEFAULT_TAXONOMY_PATH = ROOT / "data" / "sales" / "config" / "knowledge_taxonomy_v1.json"
DEFAULT_MINERU = Path(os.getenv("MINERU_EXECUTABLE", "mineru"))
DEFAULT_AI_PYTHON = Path(sys.executable)
MANIFEST_BUILDER = ROOT / "scripts" / "build_mineru_asset_manifest.py"
VISUAL_ANNOTATOR = ROOT / "scripts" / "annotate_visual_assets_qwen3vl.py"
SALES_PLAYBOOK_INGESTOR = ROOT / "scripts" / "ingest_sales_playbook.py"

DOMAIN_DEFAULTS: dict[str, dict[str, str]] = {
    "01_standard_specification": {
        "document_category": "engineering_standard",
        "source_authority": "national_or_industry_standard",
        "customer_fact_policy": "standard_reference",
    },
    "02_product_information": {
        "document_category": "enterprise_product_catalogue",
        "source_authority": "enterprise_catalogue",
        "customer_fact_policy": "enterprise_claim_or_catalogue_reference",
    },
    "03_construction_method": {
        "document_category": "enterprise_construction_method",
        "source_authority": "enterprise_construction_document",
        "customer_fact_policy": "construction_reference",
    },
    "04_node_atlas": {
        "document_category": "application_atlas",
        "source_authority": "technical_atlas",
        "customer_fact_policy": "atlas_reference",
    },
    "05_project_case": {
        "document_category": "enterprise_project_case_catalogue",
        "source_authority": "enterprise_catalogue",
        "customer_fact_policy": "enterprise_catalogue_case_reference",
    },
    "06_sales_playbook": {
        "document_category": "internal_sales_playbook",
        "source_authority": "internal_sales_enablement",
        "customer_fact_policy": "supplementary_product_reference_with_validation",
    },
}


@dataclass(frozen=True)
class IntakeOptions:
    source_path: Path
    run_root: Path = DEFAULT_RUN_ROOT
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH
    mineru_executable: Path = DEFAULT_MINERU
    python_executable: Path = DEFAULT_AI_PYTHON
    knowledge_domain: str | None = None
    document_category: str | None = None
    annotate_visuals: bool = False
    visual_limit: int = 0
    skip_existing: bool = False
    dry_run: bool = False


class IntakeState(TypedDict, total=False):
    options: IntakeOptions
    run: dict[str, Any]
    classification: dict[str, Any]
    extraction: dict[str, Any]
    visual_annotation: dict[str, Any]
    review: dict[str, Any]
    events: list[dict[str, Any]]


class IntakeOperations(Protocol):
    def validate_source(self, options: IntakeOptions) -> dict[str, Any]: ...

    def classify_document(self, options: IntakeOptions, run: dict[str, Any]) -> dict[str, Any]: ...

    def extract_local_assets(
        self, options: IntakeOptions, run: dict[str, Any], classification: dict[str, Any]
    ) -> dict[str, Any]: ...

    def annotate_local_visuals(
        self, options: IntakeOptions, extraction: dict[str, Any]
    ) -> dict[str, Any]: ...

    def build_review_package(
        self,
        options: IntakeOptions,
        run: dict[str, Any],
        classification: dict[str, Any],
        extraction: dict[str, Any],
        visual_annotation: dict[str, Any],
    ) -> dict[str, Any]: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_slug(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", value).strip("-")
    return clean[:64] or "document"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _jsonl_count(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def _propose_domain_from_name(document_name: str) -> str | None:
    lower = document_name.lower()
    if any(term in document_name for term in ("图集", "节点", "窗洞", "女儿墙", "勒脚")):
        return "04_node_atlas"
    if any(term in document_name for term in ("施工", "工艺", "干挂", "粘锚", "安装", "穿透法")):
        return "03_construction_method"
    if any(term in document_name for term in ("规范", "规程", "标准", "jgj", "jgt")) or any(
        term in lower for term in ("jgj", "jgt", "gb ")
    ):
        return "01_standard_specification"
    if any(term in document_name for term in ("案例", "工程项目", "项目汇编")):
        return "05_project_case"
    if any(term in document_name for term in ("画册", "产品", "色卡", "企业产品")):
        return "02_product_information"
    return None


class LocalIntakeOperations:
    """All actions run local subprocesses and write only under ``run_root``."""

    def validate_source(self, options: IntakeOptions) -> dict[str, Any]:
        source = options.source_path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"知识资料文件不存在：{source}")
        if source.suffix.lower() not in {".pdf", ".docx"}:
            raise ValueError("当前入库流水线仅支持 PDF 或 DOCX 文件。")
        checksum = _file_sha256(source)
        run_id = f"{_safe_slug(source.stem)}_{checksum[:10]}"
        run_dir = options.run_root.expanduser().resolve() / run_id
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "source_path": str(source),
            "source_name": source.name,
            "source_format": source.suffix.lower().lstrip("."),
            "source_sha256": checksum,
            "source_bytes": source.stat().st_size,
            "privacy": "local_only",
        }

    def classify_document(self, options: IntakeOptions, run: dict[str, Any]) -> dict[str, Any]:
        document_name = Path(str(run["source_path"])).stem
        taxonomy = _read_json(options.taxonomy_path)
        documents = taxonomy.get("documents") if isinstance(taxonomy.get("documents"), dict) else {}
        known = documents.get(document_name) if isinstance(documents.get(document_name), dict) else None

        if options.knowledge_domain:
            domain = options.knowledge_domain
            if domain not in DOMAIN_DEFAULTS:
                raise ValueError(f"未知知识域：{domain}")
            defaults = DOMAIN_DEFAULTS[domain]
            source = "operator_selected"
        elif known:
            domain = str(known.get("knowledge_domain") or "")
            defaults = {
                "document_category": str(known.get("document_category") or "unclassified"),
                "source_authority": str(known.get("source_authority") or "unknown"),
                "customer_fact_policy": str(known.get("customer_fact_policy") or "review_before_use"),
            }
            source = "existing_taxonomy"
        elif str(run["source_format"]) == "docx":
            domain = "06_sales_playbook"
            defaults = DOMAIN_DEFAULTS[domain]
            source = "safe_docx_default"
        else:
            domain = _propose_domain_from_name(document_name)
            defaults = DOMAIN_DEFAULTS.get(domain or "", {})
            source = "filename_heuristic" if domain else "unclassified"

        return {
            "knowledge_domain": domain or "unclassified",
            "document_category": options.document_category or defaults.get("document_category", "unclassified"),
            "source_authority": defaults.get("source_authority", "unknown"),
            "customer_fact_policy": defaults.get("customer_fact_policy", "review_before_use"),
            "classification_source": source,
            "review_required": True,
            "customer_shareable": False,
            "note": "仅为入库候选分类；审核通过前不进入正式客户 RAG。",
        }

    @staticmethod
    def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
        process_env = os.environ.copy()
        process_env["PYTHONUTF8"] = "1"
        if environment:
            process_env.update(environment)
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=process_env)
        if completed.returncode != 0:
            output = completed.stdout[-4000:] if completed.stdout else ""
            raise RuntimeError(f"本地入库子任务失败（exit={completed.returncode}）：\n{output}")

    @staticmethod
    def _locate_auto_directory(mineru_output: Path) -> Path:
        matches = sorted(mineru_output.rglob("*_content_list_v2.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"MinerU 解析结果不完整：{mineru_output}（找到 {len(matches)} 个内容清单）")
        return matches[0].parent

    def extract_local_assets(
        self, options: IntakeOptions, run: dict[str, Any], classification: dict[str, Any]
    ) -> dict[str, Any]:
        run_dir = Path(str(run["run_dir"]))
        source = Path(str(run["source_path"]))
        assets_root = run_dir / "assets"
        mineru_output = run_dir / "mineru"
        manifest_path = assets_root / source.stem / "manifest_metadata.json"

        if options.dry_run:
            return {
                "status": "planned",
                "assets_root": str(assets_root),
                "manifest_path": str(manifest_path),
                "text_record_count": 0,
                "visual_asset_count": 0,
            }
        run_dir.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".pdf":
            expected_manifest = assets_root / "manifest_metadata.json"
            if not (options.skip_existing and expected_manifest.is_file()):
                if not options.mineru_executable.is_file():
                    raise FileNotFoundError(f"未找到本地 MinerU：{options.mineru_executable}")
                self._run(
                    [
                        str(options.mineru_executable),
                        "-p",
                        str(source),
                        "-o",
                        str(mineru_output),
                        "-b",
                        "pipeline",
                        "-m",
                        "auto",
                        "--image-analysis",
                        "false",
                    ],
                    environment={"MINERU_MODEL_SOURCE": "modelscope"},
                )
                auto_directory = self._locate_auto_directory(mineru_output)
                self._run(
                    [
                        str(options.python_executable),
                        "-u",
                        str(MANIFEST_BUILDER),
                        str(auto_directory),
                        "--output",
                        str(assets_root),
                        "--internal-only",
                    ]
                )
            manifest_path = assets_root / "manifest_metadata.json"
        else:
            # DOCX support intentionally reuses the conservative internal-sales
            # ingestor.  It keeps the source private and emits no visual asset.
            expected_manifest = assets_root / source.stem / "manifest_metadata.json"
            if not (options.skip_existing and expected_manifest.is_file()):
                self._run(
                    [str(options.python_executable), "-u", str(SALES_PLAYBOOK_INGESTOR), "--source", str(source), "--assets-root", str(assets_root)]
                )
            manifest_path = expected_manifest

        if not manifest_path.is_file():
            raise FileNotFoundError(f"本地资产清单未生成：{manifest_path}")
        metadata = _read_json(manifest_path)
        asset_directory = manifest_path.parent
        return {
            "status": "complete",
            "asset_directory": str(asset_directory),
            "manifest_path": str(manifest_path),
            "text_evidence_path": str(asset_directory / "text_evidence.jsonl"),
            "visual_assets_path": str(asset_directory / "visual_assets.jsonl"),
            "text_record_count": int(metadata.get("text_record_count") or 0),
            "visual_asset_count": int(metadata.get("visual_asset_count") or 0),
            "page_count": metadata.get("page_count"),
            "customer_shareable": False,
        }

    def annotate_local_visuals(self, options: IntakeOptions, extraction: dict[str, Any]) -> dict[str, Any]:
        if extraction.get("status") == "planned":
            assets_root = Path(str(extraction["assets_root"]))
            visual_assets_path = assets_root / "visual_assets.jsonl"
            output = assets_root / "visual_semantics_qwen3vl.jsonl"
            return {"status": "planned", "output": str(output), "asset_count": 0}
        visual_assets_path = Path(str(extraction.get("visual_assets_path") or ""))
        output = visual_assets_path.with_name("visual_semantics_qwen3vl.jsonl")
        asset_count = _jsonl_count(visual_assets_path)
        if asset_count == 0:
            return {"status": "not_applicable", "output": str(output), "asset_count": 0}
        if not options.annotate_visuals:
            return {"status": "queued_for_local_annotation", "output": str(output), "asset_count": asset_count}
        command = [str(options.python_executable), "-u", str(VISUAL_ANNOTATOR), str(visual_assets_path), "--output", str(output), "--resume"]
        if options.visual_limit > 0:
            command.extend(["--limit", str(options.visual_limit)])
        self._run(command)
        return {
            "status": "candidate_ready",
            "output": str(output),
            "asset_count": asset_count,
            "annotated_count": _jsonl_count(output),
            "model": "local_qwen3_vl_8b_4bit",
        }

    def build_review_package(
        self,
        options: IntakeOptions,
        run: dict[str, Any],
        classification: dict[str, Any],
        extraction: dict[str, Any],
        visual_annotation: dict[str, Any],
    ) -> dict[str, Any]:
        run_dir = Path(str(run["run_dir"]))
        review_path = run_dir / "review_manifest.json"
        review_status = "dry_run_complete" if options.dry_run else "needs_human_approval"
        package = {
            "schema_version": "knowledge_intake_review_v1",
            "review_status": review_status,
            "promotion_status": "not_promoted",
            "local_only": True,
            "source": run,
            "proposed_classification": classification,
            "extraction": extraction,
            "visual_annotation": visual_annotation,
            "approval_gates": [
                "确认知识域、文档类别和资料权威性。",
                "确认哪些文字块允许作为客户事实，哪些仅保留内部参考。",
                "抽样核验图片与同页文字、标题、页码和裁剪坐标是否一致。",
                "确认企业主张、价格、工期、寿命和竞品对比不被当作标准或承诺。",
                "审核通过后，再显式执行正式 RAG 索引构建；本流水线不会自动执行。",
            ],
            "promotion_note": "当前文件尚未进入 data/sales/processed/rag_ready 或 data/sales/processed/rag_index。",
        }
        if not options.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            review_path.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"status": review_status, "review_manifest": str(review_path), **package}


def _event(name: str, state: IntakeState) -> list[dict[str, Any]]:
    prior = list(state.get("events") or [])
    return [*prior, {"node": name, "status": "complete"}]


def build_knowledge_intake_graph(operations: IntakeOperations | None = None):
    """Build a finite local-only intake graph with no checkpointer."""

    ops = operations or LocalIntakeOperations()

    def validate_source(state: IntakeState, config: RunnableConfig) -> dict[str, Any]:
        return {"run": ops.validate_source(state["options"]), "events": _event("validate_source", state)}

    def classify_document(state: IntakeState, config: RunnableConfig) -> dict[str, Any]:
        return {
            "classification": ops.classify_document(state["options"], state["run"]),
            "events": _event("classify_document", state),
        }

    def extract_local_assets(state: IntakeState, config: RunnableConfig) -> dict[str, Any]:
        return {
            "extraction": ops.extract_local_assets(state["options"], state["run"], state["classification"]),
            "events": _event("extract_local_assets", state),
        }

    def annotate_visual_assets(state: IntakeState, config: RunnableConfig) -> dict[str, Any]:
        return {
            "visual_annotation": ops.annotate_local_visuals(state["options"], state["extraction"]),
            "events": _event("annotate_visual_assets", state),
        }

    def build_review_package(state: IntakeState, config: RunnableConfig) -> dict[str, Any]:
        return {
            "review": ops.build_review_package(
                state["options"], state["run"], state["classification"], state["extraction"], state["visual_annotation"]
            ),
            "events": _event("build_review_package", state),
        }

    graph = StateGraph(IntakeState)
    graph.add_node("validate_source", validate_source)
    graph.add_node("classify_document", classify_document)
    graph.add_node("extract_local_assets", extract_local_assets)
    graph.add_node("annotate_visual_assets", annotate_visual_assets)
    graph.add_node("build_review_package", build_review_package)
    graph.add_edge(START, "validate_source")
    graph.add_edge("validate_source", "classify_document")
    graph.add_edge("classify_document", "extract_local_assets")
    graph.add_edge("extract_local_assets", "annotate_visual_assets")
    graph.add_edge("annotate_visual_assets", "build_review_package")
    graph.add_edge("build_review_package", END)
    return graph.compile()
