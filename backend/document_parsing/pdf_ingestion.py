"""Generic PDF intake with a text-PDF / scanned-PDF split.

Text-layer PDFs use bounded, layout-neutral table detection before Camelot
extracts candidate grids.  Scanned PDFs are rendered into request-scoped page
images for local vision transcription.  The latter remains a customer-review
candidate rather than canonical OCR evidence, so it cannot silently become a
business fact or final spreadsheet value.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pypdf import PdfReader

from backend.document_parsing.ingestion import (
    FinanceIntakeError,
    EvidenceBlock,
    IntakeResult,
    IntermediateCell,
    IntermediateDocument,
    IntermediateTable,
    SourcePointer,
    ValidationIssue,
    VisualAsset,
    _headers_for_rows,
    _issue,
    evidence_block,
    finalize_intermediate_evidence,
    numeric_candidates_for_text,
    standardize_tables,
)


MAX_PDF_BYTES = 30 * 1024 * 1024
MAX_PDF_PAGES = 300
TEXT_PDF_MIN_CHARACTERS_PER_PAGE = 80
MAX_SCANNED_PDF_VISION_PAGES = 4
MAX_TEXT_PDF_TABLE_PAGES = 12
# MinerU may need to start a sizeable local model.  It is therefore an
# optional enhancement, not part of the request path for ordinary text PDFs.
# Keeping the cap finite also prevents one upload from holding the web UI for
# several minutes when the local runtime is unhealthy.
MINERU_TIMEOUT_SECONDS = 90
DEFAULT_MINERU = Path(os.getenv("MINERU_EXECUTABLE", "mineru"))

StatementType = Literal["income_statement", "balance_sheet", "cash_flow_statement"]
PdfKind = Literal["text_pdf", "scanned_pdf", "mixed_pdf"]
PdfPageTextQuality = Literal["text_layer", "sparse_text", "garbled_text", "layout_suspect", "no_text"]
PdfPageParseRoute = Literal["direct_text", "vision", "hybrid"]

STATEMENT_SIGNALS: dict[StatementType, tuple[str, ...]] = {
    "income_statement": ("利润表", "损益表", "合并利润表", "income statement"),
    "balance_sheet": ("资产负债表", "合并资产负债表", "balance sheet"),
    "cash_flow_statement": ("现金流量表", "合并现金流量表", "cash flow statement"),
}


class PdfCandidatePage(BaseModel):
    page_number: int = Field(ge=1)
    statement_type: StatementType
    signals: list[str]
    table_strategy: Literal["camelot", "pp_structure_v3"]
    extraction_status: Literal["extracted", "no_table", "not_configured", "not_run"]


class PdfPageInspection(BaseModel):
    """Per-page evidence used to choose direct parsing or visual recovery."""

    page_number: int = Field(ge=1)
    text_characters: int = Field(ge=0)
    text_quality: PdfPageTextQuality
    parse_route: PdfPageParseRoute
    quality_reason: str


class PdfExtractionSummary(BaseModel):
    document_kind: PdfKind
    page_count: int = Field(ge=1)
    text_characters: int = Field(ge=0)
    page_locator: Literal["mineru", "pypdf_fallback"]
    mineru_status: Literal["completed", "unavailable"]
    candidate_pages: list[PdfCandidatePage] = Field(default_factory=list)
    page_inspections: list[PdfPageInspection] = Field(default_factory=list)
    vision_rendered_pages: int = Field(default=0, ge=0)
    # Scanned files are processed in small request-scoped batches. This keeps
    # large uploads responsive without silently discarding later pages.
    vision_rendered_page_numbers: list[int] = Field(default_factory=list)
    vision_next_page_start: int | None = Field(default=None, ge=1)


class PdfIntakeResult(IntakeResult):
    pdf: PdfExtractionSummary


_CUSTOM_GLYPH_CODE = re.compile(r"/G([0-9A-Fa-f]{2})")


def _decode_custom_glyph_text(text: str) -> str:
    """Decode PDFs that expose literal byte-valued glyph names (/G41 -> A)."""

    matches = _CUSTOM_GLYPH_CODE.findall(text)
    if len(matches) < 8 or (len(matches) * 4) / max(len(text), 1) < 0.15:
        return text

    def decode(match: re.Match[str]) -> str:
        value = int(match.group(1), 16)
        try:
            return bytes([value]).decode("cp1252")
        except UnicodeDecodeError:
            return "\ufffd"

    decoded = _CUSTOM_GLYPH_CODE.sub(decode, text)
    printable = sum(character.isprintable() or character in "\n\r\t" for character in decoded)
    return decoded if printable / max(len(decoded), 1) >= 0.9 else text


def _safe_page_text(page: Any) -> str:
    try:
        return _decode_custom_glyph_text(str(page.extract_text() or ""))
    except Exception:
        return ""


def _inspect_pdf_page(page_number: int, text: str) -> PdfPageInspection:
    """Classify a single page without treating the filename as evidence.

    The checks intentionally remain conservative: a false visual fallback is
    preferable to using garbled text as a financial source.  Final visual
    output is still a customer-review candidate, never a confirmed fact.
    """

    normalized = text.strip()
    character_count = len(normalized)
    if not normalized:
        return PdfPageInspection(
            page_number=page_number,
            text_characters=0,
            text_quality="no_text",
            parse_route="vision",
            quality_reason="No usable PDF text layer was extracted.",
        )

    replacement_or_control = normalized.count("\ufffd") + sum(
        1 for character in normalized if ord(character) < 32 and character not in "\n\r\t"
    )
    if replacement_or_control / max(character_count, 1) > 0.02:
        return PdfPageInspection(
            page_number=page_number,
            text_characters=character_count,
            text_quality="garbled_text",
            parse_route="vision",
            quality_reason="The extracted text contains replacement or control characters.",
        )

    # Some PDFs expose custom-font glyph names instead of Unicode text, for
    # example "/G41/G67/G61...". The string can be very long and contain no
    # replacement characters, so length-only checks incorrectly accept it as
    # readable text. Route glyph-code-dense pages to visual/OCR recovery.
    glyph_code_count = len(re.findall(r"/G[0-9A-Fa-f]{2}", normalized))
    if glyph_code_count >= 8 and (glyph_code_count * 4) / max(character_count, 1) >= 0.15:
        return PdfPageInspection(
            page_number=page_number,
            text_characters=character_count,
            text_quality="garbled_text",
            parse_route="vision",
            quality_reason="The extracted text is dominated by custom-font glyph codes.",
        )

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    short_line_ratio = sum(len(line) <= 2 for line in lines) / len(lines) if lines else 0.0
    if character_count >= 30 and len(lines) >= 12 and short_line_ratio >= 0.7:
        return PdfPageInspection(
            page_number=page_number,
            text_characters=character_count,
            text_quality="layout_suspect",
            parse_route="vision",
            quality_reason="Extracted text is fragmented into many very short lines; reading order may be unreliable.",
        )
    if character_count < 24:
        return PdfPageInspection(
            page_number=page_number,
            text_characters=character_count,
            text_quality="sparse_text",
            parse_route="vision",
            quality_reason="The text layer is too sparse to recover reliable document structure.",
        )
    return PdfPageInspection(
        page_number=page_number,
        text_characters=character_count,
        text_quality="text_layer",
        parse_route="direct_text",
        quality_reason="Usable text layer extracted directly from the PDF.",
    )


def _document_kind_for_pages(inspections: list[PdfPageInspection]) -> PdfKind:
    has_direct_text = any(inspection.parse_route in {"direct_text", "hybrid"} for inspection in inspections)
    has_visual_recovery = any(inspection.parse_route in {"vision", "hybrid"} for inspection in inspections)
    if has_direct_text and has_visual_recovery:
        return "mixed_pdf"
    return "text_pdf" if has_direct_text else "scanned_pdf"


def inspect_pdf_pages(path: Path) -> tuple[PdfKind, list[str], list[PdfPageInspection]]:
    """Inspect every PDF page and select the route independently per page."""

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise FinanceIntakeError("无法读取该 PDF，请确认它未损坏且未加密。") from exc
    if reader.is_encrypted:
        raise FinanceIntakeError("暂不支持加密 PDF，请先导出未加密副本后上传。")
    if not reader.pages:
        raise FinanceIntakeError("PDF 没有可解析页面。")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise FinanceIntakeError(f"PDF 超过 {MAX_PDF_PAGES} 页限制，请拆分后上传。")
    pages = [_safe_page_text(page) for page in reader.pages]
    inspections = [_inspect_pdf_page(page_number, text) for page_number, text in enumerate(pages, start=1)]
    document_kind = _document_kind_for_pages(inspections)
    return document_kind, pages, inspections


def inspect_pdf(path: Path) -> tuple[PdfKind, list[str]]:
    """Backward-compatible document-level summary; use ``inspect_pdf_pages`` for routing."""

    document_kind, pages, _inspections = inspect_pdf_pages(path)
    return document_kind, pages


def render_scanned_pdf_pages(
    source: Path,
    *,
    file_name: str,
    page_count: int,
    page_numbers: list[int] | None = None,
) -> tuple[list[VisualAsset], str | None]:
    """Render a bounded set of scan pages for generic local vision.

    Rendering creates in-memory PNG bytes only.  The bytes are attached to the
    request-scoped ``VisualAsset`` and are excluded from the returned JSON.
    No rendered page is treated as OCR truth; the later vision result remains
    a candidate the customer must review.
    """

    try:
        import fitz
    except ImportError:
        return [], "未安装 PyMuPDF，无法将扫描 PDF 页面转换为本地视觉输入。"
    assets: list[VisualAsset] = []
    selected_pages = page_numbers or list(range(1, min(page_count, MAX_SCANNED_PDF_VISION_PAGES) + 1))
    if not selected_pages:
        return [], "No PDF pages were selected for visual review."
    if len(selected_pages) > MAX_SCANNED_PDF_VISION_PAGES:
        return [], f"A visual review batch can contain at most {MAX_SCANNED_PDF_VISION_PAGES} pages."
    if len(set(selected_pages)) != len(selected_pages) or any(page_number < 1 or page_number > page_count for page_number in selected_pages):
        return [], "The requested PDF page range is invalid."

    try:
        pdf = fitz.open(str(source))
        try:
            for page_number in selected_pages:
                page = pdf.load_page(page_number - 1)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                image_bytes = pixmap.tobytes("png")
                raster_regions = []
                seen_boxes = set()
                for info in page.get_image_info():
                    box = fitz.Rect(info['bbox']) & page.rect
                    if box.is_empty or box.get_area() <= 0:
                        continue
                    normalized = tuple(round(v, 3) for v in (
                        (box.x0-page.rect.x0)/page.rect.width,(box.y0-page.rect.y0)/page.rect.height,
                        (box.x1-page.rect.x0)/page.rect.width,(box.y1-page.rect.y0)/page.rect.height))
                    if normalized not in seen_boxes:
                        seen_boxes.add(normalized)
                        raster_regions.append(list(normalized))
                source_pointer = SourcePointer(
                    source_type="pdf",
                    file_name=file_name,
                    page_number=page_number,
                    section_id=f"pdf:page:{page_number}",
                    parser="pymupdf-page-render",
                    extraction_confidence=0.95,
                )
                from backend.documents.visual_layout import raster_layout_groups
                assets.append(
                    VisualAsset(
                        visual_id=f"pdf:scan-page:{page_number}",
                        kind="document_page",
                        source=source_pointer,
                        media_type="image/png",
                        width_px=pixmap.width,
                        height_px=pixmap.height,
                        sha256=hashlib.sha256(image_bytes).hexdigest(),
                        delivery_status="ready_for_vision",
                        metadata={
                            "content_extraction": "vision_candidate_only",
                            "original_page_number": page_number,
                            "render_scale": 1.5,
                            "raster_region_boxes": raster_regions[:24],
                            "raster_region_total": len(raster_regions),
                            "raster_layout_groups": raster_layout_groups(raster_regions),
                            "raster_region_policy": "Normalized [left,top,right,bottom], origin top-left. Native raster objects, not logical figure labels. May include logos, scale bars, tiled scans or background images; verify against pixels and requested region. Vector-only figures may be absent.",
                        },
                        image_bytes=image_bytes,
                    )
                )
        finally:
            pdf.close()
    except Exception as exc:
        return [], f"扫描 PDF 页面渲染失败：{str(exc)[:160]}"
    return assets, None


def ingest_scanned_pdf_vision_batch(
    file_name: str,
    content: bytes,
    *,
    page_start: int = 1,
) -> PdfIntakeResult:
    """Render one bounded scanned or mixed-PDF visual batch for review.

    The caller re-sends the source PDF with ``page_start`` to continue.  The
    request therefore never persists customer files or rendered page images.
    """

    if not file_name.lower().endswith(".pdf"):
        raise FinanceIntakeError("The vision-batch endpoint accepts PDF files only.")
    if not content:
        raise FinanceIntakeError("The uploaded PDF is empty.")
    if len(content) > MAX_PDF_BYTES:
        raise FinanceIntakeError(f"PDF files must be at most {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    if page_start < 1:
        raise FinanceIntakeError("page_start must be at least 1.")

    issues: list[ValidationIssue] = []
    safe_file_name = Path(file_name).name
    with tempfile.TemporaryDirectory(prefix="finance-pdf-vision-batch-") as temporary_directory:
        source = Path(temporary_directory) / safe_file_name
        source.write_bytes(content)
        document_kind, pypdf_pages, page_inspections = inspect_pdf_pages(source)
        vision_page_numbers = [inspection.page_number for inspection in page_inspections if inspection.parse_route == "vision"]
        unresolved_table_pages = text_table_pages_needing_visual_recovery(source, pypdf_pages, page_inspections)
        if unresolved_table_pages:
            for page_number in unresolved_table_pages:
                current = page_inspections[page_number - 1]
                page_inspections[page_number - 1] = current.model_copy(
                    update={
                        "text_quality": "layout_suspect",
                        "parse_route": "hybrid",
                        "quality_reason": "A likely table page had usable text but its table structure could not be recovered reliably.",
                    }
                )
            vision_page_numbers = sorted(set(vision_page_numbers).union(unresolved_table_pages))
            document_kind = _document_kind_for_pages(page_inspections)
            _issue(issues, "warning", "text_table_visual_fallback", "A text table page needs visual layout recovery before it can be used.")
        if not vision_page_numbers:
            raise FinanceIntakeError("Every page has a usable text layer; this PDF does not need visual-page recovery.")
        if page_start > len(pypdf_pages):
            raise FinanceIntakeError(f"page_start exceeds this PDF's {len(pypdf_pages)} pages.")

        page_numbers = [page_number for page_number in vision_page_numbers if page_number >= page_start][:MAX_SCANNED_PDF_VISION_PAGES]
        if not page_numbers:
            raise FinanceIntakeError("There are no remaining visual-recovery pages at or after page_start.")
        visual_assets, render_error = render_scanned_pdf_pages(
            source,
            file_name=safe_file_name,
            page_count=len(pypdf_pages),
            page_numbers=page_numbers,
        )
        if render_error:
            _issue(issues, "warning", "scanned_pdf_render_unavailable", render_error)
        else:
            _issue(
                issues,
                "warning",
                "scanned_pdf_vision_pending_confirmation" if document_kind == "scanned_pdf" else "mixed_pdf_vision_pending_confirmation",
                "Rendered PDF page content is a review candidate only and cannot become a business fact or spreadsheet value without customer confirmation.",
            )

        remaining_vision_pages = [page_number for page_number in vision_page_numbers if page_number > page_numbers[-1]]
        next_page_start = remaining_vision_pages[0] if remaining_vision_pages else None
        if next_page_start is not None:
            _issue(
                issues,
                "warning",
                "scanned_pdf_more_pages_available",
                f"Pages {page_numbers[0]}-{page_numbers[-1]} were prepared. Re-upload this file with page_start={next_page_start} to continue visual review.",
            )

        blocks: list[EvidenceBlock] = []
        for page_number in page_numbers:
            page_text = pypdf_pages[page_number - 1].strip()
            inspection = page_inspections[page_number - 1]
            blocks.append(
                evidence_block(
                    block_id=f"pdf:page:{page_number}",
                    kind="page_text",
                    source=SourcePointer(
                        source_type="pdf",
                        file_name=safe_file_name,
                        page_number=page_number,
                        section_id=f"pdf:page:{page_number}",
                        parser="pypdf",
                        extraction_confidence=1.0 if inspection.parse_route == "hybrid" else 0.0,
                    ),
                    text=page_text or None,
                    metadata={
                        "page_content_kind": inspection.text_quality,
                        "parse_route": inspection.parse_route,
                        "quality_reason": inspection.quality_reason,
                        "text_characters": len(page_text),
                        "page_locator": "pypdf_fallback",
                    },
                )
            )
        for visual in visual_assets:
            blocks.append(
                evidence_block(
                    block_id=f"evidence:{visual.visual_id}",
                    kind="embedded_visual",
                    source=visual.source,
                    metadata={"visual_id": visual.visual_id, "kind": visual.kind, "delivery_status": visual.delivery_status, **visual.metadata},
                )
            )

        document = finalize_intermediate_evidence(
            IntermediateDocument(
                source_type="pdf",
                file_name=safe_file_name,
                parser="pypdf+page-render+local-vision-candidate" if document_kind == "scanned_pdf" else "pypdf+hybrid-page-routing+local-vision-candidate",
                tables=[],
                blocks=blocks,
                visual_assets=visual_assets,
            ),
            content,
        )
        standard_result = standardize_tables(document)
        return PdfIntakeResult(
            intermediate=document,
            standard=standard_result.standard,
            validation=[*issues, *standard_result.validation],
            pdf=PdfExtractionSummary(
                document_kind=document_kind,
                page_count=len(pypdf_pages),
                text_characters=sum(len(page.strip()) for page in pypdf_pages),
                page_locator="pypdf_fallback",
                mineru_status="unavailable",
                page_inspections=page_inspections,
                vision_rendered_pages=len(visual_assets),
                vision_rendered_page_numbers=[asset.source.page_number for asset in visual_assets if asset.source.page_number is not None],
                vision_next_page_start=next_page_start,
            ),
        )


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        preferred = ("text", "content", "text_content", "latex", "html")
        own = [str(value[key]) for key in preferred if isinstance(value.get(key), (str, int, float, bool))]
        nested = [_coerce_text(item) for key, item in value.items() if key not in preferred]
        return "\n".join(item for item in [*own, *nested] if item)
    if isinstance(value, list):
        return "\n".join(item for item in (_coerce_text(item) for item in value) if item)
    return ""


def _mineru_page_texts(payload: Any) -> list[str]:
    """Accept the content-list shapes emitted by current MinerU releases."""

    items = payload if isinstance(payload, list) else payload.get("pages", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []
    pages: dict[int, list[str]] = {}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        raw_page = item.get("page_idx", item.get("page_number", item.get("page", index - 1)))
        try:
            page_number = int(raw_page) + 1 if "page_idx" in item else int(raw_page)
        except (TypeError, ValueError):
            page_number = index
        page_number = page_number if page_number >= 1 else index
        text = _coerce_text(item)
        if text:
            pages.setdefault(page_number, []).append(text)
    if not pages:
        return []
    return ["\n".join(pages.get(page_number, [])) for page_number in range(1, max(pages) + 1)]


def _run_mineru(source: Path, output_directory: Path) -> list[str]:
    if not DEFAULT_MINERU.is_file():
        raise FileNotFoundError(f"未找到本地 MinerU：{DEFAULT_MINERU}")
    environment = os.environ.copy()
    environment["MINERU_MODEL_SOURCE"] = "modelscope"
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [str(DEFAULT_MINERU), "-p", str(source), "-o", str(output_directory), "-b", "pipeline", "-m", "auto", "--image-analysis", "false"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=MINERU_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        output = (completed.stdout or "")[-1_000:].replace("\r", " ").replace("\n", " ")
        raise RuntimeError(f"MinerU 退出码 {completed.returncode}：{output}")
    manifests = sorted(output_directory.rglob("*_content_list_v2.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"MinerU 未生成唯一内容清单（找到 {len(manifests)} 个）。")
    return _mineru_page_texts(json.loads(manifests[0].read_text(encoding="utf-8-sig")))


def _mineru_enabled() -> bool:
    """Opt in to the slower local layout pass only where it adds value.

    Text-layer PDFs are first located with pypdf, which is deterministic and
    fast enough for an interactive upload.  Deployments that have a healthy
    local MinerU runtime can set ``FINANCE_ENABLE_MINERU=1`` to use it as a
    fallback for text PDFs whose statement pages cannot be located that way.
    """

    return os.getenv("FINANCE_ENABLE_MINERU", "").strip().lower() in {"1", "true", "yes", "on"}


def locate_statement_pages(page_texts: list[str]) -> list[tuple[int, StatementType, list[str]]]:
    candidates: list[tuple[int, StatementType, list[str]]] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        normalised = page_text.lower()
        for statement_type, signals in STATEMENT_SIGNALS.items():
            matched = [signal for signal in signals if signal.lower() in normalised]
            if matched:
                candidates.append((page_number, statement_type, matched))
                break
    return candidates


def locate_generic_table_pages(page_texts: list[str]) -> list[tuple[int, str, list[str]]]:
    """Select likely tables without assuming a business domain.

    Camelot is comparatively expensive on a long PDF, so the lightweight text
    layer only ranks pages with a dense numeric surface or common table-layout
    labels.  It does not infer what the table means.
    """

    table_labels = (
        "table", "total", "amount", "date", "status", "actual", "budget",
        "项目", "金额", "数量", "日期", "状态", "合计", "编号", "负责人",
    )
    ranked: list[tuple[int, int, list[str]]] = []
    for page_number, text in enumerate(page_texts, start=1):
        normalised = text.lower()
        labels = [label for label in table_labels if label.lower() in normalised]
        numeric_surfaces = len(re.findall(r"(?<!\w)[¥$€£]?\s*\d+(?:[,.]\d+)?", text))
        nonempty_lines = sum(1 for line in text.splitlines() if line.strip())
        score = min(numeric_surfaces, 12) + min(len(labels) * 2, 8)
        if nonempty_lines >= 4 and (numeric_surfaces >= 5 or len(labels) >= 2) and score >= 6:
            ranked.append((page_number, score, labels))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [(page_number, "generic_table", labels) for page_number, _, labels in ranked[:MAX_TEXT_PDF_TABLE_PAGES]]


def _cell(page_number: int, row_index: int, column_index: int, value: str) -> IntermediateCell:
    display = value.strip() or None
    return IntermediateCell(
        coordinate=f"p{page_number}r{row_index}c{column_index}",
        row_index=row_index,
        column_index=column_index,
        value=display,
        display_text=display,
        numeric_candidates=numeric_candidates_for_text(display),
    )


def camelot_table_to_intermediate(table: Any, *, page_number: int, title: str, flavor: str, table_index: int) -> IntermediateTable:
    dataframe = table.df
    rows: list[list[IntermediateCell]] = []
    for row_index, values in enumerate(dataframe.itertuples(index=False, name=None), start=1):
        cells = [_cell(page_number, row_index, column_index, str(value or "")) for column_index, value in enumerate(values, start=1)]
        if any(cell.display_text for cell in cells):
            rows.append(cells)
    return IntermediateTable(
        table_id=f"pdf:{page_number}:{flavor}:{table_index}",
        title=title,
        page_number=page_number,
        range=f"page:{page_number}",
        parser=f"camelot-{flavor}",
        extraction_confidence=0.9 if flavor == "lattice" else 0.75,
        headers=_headers_for_rows(rows),
        rows=rows,
    )


def extract_text_pdf_tables(source: Path, candidates: list[tuple[int, str, list[str]]]) -> tuple[list[IntermediateTable], dict[int, str]]:
    """Run Camelot on bounded likely table pages; lattice first, then stream."""

    try:
        import camelot
    except ImportError as exc:
        raise FinanceIntakeError("PDF 表格解析依赖未安装，请先安装 camelot-py。") from exc

    tables: list[IntermediateTable] = []
    status_by_page: dict[int, str] = {}
    for page_number, table_title, _signals in candidates:
        extracted = False
        for flavor in ("lattice", "stream"):
            try:
                parsed = camelot.read_pdf(str(source), pages=str(page_number), flavor=flavor)
            except Exception:
                continue
            if not parsed:
                continue
            for table_index, table in enumerate(parsed, start=1):
                tables.append(
                    camelot_table_to_intermediate(
                        table,
                        page_number=page_number,
                        title=table_title,
                        flavor=flavor,
                        table_index=table_index,
                    )
                )
            extracted = True
            break
        status_by_page[page_number] = "extracted" if extracted else "no_table"
    return tables, status_by_page


def text_table_pages_needing_visual_recovery(
    source: Path,
    page_texts: list[str],
    inspections: list[PdfPageInspection],
) -> list[int]:
    """Recheck likely text-table pages when a later vision batch is requested."""

    direct_page_numbers = {inspection.page_number for inspection in inspections if inspection.parse_route == "direct_text"}
    statement_candidates = [candidate for candidate in locate_statement_pages(page_texts) if candidate[0] in direct_page_numbers]
    statement_pages = {candidate[0] for candidate in statement_candidates}
    generic_candidates = [
        candidate
        for candidate in locate_generic_table_pages(page_texts)
        if candidate[0] in direct_page_numbers and candidate[0] not in statement_pages
    ]
    candidates = sorted([*statement_candidates, *generic_candidates], key=lambda candidate: candidate[0])
    if not candidates:
        return []
    _tables, status_by_page = extract_text_pdf_tables(source, candidates)
    return [page_number for page_number, _title, _signals in candidates if status_by_page.get(page_number) == "no_table"]


def ingest_pdf(file_name: str, content: bytes) -> PdfIntakeResult:
    """Parse one PDF in a temporary request directory and return two JSON layers."""

    if not file_name.lower().endswith(".pdf"):
        raise FinanceIntakeError("PDF 入口仅接收 .pdf 文件。")
    if not content:
        raise FinanceIntakeError("上传的 PDF 文件为空。")
    if len(content) > MAX_PDF_BYTES:
        raise FinanceIntakeError(f"PDF 文件超过 {MAX_PDF_BYTES // (1024 * 1024)} MB 限制。")

    issues: list[ValidationIssue] = []
    with tempfile.TemporaryDirectory(prefix="finance-pdf-") as temporary_directory:
        source = Path(temporary_directory) / Path(file_name).name
        source.write_bytes(content)
        document_kind, pypdf_pages, page_inspections = inspect_pdf_pages(source)
        locator = "pypdf_fallback"
        mineru_status: Literal["completed", "unavailable"] = "unavailable"
        page_texts = pypdf_pages
        text_page_numbers = {inspection.page_number for inspection in page_inspections if inspection.parse_route == "direct_text"}
        vision_page_numbers = [inspection.page_number for inspection in page_inspections if inspection.parse_route == "vision"]
        candidates = [candidate for candidate in locate_statement_pages(page_texts) if candidate[0] in text_page_numbers]
        # Usable text does not imply that charts/photos can be discarded.
        # Detect visible raster/vector regions on CPU, preserving native text.
        try:
            import fitz
            with fitz.open(str(source)) as visual_pdf:
                for page_index, page in enumerate(visual_pdf):
                    area = max(1.0, page.rect.get_area())
                    image_area = sum(fitz.Rect(info['bbox']).get_area() for info in page.get_image_info())
                    vector_area = sum(item['rect'].get_area() for item in page.get_drawings())
                    if image_area / area >= 0.03 or vector_area / area >= 0.15:
                        vision_page_numbers.append(page_index + 1)
            vision_page_numbers = sorted(set(vision_page_numbers))
        except Exception as exc:
            _issue(issues, "warning", "visual_region_detection_unavailable", f"PDF visual coverage could not be inspected: {type(exc).__name__}")
        visual_assets: list[VisualAsset] = []
        rendered_page_numbers: list[int] = []
        next_page_start: int | None = None

        # Do not launch a model runtime on the deterministic PDF pass.  The
        # text layer is fast enough for normal documents; scanned-page vision
        # is performed later by the bounded router augmentation and stays
        # review-only.
        if document_kind == "text_pdf" and not candidates and _mineru_enabled():
            try:
                mineru_pages = _run_mineru(source, Path(temporary_directory) / "mineru")
                if mineru_pages:
                    page_texts = mineru_pages
                    locator = "mineru"
                    mineru_status = "completed"
                    candidates = [candidate for candidate in locate_statement_pages(page_texts) if candidate[0] in text_page_numbers]
                else:
                    raise RuntimeError("MinerU 内容清单中没有可用文字。")
            except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                _issue(issues, "warning", "mineru_page_locator_unavailable", f"MinerU 页面定位未完成，已使用 PDF 文字层回退定位；请复核结果。（{str(exc)[:160]}）")
        generic_table_pages: list[tuple[int, str, list[str]]] = []
        if text_page_numbers:
            financial_page_numbers = {page_number for page_number, _, _ in candidates}
            generic_table_pages = [
                candidate
                for candidate in locate_generic_table_pages(page_texts)
                if candidate[0] in text_page_numbers and candidate[0] not in financial_page_numbers
            ]
        table_candidates: list[tuple[int, str, list[str]]] = [*candidates, *generic_table_pages]
        table_candidates.sort(key=lambda candidate: candidate[0])

        intermediate_tables: list[IntermediateTable] = []
        status_by_page: dict[int, str] = {}
        if text_page_numbers and table_candidates:
            intermediate_tables, status_by_page = extract_text_pdf_tables(source, table_candidates)
            if not intermediate_tables:
                _issue(issues, "warning", "camelot_no_table", "已定位疑似财务报表页，但未提取到可用表格；请客户复核 PDF 版式或改传 Excel。")
        unresolved_table_pages = [
            page_number
            for page_number, _title, _signals in table_candidates
            if status_by_page.get(page_number) == "no_table"
        ]
        if unresolved_table_pages:
            for page_number in unresolved_table_pages:
                current = page_inspections[page_number - 1]
                page_inspections[page_number - 1] = current.model_copy(
                    update={
                        "text_quality": "layout_suspect",
                        "parse_route": "hybrid",
                        "quality_reason": "A likely table page had usable text but its table structure could not be recovered reliably.",
                    }
                )
            vision_page_numbers = sorted(set(vision_page_numbers).union(unresolved_table_pages))
            document_kind = _document_kind_for_pages(page_inspections)
            _issue(
                issues,
                "warning",
                "text_table_visual_fallback",
                "部分可读文本页的表格结构恢复失败，已追加视觉版面与表格识别候选；请核对后使用。",
            )

        if vision_page_numbers:
            selected_vision_pages = vision_page_numbers[:MAX_SCANNED_PDF_VISION_PAGES]
            visual_assets, render_error = render_scanned_pdf_pages(
                source,
                file_name=Path(file_name).name,
                page_count=len(pypdf_pages),
                page_numbers=selected_vision_pages,
            )
            if render_error:
                _issue(issues, "warning", "scanned_pdf_render_unavailable", render_error)
            elif len(vision_page_numbers) > MAX_SCANNED_PDF_VISION_PAGES:
                _issue(
                    issues,
                    "warning",
                    "scanned_pdf_pages_limited" if document_kind == "scanned_pdf" else "mixed_pdf_pages_limited",
                    f"PDF 有 {len(vision_page_numbers)} 页需要视觉恢复；本次已准备第 {selected_vision_pages[0]}-{selected_vision_pages[-1]} 页，后续可继续按页识别。",
                )
        if visual_assets:
            rendered_page_numbers = [asset.source.page_number for asset in visual_assets if asset.source.page_number is not None]
            remaining_vision_pages = [page_number for page_number in vision_page_numbers if page_number > rendered_page_numbers[-1]]
            next_page_start = remaining_vision_pages[0] if remaining_vision_pages else None
        if vision_page_numbers:
            _issue(
                issues,
                "warning",
                "scanned_pdf_vision_pending_confirmation" if document_kind == "scanned_pdf" else "mixed_pdf_vision_pending_confirmation",
                "包含扫描、图像或复杂版面页面，已准备本地视觉预览；原生文本仍单独保留。视觉候选不自动成为已核实事实。",
            )
        if not table_candidates and not vision_page_numbers:
            _issue(issues, "warning", "no_financial_statement_page", "未定位到利润表、资产负债表或现金流量表页面；请确认文件内容或改传 Excel。")

        parser_name = (
            "mineru-page-locator+camelot" if locator == "mineru"
            else "pypdf+hybrid-page-routing+camelot+local-vision-candidate" if document_kind == "mixed_pdf"
            else "pypdf+page-render+local-vision-candidate" if document_kind == "scanned_pdf"
            else "pypdf+camelot"
        )
        blocks: list[EvidenceBlock] = []
        for page_number, page_text in enumerate(page_texts, start=1):
            inspection = page_inspections[page_number - 1]
            blocks.append(
                evidence_block(
                    block_id=f"pdf:page:{page_number}",
                    kind="page_text",
                    source=SourcePointer(
                        source_type="pdf",
                        file_name=Path(file_name).name,
                        page_number=page_number,
                        section_id=f"pdf:page:{page_number}",
                        parser="mineru" if locator == "mineru" else "pypdf",
                        extraction_confidence=1.0 if inspection.parse_route in {"direct_text", "hybrid"} else 0.0,
                    ),
                    text=page_text.strip() or None,
                    metadata={
                        "page_content_kind": inspection.text_quality,
                        "parse_route": inspection.parse_route,
                        "quality_reason": inspection.quality_reason,
                        "text_characters": len(page_text.strip()),
                        "page_locator": locator,
                    },
                )
            )
        for table in intermediate_tables:
            blocks.append(
                evidence_block(
                    block_id=f"evidence:{table.table_id}",
                    kind="table",
                    source=SourcePointer(
                        source_type="pdf",
                        file_name=Path(file_name).name,
                        page_number=table.page_number,
                        section_id=table.table_id,
                        parser=table.parser,
                        extraction_confidence=table.extraction_confidence,
                    ),
                    table_id=table.table_id,
                    metadata={"title": table.title, "range": table.range, "headers": table.headers, "merged_ranges": table.merged_ranges},
                )
            )
        for visual in visual_assets:
            blocks.append(
                evidence_block(
                    block_id=f"evidence:{visual.visual_id}",
                    kind="embedded_visual",
                    source=visual.source,
                    metadata={
                        "visual_id": visual.visual_id,
                        "kind": visual.kind,
                        "delivery_status": visual.delivery_status,
                        **visual.metadata,
                    },
                )
            )
        document = finalize_intermediate_evidence(
            IntermediateDocument(
                source_type="pdf",
                file_name=Path(file_name).name,
                parser=parser_name,
                tables=intermediate_tables,
                blocks=blocks,
                visual_assets=visual_assets,
            ),
            content,
        )
        standard_result = standardize_tables(document)
        candidate_pages = [
            PdfCandidatePage(
                page_number=page_number,
                statement_type=statement_type,
                signals=signals,
                table_strategy="camelot",
                extraction_status=status_by_page.get(page_number, "not_run"),
            )
            for page_number, statement_type, signals in candidates
        ]
        return PdfIntakeResult(
            intermediate=document,
            standard=standard_result.standard,
            validation=[*issues, *standard_result.validation],
            pdf=PdfExtractionSummary(
                document_kind=document_kind,
                page_count=len(pypdf_pages),
                text_characters=sum(len(page.strip()) for page in pypdf_pages),
                page_locator=locator,
                mineru_status=mineru_status,
                candidate_pages=candidate_pages,
                page_inspections=page_inspections,
                vision_rendered_pages=len(visual_assets),
                vision_rendered_page_numbers=rendered_page_numbers,
                vision_next_page_start=next_page_start,
            ),
        )
