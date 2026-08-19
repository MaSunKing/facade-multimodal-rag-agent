"""Request-only Word and image intake for the finance assistant.

The module intentionally separates safe file parsing from financial
interpretation.  DOCX paragraphs and tables can be deterministically mapped
with the same rules as Excel/text.  Images are accepted and validated, but do
not become financial facts until a controlled OCR table pipeline is enabled.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from PIL import Image
from pydantic import BaseModel, Field

from backend.document_parsing.ingestion import (
    FinanceIntakeError,
    EvidenceBlock,
    IntakeResult,
    IntermediateCell,
    IntermediateDocument,
    IntermediateTable,
    SourcePointer,
    StandardFinancialDocument,
    StandardFinancialFact,
    ValidationIssue,
    VisualAsset,
    _headers_for_rows,
    _issue,
    _validate_facts,
    evidence_block,
    finalize_intermediate_evidence,
    numeric_candidates_for_text,
    standardize_tables,
    standardize_text,
)


MAX_WORD_BYTES = 20 * 1024 * 1024
MAX_WORD_XML_BYTES = 12 * 1024 * 1024
MAX_WORD_TABLES = 200
MAX_WORD_CELLS = 10_000
MAX_WORD_VISUAL_ASSETS = 8
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_IMAGE_SUFFIXES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

_WORD_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


class WordExtractionSummary(BaseModel):
    paragraph_count: int = Field(ge=0)
    table_count: int = Field(ge=0)


class WordIntakeResult(IntakeResult):
    word: WordExtractionSummary


class ImageExtractionSummary(BaseModel):
    media_type: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str


class ImageIntakeResult(IntakeResult):
    image: ImageExtractionSummary


def _word_text(element: ElementTree.Element) -> str:
    return "".join(part.text or "" for part in element.iter(f"{_WORD_NS}t")).strip()


def _word_text_excluding_nested_tables(element: ElementTree.Element) -> str:
    """Read a Word table cell without flattening nested tables into it.

    A nested table is a distinct grid and must keep its own rows/cells.  The
    old recursive ``Element.iter`` behaviour made a complex Word document
    look like a handful of giant single-cell tables.
    """

    fragments: list[str] = []

    def visit(node: ElementTree.Element) -> None:
        for child in list(node):
            if child.tag == f"{_WORD_NS}tbl":
                continue
            if child.tag == f"{_WORD_NS}t" and child.text:
                fragments.append(child.text)
            visit(child)

    visit(element)
    return "".join(fragments).strip()


def _word_revision_text(element: ElementTree.Element) -> str:
    """Retain deleted revision text separately from the visible paragraph."""

    tags = (f"{_WORD_NS}t", f"{_WORD_NS}delText")
    return "".join(part.text or "" for tag in tags for part in element.iter(tag)).strip()


def _word_part_blocks(
    *,
    root: ElementTree.Element,
    file_name: str,
    part_name: str,
    kind: Literal["header", "footer", "comment"],
) -> list[EvidenceBlock]:
    """Preserve non-body DOCX content rather than silently discarding it."""

    blocks: list[EvidenceBlock] = []
    for index, paragraph in enumerate(root.iter(f"{_WORD_NS}p"), start=1):
        text = _word_text(paragraph)
        if not text:
            continue
        blocks.append(
            evidence_block(
                block_id=f"word:{kind}:{part_name}:{index}",
                kind=kind,
                source=SourcePointer(
                    source_type="word",
                    file_name=file_name,
                    section_id=f"word:{kind}:{part_name}:{index}",
                    parser="docx-xml",
                    extraction_confidence=1.0,
                ),
                text=text,
                metadata={"part_name": part_name, "sequence": index},
            )
        )
    return blocks


def _word_revision_blocks(root: ElementTree.Element, file_name: str) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    revision_tags = ((f"{_WORD_NS}ins", "inserted"), (f"{_WORD_NS}del", "deleted"))
    for tag, revision_type in revision_tags:
        for index, revision in enumerate(root.iter(tag), start=1):
            text = _word_revision_text(revision)
            if not text:
                continue
            blocks.append(
                evidence_block(
                    block_id=f"word:revision:{revision_type}:{index}",
                    kind="revision",
                    source=SourcePointer(
                        source_type="word",
                        file_name=file_name,
                        section_id=f"word:revision:{revision_type}:{index}",
                        parser="docx-xml",
                        extraction_confidence=1.0,
                    ),
                    text=text,
                    metadata={"revision_type": revision_type},
                )
            )
    return blocks


def _word_table(table: ElementTree.Element, table_index: int) -> IntermediateTable:
    rows: list[list[IntermediateCell]] = []
    for row_index, row in enumerate(table.findall(f"{_WORD_NS}tr"), start=1):
        cells: list[IntermediateCell] = []
        for column_index, cell in enumerate(row.findall(f"{_WORD_NS}tc"), start=1):
            value = _word_text_excluding_nested_tables(cell)
            cells.append(
                IntermediateCell(
                    coordinate=f"word:t{table_index}:r{row_index}c{column_index}",
                    row_index=row_index,
                    column_index=column_index,
                    value=value or None,
                    display_text=value or None,
                    numeric_candidates=numeric_candidates_for_text(value),
                )
            )
        if cells:
            rows.append(cells)
    return IntermediateTable(
        table_id=f"word:table:{table_index}",
        title=f"Word 表格 {table_index}",
        sheet_name="Word 文档",
        parser="docx-xml",
        headers=_headers_for_rows(rows),
        rows=rows,
    )


def _word_visual_assets(file_name: str, media_parts: dict[str, bytes]) -> list[VisualAsset]:
    """Expose embedded raster images without claiming an in-document anchor.

    DOCX drawing anchors require relationship/layout interpretation, which is
    not reliably available from plain XML.  The part name remains an auditable
    source pointer and the local vision result is review-only.
    """

    assets: list[VisualAsset] = []
    for part_name, image_bytes in sorted(media_parts.items()):
        if len(assets) >= MAX_WORD_VISUAL_ASSETS:
            break
        media_type = _WORD_MEDIA_TYPES.get(Path(part_name).suffix.lower())
        if media_type is None:
            continue
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                image_format = image.format or Path(part_name).suffix.lstrip(".").upper()
        except Exception:
            continue
        source = SourcePointer(
            source_type="word",
            file_name=Path(file_name).name,
            section_id=f"word:media:{Path(part_name).name}",
            parser="docx-media",
            extraction_confidence=1.0,
        )
        assets.append(
            VisualAsset(
                visual_id=f"word:image:{len(assets) + 1}",
                kind="document_page",
                source=source,
                media_type=media_type,
                width_px=width,
                height_px=height,
                sha256=hashlib.sha256(image_bytes).hexdigest(),
                delivery_status="ready_for_vision",
                metadata={
                    "content_extraction": "vision_candidate_only",
                    "part_name": part_name,
                    "original_format": image_format,
                    "anchor_status": "part_only",
                },
                image_bytes=image_bytes,
            )
        )
    return assets


def parse_docx_to_intermediate(file_name: str, content: bytes) -> tuple[IntermediateDocument, WordExtractionSummary]:
    """Extract paragraphs and tables from an unencrypted modern Word document."""

    if Path(file_name).suffix.lower() != ".docx":
        raise FinanceIntakeError("Word 入口当前仅支持 .docx；请先将旧版 .doc 另存为 .docx 后上传。")
    if not content:
        raise FinanceIntakeError("上传的 Word 文件为空。")
    if len(content) > MAX_WORD_BYTES:
        raise FinanceIntakeError(f"Word 文件超过 {MAX_WORD_BYTES // (1024 * 1024)} MB 限制。")
    try:
        with ZipFile(BytesIO(content)) as archive:
            try:
                document_xml = archive.read("word/document.xml")
            except KeyError as exc:
                raise FinanceIntakeError("该文件不是可读取的 .docx 文档。") from exc
            auxiliary_parts = {
                name: archive.read(name)
                for name in archive.namelist()
                if name.startswith(("word/header", "word/footer")) or name == "word/comments.xml"
            }
            media_parts = {
                name: archive.read(name)
                for name in archive.namelist()
                if name.startswith("word/media/")
            }
    except BadZipFile as exc:
        raise FinanceIntakeError("无法读取该 Word 文件，请确认它是未加密且未损坏的 .docx。") from exc
    if len(document_xml) > MAX_WORD_XML_BYTES:
        raise FinanceIntakeError("Word 正文过大，请拆分后再上传。")
    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as exc:
        raise FinanceIntakeError("Word 文档 XML 无法解析。") from exc

    body = root.find(f"{_WORD_NS}body")
    if body is None:
        raise FinanceIntakeError("Word 文档没有可解析的正文。")
    paragraphs: list[str] = []
    tables: list[IntermediateTable] = []
    blocks: list[EvidenceBlock] = []
    table_cells = 0
    for element in list(body):
        if element.tag == f"{_WORD_NS}p":
            if text := _word_text(element):
                paragraphs.append(text)
                blocks.append(
                    evidence_block(
                        block_id=f"word:body:paragraph:{len(paragraphs)}",
                        kind="paragraph",
                        source=SourcePointer(
                            source_type="word",
                            file_name=Path(file_name).name,
                            section_id=f"word:body:paragraph:{len(paragraphs)}",
                            parser="docx-xml",
                            extraction_confidence=1.0,
                        ),
                        text=text,
                        metadata={"sequence": len(paragraphs)},
                    )
                )
        elif element.tag == f"{_WORD_NS}tbl":
            if len(tables) >= MAX_WORD_TABLES:
                raise FinanceIntakeError(f"Word 表格超过 {MAX_WORD_TABLES} 张限制，请拆分后上传。")
            table = _word_table(element, len(tables) + 1)
            table_cells += sum(len(row) for row in table.rows)
            if table_cells > MAX_WORD_CELLS:
                raise FinanceIntakeError(f"Word 表格单元格超过 {MAX_WORD_CELLS} 个限制，请拆分后上传。")
            tables.append(table)
            blocks.append(
                evidence_block(
                    block_id=table.table_id,
                    kind="table",
                    source=SourcePointer(
                        source_type="word",
                        file_name=Path(file_name).name,
                        section_id=table.table_id,
                        parser="docx-xml",
                        extraction_confidence=1.0,
                    ),
                    table_id=table.table_id,
                    metadata={"headers": table.headers, "row_count": len(table.rows), "merged_ranges": table.merged_ranges},
                )
            )

    # ``w:tbl`` can occur inside a cell of another table.  Preserve top-level
    # tables above, then add each nested grid once so it retains its own rows
    # and columns rather than being concatenated into its parent cell.
    top_level_table_ids = {id(element) for element in list(body) if element.tag == f"{_WORD_NS}tbl"}
    for element in root.iter(f"{_WORD_NS}tbl"):
        if id(element) in top_level_table_ids:
            continue
        if len(tables) >= MAX_WORD_TABLES:
            raise FinanceIntakeError(f"Word 表格超过 {MAX_WORD_TABLES} 张限制，请拆分后再上传。")
        table = _word_table(element, len(tables) + 1)
        table_cells += sum(len(row) for row in table.rows)
        if table_cells > MAX_WORD_CELLS:
            raise FinanceIntakeError(f"Word 表格单元格超过 {MAX_WORD_CELLS} 个限制，请拆分后上传。")
        tables.append(table)
        blocks.append(
            evidence_block(
                block_id=table.table_id,
                kind="table",
                source=SourcePointer(
                    source_type="word",
                    file_name=Path(file_name).name,
                    section_id=table.table_id,
                    parser="docx-xml-recursive-table",
                    extraction_confidence=1.0,
                ),
                table_id=table.table_id,
                metadata={"headers": table.headers, "row_count": len(table.rows), "merged_ranges": table.merged_ranges},
            )
        )

    blocks.extend(_word_revision_blocks(root, Path(file_name).name))
    for part_name, part_xml in auxiliary_parts.items():
        try:
            part_root = ElementTree.fromstring(part_xml)
        except ElementTree.ParseError:
            continue
        if part_name.startswith("word/header"):
            blocks.extend(_word_part_blocks(root=part_root, file_name=Path(file_name).name, part_name=part_name, kind="header"))
        elif part_name.startswith("word/footer"):
            blocks.extend(_word_part_blocks(root=part_root, file_name=Path(file_name).name, part_name=part_name, kind="footer"))
        else:
            blocks.extend(_word_part_blocks(root=part_root, file_name=Path(file_name).name, part_name=part_name, kind="comment"))

    visual_assets = _word_visual_assets(file_name, media_parts)
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
            source_type="word",
            file_name=Path(file_name).name,
            parser="docx-xml",
            tables=tables,
            raw_text="\n".join(paragraphs) or None,
            blocks=blocks,
            visual_assets=visual_assets,
        ),
        content,
    )
    return document, WordExtractionSummary(paragraph_count=len(paragraphs), table_count=len(tables))


def _deduplicate_facts(facts: list[StandardFinancialFact]) -> list[StandardFinancialFact]:
    unique: dict[tuple[str, str | None, float | None], StandardFinancialFact] = {}
    for fact in facts:
        unique.setdefault((fact.standard_item_code, fact.period, fact.value), fact)
    return list(unique.values())


def ingest_word(file_name: str, content: bytes) -> WordIntakeResult:
    """Return deterministic facts from DOCX text/tables without persisting the file."""

    document, summary = parse_docx_to_intermediate(file_name, content)
    table_result = standardize_tables(document)
    text_result = standardize_text(document) if document.raw_text else None
    facts = _deduplicate_facts([
        *table_result.standard.facts,
        *(text_result.standard.facts if text_result else []),
    ])
    issues: list[ValidationIssue] = [
        *[issue for issue in table_result.validation if issue.code != "no_mapped_financial_facts"],
        *([issue for issue in text_result.validation if issue.code != "no_mapped_financial_facts"] if text_result else []),
    ]
    if not facts:
        _issue(issues, "warning", "no_mapped_financial_facts", "未从 Word 文字或表格中识别到可确定映射的财务字段；已保留中间 JSON，需补充字段映射或客户确认。")
    _validate_facts(facts, issues)
    return WordIntakeResult(
        intermediate=document,
        standard=StandardFinancialDocument(
            company_name=text_result.standard.company_name if text_result else None,
            facts=facts,
        ),
        validation=issues,
        word=summary,
    )


def _image_media_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    media_type = _IMAGE_SUFFIXES.get(suffix)
    if media_type is None:
        allowed = "、".join(sorted(_IMAGE_SUFFIXES))
        raise FinanceIntakeError(f"不支持的图片格式。当前支持：{allowed}。")
    return media_type


def ingest_image(file_name: str, content: bytes) -> ImageIntakeResult:
    """Prepare an image for generic, review-only local visual transcription."""

    media_type = _image_media_type(file_name)
    if not content:
        raise FinanceIntakeError("上传的图片为空。")
    if len(content) > MAX_IMAGE_BYTES:
        raise FinanceIntakeError(f"图片超过 {MAX_IMAGE_BYTES // (1024 * 1024)} MB 限制。")
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            image_format = image.format or Path(file_name).suffix.lstrip(".").upper()
    except Exception as exc:
        raise FinanceIntakeError("图片损坏或无法解析。") from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise FinanceIntakeError("图片像素过大，请压缩后重试。")
    source = SourcePointer(
        source_type="image",
        file_name=Path(file_name).name,
        parser="pillow-metadata+local-vision-candidate",
        extraction_confidence=1.0,
    )
    visual = VisualAsset(
        visual_id="image:document:1",
        kind="document_page",
        source=source,
        media_type=media_type,
        width_px=width,
        height_px=height,
        sha256=hashlib.sha256(content).hexdigest(),
        delivery_status="ready_for_vision",
        metadata={"content_extraction": "vision_candidate_only", "original_format": image_format},
        image_bytes=content,
    )
    document = finalize_intermediate_evidence(
        IntermediateDocument(
            source_type="image",
            file_name=Path(file_name).name,
            parser="pillow-metadata+local-vision-candidate",
            tables=[],
            blocks=[
                evidence_block(
                    block_id="image:metadata:1",
                    kind="image_metadata",
                    source=source,
                    metadata={
                        "media_type": media_type,
                        "width": width,
                        "height": height,
                        "format": image_format,
                        "ocr_status": "vision_pending_confirmation",
                    },
                ),
                evidence_block(
                    block_id=f"evidence:{visual.visual_id}",
                    kind="embedded_visual",
                    source=source,
                    metadata={
                        "visual_id": visual.visual_id,
                        "kind": visual.kind,
                        "delivery_status": visual.delivery_status,
                        **visual.metadata,
                    },
                ),
            ],
            visual_assets=[visual],
        ),
        content,
    )
    issue = ValidationIssue(
        severity="warning",
        code="image_vision_pending_confirmation",
        message="图片已完成格式校验，将由本地视觉模型生成待确认的通用文字和表格候选；候选内容不会自动成为业务事实或最终 Excel 数据。",
        source=source,
    )
    return ImageIntakeResult(
        intermediate=document,
        standard=StandardFinancialDocument(),
        validation=[issue],
        image=ImageExtractionSummary(media_type=media_type, width=width, height=height, format=image_format),
    )


def is_supported_image_name(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in _IMAGE_SUFFIXES
