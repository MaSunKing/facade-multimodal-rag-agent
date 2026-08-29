"""Evidence Document V2 compatibility projection.

V1 remains the format-parser-facing contract. This module deterministically
projects a V1 ``IntermediateDocument`` into a typed, domain-neutral evidence
representation for retrieval and evaluation. It never mutates V1 and never
promotes OCR/VLM candidates to canonical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from backend.document_parsing.ingestion import IntermediateDocument, IntermediateTable, SourcePointer


class CoordinateSystemV2(BaseModel):
    bbox_format: Literal["xyxy_source", "xyxy_normalized"] = "xyxy_source"
    origin: Literal["top_left"] = "top_left"
    page_index_base: Literal[1] = 1


class ExcelRangeLocationV2(BaseModel):
    location_type: Literal["excel_range"] = "excel_range"
    sheet_name: str
    range: str | None = None
    supporting_cells: list[str] = Field(default_factory=list)


class PageRegionLocationV2(BaseModel):
    location_type: Literal["page_region"] = "page_region"
    page_number: int = Field(ge=1)
    raw_bbox: tuple[float, float, float, float] | None = None
    normalized_bbox: tuple[float, float, float, float] | None = None
    coordinate_system: CoordinateSystemV2 = Field(default_factory=CoordinateSystemV2)


class DocumentElementLocationV2(BaseModel):
    location_type: Literal["document_element"] = "document_element"
    section_id: str | None = None
    section_title: str | None = None
    paragraph_index: int | None = Field(default=None, ge=1)
    table_id: str | None = None


class TextSpanLocationV2(BaseModel):
    location_type: Literal["text_span"] = "text_span"
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class ImageRegionLocationV2(BaseModel):
    location_type: Literal["image_region"] = "image_region"
    image_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    raw_bbox: tuple[float, float, float, float] | None = None
    normalized_bbox: tuple[float, float, float, float] | None = None


class GenericLocationV2(BaseModel):
    location_type: Literal["generic"] = "generic"
    section_id: str | None = None


EvidenceLocationV2 = Annotated[
    ExcelRangeLocationV2
    | PageRegionLocationV2
    | DocumentElementLocationV2
    | TextSpanLocationV2
    | ImageRegionLocationV2
    | GenericLocationV2,
    Field(discriminator="location_type"),
]


class EvidenceSourceV2(BaseModel):
    source_type: Literal["text", "excel", "pdf", "word", "image", "html", "xml"]
    file_name: str | None = None
    parser: str | None = None
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    location: EvidenceLocationV2


class EvidenceAnnotationV2(BaseModel):
    source: Literal["parser_extracted", "official_dataset_gold", "model_candidate", "human_verified_model_candidate"]
    verified: bool = False
    model_name: str | None = None
    original_confidence: float | None = Field(default=None, ge=0, le=1)
    edited: bool = False


class EvidenceCellV2(BaseModel):
    cell_id: str
    coordinate: str
    row_index: int = Field(ge=1)
    column_index: int = Field(ge=1)
    value: str | int | float | bool | None = None
    display_text: str | None = None
    formula: str | None = None
    cached_value: str | int | float | bool | None = None
    numeric_candidates: list[dict[str, Any]] = Field(default_factory=list)


class BlockElementV2(BaseModel):
    element_type: Literal["block"] = "block"
    element_id: str
    legacy_element_id: str
    block_kind: str
    source: EvidenceSourceV2
    text: str | None = None
    table_id: str | None = None
    numeric_candidates: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    annotation: EvidenceAnnotationV2 = Field(default_factory=lambda: EvidenceAnnotationV2(source="parser_extracted"))


class TableElementV2(BaseModel):
    element_type: Literal["table"] = "table"
    element_id: str
    legacy_element_id: str
    parent_element_id: str | None = None
    section_id: str | None = None
    title: str | None = None
    source: EvidenceSourceV2
    header_row_index: int | None = Field(default=None, ge=1)
    header_range: str | None = None
    scope: str = "unknown"
    scope_confidence: float = Field(default=0.0, ge=0, le=1)
    headers: list[str] = Field(default_factory=list)
    rows: list[list[EvidenceCellV2]] = Field(default_factory=list)
    merged_ranges: list[str] = Field(default_factory=list)
    annotation: EvidenceAnnotationV2 = Field(default_factory=lambda: EvidenceAnnotationV2(source="parser_extracted"))


class VisualElementV2(BaseModel):
    element_type: Literal["visual"] = "visual"
    element_id: str
    legacy_element_id: str
    visual_kind: str
    source: EvidenceSourceV2
    media_type: str | None = None
    width_px: int | None = Field(default=None, ge=1)
    height_px: int | None = Field(default=None, ge=1)
    sha256: str | None = None
    delivery_status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    annotation: EvidenceAnnotationV2 = Field(default_factory=lambda: EvidenceAnnotationV2(source="parser_extracted"))


EvidenceElementV2 = Annotated[
    BlockElementV2 | TableElementV2 | VisualElementV2,
    Field(discriminator="element_type"),
]


class EvidenceFileV2(BaseModel):
    file_name: str | None = None
    file_sha256: str | None = None
    file_size_bytes: int | None = Field(default=None, ge=0)
    source_type: str
    parser: str


class EvidenceContextChunkV2(BaseModel):
    chunk_id: str
    kind: Literal["document_index", "evidence"]
    text: str
    character_count: int = Field(ge=1)
    source_refs: list[str] = Field(default_factory=list)


class EvidenceDocumentV2(BaseModel):
    evidence_schema_version: Literal["evidence-document-v2"] = "evidence-document-v2"
    location_schema_version: Literal["location-v2"] = "location-v2"
    document_id: str
    revision_id: str = "pending"
    parent_revision_id: str | None = None
    evidence_snapshot_hash: str = "pending"
    file: EvidenceFileV2
    elements: list[EvidenceElementV2] = Field(default_factory=list)
    structure_profiles: list[dict[str, Any]] = Field(default_factory=list)
    model_context_chunks: list[EvidenceContextChunkV2] = Field(default_factory=list)


def _global_id(document_id: str, legacy_id: str) -> str:
    return f"{document_id}/{legacy_id}"


def _paragraph_index(section_id: str | None) -> int | None:
    if not section_id:
        return None
    tail = section_id.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() and int(tail) >= 1 else None


def _normalised_bbox(
    bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if bbox is None or not all(math.isfinite(value) for value in bbox):
        return None
    if all(0.0 <= value <= 1.0 for value in bbox):
        return bbox
    return None


def source_location_v2(
    source: SourcePointer,
    *,
    range_override: str | None = None,
    table_id: str | None = None,
    image_id: str | None = None,
) -> EvidenceLocationV2:
    """Convert a sparse V1 source pointer without inventing coordinates."""

    if source.source_type == "excel" and source.sheet_name:
        return ExcelRangeLocationV2(
            sheet_name=source.sheet_name,
            range=range_override or source.cell,
            supporting_cells=[source.cell] if source.cell else [],
        )
    if source.source_type == "pdf" and source.page_number is not None:
        normalized = _normalised_bbox(source.bounding_box)
        return PageRegionLocationV2(
            page_number=source.page_number,
            raw_bbox=source.bounding_box,
            normalized_bbox=normalized,
            coordinate_system=CoordinateSystemV2(
                bbox_format="xyxy_normalized" if normalized is not None else "xyxy_source"
            ),
        )
    if source.source_type == "image":
        return ImageRegionLocationV2(
            image_id=image_id or source.section_id or source.file_name,
            page_number=source.page_number,
            raw_bbox=source.bounding_box,
            normalized_bbox=_normalised_bbox(source.bounding_box),
        )
    if source.source_type == "text" and source.text_span is not None:
        return TextSpanLocationV2(start=source.text_span[0], end=source.text_span[1])
    if source.source_type in {"word", "html", "xml", "text"}:
        return DocumentElementLocationV2(
            section_id=source.section_id,
            section_title=source.section_title,
            paragraph_index=_paragraph_index(source.section_id),
            table_id=table_id,
        )
    return GenericLocationV2(section_id=source.section_id)


def source_reference_v2(
    source: SourcePointer,
    *,
    range_override: str | None = None,
    table_id: str | None = None,
    image_id: str | None = None,
) -> EvidenceSourceV2:
    return EvidenceSourceV2(
        source_type=source.source_type,
        file_name=source.file_name,
        parser=source.parser,
        extraction_confidence=source.extraction_confidence,
        location=source_location_v2(
            source,
            range_override=range_override,
            table_id=table_id,
            image_id=image_id,
        ),
    )


def _table_source(document: IntermediateDocument, table: IntermediateTable) -> SourcePointer:
    return SourcePointer(
        source_type=document.source_type,
        file_name=document.file_name,
        sheet_name=table.sheet_name,
        page_number=table.page_number,
        section_id=table.section_id or table.table_id,
        section_title=table.title,
        parser=table.parser or document.parser,
        extraction_confidence=table.extraction_confidence,
    )


def _canonical_hash(value: BaseModel) -> str:
    payload = value.model_dump(
        mode="json",
        exclude={"revision_id", "evidence_snapshot_hash"},
    )

    def clean(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return {"non_finite_float": str(item)}
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items()}
        if isinstance(item, list):
            return [clean(child) for child in item]
        return item

    canonical = json.dumps(
        clean(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def convert_intermediate_to_v2(document: IntermediateDocument) -> EvidenceDocumentV2:
    """Project complete V1 canonical evidence into a deterministic V2 snapshot."""

    if not document.document_id:
        raise ValueError("IntermediateDocument must be finalised before V2 conversion.")
    document_id = document.document_id
    elements: list[EvidenceElementV2] = []

    for block in document.blocks:
        elements.append(
            BlockElementV2(
                element_id=_global_id(document_id, block.block_id),
                legacy_element_id=block.block_id,
                block_kind=block.kind,
                source=source_reference_v2(block.source, table_id=block.table_id),
                text=block.text,
                table_id=_global_id(document_id, block.table_id) if block.table_id else None,
                numeric_candidates=[item.model_dump(mode="json") for item in block.numeric_candidates],
                metadata=block.metadata,
            )
        )

    for table in document.tables:
        table_source = _table_source(document, table)
        table_element_id = _global_id(document_id, table.table_id)
        elements.append(
            TableElementV2(
                element_id=table_element_id,
                legacy_element_id=table.table_id,
                parent_element_id=(
                    _global_id(document_id, table.parent_table_id)
                    if table.parent_table_id
                    else None
                ),
                section_id=table.section_id,
                title=table.title,
                source=source_reference_v2(
                    table_source,
                    range_override=table.range,
                    table_id=table_element_id,
                ),
                header_row_index=table.header_row_index,
                header_range=table.header_range,
                scope=table.scope,
                scope_confidence=table.scope_confidence,
                headers=table.headers,
                rows=[
                    [
                        EvidenceCellV2(
                            cell_id=f"{table_element_id}/cell/{cell.coordinate}",
                            coordinate=cell.coordinate,
                            row_index=cell.row_index,
                            column_index=cell.column_index,
                            value=cell.value,
                            display_text=cell.display_text,
                            formula=cell.formula,
                            cached_value=cell.cached_value,
                            numeric_candidates=[
                                item.model_dump(mode="json")
                                for item in cell.numeric_candidates
                            ],
                        )
                        for cell in row
                    ]
                    for row in table.rows
                ],
                merged_ranges=table.merged_ranges,
            )
        )

    for visual in document.visual_assets:
        elements.append(
            VisualElementV2(
                element_id=_global_id(document_id, visual.visual_id),
                legacy_element_id=visual.visual_id,
                visual_kind=visual.kind,
                source=source_reference_v2(
                    visual.source,
                    image_id=_global_id(document_id, visual.visual_id),
                ),
                media_type=visual.media_type,
                width_px=visual.width_px,
                height_px=visual.height_px,
                sha256=visual.sha256,
                delivery_status=visual.delivery_status,
                metadata=visual.metadata,
            )
        )

    snapshot = EvidenceDocumentV2(
        document_id=document_id,
        file=EvidenceFileV2(
            file_name=document.file_name,
            file_sha256=document.file_sha256,
            file_size_bytes=document.file_size_bytes,
            source_type=document.source_type,
            parser=document.parser,
        ),
        elements=elements,
        structure_profiles=[item.model_dump(mode="json") for item in document.structure_profiles],
        model_context_chunks=[
            EvidenceContextChunkV2(**item.model_dump(mode="json"))
            for item in document.model_context_chunks
        ],
    )
    digest = _canonical_hash(snapshot)
    return snapshot.model_copy(
        update={
            "evidence_snapshot_hash": digest,
            "revision_id": f"rev-{digest[:16]}",
        }
    )
