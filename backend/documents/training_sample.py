"""Training-sample contract kept separate from runtime document evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LabelAvailability(BaseModel):
    answer: bool = False
    answer_page: bool = False
    bbox: bool = False
    element_id: bool = False
    reasoning_program: bool = False
    repair_action: bool = False


class AnnotationProvenance(BaseModel):
    source: Literal[
        "official_human_gold",
        "official_synthetic",
        "program_generated",
        "human_verified_model_candidate",
    ]
    dataset: str | None = None
    source_record_id: str | None = None
    source_document_id: str | None = None
    official_annotation_verified: bool = False
    license_reviewed: bool = False


class AdapterMapping(BaseModel):
    status: Literal[
        "not_attempted",
        "document_mapped",
        "page_mapped",
        "region_mapped",
        "element_mapped",
        "failed",
    ] = "not_attempted"
    verified: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    failure_reason: str | None = None
    # Dataset adapters may receive a question-local page window rather than a
    # proven global document order. These fields keep that distinction
    # explicit instead of presenting an arbitrary merged index as ground truth.
    document_view_id: str | None = None
    official_page_id: str | None = None
    local_page_index: int | None = Field(default=None, ge=0)
    view_page_number: int | None = Field(default=None, ge=1)
    canonical_page_number: int | None = Field(default=None, ge=1)
    order_status: Literal[
        "official_global_order",
        "canonical_order_derived",
        "official_window_order",
        "disconnected_component",
        "unavailable",
    ] | None = None


class TrainingVisualAssetReference(BaseModel):
    visual_asset_id: str
    evidence_id: str | None = None
    storage_ref: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(ge=1)
    media_type: str | None = None


class TrainingInputContext(BaseModel):
    input_modalities: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    selected_chunks: list[str] = Field(default_factory=list)
    image_refs: list[str] = Field(default_factory=list)
    visual_assets: list[TrainingVisualAssetReference] = Field(default_factory=list)


class CandidateSampling(BaseModel):
    strategy: str
    negative_evidence_ids: list[str] = Field(default_factory=list)
    deterministic_seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingSample(BaseModel):
    training_sample_schema_version: Literal["training-sample-v1"] = "training-sample-v1"
    sample_id: str
    document_id: str
    evidence_revision_id: str
    task_type: Literal[
        "evidence_selection",
        "grounded_document_qa",
        "answerability_refusal",
    ]
    question: str
    answer: str | None = None
    acceptable_answers: list[str] = Field(default_factory=list)
    answer_type: str | None = None
    answer_scale: str | None = None
    reasoning_program: str | None = None
    answerable: bool
    # ``evidence_ids`` is retained for existing consumers. New dataset
    # adapters should also populate the explicit candidate/selected fields so
    # retrieval input and gold supervision cannot be confused.
    evidence_ids: list[str] = Field(default_factory=list)
    candidate_evidence_ids: list[str] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    annotation_level: Literal["document", "page", "region", "element", "cell"] | None = None
    input_context: TrainingInputContext = Field(default_factory=TrainingInputContext)
    candidate_sampling: CandidateSampling | None = None
    label_availability: LabelAvailability = Field(default_factory=LabelAvailability)
    annotation_provenance: AnnotationProvenance
    adapter_mapping: AdapterMapping = Field(default_factory=AdapterMapping)
