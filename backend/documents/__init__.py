"""Domain-neutral document evidence contracts and format-specific source locations."""

__all__ = ["EvidenceDocumentV2", "TrainingSample", "convert_intermediate_to_v2"]


def __getattr__(name):
    """Keep parser-first imports independent of the Evidence projection cycle."""
    if name == "TrainingSample":
        from backend.documents.training_sample import TrainingSample
        return TrainingSample
    if name in {"EvidenceDocumentV2", "convert_intermediate_to_v2"}:
        from backend.documents import evidence_v2
        return getattr(evidence_v2, name)
    raise AttributeError(name)
