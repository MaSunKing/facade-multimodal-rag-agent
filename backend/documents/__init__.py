"""Domain-neutral document evidence contracts.

The format-specific parsers live under :mod:`backend.document_parsing`. This
package provides the stable, domain-neutral output contract used by retrieval,
citations and dataset adapters.
"""

from backend.documents.evidence_v2 import EvidenceDocumentV2, convert_intermediate_to_v2
from backend.documents.training_sample import TrainingSample

__all__ = ["EvidenceDocumentV2", "TrainingSample", "convert_intermediate_to_v2"]
