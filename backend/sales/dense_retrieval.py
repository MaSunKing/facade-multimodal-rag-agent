"""Local Qwen embedding and reranking helpers for private RAG retrieval.

No model or document is sent to a remote API.  Models load lazily so document
ingestion can use the GPU and then release it before the sales Copilot starts.
"""

from __future__ import annotations

import os
import gc
import threading
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"
RERANKER_MODEL_DIR = ROOT / "models" / "Qwen3-Reranker-0.6B"
RETRIEVAL_INSTRUCTION = (
    "Given a customer's Chinese facade-material question, retrieve passages that "
    "directly support an accurate answer about products, construction methods, "
    "standards, drawings, or project cases."
)

# Transformers model construction repeatedly allocates native CPU/CUDA memory.
# Re-loading both retrieval models for every question eventually fragments a
# small workstation GPU and can terminate the process without a Python
# traceback.  Keep one read-only instance of each model and serialize inference.
RETRIEVAL_INFERENCE_LOCK = threading.RLock()

# The workstation has one 16 GB GPU.  Retrieval may use it while the large
# generation model is cold, but the 0.6B embedding/reranking models must never
# remain resident beside Qwen3-VL-8B.  These references are managed explicitly
# (instead of functools.lru_cache) so a lifecycle transition can deterministically
# drop them under the same lock that serialises inference.
_embedding_model: LocalQwenEmbedding | None = None  # type: ignore[name-defined]
_reranker_model: LocalQwenReranker | None = None  # type: ignore[name-defined]
_generation_gpu_reserved = False
_lifecycle_generation = 0
_last_lifecycle_transition = "initial"
_last_lifecycle_error: str | None = None
_last_cleanup_warnings: list[str] = []


def local_device() -> str:
    requested = os.getenv("RAG_RETRIEVAL_DEVICE", "").strip().lower()
    if requested in {"cpu", "cuda"}:
        return requested if requested == "cpu" or torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def detailed_query(query: str) -> str:
    return f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery:{query}"


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), sequence_lengths
    ]


class LocalQwenEmbedding:
    """Minimal Transformers implementation of Qwen3-Embedding."""

    def __init__(self, model_dir: Path = EMBEDDING_MODEL_DIR, device: str | None = None) -> None:
        self.model_dir = model_dir
        self.device = device or local_device()
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Local embedding model was not found: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, padding_side="left")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.model_dir, dtype=dtype).to(self.device).eval()

    def encode(self, texts: Iterable[str], *, query: bool, batch_size: int = 8, max_length: int = 1024) -> np.ndarray:
        rows = [detailed_query(str(text)) if query else str(text) for text in texts]
        vectors: list[np.ndarray] = []
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs)
                embedding = F.normalize(
                    last_token_pool(output.last_hidden_state, inputs["attention_mask"]), p=2, dim=1
                )
            vectors.append(embedding.float().cpu().numpy())
        return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 1024), dtype=np.float32)


class LocalQwenReranker:
    """Official yes/no-logit reranking protocol for Qwen3-Reranker."""

    def __init__(self, model_dir: Path = RERANKER_MODEL_DIR, device: str | None = None) -> None:
        self.model_dir = model_dir
        self.device = device or local_device()
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Local reranker model was not found: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, dtype=dtype).to(self.device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
            'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            '<|im_start|>user\n'
        )
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def score(self, query: str, documents: Iterable[str], *, batch_size: int = 4, max_length: int = 2048) -> list[float]:
        pairs = [
            f"<Instruct>: {RETRIEVAL_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
            for document in documents
        ]
        scores: list[float] = []
        available_length = max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset : offset + batch_size]
            encoded = self.tokenizer(
                batch, padding=False, truncation="longest_first", return_attention_mask=False,
                max_length=available_length,
            )
            encoded["input_ids"] = [
                self.prefix_tokens + item + self.suffix_tokens for item in encoded["input_ids"]
            ]
            inputs = self.tokenizer.pad(encoded, padding=True, return_tensors="pt", max_length=max_length)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, :]
                yes_no = torch.stack((logits[:, self.token_false_id], logits[:, self.token_true_id]), dim=1)
                scores.extend(torch.log_softmax(yes_no, dim=1)[:, 1].exp().float().cpu().tolist())
        return scores


def _target_device_locked() -> str:
    """Return the only device retrieval models may use in the current phase."""

    return "cpu" if _generation_gpu_reserved else local_device()


def _cleanup_runtime_memory_locked() -> list[str]:
    """Return allocator cleanup warnings without making the service unusable."""

    warnings: list[str] = []
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # allocator cleanup differs between CUDA builds
            warnings.append(f"cuda_empty_cache:{type(exc).__name__}")
        try:
            torch.cuda.ipc_collect()
        except Exception as exc:  # unsupported by some CUDA/Windows runtimes
            warnings.append(f"cuda_ipc_collect:{type(exc).__name__}")
    return warnings


def _release_cached_models_locked() -> dict[str, object]:
    """Drop all cached retrieval models while inference is exclusively locked."""

    global _embedding_model, _reranker_model, _last_cleanup_warnings
    released_devices = sorted(
        {
            str(model.device)
            for model in (_embedding_model, _reranker_model)
            if model is not None
        }
    )
    released_count = int(_embedding_model is not None) + int(_reranker_model is not None)
    released_embedding = _embedding_model
    released_reranker = _reranker_model
    _embedding_model = None
    _reranker_model = None
    # Remove the final cache-owned references before asking CUDA to return its
    # free blocks.  A caller cannot still be using either object because every
    # encode/score operation holds RETRIEVAL_INFERENCE_LOCK.
    del released_embedding, released_reranker
    _last_cleanup_warnings = _cleanup_runtime_memory_locked()
    return {
        "released_model_count": released_count,
        "released_devices": released_devices,
        "cleanup_warnings": list(_last_cleanup_warnings),
    }


def prepare_for_generation_model() -> dict[str, object]:
    """Reserve the GPU for Qwen3-VL and force subsequent retrieval onto CPU.

    The transition is atomic relative to retrieval inference.  If preparation
    itself fails, the prior routing mode is restored so a failed generation
    load cannot strand retrieval on the wrong device.
    """

    global _generation_gpu_reserved, _lifecycle_generation
    global _last_lifecycle_transition, _last_lifecycle_error
    with RETRIEVAL_INFERENCE_LOCK:
        if _generation_gpu_reserved:
            return retrieval_runtime_status(_lock_held=True)
        previous = _generation_gpu_reserved
        try:
            _generation_gpu_reserved = True
            cleanup = _release_cached_models_locked()
        except Exception as exc:
            _generation_gpu_reserved = previous
            _last_lifecycle_transition = "prepare_failed_rolled_back"
            _last_lifecycle_error = f"{type(exc).__name__}: {exc}"
            raise
        _lifecycle_generation += 1
        _last_lifecycle_transition = "generation_gpu_reserved"
        _last_lifecycle_error = None
        return {**retrieval_runtime_status(_lock_held=True), **cleanup}


def restore_after_generation_model() -> dict[str, object]:
    """Clear CPU retrieval caches and restore the configured retrieval device.

    Restoration remains lazy: after the 8B model is unloaded, no retrieval
    model is loaded until the next query.  This keeps idle GPU/CPU memory low.
    """

    global _generation_gpu_reserved, _lifecycle_generation
    global _last_lifecycle_transition, _last_lifecycle_error
    with RETRIEVAL_INFERENCE_LOCK:
        if not _generation_gpu_reserved:
            return retrieval_runtime_status(_lock_held=True)
        try:
            cleanup = _release_cached_models_locked()
        except Exception as exc:
            # The large model has already gone away, so restore the configured
            # route even if an allocator-specific cleanup hook failed.  Cached
            # references are cleared before those hooks run.
            _generation_gpu_reserved = False
            _lifecycle_generation += 1
            _last_lifecycle_transition = "generation_gpu_restore_cleanup_failed"
            _last_lifecycle_error = f"{type(exc).__name__}: {exc}"
            raise
        _generation_gpu_reserved = False
        _lifecycle_generation += 1
        _last_lifecycle_transition = "configured_device_restored"
        _last_lifecycle_error = None
        return {**retrieval_runtime_status(_lock_held=True), **cleanup}


def retrieval_runtime_status(*, _lock_held: bool = False) -> dict[str, object]:
    """Expose lifecycle state without loading either retrieval model."""

    def snapshot() -> dict[str, object]:
        return {
            "configured_device": local_device(),
            "effective_device": _target_device_locked(),
            "generation_gpu_reserved": _generation_gpu_reserved,
            "embedding_model_loaded": _embedding_model is not None,
            "embedding_model_device": str(_embedding_model.device) if _embedding_model is not None else None,
            "reranker_model_loaded": _reranker_model is not None,
            "reranker_model_device": str(_reranker_model.device) if _reranker_model is not None else None,
            "lifecycle_generation": _lifecycle_generation,
            "last_transition": _last_lifecycle_transition,
            "last_error": _last_lifecycle_error,
            "cleanup_warnings": list(_last_cleanup_warnings),
        }

    if _lock_held:
        return snapshot()
    with RETRIEVAL_INFERENCE_LOCK:
        return snapshot()


def _discard_device_mismatch_locked(target_device: str) -> None:
    loaded = [model for model in (_embedding_model, _reranker_model) if model is not None]
    if any(str(model.device) != target_device for model in loaded):
        _release_cached_models_locked()


def get_embedding_model() -> LocalQwenEmbedding:
    global _embedding_model
    with RETRIEVAL_INFERENCE_LOCK:
        target_device = _target_device_locked()
        _discard_device_mismatch_locked(target_device)
        if _embedding_model is None:
            _embedding_model = LocalQwenEmbedding(device=target_device)
        return _embedding_model


def get_reranker_model() -> LocalQwenReranker:
    global _reranker_model
    with RETRIEVAL_INFERENCE_LOCK:
        target_device = _target_device_locked()
        _discard_device_mismatch_locked(target_device)
        if _reranker_model is None:
            _reranker_model = LocalQwenReranker(device=target_device)
        return _reranker_model
