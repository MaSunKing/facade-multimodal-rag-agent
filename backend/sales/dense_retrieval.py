"""Local Qwen embedding and reranking helpers for private RAG retrieval.

No model or document is sent to a remote API.  Models load lazily so document
ingestion can use the GPU and then release it before the sales Copilot starts.
"""

from __future__ import annotations

import os
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
