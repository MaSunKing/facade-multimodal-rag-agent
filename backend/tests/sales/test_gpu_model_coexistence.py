from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend import app as facade_app
from backend.sales import dense_retrieval


class _FakeEmbedding:
    def __init__(self, *, device: str, **_: object) -> None:
        self.device = device


class _FakeReranker:
    def __init__(self, *, device: str, **_: object) -> None:
        self.device = device


class RetrievalGpuLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        with dense_retrieval.RETRIEVAL_INFERENCE_LOCK:
            dense_retrieval._embedding_model = None
            dense_retrieval._reranker_model = None
            dense_retrieval._generation_gpu_reserved = False
            dense_retrieval._lifecycle_generation = 0
            dense_retrieval._last_lifecycle_transition = "initial"
            dense_retrieval._last_lifecycle_error = None
            dense_retrieval._last_cleanup_warnings = []

    def tearDown(self) -> None:
        with dense_retrieval.RETRIEVAL_INFERENCE_LOCK:
            dense_retrieval._embedding_model = None
            dense_retrieval._reranker_model = None
            dense_retrieval._generation_gpu_reserved = False

    def test_generation_reservation_moves_lazy_retrieval_to_cpu_then_restores_config(self) -> None:
        with patch.object(dense_retrieval, "local_device", return_value="cuda"), patch.object(
            dense_retrieval, "LocalQwenEmbedding", _FakeEmbedding
        ), patch.object(dense_retrieval, "LocalQwenReranker", _FakeReranker), patch.object(
            dense_retrieval.torch.cuda, "is_available", return_value=False
        ):
            self.assertEqual(dense_retrieval.get_embedding_model().device, "cuda")
            self.assertEqual(dense_retrieval.get_reranker_model().device, "cuda")

            prepared = dense_retrieval.prepare_for_generation_model()
            self.assertTrue(prepared["generation_gpu_reserved"])
            self.assertEqual(prepared["released_model_count"], 2)
            self.assertEqual(prepared["effective_device"], "cpu")
            self.assertEqual(dense_retrieval.get_embedding_model().device, "cpu")
            self.assertEqual(dense_retrieval.get_reranker_model().device, "cpu")

            restored = dense_retrieval.restore_after_generation_model()
            self.assertFalse(restored["generation_gpu_reserved"])
            self.assertEqual(restored["released_model_count"], 2)
            self.assertEqual(restored["effective_device"], "cuda")
            self.assertFalse(restored["embedding_model_loaded"])
            self.assertFalse(restored["reranker_model_loaded"])

    def test_prepare_waits_for_active_retrieval_inference(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        transitioned = threading.Event()

        def inference() -> None:
            with dense_retrieval.RETRIEVAL_INFERENCE_LOCK:
                entered.set()
                release.wait(timeout=2)

        def transition() -> None:
            dense_retrieval.prepare_for_generation_model()
            transitioned.set()

        with patch.object(dense_retrieval.torch.cuda, "is_available", return_value=False):
            inference_thread = threading.Thread(target=inference)
            transition_thread = threading.Thread(target=transition)
            inference_thread.start()
            self.assertTrue(entered.wait(timeout=1))
            transition_thread.start()
            time.sleep(0.05)
            self.assertFalse(transitioned.is_set())
            release.set()
            inference_thread.join(timeout=1)
            transition_thread.join(timeout=1)
            self.assertTrue(transitioned.is_set())

    def test_prepare_failure_rolls_back_device_policy(self) -> None:
        with patch.object(
            dense_retrieval,
            "_release_cached_models_locked",
            side_effect=RuntimeError("synthetic cleanup failure"),
        ):
            with self.assertRaises(RuntimeError):
                dense_retrieval.prepare_for_generation_model()
        status = dense_retrieval.retrieval_runtime_status()
        self.assertFalse(status["generation_gpu_reserved"])
        self.assertEqual(status["last_transition"], "prepare_failed_rolled_back")
        self.assertIn("synthetic cleanup failure", str(status["last_error"]))


class GenerationModelRollbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = (
            facade_app._model,
            facade_app._processor,
            facade_app._tokenizer,
            facade_app._model_last_used_monotonic,
            facade_app.MODEL_PATH,
        )
        facade_app._model = None
        facade_app._processor = None
        facade_app._tokenizer = None
        facade_app._model_last_used_monotonic = None
        facade_app.MODEL_PATH = Path(".")

    def tearDown(self) -> None:
        (
            facade_app._model,
            facade_app._processor,
            facade_app._tokenizer,
            facade_app._model_last_used_monotonic,
            facade_app.MODEL_PATH,
        ) = self.previous

    def test_failed_8b_load_restores_retrieval_device_policy(self) -> None:
        processor = SimpleNamespace(tokenizer=object())
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.empty_cache"
        ), patch("transformers.AutoProcessor.from_pretrained", return_value=processor), patch(
            "transformers.BitsAndBytesConfig"
        ), patch(
            "transformers.Qwen3VLForConditionalGeneration.from_pretrained",
            side_effect=RuntimeError("synthetic 8B load failure"),
        ), patch(
            "backend.sales.dense_retrieval.prepare_for_generation_model"
        ) as prepare, patch(
            "backend.sales.dense_retrieval.restore_after_generation_model"
        ) as restore:
            with self.assertRaisesRegex(RuntimeError, "synthetic 8B load failure"):
                facade_app.load_model()

        prepare.assert_called_once_with()
        restore.assert_called_once_with()
        self.assertIsNone(facade_app._model)
        self.assertIsNone(facade_app._processor)
        self.assertIsNone(facade_app._tokenizer)

    def test_idle_8b_unload_restores_configured_retrieval_device(self) -> None:
        facade_app._model = object()
        facade_app._processor = object()
        facade_app._tokenizer = object()
        facade_app._model_last_used_monotonic = time.monotonic() - 120
        with patch.object(facade_app, "MODEL_IDLE_UNLOAD_SECONDS", 60), patch(
            "torch.cuda.is_available", return_value=False
        ), patch(
            "backend.sales.dense_retrieval.restore_after_generation_model"
        ) as restore:
            self.assertTrue(facade_app.unload_model_if_idle())

        restore.assert_called_once_with()
        self.assertIsNone(facade_app._model)
        self.assertIsNone(facade_app._processor)
        self.assertIsNone(facade_app._tokenizer)


if __name__ == "__main__":
    unittest.main()
