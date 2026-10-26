"""Exercise result processing through real registry/history I/O without model loads."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.tools.model_registry import registry_io, verify_models


class TestVerificationRunTracking(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.history_path = self.root / "history.json"
        self.models_path = self.root / "models.json"
        self.legacy = {"model_id": "legacy", "verified_date": "2026-01-01", "extra": 7}
        self.history_path.write_text(json.dumps({"records": [self.legacy]}))
        self.models_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "model_id": "model",
                            "architecture_id": "GPT2LMHeadModel",
                            "status": 1,
                            "note": "Prior full verification",
                            "phase1_score": 100.0,
                        }
                    ]
                }
            )
        )
        for name, value in (
            ("_VERIFICATION_HISTORY_PATH", self.history_path),
            ("_SUPPORTED_MODELS_PATH", self.models_path),
        ):
            patcher = patch.object(registry_io, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(verify_models, "_CHECKPOINT_PATH", self.root / "checkpoint.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.progress = verify_models.VerificationProgress(tested=["model"])

    def result(self, phase, passed=True, severity=None, name="test", message="result"):
        return BenchmarkResult(
            name=name,
            severity=severity or BenchmarkSeverity.INFO,
            message=message,
            passed=passed,
            phase=phase,
        )

    def record(self, results, phases=None, error=None, arch="GPT2LMHeadModel"):
        verify_models._record_verification_result(
            "model",
            arch,
            phases if phases is not None else [1, 2, 3, 4],
            results,
            error,
            self.progress,
            True,
        )
        data = json.loads(self.history_path.read_text())
        self.assertEqual(data["records"][0], self.legacy)
        self.assertEqual(len(data["records"]), 2)
        return registry_io.load_verification_history().records[-1]

    def test_full_success(self):
        record = self.record([self.result(p) for p in (1, 2, 3, 4)])
        self.assertEqual(record.status, 1)
        self.assertEqual(record.phases, [1, 2, 3, 4])
        self.assertEqual(self.progress.verified, ["model"])
        self.assertTrue(registry_io.load_verification_history().is_verified("model"))

    def test_threshold_failure(self):
        record = self.record([self.result(1), self.result(1, False, name="forward_pass_logits")])
        self.assertEqual(record.status, 3)
        self.assertIn("P1=50.0%", record.notes)
        self.assertFalse(registry_io.load_verification_history().is_verified("model"))
        self.assertEqual(json.loads(self.models_path.read_text())["models"][0]["status"], 3)

    def test_exception_and_oom(self):
        for error in ("load failure", "CUDA out of memory", ""):
            with self.subTest(error=error):
                self.history_path.write_text(json.dumps({"records": [self.legacy]}))
                record = self.record([], error=error)
                self.assertEqual(record.status, 3)
                self.assertTrue(record.notes)

    def test_no_results_or_all_skipped_cannot_verify(self):
        for results in ([], [self.result(1, severity=BenchmarkSeverity.SKIPPED)]):
            with self.subTest(results=results):
                self.history_path.write_text(json.dumps({"records": [self.legacy]}))
                record = self.record(results)
                self.assertEqual(record.status, 2)
                self.assertFalse(registry_io.load_verification_history().is_verified("model"))

    def test_partial_success_does_not_claim_full_verification(self):
        record = self.record([self.result(2)], phases=[2])
        self.assertEqual(record.status, 0)
        self.assertEqual(record.phases, [2])
        self.assertFalse(registry_io.load_verification_history().is_verified("model"))
        self.assertEqual(self.progress.verified, [])
        self.assertEqual(self.progress.partial, ["model"])
        self.assertEqual(json.loads(self.models_path.read_text())["models"][0]["status"], 1)

    def test_partial_failure_preserves_unrequested_registry_scores(self):
        record = self.record([self.result(1, False, name="load_bridge")], phases=[4])
        self.assertEqual(record.status, 3)
        self.assertEqual(record.phases, [4])
        model = json.loads(self.models_path.read_text())["models"][0]
        self.assertEqual(model["phase1_score"], 100.0)
        self.assertEqual(model["status"], 1)

    def test_core_success(self):
        record = self.record([self.result(1), self.result(4)], phases=[1, 4])
        self.assertEqual(record.status, 1)
        self.assertIn("Core verification", record.notes)
        self.assertEqual(record.phases, [1, 4])

    def test_core_cannot_override_failure_in_another_requested_phase(self):
        record = self.record(
            [self.result(1), self.result(2, False), self.result(4)], phases=[1, 2, 4]
        )
        self.assertEqual(record.status, 3)
        self.assertEqual(self.progress.verified, [])

    def test_missing_core_correctness_results_cannot_verify(self):
        record = self.record([self.result(2), self.result(4)])
        self.assertEqual(record.status, 3)
        self.assertIn("Missing", record.notes)

    def test_missing_audio_results_cannot_verify(self):
        record = self.record([self.result(1)], phases=[1, 8], arch="HubertModel")
        self.assertEqual(record.status, 3)
        self.assertIn("8", record.notes)

    def test_skip_is_recorded_even_when_registry_success_is_preserved(self):
        verify_models._skip_model(
            "model", "GPT2LMHeadModel", "memory limit", self.progress, True, phases=[1, 4]
        )
        history = registry_io.load_verification_history()
        self.assertEqual(history.records[-1].status, 2)
        self.assertEqual(history.records[-1].phases, [1, 4])
        self.assertFalse(history.is_verified("model"))
        self.assertEqual(json.loads(self.models_path.read_text())["models"][0]["status"], 1)

    def test_error_notes_are_sanitized(self):
        record = self.record([], error="failed with hf_abcdefghijklmnopqrstuvwx")
        self.assertNotIn("hf_abcdefghijklmnopqrstuvwx", record.notes)
        self.assertIn("HF_TOKEN", record.notes)

    def test_progress_backward_compatibility(self):
        restored = verify_models.VerificationProgress.from_dict({"verified": ["old"]})
        self.assertEqual(restored.partial, [])
        restored.partial.append("model")
        self.assertEqual(
            verify_models.VerificationProgress.from_dict(restored.to_dict()).partial, ["model"]
        )

    def test_partial_exception_records_failure_without_overwriting_registry_status(self):
        record = self.record([], phases=[2], error="benchmark exception")
        self.assertEqual(record.status, 3)
        self.assertEqual(json.loads(self.models_path.read_text())["models"][0]["status"], 1)

    def test_skip_updates_unverified_registry_and_history(self):
        data = json.loads(self.models_path.read_text())
        data["models"][0]["status"] = 0
        self.models_path.write_text(json.dumps(data))
        verify_models._skip_model(
            "model",
            "GPT2LMHeadModel",
            "config unavailable",
            self.progress,
            True,
            phases=[1, 2, 3, 4],
        )
        self.assertEqual(registry_io.load_verification_history().records[-1].status, 2)
        self.assertEqual(json.loads(self.models_path.read_text())["models"][0]["status"], 2)

    def test_low_text_quality_remains_non_gating(self):
        result = self.result(4, passed=False)
        result.details = {"score": 20.0}
        record = self.record([self.result(1), result], phases=[1, 4])
        self.assertEqual(record.status, 1)
        self.assertIn("low text quality", record.notes)

    def test_missing_multimodal_results_cannot_verify(self):
        record = self.record(
            [self.result(1), self.result(4)],
            phases=[1, 4, 7],
            arch="LlavaForConditionalGeneration",
        )
        self.assertEqual(record.status, 3)
        self.assertIn("7", record.notes)
