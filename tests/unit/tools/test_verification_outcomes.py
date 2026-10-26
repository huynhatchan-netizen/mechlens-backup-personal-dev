"""Regression tests for explicit outcomes and preservation of legacy history."""

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transformer_lens.tools.model_registry import registry_io
from transformer_lens.tools.model_registry.validate import validate_verification_history
from transformer_lens.tools.model_registry.verification import (
    VerificationHistory,
    VerificationRecord,
)


class TestVerificationOutcomes(unittest.TestCase):
    def record(self, status=None, architecture="A"):
        return VerificationRecord("model", date(2026, 1, 1), architecture, status=status)

    def test_explicit_outcomes(self):
        for status in (None, 0, 1, 2, 3):
            with self.subTest(status=status):
                history = VerificationHistory([self.record(status)])
                self.assertEqual(history.is_verified("model"), status == 1)

    def test_failure_supersedes_success(self):
        history = VerificationHistory([self.record(1), self.record(3)])
        self.assertFalse(history.is_verified("model"))

    def test_unknown_latest_does_not_resurrect_success(self):
        history = VerificationHistory([self.record(1), self.record()])
        self.assertFalse(history.is_verified("model"))

    def test_invalidation_does_not_resurrect_success(self):
        history = VerificationHistory([self.record(1), self.record(1)])
        history.invalidate("model", "regression")
        self.assertFalse(history.is_verified("model"))

    def test_architecture_scoping(self):
        history = VerificationHistory([self.record(1, "A"), self.record(3, "B")])
        self.assertTrue(history.is_verified("model", "A"))
        self.assertFalse(history.is_verified("model", "B"))
        self.assertFalse(history.is_verified("model", "C"))

    def test_legacy_outcome_is_unknown(self):
        legacy = {"model_id": "model", "verified_date": "2026-01-01", "notes": "ok"}
        record = VerificationRecord.from_dict(legacy)
        self.assertIsNone(record.status)
        self.assertFalse(VerificationHistory([record]).is_verified("model"))
        self.assertNotIn("status", record.to_dict())
        self.assertEqual(record.notes, "ok")

    def test_explicit_outcome_round_trip(self):
        for status in (0, 1, 2, 3):
            record = self.record(status)
            self.assertEqual(VerificationRecord.from_dict(record.to_dict()), record)

    def test_append_preserves_existing_records_and_metadata(self):
        legacy = {
            "model_id": "old",
            "verified_date": "2026-01-01",
            "notes": "original",
            "custom": {"nested": [1, None]},
        }
        invalidated = dict(legacy, invalidated=True, invalidation_reason="original reason")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(
                json.dumps({"records": [legacy, invalidated], "custom_header": {"version": 7}})
            )
            with patch.object(registry_io, "_VERIFICATION_HISTORY_PATH", path), patch.object(
                registry_io, "_get_tl_version", return_value=None
            ):
                registry_io.add_verification_record("model", "A", status=3)
                registry_io.add_verification_record("unknown", "A")
            data = json.loads(path.read_text())
            self.assertEqual(data["records"][:2], [legacy, invalidated])
            self.assertEqual(data["custom_header"], {"version": 7})
            self.assertEqual(data["records"][2]["status"], 3)
            self.assertNotIn("status", data["records"][3])
            self.assertIsNotNone(data["last_updated"])

    def test_new_history_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            with patch.object(registry_io, "_VERIFICATION_HISTORY_PATH", path), patch.object(
                registry_io, "_get_tl_version", return_value=None
            ):
                registry_io.add_verification_record("model", "A", status=1)
                self.assertTrue(registry_io.load_verification_history().is_verified("model"))

    def test_validation(self):
        for status in (None, 0, 1, 2, 3):
            data = VerificationHistory([self.record(status)]).to_dict()
            self.assertTrue(validate_verification_history(data).valid)
        for status in (True, "1", 4, -1):
            data = VerificationHistory([self.record()]).to_dict()
            data["records"][0]["status"] = status
            self.assertFalse(validate_verification_history(data).valid)

    def test_load_save_preserves_sparse_legacy_and_unknown_fields(self):
        data = {
            "custom_header": {"source": ["legacy", None]},
            "last_updated": "2026-01-01T00:00:00Z",
            "records": [
                {
                    "model_id": "legacy",
                    "verified_date": "2026-01-01",
                    "custom": {"scores": [1, None]},
                },
                {"model_id": "unknown", "verified_date": "2026-01-01", "status": None},
                self.record(1).to_dict(),
                self.record(3).to_dict(),
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_text(json.dumps(data))
            with patch.object(registry_io, "_VERIFICATION_HISTORY_PATH", path):
                history = registry_io.load_verification_history()
                self.assertFalse(history.is_verified("legacy"))
                self.assertFalse(history.is_verified("unknown"))
                self.assertFalse(history.is_verified("model"))
                registry_io.save_verification_history(history)
                self.assertEqual(json.loads(path.read_text()), data)
                self.assertFalse(registry_io.load_verification_history().is_verified("model"))

    def test_round_trip_missing_history_fields(self):
        for data in ({}, {"records": []}, {"last_updated": None}):
            self.assertEqual(VerificationHistory.from_dict(data).to_dict(), data)

    def test_loaded_record_mutations_survive_without_losing_extensions(self):
        data = self.record(1).to_dict()
        data["custom"] = {"keep": True}
        history = VerificationHistory.from_dict({"records": [data]})
        history.invalidate("model", "regression")
        restored = VerificationHistory.from_dict(history.to_dict())
        self.assertFalse(restored.is_verified("model"))
        self.assertEqual(restored.to_dict()["records"][0]["custom"], {"keep": True})
        self.assertEqual(restored.records[0].invalidation_reason, "regression")
        restored.records[0].status = None
        self.assertIsNone(VerificationHistory.from_dict(restored.to_dict()).records[0].status)

    def test_malformed_numeric_status_is_not_success(self):
        for value in (True, 1.0, "1"):
            data = self.record().to_dict()
            data["status"] = value
            history = VerificationHistory.from_dict({"records": [data]})
            self.assertFalse(history.is_verified("model"))

    def test_actual_history_load_save_preserves_all_records(self):
        original = registry_io._VERIFICATION_HISTORY_PATH.read_bytes()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            path.write_bytes(original)
            with patch.object(registry_io, "_VERIFICATION_HISTORY_PATH", path):
                history = registry_io.load_verification_history()
                before = {r.model_id: history.is_verified(r.model_id) for r in history.records}
                registry_io.save_verification_history(history)
                restored = registry_io.load_verification_history()
                self.assertEqual(json.loads(path.read_bytes()), json.loads(original))
                self.assertEqual(
                    {r.model_id: restored.is_verified(r.model_id) for r in restored.records}, before
                )

    def test_outcomes_and_architectures_survive_repeated_round_trips(self):
        for status in (None, 0, 1, 2, 3):
            for invalidated in (False, True):
                latest = self.record(status, "A")
                latest.invalidated = invalidated
                history = VerificationHistory(
                    [self.record(1, "A"), latest, self.record(1, "B")]
                )
                for _ in range(3):
                    history = VerificationHistory.from_dict(
                        json.loads(json.dumps(history.to_dict()))
                    )
                    self.assertEqual(
                        history.is_verified("model", "A"), status == 1 and not invalidated
                    )
                    self.assertTrue(history.is_verified("model", "B"))

    def test_requested_phases_round_trip_and_mutation(self):
        record = self.record(3)
        record.phases = [1, 4]
        loaded = VerificationHistory.from_dict(VerificationHistory([record]).to_dict())
        self.assertEqual(loaded.records[0].phases, [1, 4])
        loaded.records[0].phases.append(7)
        restored = VerificationHistory.from_dict(loaded.to_dict())
        self.assertEqual(restored.records[0].phases, [1, 4, 7])

    def test_requested_phases_validation(self):
        for phases in (None, [], [1, 2, 3, 4], [1, 7, 8]):
            data = VerificationHistory([self.record()]).to_dict()
            data["records"][0]["phases"] = phases
            self.assertTrue(validate_verification_history(data).valid)
        for phases in ("1", [True], [5], [1.0]):
            data = VerificationHistory([self.record()]).to_dict()
            data["records"][0]["phases"] = phases
            self.assertFalse(validate_verification_history(data).valid)
