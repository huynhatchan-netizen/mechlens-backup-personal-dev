"""Regression tests for verification outcomes and legacy history."""

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
    def record(self, status=1, architecture="A"):
        return VerificationRecord("model", date(2026, 1, 1), architecture, status=status)

    def test_failure_supersedes_success(self):
        history = VerificationHistory([self.record(), self.record(3)])
        self.assertFalse(history.is_verified("model"))
        self.assertEqual(history.get_record("model").status, 3)

    def test_invalidation_does_not_resurrect_success(self):
        history = VerificationHistory([self.record(), self.record()])
        history.invalidate("model", "regression")
        self.assertFalse(history.is_verified("model"))

    def test_score_only_preserves_outcome(self):
        for status in (0, 1, 2, 3):
            history = VerificationHistory([self.record(status), self.record(None)])
            self.assertEqual(history.is_verified("model"), status == 1)

    def test_architecture_scoping(self):
        history = VerificationHistory([self.record(1, "A"), self.record(3, "B")])
        self.assertTrue(history.is_verified("model", "A"))
        self.assertFalse(history.is_verified("model", "B"))
        self.assertIsNone(history.get_record("model", "C"))

    def test_legacy_and_scores_round_trip(self):
        legacy = {"model_id": "model", "verified_date": "2026-01-01", "notes": "unchanged"}
        record = VerificationRecord.from_dict(legacy)
        self.assertEqual(record.status, 1)
        record.phase_scores = {1: 0.0, 7: None}
        restored = VerificationRecord.from_dict(record.to_dict())
        self.assertEqual(restored, record)
        self.assertEqual(restored.notes, legacy["notes"])

    def test_append_preserves_existing_records(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            legacy = self.record().to_dict()
            legacy.pop("status")
            legacy.pop("phase_scores")
            path.write_text(json.dumps({"records": [legacy]}))
            with patch.object(registry_io, "_VERIFICATION_HISTORY_PATH", path), patch.object(
                registry_io, "_get_tl_version", return_value=None
            ):
                registry_io.add_verification_record(
                    "model", "A", status=3, phase_scores={1: 50.0}
                )
                history = registry_io.load_verification_history()
            self.assertEqual(len(history.records), 2)
            for key, value in legacy.items():
                self.assertEqual(history.records[0].to_dict()[key], value)
            self.assertFalse(history.is_verified("model"))
            self.assertEqual(history.records[1].phase_scores, {1: 50.0})

    def test_validation(self):
        for status in (None, 0, 1, 2, 3):
            data = VerificationHistory([self.record(status)]).to_dict()
            self.assertTrue(validate_verification_history(data).valid)
        for status in (True, "1", 4):
            data = VerificationHistory([self.record()]).to_dict()
            data["records"][0]["status"] = status
            self.assertFalse(validate_verification_history(data).valid)
