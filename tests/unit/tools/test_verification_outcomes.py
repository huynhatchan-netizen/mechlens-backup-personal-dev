"""Regression tests for explicit outcomes and preservation of legacy history."""

import ast
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

    def test_full_run_writers_supply_explicit_outcomes(self):
        source = Path(registry_io.__file__).with_name("verify_models.py").read_text()
        tree = ast.parse(source)
        runner = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_models"
        )
        calls = [
            n
            for n in ast.walk(runner)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "add_verification_record"
        ]
        self.assertEqual(len(calls), 2)
        for call, expected in zip(sorted(calls, key=lambda n: n.lineno), (1, 3)):
            captured = []
            namespace = {
                "add_verification_record": lambda *args, **kwargs: captured.append(kwargs),
                "model_id": "model",
                "arch": "A",
                "note": "result",
                "STATUS_VERIFIED": 1,
                "STATUS_FAILED": 3,
                "_sanitize_note": lambda x: x,
            }
            exec(compile(ast.Expression(call), "verify_models.py", "eval"), namespace)
            self.assertEqual(captured[0]["status"], expected)
