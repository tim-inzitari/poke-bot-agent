"""Regression coverage for the fail-closed Elmo engine-evidence verifier."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import socket
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/run_alakazam_simulator_rules_r298_engine_verification.py"


def _module():
    spec = importlib.util.spec_from_file_location("r298_engine_verifier_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class R298EngineVerificationTests(unittest.TestCase):
    def test_manifest_binds_current_rev5_goal_and_full_case_inventory(self) -> None:
        verifier = _module()
        manifest = verifier.engine_manifest()
        self.assertEqual(manifest["goal_revision"], 5)
        self.assertEqual(
            manifest["goal_gateway_sha256"],
            "sha256:7a829abebd348d0ffdf0a73c8b559fe9c799af3d3aff49a64efdfa85a08051b6",
        )
        self.assertEqual(
            manifest["goal_contract_sha256"],
            "sha256:dbbd4dbcc057b631d61fa867e45c393d594550b3b45f306f465b6ee5b4428891",
        )
        self.assertEqual(len(manifest["case_names"]), 18)
        self.assertFalse(manifest["fixture_only_is_engine_evidence"])
        self.assertEqual(
            manifest["execution_identity_required"]["container_execution_permitted"],
            False,
        )

    def test_default_command_is_manifest_only_and_never_runs_or_authorizes_engine(self) -> None:
        verifier = _module()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = verifier.main([])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "manifest_only_no_engine_execution")
        self.assertNotIn("passed", payload)

    def test_local_or_missing_envelope_fails_closed_before_any_engine_authority(self) -> None:
        verifier = _module()
        with self.assertRaises(ValueError):
            verifier.verify_engine_envelope(
                PROJECT_ROOT / "does-not-exist.json",
                artifact_root=PROJECT_ROOT,
            )
        if socket.gethostname() != verifier.CANONICAL_HOSTNAME:
            identity = {
                "execution_host_role": "elmo",
                "canonical_execution_hostname": verifier.CANONICAL_HOSTNAME,
                "execution_hostname": verifier.CANONICAL_HOSTNAME,
                "execution_fqdn": "truenas.example.invalid",
                "host_verification": verifier.HOST_VERIFICATION,
                "container_execution_permitted": False,
            }
            with self.assertRaisesRegex(ValueError, "Elmo-only"):
                verifier._verify_live_elmo_identity(identity)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
