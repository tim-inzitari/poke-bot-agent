"""Fail-closed evidence tests for the private RTP pairing snapshot ABI."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from poke_bot.engine_rebuild.rtp_pairing_snapshot import (
    BUILD_SCHEMA,
    CAPABILITY_SCHEMA,
    PairingArtifactSet,
    PROBE_SCHEMA,
    RTPPairingSnapshotError,
    RtpPairingSnapshotEngine,
    emit_true_rng_pairing_capability,
    emit_true_rng_pairing_probe,
    file_digest,
    frozen_file_identity,
    snapshot_abi_contract,
    snapshot_abi_sha256,
    verify_pairing_case_binding,
)


def _write(path: Path, value: bytes | dict) -> Path:
    if isinstance(value, dict):
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_bytes(value)
    return path


def _digest_bytes(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifacts(
    tmp_path: Path, *, engine_path: Path | None = None
) -> PairingArtifactSet:
    engine = engine_path or _write(
        tmp_path / "libcg_rtp_pairing_snapshot.so", b"private engine"
    )
    source = _write(
        tmp_path / "source-manifest.json",
        {
            "schema": "poke_bot.recursive_turn_planner.true_rng_pairing_source_manifest/v1",
            "source_tree_sha256": _digest_bytes("source tree"),
            "files": [],
        },
    )
    patch = _write(tmp_path / "RtpPairingSnapshotExport.cpp", b"private overlay")
    engine.chmod(0o555)
    source.chmod(0o444)
    patch.chmod(0o444)
    receipt = {
        "schema": BUILD_SCHEMA,
        "status": "success",
        "engine_artifact_sha256": file_digest(engine),
        "source_artifact_sha256": file_digest(source),
        "patch_artifact_sha256": file_digest(patch),
        "canonical_abi_sha256": snapshot_abi_sha256(),
        "engine_artifact": frozen_file_identity(engine),
        "source_artifact": frozen_file_identity(source),
        "patch_artifact": frozen_file_identity(patch),
    }
    build = _write(tmp_path / "build-receipt.json", receipt)
    build.chmod(0o444)
    return PairingArtifactSet.from_paths(
        engine_path=engine,
        source_manifest_path=source,
        patch_path=patch,
        build_receipt_path=build,
    )


def _passing_probe() -> dict:
    return {
        "passed": True,
        "initial_snapshot_fingerprint_sha256": _digest_bytes("snapshot"),
        "initial_snapshot_fingerprint_bytes": 4096,
        "deterministic_transcript_sha256": _digest_bytes("transcript"),
        "transcript_steps": 4,
        "duplicate_restore_independent_handles": True,
        "device_rand_false_verified": True,
        "requested_seed_only_rejected": True,
        "delayed_restore_transcript_passed": True,
        "cross_process_restore_passed": True,
        "delayed_restore_seconds": 0.02,
    }


def test_abi_contract_requires_private_init_serialized_restore_and_no_timer() -> None:
    contract = snapshot_abi_contract()

    assert contract["version"] == 2
    assert contract["initialize_symbol"] == "RtpPairingSnapshotInitialize"
    assert contract["start_symbol"] == "RtpPairingBattleStartSeededOut"
    assert contract["observation_symbol"] == "RtpPairingSnapshotGetBattleJsonOut"
    assert contract["restore_serialized_symbol"] == "RtpPairingSnapshotRestoreSerialized"
    assert contract["requires_device_rand_false"] is True
    assert contract["requires_time_limit_zero"] is True
    assert contract["requires_pristine_process_initialization"] is True
    assert contract["serialization_compatibility"] == "exact_engine_artifact_only"
    assert contract["serialized_restore_requires_sealed_sha256"] is True


def test_capability_attestation_cross_binds_all_private_artifacts(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    probe_path = emit_true_rng_pairing_probe(
        output_path=tmp_path / "probe.json",
        artifacts=artifacts,
        deterministic_probe=_passing_probe(),
        divergent_policy_true_pairing_passed=True,
        all_arms_restored_or_replayed=True,
    )
    capability_path = emit_true_rng_pairing_capability(
        output_path=tmp_path / "capability.json",
        artifacts=artifacts,
        probe_path=probe_path,
    )
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    capability = json.loads(capability_path.read_text(encoding="utf-8"))

    assert probe["schema"] == PROBE_SCHEMA
    assert probe["cross_process_restore_passed"] is True
    assert probe["delayed_restore_transcript_passed"] is True
    assert capability["schema"] == CAPABILITY_SCHEMA
    assert capability["status"] == "available"
    assert capability["supported_rng_kinds"] == ["snapshot"]
    assert capability["abi"]["canonical_abi_sha256"] == snapshot_abi_sha256()
    assert capability["probe"]["sha256"] == file_digest(probe_path)
    assert stat.S_IMODE(probe_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(capability_path.stat().st_mode) == 0o444


def test_probe_refuses_missing_cross_process_restore_proof(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    probe = _passing_probe()
    probe["cross_process_restore_passed"] = False

    with pytest.raises(RTPPairingSnapshotError, match="cross-process"):
        emit_true_rng_pairing_probe(
            output_path=tmp_path / "probe.json",
            artifacts=artifacts,
            deterministic_probe=probe,
            divergent_policy_true_pairing_passed=True,
            all_arms_restored_or_replayed=True,
        )


def test_immutable_probe_cannot_be_reused_after_becoming_writable(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    path = tmp_path / "probe.json"
    emit_true_rng_pairing_probe(
        output_path=path,
        artifacts=artifacts,
        deterministic_probe=_passing_probe(),
        divergent_policy_true_pairing_passed=True,
        all_arms_restored_or_replayed=True,
    )
    path.chmod(0o644)

    with pytest.raises(RTPPairingSnapshotError, match="mode 0444"):
        emit_true_rng_pairing_probe(
            output_path=path,
            artifacts=artifacts,
            deterministic_probe=_passing_probe(),
            divergent_policy_true_pairing_passed=True,
            all_arms_restored_or_replayed=True,
        )


def test_case_binding_requires_cell_identity_ordered_decks_and_exact_debug_seed(
    tmp_path: Path,
) -> None:
    deck0 = _write(tmp_path / "deck0.csv", b"1\n2\n")
    deck1 = _write(tmp_path / "deck1.csv", b"3\n4\n")
    deck0.chmod(0o444)
    deck1.chmod(0o444)
    binding = {
        "schema": "poke_bot.recursive_turn_planner.r198_pairing_case_binding/v1",
        "status": "sealed",
        "cell_id": "iono-seat0-rep0",
        "case_id": "iono-seat0-rep0-case",
        "opponent_id": "iono",
        "seat": 0,
        "replicate": 0,
        "debug_seed": 123,
        "ordered_deck_identities": [
            frozen_file_identity(deck0),
            frozen_file_identity(deck1),
        ],
        "cohort_identity": {"sha256": _digest_bytes("cohort")},
        "source_exclusion_identity": {"sha256": _digest_bytes("exclusion")},
    }
    binding_path = _write(tmp_path / "case-binding.json", binding)
    binding_path.chmod(0o444)

    identity, parsed = verify_pairing_case_binding(
        frozen_file_identity(binding_path), expected_debug_seed=123
    )

    assert identity["sha256"] == file_digest(binding_path)
    assert parsed["ordered_deck_identities"][0]["sha256"] == file_digest(deck0)
    with pytest.raises(RTPPairingSnapshotError, match="debug_seed"):
        verify_pairing_case_binding(
            frozen_file_identity(binding_path), expected_debug_seed=124
        )


class _FakeFunction:
    def __init__(self, result: object = 0) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *_: object) -> object:
        self.calls += 1
        return self.result


class _FakeLibrary:
    def __init__(self) -> None:
        self.BattleFinish = _FakeFunction()
        self.Select = _FakeFunction()
        self.RtpPairingSnapshotAbiVersion = _FakeFunction(2)
        self.RtpPairingSnapshotLastError = _FakeFunction(b"")
        self.RtpPairingSnapshotInitialize = _FakeFunction(0)
        self.RtpPairingBattleStartSeededOut = _FakeFunction()
        self.RtpPairingSnapshotGetBattleJsonOut = _FakeFunction()
        self.RtpPairingSnapshotCapture = _FakeFunction()
        self.RtpPairingSnapshotRestore = _FakeFunction()
        self.RtpPairingSnapshotRestoreSerialized = _FakeFunction()
        self.RtpPairingSnapshotRelease = _FakeFunction()
        self.RtpPairingSnapshotSerializedSize = _FakeFunction()
        self.RtpPairingSnapshotSerialize = _FakeFunction()
        self.RtpPairingSnapshotFingerprintSize = _FakeFunction()
        self.RtpPairingSnapshotFingerprint = _FakeFunction()


def test_serialized_restore_never_reaches_native_without_sealed_artifact(
    tmp_path: Path,
) -> None:
    library_path = _write(tmp_path / "fake-private-engine.so", b"not a real library")
    fake = _FakeLibrary()
    engine = RtpPairingSnapshotEngine(
        library_path, library=fake, initialize=False
    )
    engine.require_bound_artifacts(_artifacts(tmp_path, engine_path=library_path))
    sealed = _write(tmp_path / "sealed-snapshot.bin", b"sealed-private-snapshot")
    sealed.chmod(0o444)
    snapshot_artifact = {
        "path": str(sealed),
        "sha256": file_digest(sealed),
        "bytes": sealed.stat().st_size,
        "mode": 0o444,
    }

    with pytest.raises(RTPPairingSnapshotError, match="do not equal"):
        engine.restore_serialized_snapshot(
            b"opaque-private-snapshot",
            snapshot_artifact=snapshot_artifact,
        )
    assert fake.RtpPairingSnapshotRestoreSerialized.calls == 0
