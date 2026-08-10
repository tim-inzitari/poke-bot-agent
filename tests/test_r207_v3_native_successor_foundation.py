"""Hermetic tests for the fail-closed r207 V3 native successor boundary."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from poke_bot.engine_rebuild.rtp_planner_successor_v3 import (
    EXPECTED_V3_EXPORTS,
    FORBIDDEN_V3_EXPORTS,
    RTP_PLANNER_V3_ABI_VERSION,
    RtpPlannerSuccessorV3Error,
    abi_contract,
    abi_sha256,
    canonical_digest,
    validate_native_exports,
)
from poke_bot.engine_rebuild.rtp_planner_successor_v3_build import (
    ENGINE_PATCH_ALLOWLIST,
    PATCHSET_SCHEMA,
    PrivateV3BuildError,
    apply_allowlisted_patchset,
    build_material,
    file_digest,
    load_patchset,
    source_manifest,
    verify_manifest_unchanged,
    verify_upstream_preimages,
)


def _write(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _private_source(tmp_path: Path) -> Path:
    source = tmp_path / ".private" / "engine-source"
    _write(source / "State.h", "before\n")
    _write(source / "Game.h", "unchanged\n")
    return source


def _diff(target: str, before: str, after: str) -> str:
    return f"--- a/{target}\n+++ b/{target}\n@@ -1 +1 @@\n-{before}\n+{after}\n"


def _patchset(
    tmp_path: Path,
    *,
    target: str,
    preimage: str,
    patch_contents: str,
    allowed_targets: list[str] | None = None,
) -> Path:
    patch = _write(tmp_path / "change.patch", patch_contents)
    payload = {
        "schema": PATCHSET_SCHEMA,
        "allowed_targets": allowed_targets if allowed_targets is not None else [target],
        "patches": [
            {
                "target": target,
                "preimage_sha256": preimage,
                "patch": patch.name,
            }
        ],
    }
    return _write(tmp_path / "patchset.json", json.dumps(payload, sort_keys=True))


def _digest_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_abi_contract_is_opaque_fresh_and_default_deny() -> None:
    contract = abi_contract()

    assert contract["version"] == RTP_PLANNER_V3_ABI_VERSION
    assert contract["fresh_native_worker_required"] is True
    assert contract["handles"]["no_native_pointer_round_trip"] is True
    assert contract["policy_visibility"] == "controlled-selection-only"
    assert contract["terminal_visibility"] == "result-only-through-node-info"
    assert contract["transition_policy"] == (
        "dynamic-provenance-clean-exact-only; default-deny-audit-required"
    )
    assert contract["chance_policy"] == "one-explicitly-armed-single-fair-coin-only"
    assert contract["operations"]["chance_outcome"] == "RtpPlannerV3ChanceOutcome"
    assert set(contract["expected_exports"]) == EXPECTED_V3_EXPORTS
    assert abi_sha256() == canonical_digest(contract)


def test_static_export_gate_rejects_public_and_unreviewed_interfaces() -> None:
    assert validate_native_exports(EXPECTED_V3_EXPORTS) == EXPECTED_V3_EXPORTS

    with pytest.raises(RtpPlannerSuccessorV3Error, match="forbidden"):
        validate_native_exports((*EXPECTED_V3_EXPORTS, "SearchBegin"))
    with pytest.raises(RtpPlannerSuccessorV3Error, match="unreviewed"):
        validate_native_exports((*EXPECTED_V3_EXPORTS, "RtpPlannerV3SecretBridge"))
    with pytest.raises(RtpPlannerSuccessorV3Error, match="missing"):
        validate_native_exports(EXPECTED_V3_EXPORTS - {"RtpPlannerV3ExpandAction"})
    assert "SearchBegin" in FORBIDDEN_V3_EXPORTS


def test_preimage_mismatch_is_rejected_before_private_staging(tmp_path: Path) -> None:
    source = _private_source(tmp_path)
    patchset_path = _patchset(
        tmp_path / "patches",
        target="State.h",
        preimage=_digest_text("different upstream bytes\n"),
        patch_contents=_diff("State.h", "before", "after"),
    )
    patchset = load_patchset(patchset_path)

    with pytest.raises(PrivateV3BuildError, match="upstream preimage mismatch"):
        verify_upstream_preimages(source, patchset)


def test_patch_outside_static_allowlist_never_reaches_staging(tmp_path: Path) -> None:
    patchset_path = _patchset(
        tmp_path / "patches",
        target="Api.h",
        preimage=_digest_text("before\n"),
        patch_contents=_diff("Api.h", "before", "after"),
        allowed_targets=["Api.h"],
    )

    with pytest.raises(PrivateV3BuildError, match="outside the engine allowlist"):
        load_patchset(patchset_path)


def test_patch_may_touch_exactly_one_declared_source_file(tmp_path: Path) -> None:
    source = _private_source(tmp_path)
    before = file_digest(source / "State.h")
    patchset_path = _patchset(
        tmp_path / "patches",
        target="State.h",
        preimage=before,
        patch_contents=_diff("State.h", "before", "after"),
    )
    patchset = load_patchset(patchset_path)
    upstream = verify_upstream_preimages(source, patchset)
    stage = tmp_path / ".private" / "stage"
    shutil.copytree(source, stage)

    patched = apply_allowlisted_patchset(stage, patchset)

    assert (stage / "State.h").read_text(encoding="utf-8") == "after\n"
    assert upstream["source_tree_sha256"] != patched["source_tree_sha256"]
    assert (stage / "Game.h").read_text(encoding="utf-8") == "unchanged\n"


def test_tampered_staging_manifest_is_detected_before_publication(tmp_path: Path) -> None:
    source = _private_source(tmp_path)
    before = file_digest(source / "State.h")
    patchset_path = _patchset(
        tmp_path / "patches",
        target="State.h",
        preimage=before,
        patch_contents=_diff("State.h", "before", "after"),
    )
    patchset = load_patchset(patchset_path)
    stage = tmp_path / ".private" / "stage"
    shutil.copytree(source, stage)
    patched = apply_allowlisted_patchset(stage, patchset)
    _write(stage / "Game.h", "tampered\n")

    with pytest.raises(PrivateV3BuildError, match="manifest changed"):
        verify_manifest_unchanged(stage, patched)


def test_multi_target_diff_is_rejected_even_when_the_declaration_is_allowlisted(
    tmp_path: Path,
) -> None:
    patchset_path = _patchset(
        tmp_path / "patches",
        target="State.h",
        preimage=_digest_text("before\n"),
        patch_contents=(
            _diff("State.h", "before", "after")
            + _diff("Game.h", "unchanged", "changed")
        ),
    )

    with pytest.raises(PrivateV3BuildError, match="exactly one file pair"):
        load_patchset(patchset_path)


def test_build_material_binds_abi_patch_overlay_compiler_and_upstream(tmp_path: Path) -> None:
    source = _private_source(tmp_path)
    upstream = source_manifest(source)
    material = build_material(
        upstream_manifest=upstream,
        patchset_manifest={"schema": "test", "sha256": _digest_text("patch")},
        overlay_identity={"sha256": _digest_text("overlay"), "bytes": 7, "mode": 0o444},
        compiler_identity={
            "path": "/private/compiler",
            "sha256": _digest_text("compiler"),
            "bytes": 8,
            "version": "test-c++",
        },
    )
    changed_flags = build_material(
        upstream_manifest=upstream,
        patchset_manifest={"schema": "test", "sha256": _digest_text("patch")},
        overlay_identity={"sha256": _digest_text("overlay"), "bytes": 7, "mode": 0o444},
        compiler_identity={
            "path": "/private/compiler",
            "sha256": _digest_text("compiler"),
            "bytes": 8,
            "version": "test-c++",
        },
        compile_flags=("-std=c++20", "-DCHANGED"),
    )

    assert material["abi_sha256"] == abi_sha256()
    assert material["expected_exports"] == sorted(EXPECTED_V3_EXPORTS)
    assert canonical_digest(material) != canonical_digest(changed_flags)


def test_reviewed_patchset_binds_only_dynamic_provenance_seams() -> None:
    root = Path(__file__).resolve().parents[1]
    patchset = load_patchset(root / "engine_patches/r207_v3/patchset.json")

    assert {patch["target"] for patch in patchset["patches"]} == {
        "CardMove.h",
        "EffectInstant.h",
        "EffectProc.h",
        "Game.h",
        "SelectProc.h",
        "State.h",
    }
    assert patchset["manifest"]["sha256"].startswith("sha256:")
    patch_text = {
        patch["target"]: patch["patch_file"].read_text(encoding="utf-8")
        for patch in patchset["patches"]
    }
    assert "PlannerTransitionAudit" in patch_text["Game.h"]
    assert "plannerCardIsPrivate" in patch_text["State.h"]
    assert "plannerRandom" in patch_text["SelectProc.h"]
    assert "markNonCoinRandom" in patch_text["CardMove.h"]
    assert "markNonCoinRandom" in patch_text["EffectProc.h"]
    assert "markNonCoinRandom" in patch_text["EffectInstant.h"]


def test_native_overlay_requires_fresh_worker_and_has_no_legacy_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "engine_patches/r207_v3/RtpPlannerSuccessorArenaV3.cpp").read_text(
        encoding="utf-8"
    )

    assert "requires a fresh engine process" in source
    assert "RtpPlannerV3ExpandAction" in source
    assert "ApplyExactSelection" in source
    assert "plannerAudit.begin" in source
    assert "BuildForcedSingleCoinChild" in source
    assert "RtpPlannerV3ChanceOutcome" in source
    assert "kBoundaryFiniteChance" in source
    for forbidden in (
        "ApiData",
        "ApiSelect",
        "ApiGetBattleData",
        "SetBattleData",
        "SearchBegin",
        "SearchStep",
        "SearchEnd",
        "RtpPairingSnapshot",
    ):
        assert forbidden not in source
    assert ENGINE_PATCH_ALLOWLIST == {
        "CardMove.h",
        "EffectInstant.h",
        "EffectProc.h",
        "Game.h",
        "SelectProc.h",
        "State.h",
        "TargetList.h",
    }
