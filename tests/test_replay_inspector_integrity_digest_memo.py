"""Regression coverage for request-time inspector integrity digest memoization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import replay_inspector.server as inspector_server
from replay_inspector.config import InspectorConfig
from replay_inspector.server import InspectorApplication, _IntegrityDigestMemo


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_file_digest_memo_restats_and_rehashes_on_signature_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    artifact = root / "submitted-runtime.tar.gz"
    artifact.write_bytes(b"first immutable bytes")
    memo = _IntegrityDigestMemo()
    calls = 0
    original = inspector_server.sha256_file

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(inspector_server, "sha256_file", counted)
    first = memo.file_digest(artifact, roots=(root,))
    second = memo.file_digest(artifact, roots=(root,))

    assert first is not None
    assert second == first
    assert calls == 1

    # A same-spelling replacement is not a hit: the current request's stat
    # signature changes, so the result is rehashed before it can pass a gate.
    artifact.write_bytes(b"replacement has a different size")
    replaced = memo.file_digest(artifact, roots=(root,))
    assert replaced is not None
    assert replaced[1] == _digest(artifact)
    assert replaced[1] != first[1]
    assert calls == 2


def test_file_digest_memo_rechecks_containment_on_a_cached_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    artifact = root / "bundle.tar.gz"
    artifact.write_bytes(b"inside")
    outside = tmp_path / "outside.tar.gz"
    outside.write_bytes(b"outside")
    memo = _IntegrityDigestMemo()
    calls = 0
    original = inspector_server.sha256_file

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(inspector_server, "sha256_file", counted)
    assert memo.file_digest(artifact, roots=(root,)) is not None
    assert calls == 1

    artifact.unlink()
    artifact.symlink_to(outside)
    # The old cached digest is not usable after exact path containment changes.
    assert memo.file_digest(artifact, roots=(root,)) is None
    assert calls == 1


def test_source_tree_digest_memo_rehashes_after_member_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    package = root / "poke_bot"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    memo = _IntegrityDigestMemo()
    calls = 0
    original = inspector_server.sha256_source_tree

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(inspector_server, "sha256_source_tree", counted)
    first = memo.source_tree_digest(root, roots=(root,))
    second = memo.source_tree_digest(root, roots=(root,))

    assert first is not None
    assert second == first
    assert calls == 1

    module.write_text("VALUE = 2\n", encoding="utf-8")
    replaced = memo.source_tree_digest(root, roots=(root,))
    assert replaced is not None
    assert replaced[1] != first[1]
    assert calls == 2


def test_baseline_game_gate_deduplicates_shared_bundle_runtime_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same artifact path is revalidated but never rehashed repeatedly."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    shared = artifact_root / "submission-and-runtime.tar.gz"
    shared.write_bytes(b"one exact immutable package")
    digest = _digest(shared)
    artifact = SimpleNamespace(expected_sha256=digest, resolved_path=shared)
    receipt = SimpleNamespace(
        artifact=artifact,
        runtime_source_tree_sha256="sha256:" + "a" * 64,
    )
    provenance = SimpleNamespace(
        checkpoint=artifact,
        bundle=artifact,
        runtime_package=artifact,
        runtime_parity_receipt=receipt,
        matchup_tree=None,
    )
    application = InspectorApplication(
        InspectorConfig(
            replay_root=tmp_path / "archive",
            rollout_root=tmp_path / "rollouts",
            artifact_roots=(artifact_root,),
            web_root=Path(__file__).resolve().parents[1] / "replay_inspector" / "web",
            game_trace_cache_enabled=False,
        )
    )
    submission = SimpleNamespace(provenance=provenance)
    entry = SimpleNamespace()
    replay = {"steps": []}
    stage = SimpleNamespace(step_index=4, factorized_stage=0)
    monkeypatch.setattr(
        application,
        "_replay_with_seat",
        lambda *_args, **_kwargs: (submission, entry, replay, "sha256:replay", 0),
    )
    monkeypatch.setattr(
        application,
        "_owner_decision_stages",
        lambda *_args, **_kwargs: [stage],
    )
    monkeypatch.setattr(application, "_trace_analysis_reason", lambda *_args: None)
    calls = 0
    original = inspector_server.sha256_file

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(inspector_server, "sha256_file", counted)
    first = application._prepare_baseline_game_materialization(
        77,
        88,
        4,
        0,
        include_setup_model_forward=True,
    )
    second = application._prepare_baseline_game_materialization(
        77,
        88,
        4,
        0,
        include_setup_model_forward=True,
    )

    assert first is not None
    assert second is not None
    assert calls == 1

    shared.write_bytes(b"changed package bytes invalidate every old digest")
    assert (
        application._prepare_baseline_game_materialization(
            77,
            88,
            4,
            0,
            include_setup_model_forward=True,
        )
        is None
    )
    assert calls == 2
