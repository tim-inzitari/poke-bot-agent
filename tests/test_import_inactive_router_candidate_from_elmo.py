from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import import_inactive_router_candidate_from_elmo as module


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_importer_refuses_remote_checksum_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    tree = {
        "schema": "poke_bot.public_matchup_decision_tree/v1",
        "runtime_enabled": False,
        "targets": ["teal-mask-ogerpon-ex"],
    }
    tree_bytes = json.dumps(tree).encode()
    ready = {
        "schema": "poke_bot.public_matchup_decision_tree_receipt/v1",
        "runtime_enabled": False,
        "artifact_sha256": "sha256:not-the-tree",
    }

    def remote_bytes(_host: str, path: str) -> bytes:
        return (
            json.dumps(ready).encode()
            if path.endswith("PUBLIC_MATCHUP_TREE_READY.json")
            else tree_bytes
        )

    monkeypatch.setattr(module, "_remote_bytes", remote_bytes)
    args = argparse.Namespace(
        host="elmo",
        remote_root="/router-v44",
        tree_output=tmp_path / "tree.json",
        audit_output=tmp_path / "audit.json",
        roster=tmp_path / "roster.json",
        corpus_root=tmp_path / "corpus",
        promotion_receipt=tmp_path / "promotion.json",
        parent_receipt=None,
        ready_archetype=[],
        minimum_precision=0.93,
        minimum_weighted_support=10_000,
    )
    try:
        module.import_candidate(args)
    except RuntimeError as error:
        assert str(error) == "remote inactive-router receipt identity failed"
    else:
        raise AssertionError("checksum mismatch was accepted")


def test_install_immutable_is_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate.json"
    module._install_immutable(output, b"candidate")
    module._install_immutable(output, b"candidate")
    assert output.read_bytes() == b"candidate"
    try:
        module._install_immutable(output, b"different")
    except RuntimeError as error:
        assert "immutable artifact differs" in str(error)
    else:
        raise AssertionError("immutable candidate was replaced")
