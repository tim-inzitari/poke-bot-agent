from __future__ import annotations

import json
from pathlib import Path

from poke_bot import checkpoint
from scripts.run_post_alakazam_core_refresh import _materialize_contract


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_materialized_core_replaces_only_alakazam_teacher(tmp_path: Path) -> None:
    old_alakazam = tmp_path / "old-alakazam.pt"
    refreshed_alakazam = tmp_path / "refreshed-alakazam.pt"
    other = tmp_path / "other.pt"
    for path, body in (
        (old_alakazam, b"old"),
        (refreshed_alakazam, b"refreshed"),
        (other, b"other"),
    ):
        path.write_bytes(body)
    template = tmp_path / "template.json"
    _write(
        template,
        {
            "schema": "poke_bot.post_specialist_core_refresh_handoff/v1",
            "core_refresh": {
                "version": 14,
                "family": "/old/core14",
                "initialization": {"checkpoint": "/old/core9/model.pt", "checksum": "sha256:old"},
                "teachers": [
                    {
                        "specialist_id": "alakazam",
                        "mode": "frozen_inference_only",
                        "checkpoint": str(old_alakazam),
                        "checksum": checkpoint.checkpoint_digest(old_alakazam),
                    },
                    {
                        "specialist_id": "archaludon-ex",
                        "mode": "frozen_inference_only",
                        "checkpoint": str(other),
                        "checksum": checkpoint.checkpoint_digest(other),
                    },
                ],
            },
            "acceptance": {"regression_result": "/old/regression.json"},
            "core_failure_fallback": {"enabled": True},
        },
    )
    registry = tmp_path / "registry.json"
    _write(
        registry,
        {
            "schema": "poke_bot.post_fleet_refresh_registry/v1",
            "refreshes": [
                {
                    "specialist_id": "alakazam",
                    "refresh_model_version": "final-format-alakazam-r79-h10-v1",
                    "checkpoint": str(refreshed_alakazam),
                    "checkpoint_checksum": checkpoint.checkpoint_digest(refreshed_alakazam),
                    "completion_authority": "measured_gate_pass",
                }
            ],
        },
    )
    pointer = tmp_path / "latest.json"
    _write(
        pointer,
        {
            "schema": "poke_bot.latest_cumulative_core_pointer/v1",
            "version": 9,
            "family": "/accepted/core9",
            "ready": "/accepted/core9-ready.json",
            "checkpoint_digest": "sha256:" + "9" * 64,
        },
    )
    output = tmp_path / "post-alakazam.json"
    value = _materialize_contract(
        template_path=template,
        registry_path=registry,
        pointer_path=pointer,
        output=output,
    )

    core = value["core_refresh"]
    teachers = {row["specialist_id"]: row for row in core["teachers"]}
    assert core["version"] == 15
    assert core["initialization"]["checksum"] == "sha256:" + "9" * 64
    assert teachers["alakazam"]["checkpoint"] == str(refreshed_alakazam)
    assert teachers["alakazam"]["checksum"] == checkpoint.checkpoint_digest(refreshed_alakazam)
    assert teachers["archaludon-ex"]["checkpoint"] == str(other)
    assert teachers["archaludon-ex"]["checksum"] == checkpoint.checkpoint_digest(other)
    assert output.is_file()


def test_materialized_core_rejects_duplicate_alakazam_teacher(tmp_path: Path) -> None:
    refreshed = tmp_path / "refreshed.pt"
    refreshed.write_bytes(b"refresh")
    template = tmp_path / "template.json"
    _write(
        template,
        {
            "schema": "poke_bot.post_specialist_core_refresh_handoff/v1",
            "core_refresh": {
                "teachers": [
                    {"specialist_id": "alakazam"},
                    {"specialist_id": "alakazam"},
                ]
            },
            "acceptance": {},
            "core_failure_fallback": {},
        },
    )
    registry = tmp_path / "registry.json"
    _write(
        registry,
        {
            "schema": "poke_bot.post_fleet_refresh_registry/v1",
            "refreshes": [
                {
                    "specialist_id": "alakazam",
                    "checkpoint": str(refreshed),
                    "checkpoint_checksum": checkpoint.checkpoint_digest(refreshed),
                }
            ],
        },
    )
    pointer = tmp_path / "latest.json"
    _write(
        pointer,
        {
            "schema": "poke_bot.latest_cumulative_core_pointer/v1",
            "version": 9,
            "family": "/accepted/core9",
            "ready": "/accepted/ready.json",
            "checkpoint_digest": "sha256:" + "9" * 64,
        },
    )

    try:
        _materialize_contract(
            template_path=template,
            registry_path=registry,
            pointer_path=pointer,
            output=tmp_path / "output.json",
        )
    except RuntimeError as exc:
        assert "one Alakazam teacher" in str(exc)
    else:
        raise AssertionError("duplicate Alakazam teachers were accepted")
