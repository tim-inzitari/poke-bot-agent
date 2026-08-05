from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.complete_post_marnie_crustle_h10 import (
    validated_population_inputs,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_crustle_completion_binds_population_bundle_and_runtime_assets(
    tmp_path: Path,
) -> None:
    checkpoint_digest = "sha256:" + "a" * 64
    expert = tmp_path / "expert.json"
    tree = tmp_path / "tree.json"
    bundle = tmp_path / "submission.tar.gz"
    expert.write_text("{}", encoding="utf-8")
    tree.write_text("{}", encoding="utf-8")
    bundle.write_bytes(b"bundle")
    registry = tmp_path / "runtime.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "poke_bot.specialist_runtime_registry/v1",
                "specialists": {
                    "crustle": {
                        "status": "ready",
                        "expert_manifest": str(expert),
                        "expert_manifest_sha256": _digest(expert).removeprefix(
                            "sha256:"
                        ),
                        "matchup_runtime_tree": str(tree),
                        "matchup_runtime_tree_sha256": _digest(tree).removeprefix(
                            "sha256:"
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bundle_digest = _digest(bundle)
    handler = {
        "submission_bundle": {
            "specialist_id": "crustle",
            "turn_order_preference": "first_if_allowed",
            "contents": {"model_sha256": checkpoint_digest},
            "sha256": bundle_digest,
        },
        "queued_submissions": [
            {
                "file": str(bundle),
                "file_sha256": bundle_digest,
                "checkpoint_checksum": checkpoint_digest,
            }
        ],
    }

    result = validated_population_inputs(
        handler=handler,
        checkpoint_digest=checkpoint_digest,
        runtime_registry_path=registry,
    )

    assert result["submission_bundle"] == str(bundle.resolve())
    assert result["submission_bundle_sha256"] == bundle_digest
    assert result["expert_manifest_sha256"] == _digest(expert)
    assert result["matchup_runtime_tree_sha256"] == _digest(tree)
