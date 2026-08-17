from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from poke_bot.feature_shards import SHARD_FORMAT, SHARD_FORMAT_VERSION
from scripts.extract_verified_specialist_records import iter_legacy_identities


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus(tmp_path: Path, *, deck: list[int]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shard = tmp_path / "legacy.features"
    with shard.open("wb") as stream:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "source_dates": ["2026-06-26"],
            },
            stream,
        )
        pickle.dump(
            SimpleNamespace(
                episode_id="123",
                seat=1,
                archetype="dudunsparce",
                deck=deck,
            ),
            stream,
        )
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "stats": {"records_kept": 1},
            },
            stream,
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "path": shard.name,
                        "sha256": _sha256(shard),
                        "source_dates": ["2026-06-26"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pointer = tmp_path / "PROTECTED_EXPERT_CORPUS.json"
    pointer.write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "manifest": manifest.name,
                "manifest_sha256": _sha256(manifest),
            }
        ),
        encoding="utf-8",
    )
    return pointer


def test_legacy_identity_index_is_exact_and_forbidden_cards_fail_closed(
    tmp_path: Path,
) -> None:
    deck = [306] + [3] * 59
    pointer = _corpus(tmp_path, deck=deck)
    assert list(
        iter_legacy_identities(
            pointer,
            specialist_id="dudunsparce",
            forbidden_card_ids=frozenset({648}),
        )
    ) == [
        {
            "episode_id": "123",
            "seat": 1,
            "day": "2026-06-26",
            "deck": deck,
            "source_shard": "legacy.features",
        }
    ]

    contaminated = _corpus(tmp_path / "contaminated", deck=[648] + [3] * 59)
    with pytest.raises(RuntimeError, match="failed validation"):
        list(
            iter_legacy_identities(
                contaminated,
                specialist_id="dudunsparce",
                forbidden_card_ids=frozenset({646, 647, 648}),
            )
        )


def test_build_log_does_not_make_a_fresh_output_look_unsealed() -> None:
    source = Path("scripts/extract_verified_specialist_records.py").read_text()
    assert 'path.name != "build.log"' in source
