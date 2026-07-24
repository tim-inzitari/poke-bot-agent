from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from poke_bot.pure_rl.model_registry import sha256
from scripts import process_kaggle_submission_queue as processor


NOW = datetime(2026, 7, 23, 18, 0, tzinfo=timezone.utc)


def _queue(tmp_path: Path) -> tuple[Path, dict]:
    bundle = tmp_path / "submission.tar.gz"
    model = b"exact frozen specialist"
    cards = list(range(1, 61))
    deck = "".join(f"{card}\n" for card in cards).encode("utf-8")
    matchup_tree = json.dumps(
        {
            "runtime_enabled": True,
            "runtime_contract": {
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        }
    ).encode("utf-8")
    with tarfile.open(bundle, "w:gz") as archive:
        for name, body in (
            ("model.pt", model),
            ("deck.csv", deck),
            ("main.py", b"def agent(_): return []\n"),
            ("cg/api.py", b"# cg\n"),
            ("matchup_tree.json", matchup_tree),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    path = tmp_path / "queue.json"
    model_digest = "sha256:" + hashlib.sha256(model).hexdigest()
    deck_digest = "sha256:" + hashlib.sha256(deck).hexdigest()
    payload = {
        "schema": "poke_bot.kaggle_submission_queue/v1",
        "daily_submission_limit": 5,
        "queue_order": "oldest_first",
        "retry_while_quota_exhausted": False,
        "quota": {},
        "queue": [
            {
                "specialist_id": "hops-trevenant",
                "copy_number": 1,
                "label": "hops-trevenant exact gate iter 25 copy 1/1 abcdef",
                "checkpoint_checksum": model_digest,
                "model_checksum": model_digest,
                "deck_file_checksum": deck_digest,
                "deck_cards_checksum": processor._canonical_digest(cards),
                "representatives_checksum": "sha256:" + "b" * 64,
                "matchup_tree_checksum": (
                    "sha256:" + hashlib.sha256(matchup_tree).hexdigest()
                ),
                "gate_id": "strong+frozen-specialists-r1",
                "iteration": 25,
                "queued_at": "2026-07-23T17:00:00+00:00",
                "queue_status": "pending",
                "competition": "pokemon-tcg-ai-battle",
                "file": str(bundle),
                "file_sha256": sha256(bundle),
                "retry_count": 0,
                "submitted_at": None,
                "submission_id": None,
                "returned_score": None,
                "failure_reason": None,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_bundle_identity_rejects_wrong_deck_even_with_updated_outer_hash(
    tmp_path: Path,
) -> None:
    queue_path, payload = _queue(tmp_path)
    bundle = Path(payload["queue"][0]["file"])
    model = b"exact frozen specialist"
    wrong_deck = "".join(f"{card}\n" for card in range(101, 161)).encode("utf-8")
    matchup_tree = json.dumps(
        {
            "runtime_enabled": True,
            "runtime_contract": {
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
            },
        }
    ).encode("utf-8")
    with tarfile.open(bundle, "w:gz") as archive:
        for name, body in (
            ("model.pt", model),
            ("deck.csv", wrong_deck),
            ("main.py", b"def agent(_): return []\n"),
            ("cg/api.py", b"# cg\n"),
            ("matchup_tree.json", matchup_tree),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    entry = payload["queue"][0]
    entry["file_sha256"] = sha256(bundle)
    with pytest.raises(RuntimeError, match="wrong specialist payload"):
        processor._verify_queued_bundle_identity(entry)


def test_successful_copy_is_submitted_once_and_later_reconciled(
    tmp_path: Path, monkeypatch
) -> None:
    queue_path, payload = _queue(tmp_path)
    submissions: list[dict[str, str]] = []
    submit_calls: list[list[str]] = []
    monkeypatch.setattr(processor, "_now", lambda: NOW)
    monkeypatch.setattr(processor, "_list_submissions", lambda *_: submissions)

    def fake_run(argv, **_kwargs):
        submit_calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="uploaded", stderr="")

    monkeypatch.setattr(processor.subprocess, "run", fake_run)
    first = processor.process_once(
        queue_path=queue_path,
        kaggle=Path("/fake/kaggle"),
        default_competition="pokemon-tcg-ai-battle",
    )
    assert first["status"] == "submitted"
    assert len(submit_calls) == 1
    stored = json.loads(queue_path.read_text(encoding="utf-8"))
    assert stored["queue"][0]["queue_status"] == "submitted"

    submissions.append(
        {
            "ref": "55000001",
            "date": "2026-07-23 18:00:01",
            "description": payload["queue"][0]["label"],
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": "812.3",
        }
    )
    second = processor.process_once(
        queue_path=queue_path,
        kaggle=Path("/fake/kaggle"),
        default_competition="pokemon-tcg-ai-battle",
    )
    assert second["status"] == "idle"
    assert len(submit_calls) == 1
    stored = json.loads(queue_path.read_text(encoding="utf-8"))
    assert stored["queue"][0]["queue_status"] == "accepted"
    assert stored["queue"][0]["submission_id"] == 55000001
    assert stored["queue"][0]["returned_score"] == 812.3


def test_daily_limit_error_queues_and_does_not_retry_same_quota_date(
    tmp_path: Path, monkeypatch
) -> None:
    queue_path, _ = _queue(tmp_path)
    legacy = json.loads(queue_path.read_text(encoding="utf-8"))
    legacy["queue"][0].pop("gate_id")
    legacy["queue"][0].pop("iteration")
    queue_path.write_text(json.dumps(legacy), encoding="utf-8")
    submit_calls = 0
    monkeypatch.setattr(processor, "_now", lambda: NOW)
    monkeypatch.setattr(processor, "_list_submissions", lambda *_: [])

    def quota_run(_argv, **_kwargs):
        nonlocal submit_calls
        submit_calls += 1
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Daily submission limit has been reached",
        )

    monkeypatch.setattr(processor.subprocess, "run", quota_run)
    first = processor.process_once(
        queue_path=queue_path,
        kaggle=Path("/fake/kaggle"),
        default_competition="pokemon-tcg-ai-battle",
    )
    second = processor.process_once(
        queue_path=queue_path,
        kaggle=Path("/fake/kaggle"),
        default_competition="pokemon-tcg-ai-battle",
    )
    assert first["status"] == "pending"
    assert second["status"] == "quota_exhausted"
    assert submit_calls == 1
    stored = json.loads(queue_path.read_text(encoding="utf-8"))
    assert stored["queue"][0]["queue_status"] == "pending"
    assert stored["queue"][0]["retry_count"] == 1
    assert stored["quota"]["quota_exhausted"] is True
    assert stored["quota"]["known_submissions_used_today"] == 5
