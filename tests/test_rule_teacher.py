from __future__ import annotations

import json

import pytest

from poke_bot.rule_teacher import (
    RULE_TEACHER_SCHEMA,
    build_rule_teacher_jobs,
    deck_digest,
    file_digest,
    resolve_protected_teacher_corpus,
    result_metadata,
    summarize_journal,
)


def _spec(identifier: str) -> dict:
    return {
        "id": identifier,
        "name": identifier,
        "dir_name": identifier,
        "group": "official",
        "source": "test",
        "path": "/ignored",
        "contract_schema": "poke_bot.portable_baseline_spec/v1",
        "content_digest": "sha256:" + identifier * 4,
    }


def test_rule_teacher_jobs_are_balanced_and_content_bound() -> None:
    deck = list(range(60))
    jobs = build_rule_teacher_jobs(
        games=8,
        seed=100,
        job_offset=10,
        teacher_spec=_spec("teacher"),
        opponent_spec=_spec("opponent"),
        teacher_deck=deck,
        archetype="alakazam",
        outcome_filter="wins",
        timeout_s=30,
    )
    assert [job["job_index"] for job in jobs] == list(range(8))
    assert [job["job_id"] for job in jobs] == list(range(10, 18))
    assert sum(job["teacher_seat"] == 0 for job in jobs) == 4
    assert sum(job["teacher_seat"] == 1 for job in jobs) == 4
    assert {job["teacher_deck_digest"] for job in jobs} == {deck_digest(deck)}


@pytest.mark.parametrize("games", [0, 1, 3])
def test_rule_teacher_jobs_reject_unbalanced_counts(games: int) -> None:
    with pytest.raises(ValueError):
        build_rule_teacher_jobs(
            games=games,
            seed=0,
            job_offset=0,
            teacher_spec=_spec("teacher"),
            opponent_spec=_spec("opponent"),
            teacher_deck=list(range(60)),
            archetype="alakazam",
            outcome_filter="wins",
            timeout_s=30,
        )


def test_result_metadata_never_journals_large_record() -> None:
    result = {
        "job_index": 1,
        "job_id": 2,
        "record_json": "x" * 1000,
        "ok": True,
        "teacher_won": True,
    }
    assert result_metadata(result) == {
        "job_id": 2,
        "ok": True,
        "teacher_won": True,
    }


def test_summarize_journal_is_seat_aware(tmp_path) -> None:
    path = tmp_path / "teacher.journal"
    rows = [
        {
            "record_written": True,
            "result": {
                "ok": True,
                "teacher_seat": 0,
                "teacher_won": True,
                "winner": 0,
                "decisions": 20,
                "wall_s": 1.0,
            },
        },
        {
            "record_written": False,
            "result": {
                "ok": True,
                "teacher_seat": 1,
                "teacher_won": False,
                "winner": 0,
                "decisions": 10,
                "wall_s": 2.0,
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = summarize_journal(path)
    assert report["jobs"] == 2
    assert report["valid_games"] == 2
    assert report["wins"] == 1
    assert report["losses"] == 1
    assert report["records_written"] == 1
    assert report["decisions"] == 30
    assert report["win_rate"] == 0.5
    assert report["seat"]["0"]["win_rate"] == 1.0
    assert report["seat"]["1"]["win_rate"] == 0.0


def test_resolve_protected_teacher_corpus_verifies_digest(tmp_path) -> None:
    corpus = tmp_path / "teacher.jsonl"
    corpus.write_text('{"episode_id":"one"}\n')
    report_path = tmp_path / "PROTECTED_RULE_TEACHER_CORPUS.json"
    report = {
        "schema": RULE_TEACHER_SCHEMA,
        "protected": True,
        "prune_policy": "never",
        "configuration": {"final_agent_runtime": "neural_only"},
        "validation": {
            "records": 1,
            "decisions": 1,
            "info_set_ok": True,
            "conversion_drops": {},
        },
        "corpus": {"path": str(corpus), "digest": file_digest(corpus)},
    }
    report_path.write_text(json.dumps(report))
    resolved, loaded = resolve_protected_teacher_corpus(report_path)
    assert resolved == corpus.resolve()
    assert loaded["protected"] is True

    corpus.write_text("changed\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_protected_teacher_corpus(report_path)
