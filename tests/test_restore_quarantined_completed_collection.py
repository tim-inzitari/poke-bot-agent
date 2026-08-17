from pathlib import Path

from scripts.restore_quarantined_completed_collection import _verified_plan


def test_restore_uses_only_managed_service_lifecycle() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/restore_quarantined_completed_collection.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("pkill", "killall", "os.kill(", "SIGKILL", "SIGTERM"):
        assert forbidden not in source
    assert '["systemctl", "--user", "stop", args.unit]' in source
    assert '["systemctl", "--user", "start", args.unit]' in source
    assert "_verified_completed_collection_across_design_chain" in source
    assert "preserve_completed_collection=False" in source


def test_verified_plan_accepts_a_completed_exact_pair(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    attempt = run / "quarantine/iter_00015/attempt_0002"
    rows = []
    for relative, content in (
        ("shards/iter_00015.jsonl", b"shard"),
        ("collection_receipts/iter_00015.json", b"receipt"),
    ):
        destination = attempt / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        rows.append(
            {
                "source": str(run / relative),
                "destination": str(destination),
                "relative_path": relative,
                "size": len(content),
                "digest": f"digest:{relative}",
            }
        )
    (attempt / "plan.json").write_text(
        __import__("json").dumps({"iteration": 15, "artifacts": rows})
    )
    (attempt / "failure.json").write_text(
        __import__("json").dumps(
            {
                "iteration": 15,
                "artifacts": rows,
                "quarantine_completed_at_utc": "2026-07-26T00:00:00+00:00",
            }
        )
    )
    # Use the recorded per-row values so the test focuses on transaction shape.
    digests = {Path(row["destination"]): row["digest"] for row in rows}
    monkeypatch.setattr(
        "scripts.restore_quarantined_completed_collection.train_pure_rl._sha256_file",
        lambda path: digests[Path(path)],
    )
    assert _verified_plan(attempt, run) == rows
