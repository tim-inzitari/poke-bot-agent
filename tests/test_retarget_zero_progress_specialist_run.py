from pathlib import Path


def test_retarget_requires_pristine_iteration_zero() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "retarget_zero_progress_specialist_run.py"
    ).read_text(encoding="utf-8")
    assert 'int(state.get("next_iteration", -1)) == 0' in source
    assert 'int(state.get("last_completed_iteration", -2)) == -1' in source
    assert 'not list((old_run / "commits").glob("iter_*.json"))' in source
    assert 'not list((old_run / "shards").glob("iter_*.jsonl"))' in source
    assert 'not list((old_run / "checkpoints").glob("iter_*.pt"))' in source
    assert "new run identity already exists" in source
