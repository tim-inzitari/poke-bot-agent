from __future__ import annotations

import json
from pathlib import Path

from scripts import select_core_regression_checkpoint as selector


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_selects_first_passing_checkpoint_in_validation_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    rows = []
    for epoch, loss in ((1, 1.4), (2, 1.2), (3, 1.3)):
        path = checkpoints / f"epoch_{epoch:02d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"epoch-{epoch}".encode())
        rows.append(
            {
                "epoch": epoch,
                "validation_loss": loss,
                "checkpoint": str(path),
                "checkpoint_digest": f"digest-{epoch}",
            }
        )
    state = tmp_path / "state.json"
    _json(
        state,
        {
            "schema": "poke_bot.multi_teacher_core_refresh_state/v1",
            "history": rows,
        },
    )
    contract = tmp_path / "contract.json"
    _json(contract, {})
    calls: list[int] = []

    monkeypatch.setattr(
        selector.checkpoint,
        "checkpoint_digest",
        lambda path: f"digest-{int(path.stem.split('_')[-1])}",
    )

    def fake_run_regression(*, candidate, output, **_kwargs):
        epoch = int(candidate.stem.split("_")[-1])
        calls.append(epoch)
        result = {"passed": epoch == 3}
        _json(output, result)
        return result

    monkeypatch.setattr(selector, "run_regression", fake_run_regression)
    result = selector.select(
        contract=contract,
        training_state=state,
        output_root=tmp_path / "results",
        receipt=tmp_path / "receipt.json",
        workers=4,
        result_prefix="core-v9",
    )
    assert calls == [2, 3]
    assert result["passed"] is True
    assert result["selected"]["epoch"] == 3
    assert result["thresholds_changed"] is False
