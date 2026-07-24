from __future__ import annotations

from pathlib import Path

import torch

from poke_bot import checkpoint
from poke_bot.matchup_adapters import MatchupAdapterBank
from scripts import run_multi_teacher_core_refresh as refresh


def _checkpoint(
    path: Path,
    value: float,
    *,
    archetype: str,
    aux_rows: int | None = None,
) -> Path:
    torch.manual_seed(17)
    bank = MatchupAdapterBank(enabled=False)
    with torch.no_grad():
        bank.experts[0].up.bias.fill_(value)
    state = {
        "encoder.weight": torch.full((3, 4), value),
        "policy_head.weight": torch.full((2, 4), value * 2),
        "integer_buffer": torch.tensor([7], dtype=torch.int64),
        **{
            f"matchup_adapter_bank.{name}": tensor.detach().clone()
            for name, tensor in bank.state_dict().items()
        },
    }
    if aux_rows is not None:
        state["aux_head.3.weight"] = torch.full((aux_rows, 4), value)
        state["aux_head.3.bias"] = torch.full((aux_rows,), value)
    checkpoint.immutable_torch_save(
        {
            "model_state_dict": state,
            "model_config": {"matchup_adapters_enabled": False},
            "archetype_id": archetype,
            "optimizer_state_dict": {"state": {"must": "be removed"}},
            "extra": {"specialist_only": True},
        },
        path,
    )
    return path


def test_multi_teacher_initialization_is_exact_mean_with_zero_adapters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initialization = _checkpoint(
        tmp_path / "core-v1.pt", 9.0, archetype="unknown"
    )
    teacher_a = _checkpoint(
        tmp_path / "alakazam.pt", 2.0, archetype="alakazam"
    )
    teacher_b = _checkpoint(
        tmp_path / "trevenant.pt", 6.0, archetype="hops-trevenant"
    )
    families = {
        "core": initialization,
        "alakazam": teacher_a,
        "trevenant": teacher_b,
    }

    def frozen(path: Path) -> dict[str, str]:
        model = families[Path(path).name]
        return {
            "family": Path(path).name,
            "model_path": str(model),
            "checkpoint_digest": checkpoint.checkpoint_digest(model),
        }

    monkeypatch.setattr(refresh, "verify_frozen_model", frozen)
    monkeypatch.setattr(refresh, "_architecture", lambda _path: {})
    output = tmp_path / "multi-teacher.pt"

    result, teachers = refresh.materialize_initialization(
        initialization_family=Path("core"),
        teacher_families=[Path("alakazam"), Path("trevenant")],
        output=output,
    )
    saved = checkpoint.load_checkpoint(output, map_location="cpu")

    assert result["teacher_count"] == 2
    assert [row["family"] for row in teachers] == ["alakazam", "trevenant"]
    assert torch.equal(
        saved["model_state_dict"]["encoder.weight"],
        torch.full((3, 4), 4.0),
    )
    assert torch.equal(
        saved["model_state_dict"]["policy_head.weight"],
        torch.full((2, 4), 8.0),
    )
    assert torch.equal(
        saved["model_state_dict"]["integer_buffer"],
        torch.tensor([7], dtype=torch.int64),
    )
    assert "optimizer_state_dict" not in saved
    assert saved["archetype_id"] == "unknown"
    assert all(
        int(value.count_nonzero().item()) == 0
        for name, value in saved["model_state_dict"].items()
        if name.startswith("matchup_adapter_bank.")
        and (name.endswith(".up.weight") or name.endswith(".up.bias"))
    )
    record = saved["extra"]["multi_teacher_core_initialization"]
    assert record["averaging"] == "equal_weight_parameter_space_mean"
    assert [row["weight"] for row in record["teachers"]] == [0.5, 0.5]

    recovered, recovered_teachers = refresh.materialize_initialization(
        initialization_family=Path("core"),
        teacher_families=[Path("alakazam"), Path("trevenant")],
        output=output,
    )
    assert recovered == result
    assert recovered_teachers == teachers


def test_legacy_auxiliary_rows_are_aligned_without_losing_teacher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initialization = _checkpoint(
        tmp_path / "core-v2.pt", 9.0, archetype="unknown", aux_rows=23
    )
    legacy = _checkpoint(
        tmp_path / "alakazam.pt", 2.0, archetype="alakazam", aux_rows=20
    )
    current = _checkpoint(
        tmp_path / "lucario.pt", 6.0, archetype="lucario", aux_rows=23
    )
    families = {"core": initialization, "alakazam": legacy, "lucario": current}

    def frozen(path: Path) -> dict[str, str]:
        model = families[Path(path).name]
        return {
            "family": Path(path).name,
            "model_path": str(model),
            "checkpoint_digest": checkpoint.checkpoint_digest(model),
        }

    monkeypatch.setattr(refresh, "verify_frozen_model", frozen)
    monkeypatch.setattr(refresh, "_architecture", lambda _path: {})
    output = tmp_path / "aligned.pt"
    refresh.materialize_initialization(
        initialization_family=Path("core"),
        teacher_families=[Path("alakazam"), Path("lucario")],
        output=output,
    )
    saved = checkpoint.load_checkpoint(output, map_location="cpu")
    weight = saved["model_state_dict"]["aux_head.3.weight"]
    assert torch.equal(weight[:19], torch.full((19, 4), 4.0))
    assert torch.equal(weight[19:22], torch.full((3, 4), 7.5))
    assert torch.equal(weight[-1], torch.full((4,), 4.0))
    alignment = saved["extra"]["multi_teacher_core_initialization"][
        "architecture_alignment"
    ]
    assert len(alignment) == 1
    assert alignment[0]["tensors"][0]["retained_initialization_rows"] == [
        "dudunsparce",
        "hops-trevenant",
        "walrein",
    ]
