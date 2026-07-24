from __future__ import annotations

from pathlib import Path

import pytest
import torch

from poke_bot import archetypes, checkpoint
from scripts import run_starmie_expert_bootstrap as bootstrap


ROOT = Path(__file__).resolve().parents[1]


def _argv(tmp_path: Path, *, epochs: int) -> list[str]:
    return [
        "--expert-corpus",
        str(tmp_path / "corpus.json"),
        "--archetype",
        "hops-trevenant",
        "--core-family",
        str(tmp_path / "core"),
        "--registry-root",
        str(tmp_path / "registry"),
        "--ready",
        str(tmp_path / "ready.json"),
        "--run-name",
        "trevenant-bootstrap-test",
        "--run-dir",
        str(tmp_path / "run"),
        "--epochs",
        str(epochs),
        "--cpu-pack-root",
        str(tmp_path / "cpu-pack"),
    ]


def test_specialist_bootstrap_rejects_non_exact_epoch_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 25 epochs"):
        bootstrap.main(_argv(tmp_path, epochs=24))


def test_diagnostic_patience_cannot_end_bootstrap_early() -> None:
    source = (ROOT / "scripts/run_starmie_expert_bootstrap.py").read_text(
        encoding="utf-8"
    )
    assert "if bad_epochs >= int(args.patience)" not in source
    assert "for epoch in range(start_epoch, int(args.epochs) + 1)" in source


def test_specialist_bootstrap_materializes_all_head_targets() -> None:
    source = (ROOT / "scripts/run_starmie_expert_bootstrap.py").read_text(
        encoding="utf-8"
    )
    assert "belief_card_vocab_from_state" in source
    assert "belief_card_vocab=belief_card_vocab" in source
    assert "if not corpus.has_exact_targets" in source


def test_rehearsal_expands_historical_archetype_head_before_training() -> None:
    source = (ROOT / "poke_bot/train.py").read_text(encoding="utf-8")
    load = source.index("model = load_model_from_checkpoint(base_path, device=device)")
    expand = source.index(
        "aux_head_expanded = expand_aux_head_to_current_registry(model)",
        load,
    )
    optimizer = source.index("optimizer = torch.optim.AdamW(", expand)
    assert load < expand < optimizer
    assert "if warm_started_heads_before or aux_head_expanded:" in source


def test_generic_entrypoint_uses_the_audited_bootstrap() -> None:
    source = (ROOT / "scripts/run_specialist_expert_bootstrap.py").read_text(
        encoding="utf-8"
    )
    assert "from scripts.run_starmie_expert_bootstrap import main" in source


def test_specialist_hot_start_append_expands_archetype_head(
    tmp_path: Path,
) -> None:
    old_ids = list(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS)
    target_ids = list(archetypes.archetype_ids())
    width = 4
    old_weight = torch.arange(
        (len(old_ids) + 1) * width, dtype=torch.float32
    ).reshape(len(old_ids) + 1, width)
    old_bias = torch.arange(len(old_ids) + 1, dtype=torch.float32)
    core = tmp_path / "core.pt"
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": {
                "aux_head.3.weight": old_weight,
                "aux_head.3.bias": old_bias,
            },
            "optimizer_state_dict": {"must": "not survive"},
            "step": 99,
            "epoch": 7,
            "rl_iteration": 5,
        },
        core,
    )

    hot_start, hot_digest, expansion = (
        bootstrap._specialist_hot_start_from_core(
            core,
            run_dir=tmp_path / "run",
            archetype="hops-trevenant",
        )
    )
    payload = checkpoint.load_checkpoint(hot_start, map_location="cpu")
    state = payload["model_state_dict"]

    assert state["aux_head.3.weight"].shape == (len(target_ids) + 1, width)
    for old_index, name in enumerate(old_ids):
        new_index = target_ids.index(name)
        assert torch.equal(
            state["aux_head.3.weight"][new_index], old_weight[old_index]
        )
        assert torch.equal(
            state["aux_head.3.bias"][new_index], old_bias[old_index]
        )
    assert torch.equal(state["aux_head.3.weight"][-1], old_weight[-1])
    assert torch.equal(state["aux_head.3.bias"][-1], old_bias[-1])
    assert expansion["newly_initialized_rows"] == [
        name for name in target_ids if name not in old_ids
    ]
    assert expansion["unknown_row_moved_to_final"] is True
    assert payload["step"] == payload["epoch"] == payload["rl_iteration"] == 0
    assert "optimizer_state_dict" not in payload
    assert hot_digest == checkpoint.checkpoint_digest(hot_start)

    repeated_path, repeated_digest, repeated_expansion = (
        bootstrap._specialist_hot_start_from_core(
            core,
            run_dir=tmp_path / "run",
            archetype="hops-trevenant",
        )
    )
    assert repeated_path == hot_start
    assert repeated_digest == hot_digest
    assert repeated_expansion == expansion


def test_specialist_hot_start_expands_cumulative_v4_archetype_order(
    tmp_path: Path,
) -> None:
    old_ids = list(archetypes.CUMULATIVE_V4_AUX_ARCHETYPE_IDS)
    target_ids = list(archetypes.archetype_ids())
    width = 3
    old_weight = torch.arange(
        (len(old_ids) + 1) * width, dtype=torch.float32
    ).reshape(len(old_ids) + 1, width)
    old_bias = torch.arange(len(old_ids) + 1, dtype=torch.float32)
    core = tmp_path / "cumulative-v4.pt"
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": {
                "aux_head.3.weight": old_weight,
                "aux_head.3.bias": old_bias,
            },
            "extra": {
                "matchup_adapter_config": {"expert_ids": old_ids},
            },
        },
        core,
    )

    hot_start, _, expansion = bootstrap._specialist_hot_start_from_core(
        core,
        run_dir=tmp_path / "run",
        archetype="dragapult-dusknoir",
    )
    state = checkpoint.load_checkpoint(
        hot_start, map_location="cpu"
    )["model_state_dict"]

    for old_index, name in enumerate(old_ids):
        new_index = target_ids.index(name)
        assert torch.equal(
            state["aux_head.3.weight"][new_index], old_weight[old_index]
        )
        assert torch.equal(
            state["aux_head.3.bias"][new_index], old_bias[old_index]
        )
    assert torch.equal(state["aux_head.3.weight"][-1], old_weight[-1])
    assert torch.equal(state["aux_head.3.bias"][-1], old_bias[-1])
    assert expansion["source_archetype_ids"] == old_ids
    assert expansion["unknown_row_moved_to_final"] is True
