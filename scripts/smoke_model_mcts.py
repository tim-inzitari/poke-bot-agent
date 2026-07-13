"""Phase 2A smoke: TemporalCabtTransformer + short MCTS + checkpoint round-trip.

Exercises:
  1. Load Hammer-Pult deck; build model on training_device.
  2. One forward pass (board + options) from a live observation.
  3. Short MCTS (few sims) via official Search API.
  4. Save + load a checkpoint; verify weights reload.

Use the conda env (NOT .venv)::

    /home/inzi/miniconda3/envs/poke-bot-agent/bin/python scripts/smoke_model_mcts.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from poke_bot import (  # noqa: E402
    archetypes,
    cg_env,
    checkpoint,
    config,
    deck_pool,
    device,
    features,
    matchup_id,
    mcts,
    model,
    paths,
)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def advance_to_decision(deck: list[int], rng: random.Random, n_steps: int = 4) -> dict:
    obs, start = cg_env.battle_start(deck, deck)
    if obs is None or getattr(start, "errorPlayer", -1) >= 0:
        raise RuntimeError("battle_start failed")
    for _ in range(n_steps):
        if cg_env.is_finished(obs):
            break
        sel = obs.get("select")
        if sel is None:
            break
        obs = cg_env.battle_select(cg_env.random_legal_select(obs, rng))
    return obs


def main() -> int:
    rng = random.Random(7)
    paths.ensure_runtime_dirs()

    section("runtime")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  cg: {cg_env.ensure_cg_importable()}")
    print(f"  devices: {device.describe()}")
    print(
        f"  MODEL max_context={config.MODEL.max_context} "
        f"temporal_pos={config.MODEL.temporal_pos} kv_cache={config.MODEL.kv_cache}"
    )

    section("deck")
    deck = deck_pool.primary_deck()
    print(f"  size={len(deck)} archetype={archetypes.classify_deck(deck)}")
    assert archetypes.classify_deck(deck) == "dragapult"
    assert not archetypes.is_hammer_signature(deck)

    section("model forward")
    train_dev = device.training_device(allow_cpu=True)
    print(f"  training_device={train_dev}")
    net = model.build_model(device=train_dev)
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  TemporalCabtTransformer params={n_params:,}")

    obs = advance_to_decision(deck, rng)
    features.assert_info_set(obs)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = net.forward_from_obs(obs, deck, append_cache=True)
    fwd_s = time.perf_counter() - t0
    logits = out["policy_logits"]
    value = out["value"]
    combos = out["action_combos"]
    print(
        f"  forward ok: n_options={len(combos)} logits={tuple(logits.shape)} "
        f"value={float(value[0]):.4f} ({fwd_s:.3f}s)"
    )
    assert logits.shape[-1] >= 1
    assert value.shape == (1,)
    kv = out["kv_cache"]
    assert kv is not None and kv.length == 1, "KV cache should append one CLS"
    print(f"  kv_cache length={kv.length} layers={len(kv.layers)}")

    section("matchup id")
    id_net = matchup_id.build_matchup_id_net(
        device=train_dev, shared_board_bag=net.board_bag, freeze_shared_bag=True
    )
    id_out = id_net.forward_from_obs(obs)
    label, conf = id_net.predict_label(obs)
    print(
        f"  classes={id_net.num_classes} pred={label} conf={conf:.3f} "
        f"probs_shape={tuple(id_out['probs'].shape)}"
    )

    section("mcts (short)")
    leaf_dev = device.leaf_eval_device(allow_cpu=True)
    print(f"  leaf_eval_device={leaf_dev}")
    # Move model to leaf device for search eval if different.
    if leaf_dev != train_dev:
        net = net.to(leaf_dev)
    t0 = time.perf_counter()
    result = mcts.run_mcts(net, obs, deck, max_sims=4, device=leaf_dev)
    print(
        f"  mcts ok: select={result.select} sims={result.sims_run} "
        f"elapsed={result.elapsed_s:.3f}s root_value={result.target.value:.4f}"
    )
    print(
        f"  visit policy (first 5)={result.target.policy[:5]} "
        f"visits={result.target.visits[:5]}"
    )
    assert result.sims_run >= 1 or result.select is not None
    assert len(result.target.policy) == len(result.target.visits)
    cg_env.battle_finish()

    section("checkpoint save/load")
    run = "smoke_phase2a"
    ckpt = checkpoint.build_checkpoint(
        model=net,
        step=1,
        epoch=0,
        model_config=config.MODEL,
        archetype_id="dragapult",
        model_id="temporal_cabt_v1",
        extra={"smoke": True},
    )
    saved = checkpoint.save_checkpoint(ckpt, run, write_step_copy=True)
    print(f"  saved: { {k: str(v) for k, v in saved.items()} }")

    net2 = model.build_model(device=leaf_dev)
    loaded = checkpoint.load_checkpoint(saved["latest"], map_location=leaf_dev)
    meta = checkpoint.apply_checkpoint(loaded, model=net2, restore_rng=True)
    # Compare a weight tensor.
    p1 = next(net.parameters()).detach().cpu()
    p2 = next(net2.parameters()).detach().cpu()
    max_diff = float((p1 - p2).abs().max())
    print(f"  reloaded step={meta['step']} max_weight_diff={max_diff:.2e}")
    assert max_diff < 1e-6

    resume = checkpoint.resolve_resume_path(run, "auto")
    print(f"  resume auto -> {resume}")
    assert resume is not None and resume.is_file()

    print("\nSMOKE TEST PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
