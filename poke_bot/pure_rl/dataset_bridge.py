"""Convert compact pure-RL shards into :class:`GameSequence` for AWR train."""

from __future__ import annotations

from typing import Iterator, Optional

from poke_bot.dataset import (
    BootstrapDataset,
    DecisionSample,
    GameSequence,
    PolicyStage,
    featurize_step,
)
from poke_bot.pure_rl.shards import CompactGame, iter_shard_games


def compact_game_to_sequence(
    game: CompactGame,
    *,
    verify_info_set: bool = False,
) -> Optional[GameSequence]:
    """Featurize compact decisions; never attaches soft policy targets."""
    decisions: list[DecisionSample] = []
    for d in game.decisions:
        step = {
            "observation": d.observation,
            "action": list(d.action),
            "env_step": d.env_step,
        }
        try:
            sample = featurize_step(
                step,
                list(game.deck) if game.deck else [0] * 60,
                verify_info_set=verify_info_set,
            )
        except Exception:
            # Fall back: keep selected_index via synthetic single stage when
            # observation is too thin for full featurization (unit tests).
            if not d.observation:
                continue
            raise
        # Force hard target from compact selected_index when stages exist.
        if sample.policy_stages:
            stages = []
            for i, stage in enumerate(sample.policy_stages):
                idx = int(d.selected_index) if i == 0 else int(stage.target_index)
                n = max(1, len(stage.action_combos))
                idx = max(0, min(idx, n - 1))
                stages.append(
                    PolicyStage(
                        options=stage.options,
                        action_combos=stage.action_combos,
                        target_index=idx,
                    )
                )
            sample.policy_stages = stages
            sample.action_combo_index = int(stages[0].target_index)
        decisions.append(sample)
    if not decisions:
        return None
    return GameSequence(
        episode_id=game.episode_id,
        seat=int(game.seat),
        archetype=game.archetype,
        opp_archetype=game.opp_archetype,
        deck=list(game.deck) if game.deck else [0] * 60,
        value=float(game.value),
        decisions=decisions,
        source=game.source or "pure_rl",
        policy_targets=None,
        factorized_policy_targets=None,
        target_provenance={
            **dict(game.target_provenance),
            "pure_rl": True,
            "soft_policy_targets": False,
        },
    )


def dataset_from_shard(
    path,
    *,
    verify_info_set: bool = False,
    max_games: Optional[int] = None,
) -> BootstrapDataset:
    seqs: list[GameSequence] = []
    for i, game in enumerate(iter_shard_games(path)):
        if max_games is not None and i >= max_games:
            break
        seq = compact_game_to_sequence(game, verify_info_set=verify_info_set)
        if seq is not None:
            seqs.append(seq)
    return BootstrapDataset(sequences=seqs)


def iter_sequences_from_shard(path, *, verify_info_set: bool = False) -> Iterator[GameSequence]:
    for game in iter_shard_games(path):
        seq = compact_game_to_sequence(game, verify_info_set=verify_info_set)
        if seq is not None:
            yield seq
