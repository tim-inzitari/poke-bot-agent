from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from poke_bot import config
from poke_bot.pure_rl.eval_public import OFFICIAL_BASELINE_IDS
from scripts import train_pure_rl


ROOT = Path(__file__).resolve().parents[1]


def test_scoped_pure_rl_env_exact_name_wins(monkeypatch) -> None:
    monkeypatch.setenv("PURE_RL_SELF_PLAY_FRAC", "0.50")
    monkeypatch.setenv("POKEBOT_PURE_RL_SELF_PLAY_FRAC", "0.85")
    assert config._env_float("PURE_RL_SELF_PLAY_FRAC", 1.0) == 0.50


def test_scoped_pure_rl_env_keeps_legacy_fallback(monkeypatch) -> None:
    monkeypatch.delenv("PURE_RL_SELF_PLAY_FRAC", raising=False)
    monkeypatch.setenv("POKEBOT_PURE_RL_SELF_PLAY_FRAC", "0.60")
    assert config._env_float("PURE_RL_SELF_PLAY_FRAC", 1.0) == 0.60


def test_unscoped_config_keeps_pokebot_contract(monkeypatch) -> None:
    monkeypatch.setenv("CARD_VOCAB", "9999")
    monkeypatch.setenv("POKEBOT_CARD_VOCAB", "2048")
    assert config._env_int("CARD_VOCAB", 1268) == 2048


def test_fresh_process_resolves_intended_pure_rl_config() -> None:
    env = dict(os.environ)
    env["PURE_RL_SELF_PLAY_FRAC"] = "0.50"
    env["PURE_RL_REPLAY_WINDOW_SHARDS"] = "2"
    env["PURE_RL_ALLOW_SINGLE_GPU"] = "1"
    env.pop("POKEBOT_PURE_RL_SELF_PLAY_FRAC", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from poke_bot.config import PURE_RL; "
                "print(PURE_RL.self_play_frac, PURE_RL.replay_window_shards, "
                "PURE_RL.allow_single_gpu)"
            ),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert proc.stdout.strip() == "0.5 2 True"


def test_intended_mix_allocates_half_public_without_official_holdout(
    tmp_path: Path,
) -> None:
    baseline_dirs = [tmp_path / opponent_id for opponent_id in ("public-a", "public-b")]
    for baseline_dir in baseline_dirs:
        baseline_dir.mkdir()
    specs = [
        SimpleNamespace(
            id=baseline_dir.name,
            name=baseline_dir.name,
            dir_name=baseline_dir.name,
            group="test",
            source="test",
            path=baseline_dir,
        )
        for baseline_dir in baseline_dirs
    ]
    self_play, public = train_pure_rl._build_collect_jobs(
        n_games=8192,
        ckpt=Path("/tmp/learner.pt"),
        digest="sha256:test",
        model_generation=1,
        decks=[("alakazam", [1] * 60), ("crustle", [2] * 60)],
        specs=specs,
        seed=26,
        game_timeout_s=600,
        mode="core",
        self_play_frac=0.50,
    )
    assert len(self_play) == 4096
    assert len(public) == 4096
    assert not ({job["opponent_id"] for job in public} & set(OFFICIAL_BASELINE_IDS))
