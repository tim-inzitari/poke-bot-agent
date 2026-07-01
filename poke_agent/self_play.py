from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm.auto import tqdm

from poke_agent.checkpoint import save_checkpoint
from poke_agent.dataset import prepare_training_tensors
from poke_agent.deck_pool import (
    FieldMatchup,
    choose_agent_vs_field_matchup,
    filter_pool_by_max_placement,
    mirror_matchup,
    resolve_field_pool,
)
from poke_agent.device import torch_device
from poke_agent.policy_agent import PolicyRuntime, PolicySession, make_policy_fn
from poke_agent.rule_baselines import BaselineOpponent, baseline_opponents_from_names
from poke_agent.rollout import make_random_agent, play_match
from poke_agent.simulator import SimulatorState
from poke_agent.kaggle_submit import (
    DEFAULT_SUBMISSION_MESSAGE,
    champion_checkpoint_from_manifest,
    submit_champion_checkpoint,
)
from poke_agent.training import build_model, train_model


@dataclass
class OpponentPool:
    """Fictitious self-play pool of past checkpoints."""

    max_size: int = 5
    checkpoints: list[Path] = field(default_factory=list)

    def add(self, checkpoint: Path) -> None:
        checkpoint = Path(checkpoint)
        self.checkpoints = [checkpoint, *[path for path in self.checkpoints if path != checkpoint]]
        self.checkpoints = self.checkpoints[: self.max_size]

    def sample(self, *, exclude: Path | None = None) -> Path | None:
        choices = [path for path in self.checkpoints if exclude is None or path != exclude]
        if not choices:
            return None
        return random.choice(choices)


@dataclass
class SelfPlaySettings:
    games_per_iteration: int
    eval_games: int
    iterations: int
    opponent_pool_size: int
    use_beam: bool
    output_path: Path
    checkpoint_dir: Path
    train_after_collect: bool = True
    field_deck_dir: str | None = None
    matchup_mode: str = "sample"
    field_pool: list[tuple[str, list[int]]] = field(default_factory=list)
    use_field: bool = False
    target_rank: int = 1000
    target_win_rate: float = 0.55
    plateau_patience: int = 3
    target_eval_pool: list[tuple[str, list[int]]] = field(default_factory=list)
    baseline_names: list[str] = field(default_factory=list)
    baseline_opponents: list[BaselineOpponent] = field(default_factory=list)
    train_vs_baselines: bool = False


def summarize_results(results: list[int], *, seat_index: int) -> dict[str, float]:
    """Summarize game outcomes from one seat's perspective (0=player0 wins)."""
    wins = sum(1 for result in results if result == seat_index)
    losses = sum(1 for result in results if result >= 0 and result != 2 and result != seat_index)
    draws = sum(1 for result in results if result == 2)
    decided = wins + losses
    win_rate = (wins / decided) if decided else 0.0
    return {
        "games": float(len(results)),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": win_rate,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"iterations": [], "champion": None}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def resolve_matchup(
    episode: int,
    agent_name: str,
    agent_deck: list[int],
    settings: SelfPlaySettings,
) -> FieldMatchup:
    if settings.use_field and settings.field_pool:
        return choose_agent_vs_field_matchup(
            episode,
            agent_name,
            agent_deck,
            settings.field_pool,
            mode=settings.matchup_mode,
        )
    return mirror_matchup(agent_name, agent_deck)


def resolve_target_eval_matchup(
    episode: int,
    agent_name: str,
    agent_deck: list[int],
    settings: SelfPlaySettings,
) -> FieldMatchup:
    pool = settings.target_eval_pool or settings.field_pool
    if pool:
        return choose_agent_vs_field_matchup(
            episode,
            agent_name,
            agent_deck,
            pool,
            mode=settings.matchup_mode,
        )
    return mirror_matchup(agent_name, agent_deck)


def annotate_matchup_metadata(rows: list[dict[str, Any]], matchup: FieldMatchup) -> None:
    """JSONL metadata only — not used in feature encoding."""
    if matchup.opponent_placement is None:
        return
    for row in rows:
        row["opponent_placement"] = matchup.opponent_placement


def _mark_retained_terminal(rows: list[dict[str, Any]]) -> None:
    """Keep our-turn-only rows complete when opponent took the final action."""
    if not rows:
        return
    for row in rows:
        row["terminal"] = False
    rows[-1]["terminal"] = True
    rows[-1]["reward"] = rows[-1].get("value", rows[-1].get("reward", 0.0))


def _baseline_for_episode(settings: SelfPlaySettings, episode: int) -> BaselineOpponent:
    if not settings.baseline_opponents:
        raise ValueError("baseline opponent pool is empty")
    return settings.baseline_opponents[episode % len(settings.baseline_opponents)]


def _seat_agent(
    runtime: PolicyRuntime,
    session: PolicySession,
    matchup: FieldMatchup,
    seat: int,
    *,
    use_beam: bool,
) -> Callable[[dict], list[int]]:
    deck = matchup.deck_for_seat(seat)
    return make_policy_fn(runtime, session, deck, use_beam=use_beam)


def evaluate_agent_vs_field(
    simulator: SimulatorState,
    agent_fn: Callable[[dict], list[int]],
    *,
    agent_name: str,
    agent_deck: list[int],
    opponent_fn: Callable[[dict], list[int]],
    games: int,
    settings: SelfPlaySettings,
    start_episode: int = 0,
    use_target_pool: bool = False,
) -> dict[str, Any]:
    if games <= 0:
        return {
            "agent_a": {"games": 0.0, "wins": 0.0, "losses": 0.0, "draws": 0.0, "win_rate": 0.0},
            "results": [],
            "matchups": [],
        }

    results: list[int] = []
    matchup_names: list[str] = []
    agent_a_wins = 0
    agent_a_losses = 0
    draws = 0

    for game_index in range(games):
        episode = start_episode + game_index
        if use_target_pool:
            matchup = resolve_target_eval_matchup(episode, agent_name, agent_deck, settings)
        else:
            matchup = resolve_matchup(episode, agent_name, agent_deck, settings)
        matchup_names.append(
            f"{matchup.field_name}@{matchup.opponent_placement}"
            if matchup.opponent_placement is not None
            else matchup.field_name
        )
        agent_is_seat0 = matchup.agent_seat == 0
        seat0 = agent_fn if agent_is_seat0 else opponent_fn
        seat1 = opponent_fn if agent_is_seat0 else agent_fn
        rows = play_match(
            episode,
            matchup.deck0,
            matchup.deck1,
            simulator,
            seat0,
            seat1,
            deck0_name=matchup.deck0_name,
            deck1_name=matchup.deck1_name,
        )
        result = int(next(row for row in reversed(rows) if row.get("terminal")).get("result", -1))
        results.append(result)
        if result == 2:
            draws += 1
        elif result == matchup.agent_seat:
            agent_a_wins += 1
        elif result >= 0:
            agent_a_losses += 1

    decided = agent_a_wins + agent_a_losses
    return {
        "agent_a": {
            "games": float(len(results)),
            "wins": float(agent_a_wins),
            "losses": float(agent_a_losses),
            "draws": float(draws),
            "win_rate": (agent_a_wins / decided) if decided else 0.0,
        },
        "results": results,
        "matchups": matchup_names,
    }


def evaluate_checkpoint_vs_baselines(
    simulator: SimulatorState,
    checkpoint_path: Path,
    *,
    device: torch.device,
    agent_name: str,
    agent_deck: list[int],
    games: int,
    settings: SelfPlaySettings,
    start_episode: int = 0,
) -> dict[str, Any]:
    if games <= 0:
        return {
            "agent_a": {"games": 0.0, "wins": 0.0, "losses": 0.0, "draws": 0.0, "win_rate": 0.0},
            "by_baseline": {},
            "results": [],
        }

    runtime = PolicyRuntime(checkpoint_path, device=device)
    wins = losses = draws = 0
    results: list[int] = []
    by_baseline: dict[str, dict[str, int]] = {}
    iterator = tqdm(range(games), desc="eval vs public baselines", unit="game")
    for offset in iterator:
        episode = start_episode + offset
        baseline = _baseline_for_episode(settings, episode)
        agent_seat = offset % 2
        session = runtime.new_session()
        neural_agent = make_policy_fn(runtime, session, agent_deck, use_beam=settings.use_beam)
        baseline_agent = baseline.make_agent()
        deck0 = agent_deck if agent_seat == 0 else baseline.deck
        deck1 = baseline.deck if agent_seat == 0 else agent_deck
        rows = play_match(
            episode,
            deck0,
            deck1,
            simulator,
            neural_agent if agent_seat == 0 else baseline_agent,
            baseline_agent if agent_seat == 0 else neural_agent,
            deck0_name=agent_name if agent_seat == 0 else baseline.name,
            deck1_name=baseline.name if agent_seat == 0 else agent_name,
        )
        if not rows:
            continue
        result = int(rows[-1].get("result", -1))
        results.append(result)
        stats = by_baseline.setdefault(baseline.name, {"games": 0, "wins": 0, "losses": 0, "draws": 0})
        stats["games"] += 1
        if result == 2:
            draws += 1
            stats["draws"] += 1
        elif result == agent_seat:
            wins += 1
            stats["wins"] += 1
        elif result >= 0:
            losses += 1
            stats["losses"] += 1
        decided = wins + losses
        iterator.set_postfix(win_rate=f"{(wins / decided) if decided else 0.0:.1%}", baseline=baseline.name)

    decided = wins + losses
    by_baseline_rates = {
        name: {
            **stats,
            "win_rate": (stats["wins"] / max(1, stats["wins"] + stats["losses"])),
        }
        for name, stats in by_baseline.items()
    }
    return {
        "agent_a": {
            "games": float(len(results)),
            "wins": float(wins),
            "losses": float(losses),
            "draws": float(draws),
            "win_rate": (wins / decided) if decided else 0.0,
        },
        "by_baseline": by_baseline_rates,
        "results": results,
    }


def collect_self_play_games(
    simulator: SimulatorState,
    agent_deck: list[int],
    *,
    agent_name: str,
    games: int,
    start_episode: int,
    current_runtime: PolicyRuntime,
    opponent_runtime: PolicyRuntime,
    settings: SelfPlaySettings,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_session = current_runtime.new_session()
    opponent_session = opponent_runtime.new_session()
    use_beam = settings.use_beam

    for offset in range(games):
        episode = start_episode + offset
        current_session.reset()
        opponent_session.reset()
        matchup = resolve_matchup(episode, agent_name, agent_deck, settings)
        current_plays_seat0 = offset % 2 == 0

        if current_plays_seat0:
            agent0 = _seat_agent(current_runtime, current_session, matchup, 0, use_beam=use_beam)
            agent1 = _seat_agent(opponent_runtime, opponent_session, matchup, 1, use_beam=use_beam)
        else:
            agent0 = _seat_agent(opponent_runtime, opponent_session, matchup, 0, use_beam=use_beam)
            agent1 = _seat_agent(current_runtime, current_session, matchup, 1, use_beam=use_beam)

        match_rows = play_match(
            episode,
            matchup.deck0,
            matchup.deck1,
            simulator,
            agent0,
            agent1,
            deck0_name=matchup.deck0_name,
            deck1_name=matchup.deck1_name,
        )
        annotate_matchup_metadata(match_rows, matchup)
        rows.extend(match_rows)
    return rows


def collect_baseline_games(
    simulator: SimulatorState,
    agent_deck: list[int],
    *,
    agent_name: str,
    games: int,
    start_episode: int,
    current_runtime: PolicyRuntime,
    settings: SelfPlaySettings,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    iterator = tqdm(range(games), desc="active CABT simulations vs baselines", unit="game")
    for offset in iterator:
        episode = start_episode + offset
        baseline = _baseline_for_episode(settings, episode)
        agent_seat = offset % 2
        session = current_runtime.new_session()
        neural_agent = make_policy_fn(current_runtime, session, agent_deck, use_beam=settings.use_beam)
        baseline_agent = baseline.make_agent()

        deck0 = agent_deck if agent_seat == 0 else baseline.deck
        deck1 = baseline.deck if agent_seat == 0 else agent_deck
        match_rows = play_match(
            episode,
            deck0,
            deck1,
            simulator,
            neural_agent if agent_seat == 0 else baseline_agent,
            baseline_agent if agent_seat == 0 else neural_agent,
            deck0_name=agent_name if agent_seat == 0 else baseline.name,
            deck1_name=baseline.name if agent_seat == 0 else agent_name,
            rewards=settings.__dict__.get("rewards"),
        )
        neural_rows = [row for row in match_rows if int(row.get("player", -1)) == agent_seat]
        if neural_rows:
            _mark_retained_terminal(neural_rows)
        for row in neural_rows:
            row["opponent_baseline"] = baseline.name
            row["opponent_family"] = baseline.family
            row["opponent_source"] = baseline.source
        rows.extend(neural_rows)

        if match_rows:
            result = int(match_rows[-1].get("result", -1))
            won = result == agent_seat
            iterator.set_postfix(baseline=baseline.name, result="W" if won else "L")
    return rows


def train_on_rollouts(
    config: dict[str, Any],
    device: torch.device,
    *,
    data_path: Path,
    checkpoint_path: Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Retrain on accumulated self-play JSONL (all games in file, not DATASET_GAMES cap)."""
    train_config = dict(config)
    train_config["training_data_path"] = data_path
    train_config["generated_path"] = data_path
    train_config["dataset_games"] = None
    train_config["require_cabt_eval_data"] = False
    train_config["require_training_matchup_diversity"] = False

    sp = dict(config.get("self_play", {}))
    train_cfg = dict(train_config.get("training", {}))
    if sp.get("train_epochs") is not None:
        train_cfg["epochs"] = int(sp["train_epochs"])
    train_config["training"] = train_cfg

    row_count = sum(1 for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"self-play retrain: {data_path} ({row_count:,} rows, all collected games)")

    tensors = prepare_training_tensors(train_config, device)
    model = build_model(train_config, tensors, device)
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded weights from {checkpoint_path.name} before self-play training")
    training_report = train_model(model, tensors, train_config, device)
    return model, tensors, training_report


def run_self_play_iteration(
    *,
    iteration: int,
    config: dict[str, Any],
    simulator: SimulatorState,
    agent_deck: list[int],
    agent_name: str,
    settings: SelfPlaySettings,
    device: torch.device,
    current_checkpoint: Path,
    pool: OpponentPool,
    start_episode: int,
) -> dict[str, Any]:
    if not simulator.available:
        raise RuntimeError("CABT simulator is required for self-play")

    current_runtime = PolicyRuntime(current_checkpoint, device=device)
    opponent_path = pool.sample(exclude=current_checkpoint) or current_checkpoint
    opponent_runtime = PolicyRuntime(opponent_path, device=device) if not settings.train_vs_baselines else None

    if settings.train_vs_baselines:
        field_note = f"public baselines={len(settings.baseline_opponents)}"
    else:
        field_note = (
            f"field={len(settings.field_pool)} decks ({settings.matchup_mode})"
            if settings.use_field
            else "mirror deck"
        )
    print(
        f"self-play iter {iteration}: collect {settings.games_per_iteration} games "
        f"(current={current_checkpoint.name}, opponent={opponent_path.name}, "
        f"beam={settings.use_beam}, {field_note})"
    )
    if settings.train_vs_baselines:
        rows = collect_baseline_games(
            simulator,
            agent_deck,
            agent_name=agent_name,
            games=settings.games_per_iteration,
            start_episode=start_episode,
            current_runtime=current_runtime,
            settings=settings,
        )
    else:
        assert opponent_runtime is not None
        rows = collect_self_play_games(
            simulator,
            agent_deck,
            agent_name=agent_name,
            games=settings.games_per_iteration,
            start_episode=start_episode,
            current_runtime=current_runtime,
            opponent_runtime=opponent_runtime,
            settings=settings,
        )
    unique_matchups = len({(row.get("deck0"), row.get("deck1")) for row in rows if int(row.get("step", -1)) == 0})
    append = settings.output_path.exists() and iteration > 1
    write_jsonl(settings.output_path, rows, append=append)
    print(
        f"self-play data -> {settings.output_path} ({len(rows):,} rows, "
        f"{unique_matchups} unique deck pairings, append={append})"
    )

    if settings.train_vs_baselines:
        eval_report = evaluate_checkpoint_vs_baselines(
            simulator,
            current_checkpoint,
            device=device,
            agent_name=agent_name,
            agent_deck=agent_deck,
            games=settings.eval_games,
            settings=settings,
            start_episode=start_episode + settings.games_per_iteration,
        )
        print(
            "eval vs public baselines: "
            f"win_rate={eval_report['agent_a']['win_rate']:.1%} "
            f"({int(eval_report['agent_a']['wins'])}/{int(eval_report['agent_a']['games'])})"
        )
        for name, stats in sorted(eval_report.get("by_baseline", {}).items()):
            print(
                f"  {name}: {stats['win_rate']:.1%} "
                f"({stats['wins']}/{stats['games']}, draws={stats['draws']})"
            )
    else:
        random_agent = make_random_agent(simulator.to_observation_class)
        current_session = current_runtime.new_session()
        current_fn = make_policy_fn(
            current_runtime,
            current_session,
            agent_deck,
            use_beam=settings.use_beam,
        )

        eval_report = evaluate_agent_vs_field(
            simulator,
            current_fn,
            agent_name=agent_name,
            agent_deck=agent_deck,
            opponent_fn=random_agent,
            games=settings.eval_games,
            settings=settings,
            start_episode=start_episode + settings.games_per_iteration,
            use_target_pool=True,
        )
        sample_matchups = eval_report.get("matchups", [])[:5]
        print(
            f"eval vs random (placement<={settings.target_rank}): "
            f"win_rate={eval_report['agent_a']['win_rate']:.1%} "
            f"({int(eval_report['agent_a']['wins'])}/{int(eval_report['agent_a']['games'])}) "
            f"sample opponents={sample_matchups}"
        )

    training_report: dict[str, Any] | None = None
    saved_checkpoint = current_checkpoint
    if settings.train_after_collect:
        print("training on self-play rollouts...")
        model, tensors, training_report = train_on_rollouts(
            config,
            device,
            data_path=settings.output_path,
            checkpoint_path=current_checkpoint,
        )
        iter_path = settings.checkpoint_dir / f"iter_{iteration:03d}.pt"
        training_report = save_checkpoint(
            model=model,
            tensors=tensors,
            config=config,
            training_report=training_report,
            output_path=iter_path,
        )
        saved_checkpoint = iter_path
        pool.add(iter_path)
        print(f"saved iteration checkpoint -> {iter_path}")

    return {
        "iteration": iteration,
        "rows_collected": len(rows),
        "unique_matchups": unique_matchups,
        "data_path": str(settings.output_path),
        "current_checkpoint": str(current_checkpoint),
        "opponent_checkpoint": str(opponent_path),
        "saved_checkpoint": str(saved_checkpoint),
        "eval_vs_random": eval_report["agent_a"],
        "eval_vs_baselines": eval_report.get("by_baseline", {}),
        "field_decks": len(settings.field_pool),
        "baseline_opponents": [baseline.name for baseline in settings.baseline_opponents],
        "training_report": training_report,
    }


def run_self_play_loop(
    *,
    config: dict[str, Any],
    simulator: SimulatorState,
    agent_deck: list[int],
    agent_name: str,
    settings: SelfPlaySettings,
    device: torch.device | None = None,
    initial_checkpoint: Path | None = None,
    submit_on_stop: bool = False,
    submission_message: str = DEFAULT_SUBMISSION_MESSAGE,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    device = device or torch_device()
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = settings.checkpoint_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest_iterations = manifest.setdefault("iterations", [])
    pool = OpponentPool(max_size=settings.opponent_pool_size)
    last_manifest_checkpoint: Path | None = None
    for entry in manifest_iterations:
        checkpoint = entry.get("saved_checkpoint")
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                pool.add(checkpoint_path)
                last_manifest_checkpoint = checkpoint_path

    current_checkpoint = Path(initial_checkpoint) if initial_checkpoint is not None else (
        last_manifest_checkpoint or Path(config["output_path"])
    )
    if not current_checkpoint.exists():
        raise FileNotFoundError(f"initial checkpoint not found: {current_checkpoint}")

    pool.add(current_checkpoint)
    start_episode = int(manifest.get("next_episode", 0))
    start_iteration = int(manifest.get("next_iteration", len(manifest_iterations) + 1))
    reports: list[dict[str, Any]] = []
    best_target_rate = float((manifest.get("champion") or {}).get("eval_vs_random", {}).get("win_rate", -1.0))
    plateau_count = int(manifest.get("plateau_count", 0))
    stop_reason: str | None = None

    if start_iteration > settings.iterations:
        print(
            f"self-play manifest already complete through iteration {start_iteration - 1}; "
            f"requested max iteration is {settings.iterations}"
        )
        return reports

    print(
        f"self-play resume: start_iteration={start_iteration} "
        f"next_episode={start_episode} checkpoint={current_checkpoint}"
    )

    for iteration in range(start_iteration, settings.iterations + 1):
        report = run_self_play_iteration(
            iteration=iteration,
            config=config,
            simulator=simulator,
            agent_deck=agent_deck,
            agent_name=agent_name,
            settings=settings,
            device=device,
            current_checkpoint=current_checkpoint,
            pool=pool,
            start_episode=start_episode,
        )
        reports.append(report)
        start_episode += settings.games_per_iteration
        current_checkpoint = Path(report["saved_checkpoint"])
        manifest["next_episode"] = start_episode
        manifest["next_iteration"] = iteration + 1

        target_rate = float(report["eval_vs_random"]["win_rate"])
        manifest_iterations.append(report)
        if target_rate >= best_target_rate + 1e-6:
            best_target_rate = target_rate
            plateau_count = 0
            manifest["champion"] = report
        else:
            plateau_count += 1
        manifest["plateau_count"] = plateau_count
        save_manifest(manifest_path, manifest)

        if target_rate >= settings.target_win_rate:
            stop_reason = f"target win rate {settings.target_win_rate:.0%} reached ({target_rate:.1%})"
            print(stop_reason)
            break
        if plateau_count >= settings.plateau_patience:
            stop_reason = (
                f"plateau: no eval improvement for {settings.plateau_patience} iterations "
                f"(best={best_target_rate:.1%})"
            )
            print(stop_reason)
            break

    if stop_reason:
        manifest["stop_reason"] = stop_reason
        save_manifest(manifest_path, manifest)

    if submit_on_stop and reports:
        submit_root = root or settings.checkpoint_dir.parent.parent.parent
        champion_ckpt = champion_checkpoint_from_manifest(manifest, current_checkpoint)
        try:
            submission = submit_champion_checkpoint(
                champion_ckpt,
                root=submit_root,
                message=submission_message,
            )
            manifest["kaggle_submission"] = submission
            save_manifest(manifest_path, manifest)
        except subprocess.CalledProcessError as exc:
            manifest["kaggle_submission_error"] = {
                "checkpoint": str(champion_ckpt),
                "returncode": exc.returncode,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            }
            save_manifest(manifest_path, manifest)
            raise

    return reports


def self_play_settings_from_config(
    config: dict[str, Any],
    root: Path,
    *,
    agent_name: str,
    agent_deck: list[int],
) -> SelfPlaySettings:
    sp = dict(config.get("self_play", {}))
    field_dir = sp.get("field_deck_dir")
    target_rank = int(sp.get("target_rank", 1000))
    field_pool, use_field = resolve_field_pool(field_dir, root=root, agent_name=agent_name, agent_deck=agent_deck)
    if use_field:
        print(f"field deck pool: {len(field_pool)} decks from {field_dir}")

    target_eval_pool = filter_pool_by_max_placement(field_pool, target_rank)
    if not target_eval_pool:
        try:
            rest_pool, _ = resolve_field_pool(
                "decks/competitive/the_rest",
                root=root,
                agent_name=agent_name,
                agent_deck=agent_deck,
            )
            target_eval_pool = filter_pool_by_max_placement(rest_pool, target_rank)
            if target_eval_pool:
                print(
                    f"target eval pool: {len(target_eval_pool)} decks "
                    f"with placement<={target_rank} (from the_rest)"
                )
        except (FileNotFoundError, ValueError):
            target_eval_pool = list(field_pool)
    else:
        print(
            f"target eval pool: {len(target_eval_pool)} decks with placement<={target_rank}"
        )

    baseline_names_raw = sp.get("baseline_names", [])
    baseline_opponents = baseline_opponents_from_names(baseline_names_raw, root=root) if baseline_names_raw else []
    if baseline_opponents:
        print(
            "public baseline pool:",
            ", ".join(f"{baseline.name}({baseline.source})" for baseline in baseline_opponents),
        )

    return SelfPlaySettings(
        games_per_iteration=int(sp.get("games_per_iteration", 20)),
        eval_games=int(sp.get("eval_games", 10)),
        iterations=int(sp.get("iterations", 3)),
        opponent_pool_size=int(sp.get("opponent_pool_size", 5)),
        use_beam=bool(sp.get("use_beam", True)),
        output_path=root / sp.get("output_path", "outputs/rollouts/self_play_rollouts.jsonl"),
        checkpoint_dir=root / sp.get("checkpoint_dir", "outputs/checkpoints/self_play"),
        train_after_collect=bool(sp.get("train_after_collect", True)),
        field_deck_dir=field_dir,
        matchup_mode=str(sp.get("matchup_mode", "sample")),
        field_pool=field_pool,
        use_field=use_field,
        target_rank=target_rank,
        target_win_rate=float(sp.get("target_win_rate", 0.55)),
        plateau_patience=int(sp.get("plateau_patience", 3)),
        target_eval_pool=target_eval_pool,
        baseline_names=[baseline.name for baseline in baseline_opponents],
        baseline_opponents=baseline_opponents,
        train_vs_baselines=bool(sp.get("train_vs_baselines", False)) or bool(baseline_opponents),
    )
