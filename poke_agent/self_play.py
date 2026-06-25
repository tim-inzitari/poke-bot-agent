from __future__ import annotations

import json
import multiprocessing as mp
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm.auto import tqdm

from poke_agent.checkpoint import save_checkpoint
from poke_agent.baseline_agents import BaselineAgent, load_baseline_agents, summarize_baseline_eval
from poke_agent.beam_search import BeamSearchConfig

# Max games per worker task — smaller chunks = more frequent tqdm updates during parallel collect.
PARALLEL_PROGRESS_CHUNK_GAMES = 4
from poke_agent.data_pipeline import default_self_play_workers, episode_chunks
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
from poke_agent.rollout import make_random_agent, play_match
from poke_agent.simulator import SimulatorState, load_simulator
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
    workers: int | None = None
    baseline_dir: str | None = None
    baseline_win_rate_threshold: float = 0.60
    baseline_agents: list[BaselineAgent] = field(default_factory=list)
    agent_deck_pool: list[tuple[str, list[int]]] = field(default_factory=list)
    per_deck_checkpoint_dir: Path | None = None
    train_window_games: int | None = None
    trim_rollout_file: bool = True
    warmup_iterations: int = 10
    warmup_lr_multiplier: float = 25.0
    beam_config: BeamSearchConfig | None = None


def resolve_self_play_workers(settings: SelfPlaySettings, *, games: int) -> int:
    if settings.workers is not None and int(settings.workers) > 0:
        return max(1, min(int(settings.workers), games))
    return default_self_play_workers(games=games)


def _baseline_worker_settings(settings: SelfPlaySettings) -> SelfPlaySettings:
    """Pickle-friendly settings for baseline collection workers."""
    return SelfPlaySettings(
        games_per_iteration=0,
        eval_games=0,
        iterations=0,
        opponent_pool_size=0,
        use_beam=settings.use_beam,
        output_path=Path("."),
        checkpoint_dir=Path("."),
        beam_config=settings.beam_config,
        agent_deck_pool=list(settings.agent_deck_pool),
        baseline_dir=settings.baseline_dir,
    )


def _matchup_settings(settings: SelfPlaySettings) -> SelfPlaySettings:
    """Pickle-friendly settings subset for worker processes."""
    return SelfPlaySettings(
        games_per_iteration=0,
        eval_games=0,
        iterations=0,
        opponent_pool_size=0,
        use_beam=settings.use_beam,
        output_path=Path("."),
        checkpoint_dir=Path("."),
        matchup_mode=settings.matchup_mode,
        field_pool=list(settings.field_pool),
        use_field=settings.use_field,
        target_eval_pool=list(settings.target_eval_pool),
        beam_config=settings.beam_config,
    )


def _terminal_result(match_rows: list[dict[str, Any]]) -> int | None:
    if not match_rows:
        return None
    return int(next(row for row in reversed(match_rows) if row.get("terminal")).get("result", -1))


def _record_seat_outcome(
    result: int,
    our_seat: int,
    *,
    wins: int,
    losses: int,
    draws: int,
) -> tuple[int, int, int]:
    if result == 2:
        return wins, losses, draws + 1
    if result == our_seat:
        return wins + 1, losses, draws
    if result >= 0:
        return wins, losses + 1, draws
    return wins, losses, draws


def _win_rate_postfix(wins: int, losses: int, *, draws: int = 0, **extra: Any) -> dict[str, Any]:
    decided = wins + losses
    postfix: dict[str, Any] = {
        "wr": f"{(wins / decided):.0%}" if decided else "—",
        "W": wins,
        "L": losses,
    }
    if draws:
        postfix["D"] = draws
    postfix.update(extra)
    return postfix


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


def maybe_trim_rollout_file(settings: SelfPlaySettings) -> None:
    if not settings.trim_rollout_file:
        return
    from poke_agent.dataset import trim_rollout_jsonl

    trim_rollout_jsonl(settings.output_path, settings.train_window_games)


def rollout_buffer_overwrites(settings: SelfPlaySettings) -> bool:
    """True when each iteration should replace the JSONL with only this batch."""
    if not settings.trim_rollout_file:
        return False
    window = settings.train_window_games
    if window is None or int(window) <= 0:
        return False
    return int(settings.games_per_iteration) == int(window)


def write_rollout_buffer(
    settings: SelfPlaySettings,
    rows: list[dict[str, Any]],
    *,
    iteration: int,
) -> tuple[bool, int]:
    """Persist collected rows; returns (overwrote_file, games_on_disk)."""
    if rollout_buffer_overwrites(settings):
        write_jsonl(settings.output_path, rows, append=False)
        kept_games = _count_rollout_games(settings.output_path)
        return True, kept_games

    append = settings.output_path.exists() and iteration > 1
    write_jsonl(settings.output_path, rows, append=append)
    maybe_trim_rollout_file(settings)
    return append, _count_rollout_games(settings.output_path)


def _count_rollout_games(path: Path) -> int:
    if not path.exists():
        return 0
    episodes: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episodes.add(int(json.loads(line)["episode"]))
    return len(episodes)


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


def choose_pool_deck(pool: list[tuple[str, list[int]]], episode: int) -> tuple[str, list[int]]:
    if not pool:
        raise ValueError("agent deck pool is empty")
    return pool[episode % len(pool)]


def _evaluate_vs_fixed_opponent(
    simulator: SimulatorState,
    *,
    agent_fn: Callable[[dict], list[int]],
    opponent_fn: Callable[[dict], list[int]],
    agent_deck: list[int],
    agent_name: str,
    opponent_deck: list[int],
    opponent_name: str,
    games: int,
    start_episode: int,
    eval_batch_start: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    results: list[int] = []
    agent_a_wins = 0
    agent_a_losses = 0
    draws = 0
    seat_epoch_base = start_episode if eval_batch_start is None else eval_batch_start

    game_range: Any = range(games)
    if progress_desc is not None:
        game_range = tqdm(range(games), desc=progress_desc, unit="game", leave=False)

    for game_index in game_range:
        episode = start_episode + game_index
        agent_seat = (episode - seat_epoch_base) % 2
        if agent_seat == 0:
            deck0, deck0_name = agent_deck, agent_name
            deck1, deck1_name = opponent_deck, opponent_name
            seat0, seat1 = agent_fn, opponent_fn
        else:
            deck0, deck0_name = opponent_deck, opponent_name
            deck1, deck1_name = agent_deck, agent_name
            seat0, seat1 = opponent_fn, agent_fn

        rows = play_match(
            episode,
            deck0,
            deck1,
            simulator,
            seat0,
            seat1,
            deck0_name=deck0_name,
            deck1_name=deck1_name,
        )
        if not rows:
            continue
        result = int(next(row for row in reversed(rows) if row.get("terminal")).get("result", -1))
        results.append(result)
        if result == 2:
            draws += 1
        elif result == agent_seat:
            agent_a_wins += 1
        elif result >= 0:
            agent_a_losses += 1

        if progress_desc is not None and isinstance(game_range, tqdm):
            decided = agent_a_wins + agent_a_losses
            game_range.set_postfix(
                wins=agent_a_wins,
                wr=f"{(agent_a_wins / decided) if decided else 0:.0%}",
            )

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
        "opponent": opponent_name,
    }


def _collect_games_vs_baselines_sequential(
    simulator: SimulatorState,
    *,
    baselines: list[BaselineAgent],
    agent_deck_pool: list[tuple[str, list[int]]],
    current_runtime: PolicyRuntime,
    games: int,
    start_episode: int,
    collect_start_episode: int,
    use_beam: bool,
    beam_config: BeamSearchConfig | None = None,
    progress_desc: str | None = "baseline games",
) -> tuple[list[dict[str, Any]], str, str]:
    rows: list[dict[str, Any]] = []
    session = current_runtime.new_session()
    deck_name = ""
    baseline_name = ""
    wins = losses = draws = 0

    game_range: Any = range(games)
    if progress_desc is not None:
        game_range = tqdm(game_range, desc=progress_desc, unit="game", leave=False, dynamic_ncols=True)

    for offset in game_range:
        episode = start_episode + offset
        baseline = baselines[episode % len(baselines)]
        deck_name, our_deck = choose_pool_deck(agent_deck_pool, episode)
        baseline_name = baseline.name
        session.reset()
        agent_seat = (episode - collect_start_episode) % 2
        our_fn = make_policy_fn(
            current_runtime, session, our_deck, use_beam=use_beam, beam_config=beam_config
        )

        if agent_seat == 0:
            deck0, deck0_name = our_deck, deck_name
            deck1, deck1_name = baseline.deck, baseline.agent_id
            agent0, agent1 = our_fn, baseline.act
        else:
            deck0, deck0_name = baseline.deck, baseline.agent_id
            deck1, deck1_name = our_deck, deck_name
            agent0, agent1 = baseline.act, our_fn

        match_rows = play_match(
            episode,
            deck0,
            deck1,
            simulator,
            agent0,
            agent1,
            deck0_name=deck0_name,
            deck1_name=deck1_name,
        )
        rows.extend(match_rows)
        result = _terminal_result(match_rows)
        if result is not None:
            wins, losses, draws = _record_seat_outcome(
                result, agent_seat, wins=wins, losses=losses, draws=draws
            )
        if progress_desc is not None and isinstance(game_range, tqdm):
            game_range.set_postfix(
                **_win_rate_postfix(wins, losses, draws=draws, our=deck_name[:10], vs=baseline.agent_id)
            )

    return rows, deck_name, baseline_name


def _collect_baselines_range(task: tuple[Any, ...]) -> tuple[int, list[dict[str, Any]]]:
    import warnings

    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    (
        root_str,
        start_episode,
        game_count,
        collect_start_episode,
        agent_checkpoint,
        inference_device,
        cg_lib_path,
        worker_settings,
    ) = task
    root = Path(root_str)
    simulator = load_simulator(root)
    if not simulator.available:
        raise RuntimeError("CABT simulator is not available in baseline worker")

    baselines = load_baseline_agents(
        root,
        baseline_dir=worker_settings.baseline_dir or "baselines/official",
        cg_lib_path=cg_lib_path,
        quiet=True,
    )
    runtime = PolicyRuntime(Path(agent_checkpoint), device=torch.device(inference_device))
    rows, _, _ = _collect_games_vs_baselines_sequential(
        simulator,
        baselines=baselines,
        agent_deck_pool=worker_settings.agent_deck_pool,
        current_runtime=runtime,
        games=game_count,
        start_episode=start_episode,
        collect_start_episode=collect_start_episode,
        use_beam=worker_settings.use_beam,
        beam_config=worker_settings.beam_config,
        progress_desc=None,
    )
    return start_episode, rows


def collect_games_vs_baselines(
    simulator: SimulatorState,
    *,
    baselines: list[BaselineAgent],
    agent_deck_pool: list[tuple[str, list[int]]],
    current_runtime: PolicyRuntime,
    games: int,
    start_episode: int,
    use_beam: bool,
    beam_config: BeamSearchConfig | None = None,
    settings: SelfPlaySettings | None = None,
    root: Path | None = None,
    current_checkpoint: Path | None = None,
    device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Play our transformer vs official baseline agents; rotate our deck each game."""
    if not baselines:
        raise ValueError("baseline agent list is empty")

    workers = resolve_self_play_workers(settings, games=games) if settings is not None else 1
    if workers <= 1 or root is None or current_checkpoint is None or settings is None:
        return _collect_games_vs_baselines_sequential(
            simulator,
            baselines=baselines,
            agent_deck_pool=agent_deck_pool,
            current_runtime=current_runtime,
            games=games,
            start_episode=start_episode,
            collect_start_episode=start_episode,
            use_beam=use_beam,
            beam_config=beam_config,
        )

    worker_settings = _baseline_worker_settings(settings)
    inference_device = str(device or torch.device("cpu"))
    chunks = episode_chunks(games, workers, max_chunk_size=PARALLEL_PROGRESS_CHUNK_GAMES)
    tasks = [
        (
            str(root),
            start_episode + offset,
            stop - offset,
            start_episode,
            str(current_checkpoint),
            inference_device,
            simulator.lib_path,
            worker_settings,
        )
        for offset, stop in chunks
    ]
    chunk_rows: list[tuple[int, list[dict[str, Any]]]] = []
    tqdm.write(
        f"baseline games: {workers} workers, {len(tasks)} tasks "
        f"(up to {max(stop - start for start, stop in chunks)} games per progress tick)"
    )
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(_collect_baselines_range, tasks)
        wins = losses = draws = 0
        with tqdm(
            total=games,
            desc="baseline games",
            unit="game",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.3,
            file=sys.stderr,
        ) as progress:
            for chunk_start, chunk in iterator:
                chunk_rows.append((chunk_start, chunk))
                for episode in sorted({int(row["episode"]) for row in chunk}):
                    ep_rows = [row for row in chunk if int(row["episode"]) == episode]
                    result = _terminal_result(ep_rows)
                    if result is None:
                        continue
                    agent_seat = (episode - start_episode) % 2
                    wins, losses, draws = _record_seat_outcome(
                        result, agent_seat, wins=wins, losses=losses, draws=draws
                    )
                progress.update(len({int(row["episode"]) for row in chunk}))
                progress.set_postfix(**_win_rate_postfix(wins, losses, draws=draws), refresh=True)
    chunk_rows.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    for _, chunk in chunk_rows:
        rows.extend(chunk)

    last_episode = start_episode + max(0, games - 1)
    last_baseline = baselines[last_episode % len(baselines)]
    last_deck_name, _ = choose_pool_deck(agent_deck_pool, last_episode)
    return rows, last_deck_name, last_baseline.name


def _merge_fixed_opponent_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[int] = []
    agent_a_wins = 0
    agent_a_losses = 0
    draws = 0
    opponent = reports[0]["opponent"] if reports else ""
    for report in reports:
        results.extend(report["results"])
        agent_a_wins += int(report["agent_a"]["wins"])
        agent_a_losses += int(report["agent_a"]["losses"])
        draws += int(report["agent_a"]["draws"])
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
        "opponent": opponent,
    }


def _evaluate_vs_baseline_range(task: tuple[Any, ...]) -> tuple[str, int, int, dict[str, Any]]:
    import warnings

    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    (
        root_str,
        chunk_start_episode,
        game_count,
        eval_batch_start,
        agent_checkpoint,
        agent_deck,
        agent_name,
        opponent_agent_id,
        opponent_name,
        inference_device,
        cg_lib_path,
        worker_settings,
    ) = task
    root = Path(root_str)
    simulator = load_simulator(root)
    if not simulator.available:
        raise RuntimeError("CABT simulator is not available in baseline eval worker")

    baselines = load_baseline_agents(
        root,
        baseline_dir=worker_settings.baseline_dir or "baselines/official",
        cg_lib_path=cg_lib_path,
        quiet=True,
    )
    baseline = next(agent for agent in baselines if agent.agent_id == opponent_agent_id)
    runtime = PolicyRuntime(Path(agent_checkpoint), device=torch.device(inference_device))
    session = runtime.new_session()
    agent_fn = make_policy_fn(
        runtime,
        session,
        agent_deck,
        use_beam=worker_settings.use_beam,
        beam_config=worker_settings.beam_config,
    )
    report = _evaluate_vs_fixed_opponent(
        simulator,
        agent_fn=agent_fn,
        opponent_fn=baseline.act,
        agent_deck=agent_deck,
        agent_name=agent_name,
        opponent_deck=baseline.deck,
        opponent_name=opponent_name,
        games=game_count,
        start_episode=chunk_start_episode,
        eval_batch_start=eval_batch_start,
        progress_desc=None,
    )
    return opponent_agent_id, eval_batch_start, chunk_start_episode, report


def evaluate_vs_baselines(
    simulator: SimulatorState,
    *,
    agent_fn: Callable[[dict], list[int]],
    agent_deck: list[int],
    agent_name: str,
    baselines: list[BaselineAgent],
    games: int,
    start_episode: int,
    settings: SelfPlaySettings | None = None,
    root: Path | None = None,
    agent_checkpoint: Path | None = None,
    device: torch.device | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Win rate vs each official baseline agent."""
    per_baseline: dict[str, dict[str, float]] = {}
    games_each = max(1, games // max(1, len(baselines)))
    workers = resolve_self_play_workers(settings, games=games) if settings is not None else 1

    if workers <= 1 or root is None or agent_checkpoint is None or settings is None:
        for index, baseline in enumerate(baselines):
            report = _evaluate_vs_fixed_opponent(
                simulator,
                agent_fn=agent_fn,
                opponent_fn=baseline.act,
                agent_deck=agent_deck,
                agent_name=agent_name,
                opponent_deck=baseline.deck,
                opponent_name=baseline.name,
                games=games_each,
                start_episode=start_episode + index * games_each,
                progress_desc=f"eval vs {baseline.agent_id}",
            )
            per_baseline[baseline.agent_id] = report["agent_a"]
        return per_baseline, summarize_baseline_eval(per_baseline)

    worker_settings = _baseline_worker_settings(settings)
    inference_device = str(device or torch.device("cpu"))
    total_eval_games = games_each * len(baselines)
    tasks: list[tuple[Any, ...]] = []
    for index, baseline in enumerate(baselines):
        eval_batch_start = start_episode + index * games_each
        for offset, stop in episode_chunks(
            games_each,
            workers,
            max_chunk_size=PARALLEL_PROGRESS_CHUNK_GAMES,
        ):
            tasks.append(
                (
                    str(root),
                    eval_batch_start + offset,
                    stop - offset,
                    eval_batch_start,
                    str(agent_checkpoint),
                    agent_deck,
                    agent_name,
                    baseline.agent_id,
                    baseline.name,
                    inference_device,
                    simulator.lib_path,
                    worker_settings,
                )
            )

    tqdm.write(
        f"baseline eval: {workers} workers, {len(tasks)} tasks "
        f"({total_eval_games} games vs {len(baselines)} baselines)"
    )
    reports_by_baseline: dict[str, list[dict[str, Any]]] = {baseline.agent_id: [] for baseline in baselines}
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(_evaluate_vs_baseline_range, tasks)
        wins = losses = draws = 0
        with tqdm(
            total=total_eval_games,
            desc="baseline eval",
            unit="game",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.3,
            file=sys.stderr,
        ) as progress:
            for opponent_id, _batch_start, _chunk_start, report in iterator:
                reports_by_baseline[opponent_id].append(report)
                chunk_games = int(report["agent_a"]["games"])
                chunk_wins = int(report["agent_a"]["wins"])
                chunk_losses = int(report["agent_a"]["losses"])
                chunk_draws = int(report["agent_a"]["draws"])
                wins += chunk_wins
                losses += chunk_losses
                draws += chunk_draws
                progress.update(chunk_games)
                progress.set_postfix(**_win_rate_postfix(wins, losses, draws=draws), refresh=True)

    for baseline in baselines:
        merged = _merge_fixed_opponent_reports(reports_by_baseline[baseline.agent_id])
        per_baseline[baseline.agent_id] = merged["agent_a"]
    return per_baseline, summarize_baseline_eval(per_baseline)


def annotate_matchup_metadata(rows: list[dict[str, Any]], matchup: FieldMatchup) -> None:
    """JSONL metadata only — not used in feature encoding."""
    if matchup.opponent_placement is None:
        return
    for row in rows:
        row["opponent_placement"] = matchup.opponent_placement


def _seat_agent(
    runtime: PolicyRuntime,
    session: PolicySession,
    matchup: FieldMatchup,
    seat: int,
    *,
    use_beam: bool,
    beam_config: BeamSearchConfig | None = None,
) -> Callable[[dict], list[int]]:
    deck = matchup.deck_for_seat(seat)
    return make_policy_fn(runtime, session, deck, use_beam=use_beam, beam_config=beam_config)


def _evaluate_games_sequential(
    simulator: SimulatorState,
    *,
    agent_fn: Callable[[dict], list[int]],
    opponent_fn: Callable[[dict], list[int]],
    agent_name: str,
    agent_deck: list[int],
    games: int,
    settings: SelfPlaySettings,
    start_episode: int,
    use_target_pool: bool,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    results: list[int] = []
    matchup_names: list[str] = []
    agent_a_wins = 0
    agent_a_losses = 0
    draws = 0

    game_range = range(games)
    if progress_desc is not None:
        game_range = tqdm(game_range, desc=progress_desc, unit="game", leave=False)

    for game_index in game_range:
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
        if not rows:
            continue
        result = int(next(row for row in reversed(rows) if row.get("terminal")).get("result", -1))
        results.append(result)
        if result == 2:
            draws += 1
        elif result == matchup.agent_seat:
            agent_a_wins += 1
        elif result >= 0:
            agent_a_losses += 1

        if progress_desc is not None and isinstance(game_range, tqdm):
            game_range.set_postfix(
                wins=agent_a_wins,
                losses=agent_a_losses,
                wr=f"{(agent_a_wins / max(1, agent_a_wins + agent_a_losses)):.0%}",
            )

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


def _eval_agent_vs_field_range(task: tuple[Any, ...]) -> tuple[int, dict[str, Any]]:
    (
        root_str,
        start_episode,
        game_count,
        agent_name,
        agent_deck,
        agent_checkpoint,
        use_target_pool,
        settings,
    ) = task
    simulator = load_simulator(Path(root_str))
    if not simulator.available or simulator.to_observation_class is None:
        raise RuntimeError("CABT simulator is not available in eval worker")

    matchup_settings = _matchup_settings(settings)
    agent_runtime = PolicyRuntime(Path(agent_checkpoint), device=torch.device("cpu"))
    agent_session = agent_runtime.new_session()
    agent_fn = make_policy_fn(
        agent_runtime,
        agent_session,
        agent_deck,
        use_beam=matchup_settings.use_beam,
        beam_config=matchup_settings.beam_config,
    )
    opponent_fn = make_random_agent(simulator.to_observation_class)
    report = _evaluate_games_sequential(
        simulator,
        agent_fn=agent_fn,
        opponent_fn=opponent_fn,
        agent_name=agent_name,
        agent_deck=agent_deck,
        games=game_count,
        settings=matchup_settings,
        start_episode=start_episode,
        use_target_pool=use_target_pool,
    )
    return start_episode, report


def _merge_eval_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[int] = []
    matchups: list[str] = []
    agent_a_wins = 0
    agent_a_losses = 0
    draws = 0
    for report in reports:
        results.extend(report["results"])
        matchups.extend(report["matchups"])
        agent_a_wins += int(report["agent_a"]["wins"])
        agent_a_losses += int(report["agent_a"]["losses"])
        draws += int(report["agent_a"]["draws"])
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
        "matchups": matchups,
    }


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
    root: Path | None = None,
    agent_checkpoint: Path | None = None,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    if games <= 0:
        return {
            "agent_a": {"games": 0.0, "wins": 0.0, "losses": 0.0, "draws": 0.0, "win_rate": 0.0},
            "results": [],
            "matchups": [],
        }

    workers = resolve_self_play_workers(settings, games=games)
    if workers <= 1 or root is None or agent_checkpoint is None:
        return _evaluate_games_sequential(
            simulator,
            agent_fn=agent_fn,
            opponent_fn=opponent_fn,
            agent_name=agent_name,
            agent_deck=agent_deck,
            games=games,
            settings=settings,
            start_episode=start_episode,
            use_target_pool=use_target_pool,
            progress_desc=progress_desc,
        )

    matchup_settings = _matchup_settings(settings)
    tasks = [
        (
            str(root),
            start_episode + offset,
            stop - offset,
            agent_name,
            agent_deck,
            str(agent_checkpoint),
            use_target_pool,
            matchup_settings,
        )
        for offset, stop in episode_chunks(games, workers)
    ]
    chunk_reports: list[tuple[int, dict[str, Any]]] = []
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(_eval_agent_vs_field_range, tasks)
        if progress_desc is not None:
            with tqdm(total=games, desc=progress_desc, unit="game", leave=False) as progress:
                for start_ep, report in iterator:
                    chunk_reports.append((start_ep, report))
                    progress.update(int(report["agent_a"]["games"]))
        else:
            chunk_reports = list(iterator)
    chunk_reports.sort(key=lambda item: item[0])
    return _merge_eval_reports([report for _, report in chunk_reports])


def _collect_self_play_games_sequential(
    simulator: SimulatorState,
    agent_deck: list[int],
    *,
    agent_name: str,
    games: int,
    start_episode: int,
    collect_start_episode: int,
    current_runtime: PolicyRuntime,
    opponent_runtime: PolicyRuntime,
    settings: SelfPlaySettings,
    progress_desc: str | None = "self-play games",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_session = current_runtime.new_session()
    opponent_session = opponent_runtime.new_session()
    use_beam = settings.use_beam
    beam_config = settings.beam_config
    wins = losses = draws = 0

    game_range: Any = range(games)
    if progress_desc is not None:
        game_range = tqdm(range(games), desc=progress_desc, unit="game", leave=False, dynamic_ncols=True)

    for offset in game_range:
        episode = start_episode + offset
        current_session.reset()
        opponent_session.reset()
        matchup = resolve_matchup(episode, agent_name, agent_deck, settings)
        game_index = episode - collect_start_episode
        current_plays_seat0 = game_index % 2 == 0

        if current_plays_seat0:
            agent0 = _seat_agent(
                current_runtime, current_session, matchup, 0, use_beam=use_beam, beam_config=beam_config
            )
            agent1 = _seat_agent(
                opponent_runtime, opponent_session, matchup, 1, use_beam=use_beam, beam_config=beam_config
            )
        else:
            agent0 = _seat_agent(
                opponent_runtime, opponent_session, matchup, 0, use_beam=use_beam, beam_config=beam_config
            )
            agent1 = _seat_agent(
                current_runtime, current_session, matchup, 1, use_beam=use_beam, beam_config=beam_config
            )

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
        our_seat = 0 if current_plays_seat0 else 1
        result = _terminal_result(match_rows)
        if result is not None:
            wins, losses, draws = _record_seat_outcome(
                result, our_seat, wins=wins, losses=losses, draws=draws
            )
        if progress_desc is not None and isinstance(game_range, tqdm):
            game_range.set_postfix(**_win_rate_postfix(wins, losses, draws=draws))
    return rows


def _collect_self_play_range(task: tuple[Any, ...]) -> tuple[int, list[dict[str, Any]]]:
    import warnings

    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    (
        root_str,
        agent_name,
        agent_deck,
        start_episode,
        game_count,
        collect_start_episode,
        current_checkpoint,
        opponent_checkpoint,
        settings,
    ) = task
    simulator = load_simulator(Path(root_str))
    if not simulator.available:
        raise RuntimeError("CABT simulator is not available in self-play worker")

    matchup_settings = _matchup_settings(settings)
    device = torch.device("cpu")
    current_runtime = PolicyRuntime(Path(current_checkpoint), device=device)
    opponent_runtime = PolicyRuntime(Path(opponent_checkpoint), device=device)
    rows = _collect_self_play_games_sequential(
        simulator,
        agent_deck,
        agent_name=agent_name,
        games=game_count,
        start_episode=start_episode,
        collect_start_episode=collect_start_episode,
        current_runtime=current_runtime,
        opponent_runtime=opponent_runtime,
        settings=matchup_settings,
        progress_desc=None,
    )
    return start_episode, rows


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
    root: Path | None = None,
    current_checkpoint: Path | None = None,
    opponent_checkpoint: Path | None = None,
    progress_desc: str | None = "self-play games",
) -> list[dict[str, Any]]:
    workers = resolve_self_play_workers(settings, games=games)
    if workers <= 1:
        return _collect_self_play_games_sequential(
            simulator,
            agent_deck,
            agent_name=agent_name,
            games=games,
            start_episode=start_episode,
            collect_start_episode=start_episode,
            current_runtime=current_runtime,
            opponent_runtime=opponent_runtime,
            settings=settings,
            progress_desc=progress_desc,
        )

    if root is None or current_checkpoint is None or opponent_checkpoint is None:
        raise ValueError("parallel self-play requires root and checkpoint paths")

    matchup_settings = _matchup_settings(settings)
    chunks = episode_chunks(games, workers, max_chunk_size=PARALLEL_PROGRESS_CHUNK_GAMES)
    tasks = [
        (
            str(root),
            agent_name,
            agent_deck,
            start_episode + offset,
            stop - offset,
            start_episode,
            str(current_checkpoint),
            str(opponent_checkpoint),
            matchup_settings,
        )
        for offset, stop in chunks
    ]
    chunk_rows: list[tuple[int, list[dict[str, Any]]]] = []
    if progress_desc is not None:
        tqdm.write(
            f"{progress_desc}: {workers} workers, {len(tasks)} tasks "
            f"(up to {max(stop - start for start, stop in chunks)} games per progress tick)"
        )
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(_collect_self_play_range, tasks)
        if progress_desc is not None:
            wins = losses = draws = 0
            with tqdm(
                total=games,
                desc=progress_desc,
                unit="game",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.3,
                file=sys.stderr,
            ) as progress:
                for start_ep, chunk in iterator:
                    chunk_rows.append((start_ep, chunk))
                    for episode in sorted({int(row["episode"]) for row in chunk}):
                        ep_rows = [row for row in chunk if int(row["episode"]) == episode]
                        result = _terminal_result(ep_rows)
                        if result is None:
                            continue
                        game_index = episode - start_episode
                        our_seat = 0 if game_index % 2 == 0 else 1
                        wins, losses, draws = _record_seat_outcome(
                            result, our_seat, wins=wins, losses=losses, draws=draws
                        )
                    progress.update(len({int(row["episode"]) for row in chunk}))
                    progress.set_postfix(**_win_rate_postfix(wins, losses, draws=draws), refresh=True)
        else:
            chunk_rows = list(iterator)
    chunk_rows.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    for _, chunk in chunk_rows:
        rows.extend(chunk)
    return rows


def train_on_rollouts(
    config: dict[str, Any],
    device: torch.device,
    *,
    data_path: Path,
    checkpoint_path: Path | None = None,
    iteration: int = 1,
    apply_lr_warmup: bool = False,
    periodic_checkpoint_path: Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Retrain on self-play JSONL, optionally capped to the most recent games."""
    train_config = dict(config)
    train_config["training_data_path"] = data_path
    train_config["generated_path"] = data_path
    train_config["require_cabt_eval_data"] = False
    train_config["require_training_matchup_diversity"] = False
    # Self-play / baseline rollouts are not top-of-ladder replays; that data-quality
    # gate only applies to the bootstrap corpus.
    train_config["require_top_of_ladder_data"] = False

    sp = dict(config.get("self_play", {}))
    train_cfg = dict(train_config.get("training", {}))
    window = sp.get("train_window_games")
    if window is not None and int(window) > 0:
        train_cfg["games"] = int(window)
        train_cfg["recent_games"] = True
        window_note = f"last {window} games"
    else:
        train_config["dataset_games"] = None
        window_note = "all collected games"
    if sp.get("train_epochs") is not None:
        train_cfg["epochs"] = int(sp["train_epochs"])
    train_config["training"] = train_cfg

    base_lr = float(train_config["model"]["learning_rate"])
    warmup_iters = int(sp.get("warmup_iterations", 0) or 0)
    warmup_mult = float(sp.get("warmup_lr_multiplier", 1.0) or 1.0)
    if apply_lr_warmup and warmup_iters > 0 and iteration <= warmup_iters and warmup_mult > 1.0:
        model_cfg = dict(train_config["model"])
        model_cfg["learning_rate"] = base_lr * warmup_mult
        train_config["model"] = model_cfg
        lr_note = (
            f"warmup lr {model_cfg['learning_rate']:g} "
            f"({warmup_mult:g}x, iter {iteration}/{warmup_iters})"
        )
    else:
        lr_note = f"lr {base_lr:g}"

    row_count = sum(1 for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(
        f"self-play retrain on {device}: {data_path} "
        f"({row_count:,} rows on disk, train on {window_note}, {lr_note})"
    )

    tensors = prepare_training_tensors(train_config, device)
    model = build_model(train_config, tensors, device)
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded weights from {checkpoint_path.name} before self-play training")
    # Periodic .latest.pt beside the iteration checkpoint; no cross-iteration resume.
    checkpoint_every = (
        int(train_cfg.get("checkpoint_every", 0)) if periodic_checkpoint_path is not None else 0
    )
    training_report = train_model(
        model,
        tensors,
        train_config,
        device,
        checkpoint_path=periodic_checkpoint_path,
        checkpoint_every_epochs=checkpoint_every,
        resume_path=None,
    )
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
    root: Path,
) -> dict[str, Any]:
    if not simulator.available:
        raise RuntimeError("CABT simulator is required for self-play")

    workers = resolve_self_play_workers(settings, games=settings.games_per_iteration)
    opponent_path = pool.sample(exclude=current_checkpoint) or current_checkpoint
    inference_device = device if workers <= 1 else torch.device("cpu")
    current_runtime = PolicyRuntime(current_checkpoint, device=inference_device)
    opponent_runtime = PolicyRuntime(opponent_path, device=inference_device)

    field_note = (
        f"field={len(settings.field_pool)} decks ({settings.matchup_mode})"
        if settings.use_field
        else "mirror deck"
    )
    worker_note = f"workers={workers}" + (" (inference on cpu)" if workers > 1 else "")
    print(
        f"self-play iter {iteration}: collect {settings.games_per_iteration} games "
        f"({worker_note}, train_device={device}, current={current_checkpoint.name}, "
        f"opponent={opponent_path.name}, beam={settings.use_beam}, {field_note})"
    )
    rows = collect_self_play_games(
        simulator,
        agent_deck,
        agent_name=agent_name,
        games=settings.games_per_iteration,
        start_episode=start_episode,
        current_runtime=current_runtime,
        opponent_runtime=opponent_runtime,
        settings=settings,
        root=root,
        current_checkpoint=current_checkpoint,
        opponent_checkpoint=opponent_path,
        progress_desc="self-play",
    )
    unique_matchups = len({(row.get("deck0"), row.get("deck1")) for row in rows if int(row.get("step", -1)) == 0})
    overwrite, kept_games = write_rollout_buffer(settings, rows, iteration=iteration)
    print(
        f"self-play data -> {settings.output_path} ({len(rows):,} rows, "
        f"{unique_matchups} unique deck pairings, "
        f"mode={'rolling' if overwrite else 'accumulate'}, "
        f"buffer={kept_games} games, train_window={settings.train_window_games})"
    )

    random_agent = make_random_agent(simulator.to_observation_class)
    current_session = current_runtime.new_session()
    current_fn = make_policy_fn(
        current_runtime,
        current_session,
        agent_deck,
        use_beam=settings.use_beam,
        beam_config=settings.beam_config,
    )

    eval_vs_random = evaluate_agent_vs_field(
        simulator,
        current_fn,
        agent_name=agent_name,
        agent_deck=agent_deck,
        opponent_fn=random_agent,
        games=settings.eval_games,
        settings=settings,
        start_episode=start_episode + settings.games_per_iteration,
        use_target_pool=True,
        root=root,
        agent_checkpoint=current_checkpoint,
        progress_desc=f"iter {iteration} eval",
    )
    sample_matchups = eval_vs_random.get("matchups", [])[:5]
    print(
        f"eval vs random (placement<={settings.target_rank}): "
        f"win_rate={eval_vs_random['agent_a']['win_rate']:.1%} "
        f"({int(eval_vs_random['agent_a']['wins'])}/{int(eval_vs_random['agent_a']['games'])}) "
        f"sample opponents={sample_matchups}"
    )

    training_report: dict[str, Any] | None = None
    saved_checkpoint = current_checkpoint
    if settings.train_after_collect:
        print("training on self-play rollouts...")
        iter_path = settings.checkpoint_dir / f"iter_{iteration:03d}.pt"
        model, tensors, training_report = train_on_rollouts(
            config,
            device,
            data_path=settings.output_path,
            checkpoint_path=current_checkpoint,
            iteration=iteration,
            apply_lr_warmup=True,
            periodic_checkpoint_path=iter_path,
        )
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
        "phase": "transformer",
        "rows_collected": len(rows),
        "unique_matchups": unique_matchups,
        "data_path": str(settings.output_path),
        "current_checkpoint": str(current_checkpoint),
        "opponent_checkpoint": str(opponent_path),
        "saved_checkpoint": str(saved_checkpoint),
        "eval_vs_random": eval_vs_random["agent_a"],
        "field_decks": len(settings.field_pool),
        "training_report": training_report,
    }


def _save_per_deck_checkpoint(
    *,
    source_checkpoint: Path,
    per_deck_dir: Path | None,
    deck_name: str,
    model: Any | None,
    tensors: Any | None,
    config: dict[str, Any],
    training_report: dict[str, Any] | None,
) -> Path | None:
    if per_deck_dir is None:
        return None
    per_deck_dir.mkdir(parents=True, exist_ok=True)
    slug = Path(deck_name).stem if deck_name else "unknown"
    target = per_deck_dir / f"{slug}.pt"
    if model is not None and tensors is not None and training_report is not None:
        save_checkpoint(
            model=model,
            tensors=tensors,
            config=config,
            training_report=training_report,
            output_path=target,
        )
    elif source_checkpoint.exists():
        target.write_bytes(source_checkpoint.read_bytes())
    else:
        return None
    print(f"per-deck checkpoint -> {target}")
    return target


def run_baseline_iteration(
    *,
    iteration: int,
    config: dict[str, Any],
    simulator: SimulatorState,
    agent_deck: list[int],
    agent_name: str,
    settings: SelfPlaySettings,
    device: torch.device,
    current_checkpoint: Path,
    start_episode: int,
    root: Path | None = None,
) -> dict[str, Any]:
    if not simulator.available:
        raise RuntimeError("CABT simulator is required for baseline training")
    if not settings.baseline_agents:
        raise RuntimeError("baseline agents not loaded — see baselines/README.md")
    if not settings.agent_deck_pool:
        raise RuntimeError("agent deck pool is empty — set SELF_PLAY_AGENT_DECK_DIR")

    current_runtime = PolicyRuntime(current_checkpoint, device=device)
    workers = resolve_self_play_workers(settings, games=settings.games_per_iteration)
    play_root = root or settings.checkpoint_dir.parent.parent
    worker_note = f"workers={workers}"
    if workers > 1:
        worker_note += f", inference={device}"
    print(
        f"baseline iter {iteration}: collect {settings.games_per_iteration} games "
        f"({worker_note}) vs {len(settings.baseline_agents)} official agents, "
        f"our deck pool={len(settings.agent_deck_pool)}, checkpoint={current_checkpoint.name}"
    )
    rows, last_deck_name, last_baseline = collect_games_vs_baselines(
        simulator,
        baselines=settings.baseline_agents,
        agent_deck_pool=settings.agent_deck_pool,
        current_runtime=current_runtime,
        games=settings.games_per_iteration,
        start_episode=start_episode,
        use_beam=settings.use_beam,
        beam_config=settings.beam_config,
        settings=settings,
        root=play_root,
        current_checkpoint=current_checkpoint,
        device=device,
    )
    overwrite, kept_games = write_rollout_buffer(settings, rows, iteration=iteration)
    print(
        f"baseline data -> {settings.output_path} ({len(rows):,} rows, "
        f"last our={last_deck_name} vs {last_baseline}, "
        f"mode={'rolling' if overwrite else 'accumulate'}, "
        f"buffer={kept_games} games, train_window={settings.train_window_games})"
    )

    eval_deck_name, eval_deck = choose_pool_deck(settings.agent_deck_pool, start_episode)
    eval_session = current_runtime.new_session()
    eval_fn = make_policy_fn(
        current_runtime,
        eval_session,
        eval_deck,
        use_beam=settings.use_beam,
        beam_config=settings.beam_config,
    )
    per_baseline, aggregate = evaluate_vs_baselines(
        simulator,
        agent_fn=eval_fn,
        agent_deck=eval_deck,
        agent_name=eval_deck_name,
        baselines=settings.baseline_agents,
        games=settings.eval_games,
        start_episode=start_episode + settings.games_per_iteration,
        settings=settings,
        root=play_root,
        agent_checkpoint=current_checkpoint,
        device=device,
    )
    for baseline_id, stats in per_baseline.items():
        print(
            f"  vs {baseline_id}: {stats['win_rate']:.1%} "
            f"({int(stats['wins'])}/{int(stats['games'])})"
        )
    print(
        f"baseline eval aggregate: {aggregate['win_rate']:.1%} "
        f"(gate {settings.baseline_win_rate_threshold:.0%})"
    )

    training_report: dict[str, Any] | None = None
    saved_checkpoint = current_checkpoint
    model = None
    tensors = None
    if settings.train_after_collect:
        print("training on baseline rollout data...")
        iter_path = settings.checkpoint_dir / f"baseline_{iteration:03d}.pt"
        model, tensors, training_report = train_on_rollouts(
            config,
            device,
            data_path=settings.output_path,
            checkpoint_path=current_checkpoint,
            iteration=iteration,
            periodic_checkpoint_path=iter_path,
        )
        training_report = save_checkpoint(
            model=model,
            tensors=tensors,
            config=config,
            training_report=training_report,
            output_path=iter_path,
        )
        saved_checkpoint = iter_path

    per_deck_path = _save_per_deck_checkpoint(
        source_checkpoint=saved_checkpoint,
        per_deck_dir=settings.per_deck_checkpoint_dir,
        deck_name=last_deck_name,
        model=model,
        tensors=tensors,
        config=config,
        training_report=training_report,
    )

    return {
        "iteration": iteration,
        "phase": "baseline",
        "rows_collected": len(rows),
        "data_path": str(settings.output_path),
        "current_checkpoint": str(current_checkpoint),
        "saved_checkpoint": str(saved_checkpoint),
        "our_deck_pool_size": len(settings.agent_deck_pool),
        "last_our_deck": last_deck_name,
        "last_baseline": last_baseline,
        "eval_deck": eval_deck_name,
        "eval_vs_baselines": per_baseline,
        "eval_vs_baselines_aggregate": aggregate,
        "per_deck_checkpoint": str(per_deck_path) if per_deck_path else None,
        "training_report": training_report,
    }


def run_baseline_phase_loop(
    *,
    config: dict[str, Any],
    simulator: SimulatorState,
    agent_deck: list[int],
    agent_name: str,
    settings: SelfPlaySettings,
    device: torch.device | None = None,
    initial_checkpoint: Path | None = None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    device = device or torch_device()
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if settings.per_deck_checkpoint_dir is not None:
        settings.per_deck_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = settings.checkpoint_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest["phase"] = "baseline"
    print(
        f"baseline training window: {settings.train_window_games} games "
        f"(collect {settings.games_per_iteration} per iteration, "
        f"rollout buffer={'rolling overwrite' if rollout_buffer_overwrites(settings) else 'accumulate+trim'})"
    )
    current_checkpoint = Path(initial_checkpoint or config["output_path"])
    if not current_checkpoint.exists():
        raise FileNotFoundError(f"initial checkpoint not found: {current_checkpoint}")

    start_episode = int(manifest.get("next_episode", 0))
    reports: list[dict[str, Any]] = []
    stop_reason: str | None = None

    for iteration in range(1, settings.iterations + 1):
        print(f"baseline iteration {iteration}/{settings.iterations}")
        report = run_baseline_iteration(
            iteration=iteration,
            config=config,
            simulator=simulator,
            agent_deck=agent_deck,
            agent_name=agent_name,
            settings=settings,
            device=device,
            current_checkpoint=current_checkpoint,
            start_episode=start_episode,
            root=root,
        )
        reports.append(report)
        start_episode += settings.games_per_iteration
        current_checkpoint = Path(report["saved_checkpoint"])
        manifest["next_episode"] = start_episode
        manifest.setdefault("baseline_iterations", []).append(report)
        aggregate_rate = float(report["eval_vs_baselines_aggregate"]["win_rate"])
        if aggregate_rate >= settings.baseline_win_rate_threshold:
            stop_reason = (
                f"baseline gate passed: {aggregate_rate:.1%} >= "
                f"{settings.baseline_win_rate_threshold:.0%} vs official agents"
            )
            manifest["champion"] = report
            print(stop_reason)
            break
        if iteration >= settings.iterations:
            stop_reason = f"baseline iterations exhausted ({settings.iterations})"
            print(stop_reason)

    if stop_reason:
        manifest["stop_reason"] = stop_reason
    manifest["phase"] = (
        "transformer"
        if reports
        and float(reports[-1]["eval_vs_baselines_aggregate"]["win_rate"])
        >= settings.baseline_win_rate_threshold
        else "baseline"
    )
    save_manifest(manifest_path, manifest)
    return reports


def run_curriculum_self_play(
    *,
    config: dict[str, Any],
    simulator: SimulatorState,
    agent_deck: list[int],
    agent_name: str,
    settings: SelfPlaySettings,
    device: torch.device | None = None,
    initial_checkpoint: Path | None = None,
    submit_on_stop: bool = False,
    submit_after_baseline: bool = False,
    submission_message: str = DEFAULT_SUBMISSION_MESSAGE,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Baseline phase vs official Kaggle agents, then transformer self-play at 60% gate."""
    device = device or torch_device()
    play_root = root or Path(config["output_path"]).resolve().parents[2]
    manifest_path = settings.checkpoint_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    phase = str(manifest.get("phase", "baseline"))

    if phase == "transformer":
        print("curriculum: resuming transformer self-play phase")
        return run_self_play_loop(
            config=config,
            simulator=simulator,
            agent_deck=agent_deck,
            agent_name=agent_name,
            settings=settings,
            device=device,
            initial_checkpoint=initial_checkpoint,
            submit_on_stop=submit_on_stop,
            submission_message=submission_message,
            root=play_root,
        )

    if not settings.baseline_agents:
        settings.baseline_agents = load_baseline_agents(
            play_root,
            baseline_dir=settings.baseline_dir or "baselines/official",
            cg_lib_path=simulator.lib_path,
        )

    print(
        f"curriculum phase 1: baseline agents ({len(settings.baseline_agents)}), "
        f"our deck pool={len(settings.agent_deck_pool)}, "
        f"gate={settings.baseline_win_rate_threshold:.0%}"
    )
    baseline_reports = run_baseline_phase_loop(
        config=config,
        simulator=simulator,
        agent_deck=agent_deck,
        agent_name=agent_name,
        settings=settings,
        device=device,
        initial_checkpoint=initial_checkpoint,
        root=play_root,
    )

    if not baseline_reports:
        return baseline_reports

    aggregate = float(baseline_reports[-1]["eval_vs_baselines_aggregate"]["win_rate"])
    if aggregate < settings.baseline_win_rate_threshold:
        print(
            f"baseline gate not reached ({aggregate:.1%} < "
            f"{settings.baseline_win_rate_threshold:.0%}); staying in baseline phase"
        )
        return baseline_reports

    champion_ckpt = Path(baseline_reports[-1]["saved_checkpoint"])

    if submit_after_baseline:
        print(f"baseline gate beaten ({aggregate:.1%}); submitting {champion_ckpt.name} to Kaggle")
        manifest = load_manifest(manifest_path)
        try:
            submission = submit_champion_checkpoint(
                champion_ckpt,
                root=play_root,
                message=f"{submission_message} (baseline gate {aggregate:.0%})",
            )
            manifest["kaggle_submission_baseline"] = submission
        except subprocess.CalledProcessError as exc:
            manifest["kaggle_submission_baseline_error"] = {
                "checkpoint": str(champion_ckpt),
                "returncode": exc.returncode,
                "output": (exc.stdout or "") + (exc.stderr or ""),
            }
            print(f"WARN: baseline-gate Kaggle submission failed: {exc}")
        save_manifest(manifest_path, manifest)

    print("curriculum phase 2: pure transformer self-play")
    if settings.output_path.exists():
        settings.output_path.unlink()
        print(f"cleared baseline rollout buffer -> {settings.output_path}")
    transformer_reports = run_self_play_loop(
        config=config,
        simulator=simulator,
        agent_deck=agent_deck,
        agent_name=agent_name,
        settings=settings,
        device=device,
        initial_checkpoint=champion_ckpt,
        submit_on_stop=submit_on_stop,
        submission_message=submission_message,
        root=play_root,
    )
    return baseline_reports + transformer_reports


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
    print(f"self-play device: {device}")
    window = settings.train_window_games
    window_note = f"last {window} games" if window else "all collected games"
    print(
        f"self-play training buffer: {window_note} "
        f"(collect {settings.games_per_iteration} per iteration), "
        f"warmup={settings.warmup_iterations} iters at {settings.warmup_lr_multiplier:g}x lr"
    )
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = settings.checkpoint_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    pool = OpponentPool(max_size=settings.opponent_pool_size)
    for entry in manifest.get("iterations", []):
        checkpoint = entry.get("saved_checkpoint")
        if checkpoint:
            pool.add(Path(checkpoint))

    current_checkpoint = Path(initial_checkpoint or config["output_path"])
    if not current_checkpoint.exists():
        raise FileNotFoundError(f"initial checkpoint not found: {current_checkpoint}")

    pool.add(current_checkpoint)
    start_episode = int(manifest.get("next_episode", 0))
    reports: list[dict[str, Any]] = []
    best_target_rate = float((manifest.get("champion") or {}).get("eval_vs_random", {}).get("win_rate", -1.0))
    plateau_count = int(manifest.get("plateau_count", 0))
    stop_reason: str | None = None
    play_root = root or settings.checkpoint_dir.parent.parent.parent

    for iteration in range(1, settings.iterations + 1):
        print(f"self-play iteration {iteration}/{settings.iterations}")
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
            root=play_root,
        )
        reports.append(report)
        start_episode += settings.games_per_iteration
        current_checkpoint = Path(report["saved_checkpoint"])
        manifest["next_episode"] = start_episode

        target_rate = float(report["eval_vs_random"]["win_rate"])
        manifest.setdefault("iterations", []).append(report)
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
        submit_root = play_root
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

    agent_deck_pool = _resolve_agent_deck_pool(config, root, agent_name=agent_name, agent_deck=agent_deck)
    beam_config = BeamSearchConfig.from_self_play_config(config)
    if sp.get("use_beam", True):
        print(
            f"self-play beam: width={beam_config.width} "
            f"budget_ms={beam_config.time_budget_ms} max_steps={beam_config.max_search_steps}"
        )

    per_deck_dir = sp.get("per_deck_checkpoint_dir")
    per_deck_checkpoint_dir = (root / per_deck_dir) if per_deck_dir else None

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
        workers=sp.get("workers"),
        baseline_dir=str(sp.get("baseline_dir", "baselines/official")),
        baseline_win_rate_threshold=float(sp.get("baseline_win_rate", 0.60)),
        agent_deck_pool=agent_deck_pool,
        per_deck_checkpoint_dir=per_deck_checkpoint_dir,
        train_window_games=sp.get("train_window_games"),
        trim_rollout_file=bool(sp.get("trim_rollout_file", True)),
        warmup_iterations=int(sp.get("warmup_iterations", 10) or 0),
        warmup_lr_multiplier=float(sp.get("warmup_lr_multiplier", 25.0) or 1.0),
        beam_config=beam_config,
    )


def _resolve_agent_deck_pool(
    config: dict[str, Any],
    root: Path,
    *,
    agent_name: str,
    agent_deck: list[int],
) -> list[tuple[str, list[int]]]:
    sp = dict(config.get("self_play", {}))
    if sp.get("baseline_archetype_decks_only", True):
        try:
            from poke_agent.baseline_agents import resolve_baseline_archetype_deck_pool

            pool = resolve_baseline_archetype_deck_pool(
                root,
                baseline_dir=str(sp.get("baseline_dir", "baselines/official")),
                top_decks_per_archetype=int(sp.get("baseline_top_decks_per_archetype", 3)),
            )
            print(
                f"our-side deck pool: {len(pool)} decks "
                f"(top {int(sp.get('baseline_top_decks_per_archetype', 3))} placements per baseline archetype + official)"
            )
            return pool
        except (FileNotFoundError, ValueError) as exc:
            print(f"baseline archetype deck pool unavailable ({exc}); falling back to agent_deck_dir")

    deck_dir = sp.get("agent_deck_dir")
    if not deck_dir:
        return [(agent_name, agent_deck)]
    try:
        from poke_agent.deck_pool import read_deck_pool

        pool = read_deck_pool(deck_dir, root=root)
        agent_key = tuple(agent_deck)
        filtered = [(name, deck) for name, deck in pool if tuple(deck) != agent_key]
        if filtered:
            print(f"our-side fine-tune deck pool: {len(filtered)} decks from {deck_dir}")
            return filtered
    except (FileNotFoundError, ValueError) as exc:
        print(f"agent deck pool unavailable ({exc}); using submission deck only")
    return [(agent_name, agent_deck)]
