#!/usr/bin/env python3
"""Offline reanalyse: refresh search-improved value/policy targets on hard states.

MuZero-Unplugged / Reanalyse-style relabeling. Streams a rollout JSONL episode by
episode (low RAM), and for the hardest ``--fraction`` of decision states re-runs the
CABT simulator beam with a stronger budget using the given checkpoint. The improved
``search_value`` / ``search_policy_target`` / ``search_policy_kl`` are written back to
an output JSONL; every other row is passed through unchanged.

Hardness (cheap, no model needed for the threshold pass) is the legal-action count plus
a bonus when the prize race is close. Only rows that carry a stored ``observation`` with
a live ``search_begin_input`` handle and a known seat deck can be re-searched; all other
rows are copied through untouched.

Requires the CABT simulator (cg-lib) and a trained checkpoint. This is a batch tool:
it is intentionally NOT wired into the self-play loop.

Examples
--------
    python scripts/reanalyse_rollouts.py data/dragapult_ladder.jsonl \
        --checkpoint outputs/checkpoints/dragapult_fresh.pt \
        --out data/dragapult_ladder_reanalysed.jsonl --fraction 0.2

    python scripts/reanalyse_rollouts.py outputs/rollouts/dragapult_self_play.jsonl \
        --checkpoint outputs/checkpoints/dragapult_fresh.pt --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.config import build_config
from poke_agent.features import seat_deck_from_row
from poke_agent.paths import resolve_root
from poke_agent.simulator import load_simulator


def _iter_episodes(path: Path) -> Iterator[list[dict[str, Any]]]:
    """Yield one episode's rows at a time; JSONL must be grouped by episode id."""
    current: int | None = None
    batch: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            episode = int(row["episode"])
            if current is not None and episode != current:
                yield batch
                batch = []
            current = episode
            batch.append(row)
    if batch:
        yield batch


def _eligible(row: dict[str, Any]) -> bool:
    """A row can be re-searched only with a live search handle, options, and a deck."""
    obs = row.get("observation")
    if not obs or not obs.get("search_begin_input"):
        return False
    select = obs.get("select") or {}
    if not (select.get("option") or []):
        return False
    return seat_deck_from_row(row) is not None


def _row_hardness(row: dict[str, Any]) -> float:
    """Cheap hardness score: branching factor, with a close-prize-race bonus."""
    obs = row.get("observation") or {}
    select = obs.get("select") or {}
    options = select.get("option") or []
    legal = row.get("legal_action_count")
    score = float(int(legal) if legal is not None else len(options))
    self_prize = row.get("self_prize_remaining")
    opp_prize = row.get("opp_prize_remaining")
    if self_prize is not None and opp_prize is not None:
        if abs(int(self_prize) - int(opp_prize)) <= 2:
            score += 5.0
    return score


def _hardness_threshold(in_path: Path, fraction: float) -> tuple[float, int]:
    """First pass: return (threshold, eligible_count) for the top-``fraction`` states."""
    if fraction >= 1.0:
        return float("-inf"), -1
    if fraction <= 0.0:
        return float("inf"), 0
    scores: list[float] = []
    for episode_rows in _iter_episodes(in_path):
        for row in episode_rows:
            if _eligible(row):
                scores.append(_row_hardness(row))
    if not scores:
        return float("inf"), 0
    threshold = float(np.quantile(np.asarray(scores, dtype=np.float64), 1.0 - fraction))
    return threshold, len(scores)


def _build_beam_config(config: dict[str, Any], args: argparse.Namespace) -> Any:
    from poke_agent.beam_search import BeamSearchConfig

    beam_cfg = BeamSearchConfig.from_self_play_config(config)
    beam_cfg.width = int(args.beam_width)
    beam_cfg.num_determinizations = max(1, int(args.determinizations))
    beam_cfg.time_budget_ms = int(args.time_budget_ms)
    beam_cfg.max_search_steps = int(args.max_search_steps)
    beam_cfg.rollout_policy_width = int(args.rollout_policy_width)
    beam_cfg.sim_mode = True  # offline: never gate on the competition clock
    beam_cfg.min_remaining_sec = 0
    return beam_cfg


def reanalyse(args: argparse.Namespace) -> int:
    import torch

    in_path = Path(args.data)
    if not in_path.is_file():
        print(f"ERROR: input rollouts not found: {in_path}", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_reanalysed.jsonl")

    root = resolve_root()
    config = build_config(root)

    simulator = load_simulator(root)
    if not simulator.available:
        print(
            "ERROR: CABT simulator (cg-lib) unavailable; reanalyse needs the simulator to "
            f"re-run search. error={simulator.error}",
            file=sys.stderr,
        )
        return 1

    device = str(args.device)
    if device == "cpu" and args.workers:
        torch.set_num_threads(max(1, int(args.workers)))

    from poke_agent.policy_agent import PolicyRuntime

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    runtime = PolicyRuntime(checkpoint, device=device)
    beam_cfg = _build_beam_config(config, args)

    print(f"reanalyse: {in_path} -> {out_path}")
    print(
        f"checkpoint={checkpoint.name} device={device} fraction={args.fraction} "
        f"beam_width={beam_cfg.width} determinizations={beam_cfg.num_determinizations} "
        f"time_budget_ms={beam_cfg.time_budget_ms}"
    )

    started = time.perf_counter()
    threshold, eligible_count = _hardness_threshold(in_path, float(args.fraction))
    if eligible_count == 0:
        print("reanalyse: no eligible (searchable) states found; copying file unchanged.")
    else:
        target = "all eligible" if eligible_count < 0 else f"top {args.fraction:.0%} of {eligible_count:,}"
        print(f"reanalyse: hardness threshold={threshold:.2f} ({target} eligible states)")

    episodes = 0
    rows_total = 0
    updated = 0
    searched = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for episode_rows in _iter_episodes(in_path):
            episodes += 1
            if args.limit_episodes and episodes > int(args.limit_episodes):
                # Still pass remaining episodes through unchanged.
                for row in episode_rows:
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows_total += 1
                continue

            by_seat: dict[int, list[dict[str, Any]]] = {}
            for row in episode_rows:
                by_seat.setdefault(int(row.get("player", 0)), []).append(row)

            for seat_rows in by_seat.values():
                seat_rows.sort(key=lambda item: int(item.get("step", 0)))
                session = runtime.new_session()
                for row in seat_rows:
                    obs = row.get("observation")
                    if not obs:
                        continue
                    deck = seat_deck_from_row(row)
                    do_beam = (
                        deck is not None
                        and _eligible(row)
                        and _row_hardness(row) >= threshold
                    )
                    try:
                        runtime.choose_action(
                            obs,
                            session,
                            our_deck=deck,
                            use_beam=do_beam,
                            beam_config=beam_cfg,
                        )
                    except Exception:
                        continue
                    if do_beam:
                        searched += 1
                        diagnostics = session.last_search_diagnostics
                        if diagnostics is not None:
                            row.update(diagnostics.to_row_fields())
                            updated += 1

            for row in episode_rows:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                rows_total += 1
            if episodes % 100 == 0:
                rate = episodes / max(1e-6, time.perf_counter() - started)
                print(f"  ...{episodes:,} episodes ({rate:.1f}/s), {updated:,} states relabeled")

    elapsed = time.perf_counter() - started
    print(
        f"reanalyse done: {episodes:,} episodes, {rows_total:,} rows, "
        f"{searched:,} states searched, {updated:,} relabeled in {elapsed:.1f}s -> {out_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline reanalyse of rollout JSONL hard states.")
    parser.add_argument("data", type=Path, help="input rollout JSONL (grouped by episode)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="model checkpoint .pt")
    parser.add_argument("--out", type=Path, default=None, help="output JSONL (default: <stem>_reanalysed.jsonl)")
    parser.add_argument("--fraction", type=float, default=0.2, help="top fraction of hard states to re-search")
    parser.add_argument("--device", default="cpu", help="torch device for model inference (cpu|cuda)")
    parser.add_argument("--workers", type=int, default=0, help="CPU torch threads (0 = torch default)")
    parser.add_argument("--beam-width", type=int, default=16, help="beam width for re-search")
    parser.add_argument("--determinizations", type=int, default=8, help="hidden-info samples per action")
    parser.add_argument("--time-budget-ms", type=int, default=2500, help="per-move search budget")
    parser.add_argument("--max-search-steps", type=int, default=128, help="max simulator steps per rollout")
    parser.add_argument("--rollout-policy-width", type=int, default=12, help="policy width inside rollouts")
    parser.add_argument("--limit-episodes", type=int, default=0, help="only reanalyse the first N episodes (0 = all)")
    args = parser.parse_args()
    return reanalyse(args)


if __name__ == "__main__":
    raise SystemExit(main())
