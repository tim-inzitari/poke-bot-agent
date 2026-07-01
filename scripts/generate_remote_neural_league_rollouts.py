#!/usr/bin/env python3
"""Generate Docker/CABT rollouts for a league of Mac-hosted neural policies."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.features import features_from_observation
from poke_agent.game_tracker import GameEventTracker
from poke_agent.rewards import assign_episode_values, is_complete_episode


def read_deck(path: Path) -> list[int]:
    deck = [int(token) for token in path.read_text(encoding="utf-8").replace(",", "\n").split()]
    if len(deck) != 60:
        raise ValueError(f"{path} must contain 60 card ids, found {len(deck)}")
    return deck


def json_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, separators=(",", ":")))


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._") or "agent"


def post_json(url: str, payload: dict[str, Any], *, timeout: float = 180.0) -> dict[str, Any]:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"policy server HTTP {exc.code}: {detail}") from exc


class RemotePolicyAgent:
    def __init__(self, policy_url: str, session_id: str, *, timeout: float = 180.0):
        self.policy_url = policy_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout
        self.fallbacks = 0
        try:
            post_json(f"{self.policy_url}/reset", {"session_id": session_id}, timeout=timeout)
        except Exception:
            self.fallbacks += 1

    def __call__(self, obs_dict: dict[str, Any]) -> list[int]:
        try:
            response = post_json(
                f"{self.policy_url}/act",
                {"session_id": self.session_id, "observation": obs_dict},
                timeout=self.timeout,
            )
            if "action" not in response:
                raise RuntimeError(f"policy server response missing action: {response}")
            return list(response["action"])
        except Exception:
            self.fallbacks += 1
            select = obs_dict.get("select") or {}
            options = select.get("option") or []
            count = min(int(select.get("maxCount", 1)), len(options))
            min_count = min(int(select.get("minCount", count)), count)
            if count <= 0:
                return []
            return random.sample(range(len(options)), max(min_count, count))


def mark_retained_terminal(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    for row in rows:
        row["terminal"] = False
    rows[-1]["terminal"] = True
    rows[-1]["reward"] = rows[-1].get("value", rows[-1].get("reward", 0.0))


def load_agents(path: Path) -> list[dict[str, Any]]:
    agents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(agents, list) or len(agents) < 2:
        raise ValueError("agents JSON must contain at least two agents")
    seen: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for raw in agents:
        name = safe_name(str(raw["name"]))
        if name in seen:
            raise ValueError(f"duplicate agent name after sanitizing: {name}")
        seen.add(name)
        deck_path = Path(raw["deck"])
        if not deck_path.is_absolute():
            deck_path = ROOT / deck_path
        loaded.append({
            "name": name,
            "deck_path": deck_path,
            "deck": read_deck(deck_path),
            "policy_url": str(raw["policy_url"]),
            "checkpoint": str(raw.get("checkpoint", "")),
        })
    return loaded


def play_league_game(
    *,
    episode: int,
    seat0: dict[str, Any],
    seat1: dict[str, Any],
    policy_timeout: float,
    max_steps: int,
) -> tuple[dict[str, list[dict[str, Any]]], int, dict[str, int]]:
    from cg.game import battle_finish, battle_select, battle_start

    agent0 = RemotePolicyAgent(
        seat0["policy_url"],
        f"league-{episode}-{seat0['name']}-seat0",
        timeout=policy_timeout,
    )
    agent1 = RemotePolicyAgent(
        seat1["policy_url"],
        f"league-{episode}-{seat1['name']}-seat1",
        timeout=policy_timeout,
    )
    agents = [agent0, agent1]
    seats = [seat0, seat1]

    all_rows: list[dict[str, Any]] = []
    rows_by_agent: dict[str, list[dict[str, Any]]] = {seat0["name"]: [], seat1["name"]: []}
    tracker = GameEventTracker()
    obs, start_data = battle_start(seat0["deck"], seat1["deck"])
    if getattr(start_data, "errorPlayer", -1) >= 0:
        raise ValueError(f"deck error type={start_data.errorType} player={start_data.errorPlayer}")
    try:
        step = 0
        truncated = False
        while obs["current"]["result"] < 0 and step < max_steps:
            select = obs.get("select") or {}
            options = select.get("option") or []
            player_index = int(obs["current"]["yourIndex"])
            acting_agent = seats[player_index]
            opponent_agent = seats[1 - player_index]
            action = agents[player_index](obs)
            next_obs = battle_select(action)
            terminal = int((next_obs.get("current") or {}).get("result", -1)) >= 0
            next_tracker = copy.deepcopy(tracker)
            row = {
                "episode": episode,
                "step": step,
                "features": features_from_observation(obs, tracker),
                "next_features": features_from_observation(next_obs, next_tracker),
                "observation": json_snapshot(obs),
                "action": json_snapshot(action),
                "next_observation": json_snapshot(next_obs),
                "legal_action_count": len(options),
                "select_min_count": int(select.get("minCount", 0)),
                "select_max_count": int(select.get("maxCount", 0)),
                "terminal": terminal,
                "reward": 0.0,
                "player": player_index,
                "deck0": seat0["name"],
                "deck1": seat1["name"],
                "deck0_cards": list(seat0["deck"]),
                "deck1_cards": list(seat1["deck"]),
                "agent_name": acting_agent["name"],
                "opponent_name": opponent_agent["name"],
                "league_pair": f"{seat0['name']}__vs__{seat1['name']}",
                "policy_fallbacks_so_far": agents[player_index].fallbacks,
            }
            all_rows.append(row)
            rows_by_agent[acting_agent["name"]].append(row)
            obs = next_obs
            step += 1
        if obs["current"]["result"] < 0:
            truncated = True
        result = int(obs["current"]["result"])
        if truncated or not is_complete_episode(result, terminal_obs=obs, truncated=truncated):
            return {}, result, {seat0["name"]: agent0.fallbacks, seat1["name"]: agent1.fallbacks}
        assign_episode_values(
            all_rows,
            result,
            terminal_obs=obs,
            value_win=1.0,
            value_not_win=-1.0,
            value_timeout=-2.0,
        )
        for rows in rows_by_agent.values():
            for row in rows:
                row["complete"] = True
                row["truncated"] = False
            mark_retained_terminal(rows)
        return rows_by_agent, result, {seat0["name"]: agent0.fallbacks, seat1["name"]: agent1.fallbacks}
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural-vs-neural CABT league rollouts via remote policy servers.")
    parser.add_argument("--agents-json", type=Path, required=True)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/rollouts/neural_league")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--policy-timeout", type=float, default=180.0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    agents = load_agents(args.agents_json)
    pairs = [(i, j) for i in range(len(agents)) for j in range(i + 1, len(agents))]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    handles = {
        agent["name"]: (args.out_dir / f"{agent['name']}.jsonl").open(mode, encoding="utf-8")
        for agent in agents
    }

    stats: dict[str, dict[str, int]] = {
        agent["name"]: {"games": 0, "wins": 0, "losses": 0, "draws": 0, "rows": 0, "fallbacks": 0}
        for agent in agents
    }
    errors = 0
    try:
        progress = tqdm(
            range(args.games),
            desc="neural league CABT simulations",
            unit="game",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
        )
        for local_index in progress:
            episode = args.episode_offset + local_index
            pair_index = episode % len(pairs)
            left, right = pairs[pair_index]
            if (episode // len(pairs)) % 2:
                left, right = right, left
            seat0 = agents[left]
            seat1 = agents[right]
            try:
                rows_by_agent, result, fallbacks = play_league_game(
                    episode=episode,
                    seat0=seat0,
                    seat1=seat1,
                    policy_timeout=args.policy_timeout,
                    max_steps=args.max_steps,
                )
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                errors += 1
                progress.write(
                    f"episode {episode} {seat0['name']} vs {seat1['name']}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            if not rows_by_agent:
                continue

            for name, rows in rows_by_agent.items():
                handle = handles[name]
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                stats[name]["games"] += 1
                stats[name]["rows"] += len(rows)
                stats[name]["fallbacks"] += int(fallbacks.get(name, 0))

            if result == 2:
                stats[seat0["name"]]["draws"] += 1
                stats[seat1["name"]]["draws"] += 1
            elif result == 0:
                stats[seat0["name"]]["wins"] += 1
                stats[seat1["name"]]["losses"] += 1
            elif result == 1:
                stats[seat1["name"]]["wins"] += 1
                stats[seat0["name"]]["losses"] += 1

            progress.set_postfix(
                pair=f"{seat0['name']} vs {seat1['name']}",
                errors=errors,
                rows=sum(item["rows"] for item in stats.values()),
            )
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "games": args.games,
        "episode_offset": args.episode_offset,
        "errors": errors,
        "agents": {
            name: {
                **item,
                "win_rate": item["wins"] / max(1, item["wins"] + item["losses"]),
            }
            for name, item in stats.items()
        },
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
