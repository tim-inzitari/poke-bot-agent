#!/usr/bin/env python
"""Round-robin baseline-vs-baseline strength ladder (separate from hammer RL).

Pits every surviving installed baseline against every other baseline, ranks them
by Elo + overall WR / SoS, partitions into a **5-tier** ladder, and writes
artifacts the RL loop can later consume for opponent scheduling / loss weighting.

Matchup convention
------------------
For each **unordered** pair ``{A, B}`` we play ``--games-per-pair`` games
(default **100**) with seat-swap: half with A in seat 0, half with B in seat 0
(50/50 when N is even). We do **not** play 100 each direction (that would be
200/pair). Total games ≈ C(n,2) × 100.

Outputs
-------
- ``outputs/eval/baseline_rank.json`` — full matrix, Elo, ranked list, tiers
- ``outputs/eval/baseline_rank.md`` — human-readable ladder
- ``outputs/eval/baseline_tiers.json`` — RL-facing tier + sampling/loss weights
- ``outputs/eval/baseline_rank.checkpoint.json`` — resumable per-pair aggregates

Resource note
-------------
CPU-only workers (``POKEBOT_WORKER_CPU_ONLY=1``). Default ``--workers 8`` so
the live hammer RR (Blackwell, many sim workers) and core-kernel are not starved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from poke_bot import config, paths
from poke_bot.baselines_runtime import (
    BaselineSpec,
    baseline_spec_payload,
    ensure_baselines_installed,
    filter_loadable_baselines,
    load_manifest,
    resolve_baseline_spec_payload,
)
from poke_bot.eval_metrics import wilson_interval, wilson_lower
from poke_bot.worker_pool import WorkerPool

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

EVAL_DIR = paths.OUTPUTS_DIR / "eval"
OUT_JSON = EVAL_DIR / "baseline_rank.json"
OUT_MD = EVAL_DIR / "baseline_rank.md"
OUT_TIERS = EVAL_DIR / "baseline_tiers.json"
CHECKPOINT = EVAL_DIR / "baseline_rank.checkpoint.json"
BROKEN_PATH = EVAL_DIR / "broken_baselines.json"
LOG_HINT = paths.OUTPUTS_DIR / "logs" / "baseline_rank.log"

N_TIERS = 5
ELO_INIT = 1500.0
ELO_K = 20.0
ELO_EPOCHS = 8

# Stronger tiers → higher sampling + loss weight so RL focuses on hard opps.
# Tier 5 is deliberately soft so we don't overfit easy wins.
DEFAULT_TIER_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"sampling_weight": 1.00, "loss_multiplier": 1.00, "awr_advantage_scale": 1.00},
    2: {"sampling_weight": 0.75, "loss_multiplier": 0.85, "awr_advantage_scale": 0.90},
    3: {"sampling_weight": 0.50, "loss_multiplier": 0.70, "awr_advantage_scale": 0.75},
    4: {"sampling_weight": 0.35, "loss_multiplier": 0.50, "awr_advantage_scale": 0.55},
    5: {"sampling_weight": 0.20, "loss_multiplier": 0.25, "awr_advantage_scale": 0.35},
}

TIER_LABELS = {
    1: "S / elite",
    2: "strong",
    3: "mid",
    4: "soft",
    5: "weakest",
}


# ---------------------------------------------------------------------------
# Crash / blacklist helpers (mirror train_round_robin policy)
# ---------------------------------------------------------------------------


def _classify_error(error: str | None) -> str:
    t = (error or "").lower()
    if (
        "outofmemory" in t
        or "out of memory" in t
        or "cuda error" in t
        or "cublas" in t
        or "cudnn" in t
        or "memoryerror" in t
    ):
        return "resource"
    if "timeouterror" in t or "timed out" in t or "exceeded" in t:
        return "timeout"
    return "code"


def _entry_is_resource(entry: object) -> bool:
    if isinstance(entry, dict):
        kind = str(entry.get("kind", "")).lower()
        if kind in ("resource", "timeout"):
            return True
        err = str(entry.get("error", ""))
    else:
        err = str(entry)
    return _classify_error(err) in ("resource", "timeout")


def _pair_key(a: str, b: str) -> str:
    x, y = sorted((a, b))
    return f"{x}|{y}"


def _spec_payload(spec: BaselineSpec) -> dict[str, Any]:
    return baseline_spec_payload(spec)


# ---------------------------------------------------------------------------
# Worker job: one baseline-vs-baseline game (CPU only)
# ---------------------------------------------------------------------------


def _game_job(payload: dict) -> dict:
    """Play one baseline-vs-baseline game. Never raises (isolates crashes)."""
    import random
    import signal

    from poke_bot import config as _config
    from poke_bot.agent import install_quiet_stdout, play_game
    from poke_bot.baselines_runtime import (
        load_baseline_agent,
        resolve_baseline_spec_payload,
    )

    install_quiet_stdout(_config.agent_verbose())

    a_id = payload["a_id"]
    b_id = payload["b_id"]
    a_seat = int(payload["a_seat"])  # seat occupied by agent A
    seed = int(payload["seed"])
    timeout_s = int(payload.get("timeout_s", 180))

    def _ok(**over: Any) -> dict:
        r = {
            "a_id": a_id,
            "b_id": b_id,
            "a_seat": a_seat,
            "winner": 2,
            "steps": 0,
            "seed": seed,
            "baseline_failed": None,
            "resource_error": False,
            "error": None,
        }
        r.update(over)
        return r

    def _fail(failed_id: str, error: str) -> dict:
        kind = _classify_error(error)
        if kind in ("resource", "timeout"):
            return _ok(resource_error=True, error=f"[{kind}] {error}")
        # Code fault: failed agent forfeits → other wins.
        # Winner is seat of the non-failed agent.
        if failed_id == a_id:
            winner_seat = 1 - a_seat
        else:
            winner_seat = a_seat
        return _ok(
            baseline_failed=failed_id,
            winner=winner_seat,
            error=error,
        )

    try:
        a_spec = resolve_baseline_spec_payload(payload["a_spec"])
        b_spec = resolve_baseline_spec_payload(payload["b_spec"])
        try:
            a_fn, a_deck = load_baseline_agent(a_spec)
        except Exception as exc:  # noqa: BLE001
            return _fail(a_id, f"load: {type(exc).__name__}: {exc}")
        try:
            b_fn, b_deck = load_baseline_agent(b_spec)
        except Exception as exc:  # noqa: BLE001
            return _fail(b_id, f"load: {type(exc).__name__}: {exc}")

        # Seat assignment: a_seat tells which physical seat A occupies.
        if a_seat == 0:
            agent0, agent1 = a_fn, b_fn
            deck0, deck1 = a_deck, b_deck
        else:
            agent0, agent1 = b_fn, a_fn
            deck0, deck1 = b_deck, a_deck

        def _on_timeout(signum, frame):  # noqa: ARG001
            raise TimeoutError(f"game exceeded {timeout_s}s")

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(timeout_s)
        try:
            result = play_game(agent0, agent1, deck0, deck1)
        finally:
            if had_alarm:
                signal.alarm(0)

        failed_seat = result.get("failed_seat")
        if failed_seat is not None:
            failed_id = a_id if failed_seat == a_seat else b_id
            return _fail(failed_id, f"in-game: {result.get('error')}")

        return _ok(winner=int(result["winner"]), steps=int(result.get("steps") or 0))
    except BaseException as exc:  # noqa: BLE001
        # Ambiguous worker fault — treat as resource/retryable if OOM/timeout,
        # otherwise report without blaming a specific baseline (skip game).
        err = f"worker: {type(exc).__name__}: {exc}"
        kind = _classify_error(err)
        if kind in ("resource", "timeout"):
            return _ok(resource_error=True, error=f"[{kind}] {err}")
        return _ok(error=err, resource_error=False)


# ---------------------------------------------------------------------------
# Aggregates / Elo / SoS / tiers
# ---------------------------------------------------------------------------


def _empty_pair(a: str, b: str) -> dict[str, Any]:
    x, y = sorted((a, b))
    return {
        "a": x,
        "b": y,
        "games": 0,
        "a_score": 0.0,  # score from a's perspective (wins + 0.5 draws)
        "b_score": 0.0,
        "draws": 0,
        "a_seat0_games": 0,
        "a_seat0_score": 0.0,
        "results": [],  # compact: [a_seat, winner_seat, seed]
    }


def _record_game(pair: dict[str, Any], res: dict) -> None:
    a, b = pair["a"], pair["b"]
    # Normalize to stored order (lexicographic a < b).
    if res["a_id"] == a:
        a_seat = int(res["a_seat"])
        winner = int(res["winner"])
    else:
        # Payload used swapped labels — rewrite into stored order.
        a_seat = 1 - int(res["a_seat"])
        winner = int(res["winner"])
    pair["games"] += 1
    if winner == 2:
        pair["draws"] += 1
        a_pts = 0.5
    elif winner == a_seat:
        a_pts = 1.0
    else:
        a_pts = 0.0
    pair["a_score"] += a_pts
    pair["b_score"] += 1.0 - a_pts
    if a_seat == 0:
        pair["a_seat0_games"] += 1
        pair["a_seat0_score"] += a_pts
    pair["results"].append([a_seat, winner, int(res["seed"])])


def _elo_from_pairs(agent_ids: list[str], pairs: dict[str, dict]) -> dict[str, float]:
    rating = {aid: ELO_INIT for aid in agent_ids}
    # Flatten decisive/drawn games into (winner_id, loser_id, score_w) tuples.
    games: list[tuple[str, str, float]] = []
    for p in pairs.values():
        a, b = p["a"], p["b"]
        for a_seat, winner, _seed in p.get("results") or []:
            if winner == 2:
                # Draw: each scores 0.5 — update both symmetrically via half steps.
                games.append((a, b, 0.5))
                games.append((b, a, 0.5))
            elif winner == a_seat:
                games.append((a, b, 1.0))
            else:
                games.append((b, a, 1.0))
    if not games:
        return rating
    for _ in range(ELO_EPOCHS):
        for winner, loser, score in games:
            # score is expected score of `winner` vs `loser` (1/0.5).
            ra, rb = rating[winner], rating[loser]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            # When score==0.5 we processed both orderings; K is halved net effect.
            k = ELO_K * (0.5 if score == 0.5 else 1.0)
            rating[winner] = ra + k * (score - ea)
            # Mirror update for loser only on decisive games (draw handled by dual entry).
            if score != 0.5:
                eb = 1.0 - ea
                rating[loser] = rb + k * ((1.0 - score) - eb)
    return rating


def _overall_stats(
    agent_ids: list[str], pairs: dict[str, dict]
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {
        aid: {
            "id": aid,
            "games": 0,
            "score": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "draws": 0,
            "opp_scores": [],  # list of (opp_id, pts_scored_against_them, games)
        }
        for aid in agent_ids
    }
    for p in pairs.values():
        a, b = p["a"], p["b"]
        g = int(p["games"])
        if g <= 0:
            continue
        ascore = float(p["a_score"])
        bscore = float(p["b_score"])
        draws = int(p["draws"])
        # Approximate W/L from score + draws.
        a_wins = ascore - 0.5 * draws
        b_wins = bscore - 0.5 * draws
        for aid, score, wins, opp, opp_score in (
            (a, ascore, a_wins, b, bscore),
            (b, bscore, b_wins, a, ascore),
        ):
            if aid not in stats:
                continue
            st = stats[aid]
            st["games"] += g
            st["score"] += score
            st["wins"] += wins
            st["losses"] += g - wins - draws
            st["draws"] += draws
            st["opp_scores"].append((opp, score, g, opp_score / g if g else 0.5))
    for st in stats.values():
        g = st["games"]
        st["wr"] = (st["score"] / g) if g else 0.0
        st["wilson_lo"] = wilson_lower(st["score"], g) if g else 0.0
        # Strength of schedule = mean opponent overall WR (of finished opponents).
        # Use opponent's score/games across the whole tournament when available.
    # Second pass once overall WRs known.
    wr_map = {aid: stats[aid]["wr"] for aid in agent_ids}
    for st in stats.values():
        if not st["opp_scores"]:
            st["sos"] = 0.5
            continue
        # Weight by games vs that opp.
        num = 0.0
        den = 0.0
        for opp, _pts, g, _opp_pair_wr in st["opp_scores"]:
            num += wr_map.get(opp, 0.5) * g
            den += g
        st["sos"] = num / den if den else 0.5
        # SoS-adjusted score: WR * SoS (simple)
        st["sos_adj"] = st["wr"] * st["sos"]
    return stats


def _tier_sizes(n: int, n_tiers: int = N_TIERS) -> list[int]:
    """Split n agents into n_tiers buckets of size ~n/n_tiers; extras to weakest."""
    if n <= 0:
        return [0] * n_tiers
    base = n // n_tiers
    rem = n % n_tiers
    sizes = [base] * n_tiers
    # Put remainder on the last (weakest) tier first, then climb if needed.
    for i in range(rem):
        sizes[n_tiers - 1 - (i % n_tiers)] += 1
    # Prefer something like 5,5,5,5,6 for n=26.
    return sizes


def _partition_tiers(ranked_ids: list[str]) -> list[dict[str, Any]]:
    sizes = _tier_sizes(len(ranked_ids))
    tiers: list[dict[str, Any]] = []
    idx = 0
    for t, sz in enumerate(sizes, start=1):
        agents = ranked_ids[idx : idx + sz]
        idx += sz
        w = DEFAULT_TIER_WEIGHTS[t]
        tiers.append(
            {
                "tier": t,
                "label": TIER_LABELS[t],
                "agents": agents,
                "n": len(agents),
                **w,
            }
        )
    return tiers


def _rank_agents(
    agent_ids: list[str], pairs: dict[str, dict]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    elo = _elo_from_pairs(agent_ids, pairs)
    stats = _overall_stats(agent_ids, pairs)
    rows = []
    for aid in agent_ids:
        st = stats[aid]
        rows.append(
            {
                "id": aid,
                "elo": round(elo[aid], 2),
                "wr": st["wr"],
                "wilson_lo": st["wilson_lo"],
                "sos": st["sos"],
                "sos_adj": st.get("sos_adj", st["wr"] * st["sos"]),
                "games": st["games"],
                "score": st["score"],
                "wins": st["wins"],
                "losses": st["losses"],
                "draws": st["draws"],
            }
        )
    # Primary: Elo; tie-break: overall WR, then SoS-adjusted, then id.
    rows.sort(key=lambda r: (-r["elo"], -r["wr"], -r["sos_adj"], r["id"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows, elo


def _wr_matrix(agent_ids: list[str], pairs: dict[str, dict]) -> dict[str, dict[str, Any]]:
    """Directed WR: matrix[row][col] = row's WR vs col (None on diagonal)."""
    mat: dict[str, dict[str, Any]] = {a: {} for a in agent_ids}
    for a in agent_ids:
        for b in agent_ids:
            if a == b:
                mat[a][b] = None
                continue
            key = _pair_key(a, b)
            p = pairs.get(key)
            if not p or p["games"] <= 0:
                mat[a][b] = {"games": 0, "wr": None, "wilson_lo": None}
                continue
            if p["a"] == a:
                score, g = float(p["a_score"]), int(p["games"])
            else:
                score, g = float(p["b_score"]), int(p["games"])
            _c, lo, _hi = wilson_interval(score, g)
            mat[a][b] = {
                "games": g,
                "score": score,
                "wr": score / g,
                "wilson_lo": lo,
            }
    return mat


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _build_tiers_artifact(
    ranked: list[dict[str, Any]],
    tiers: list[dict[str, Any]],
    *,
    ranking_metric: str,
) -> dict[str, Any]:
    by_agent = {r["id"]: r for r in ranked}
    tier_by_agent: dict[str, int] = {}
    for t in tiers:
        for aid in t["agents"]:
            tier_by_agent[aid] = int(t["tier"])

    agents_out = []
    for r in ranked:
        tid = tier_by_agent[r["id"]]
        tw = DEFAULT_TIER_WEIGHTS[tid]
        agents_out.append(
            {
                "id": r["id"],
                "rank": r["rank"],
                "tier": tid,
                "elo": r["elo"],
                "wr": r["wr"],
                "wilson_lo": r["wilson_lo"],
                "sos": r["sos"],
                **tw,
            }
        )

    return {
        "schema": "poke_bot.baseline_tiers.v1",
        "n_tiers": N_TIERS,
        "ranking_metric": ranking_metric,
        "weight_curve": {
            "description": (
                "Sampling + loss-importance rise with opponent strength so RL "
                "focuses on hard baselines. Tier 5 is soft (0.2 / 0.25) to avoid "
                "overfitting easy wins. AWR advantage scale follows the same curve."
            ),
            "by_tier": {
                str(t): dict(DEFAULT_TIER_WEIGHTS[t]) for t in range(1, N_TIERS + 1)
            },
        },
        "rr_usage": {
            "opponent_scheduling": (
                "Sample opponents with probability ∝ sampling_weight[tier(opp)]. "
                "Optionally hard-cap fraction of games vs Tier 5 (e.g. ≤15%)."
            ),
            "loss_weighting": (
                "Multiply per-game policy/value loss by loss_multiplier[tier(opp)]. "
                "Games vs Tier 5 contribute little gradient; Tier 1/2 dominate."
            ),
            "awr_advantage": (
                "Scale advantages / soft priors by awr_advantage_scale[tier(opp)] "
                "so beating elite baselines counts more than stomping weak ones."
            ),
            "note": (
                "Artifacts only — do not hot-rewire a live train_round_robin loop; "
                "wire on next restart / curriculum switch."
            ),
        },
        "tiers": tiers,
        "agents": agents_out,
        "tier_by_agent": tier_by_agent,
    }


def _write_md(
    path: Path,
    *,
    ranked: list[dict[str, Any]],
    tiers: list[dict[str, Any]],
    agent_ids: list[str],
    pairs: dict[str, dict],
    meta: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Baseline strength ladder")
    lines.append("")
    lines.append(
        f"Generated `{meta.get('finished_at', '')}` · agents={len(agent_ids)} · "
        f"pairs={meta.get('n_pairs')} · games/pair={meta.get('games_per_pair')} "
        f"(50/50 seat-swap) · ranking=`{meta.get('ranking_metric')}`"
    )
    lines.append("")
    lines.append("## 5-tier ladder (strongest → weakest)")
    lines.append("")
    for t in tiers:
        w = DEFAULT_TIER_WEIGHTS[int(t["tier"])]
        lines.append(
            f"### Tier {t['tier']} — {t['label']} "
            f"(n={t['n']}, sample={w['sampling_weight']}, "
            f"loss×={w['loss_multiplier']})"
        )
        lines.append("")
        lines.append("| Rank | Agent | Elo | WR | Wilson lo | SoS | Games |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for aid in t["agents"]:
            r = next(x for x in ranked if x["id"] == aid)
            lines.append(
                f"| {r['rank']} | `{aid}` | {r['elo']:.1f} | {r['wr']:.1%} | "
                f"{r['wilson_lo']:.1%} | {r['sos']:.3f} | {r['games']} |"
            )
        lines.append("")

    lines.append("## Full ranking")
    lines.append("")
    lines.append("| Rank | Tier | Agent | Elo | WR | Wilson lo | SoS | Games |")
    lines.append("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    tier_of = {aid: t["tier"] for t in tiers for aid in t["agents"]}
    for r in ranked:
        lines.append(
            f"| {r['rank']} | {tier_of[r['id']]} | `{r['id']}` | {r['elo']:.1f} | "
            f"{r['wr']:.1%} | {r['wilson_lo']:.1%} | {r['sos']:.3f} | {r['games']} |"
        )
    lines.append("")

    lines.append("## Pairwise WR summary (row vs column)")
    lines.append("")
    lines.append(
        "Cell = row player's win rate vs column (draws=0.5). "
        "Diagonal blank. Compact top of matrix shown for readability — "
        "full matrix is in `baseline_rank.json`."
    )
    lines.append("")
    # Show truncated matrix if large: first 12 agents by rank.
    show = [r["id"] for r in ranked[:12]]
    header = "| | " + " | ".join(f"`{a[:10]}`" for a in show) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in show) + " |"
    lines.append(header)
    lines.append(sep)
    for a in show:
        cells = [f"`{a[:10]}`"]
        for b in show:
            if a == b:
                cells.append("—")
                continue
            p = pairs.get(_pair_key(a, b))
            if not p or p["games"] <= 0:
                cells.append("")
                continue
            score = float(p["a_score"]) if p["a"] == a else float(p["b_score"])
            wr = score / p["games"]
            cells.append(f"{wr:.0%}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## RL weight curve")
    lines.append("")
    lines.append(
        "See `baseline_tiers.json`. Sampling / loss multipliers by tier: "
        + ", ".join(
            f"T{t}=({DEFAULT_TIER_WEIGHTS[t]['sampling_weight']}/"
            f"{DEFAULT_TIER_WEIGHTS[t]['loss_multiplier']})"
            for t in range(1, N_TIERS + 1)
        )
        + "."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def emit_artifacts(
    agent_ids: list[str],
    pairs: dict[str, dict],
    *,
    meta: dict[str, Any],
) -> dict[str, Any]:
    ranked, elo = _rank_agents(agent_ids, pairs)
    tiers = _partition_tiers([r["id"] for r in ranked])
    ranking_metric = "elo_primary_wr_sos_tiebreak"
    matrix = _wr_matrix(agent_ids, pairs)

    # Compact pairs for JSON (drop bulky per-game seeds if huge — keep aggregates).
    pairs_out = {}
    for k, p in pairs.items():
        pairs_out[k] = {
            "a": p["a"],
            "b": p["b"],
            "games": p["games"],
            "a_score": p["a_score"],
            "b_score": p["b_score"],
            "draws": p["draws"],
            "a_wr": (p["a_score"] / p["games"]) if p["games"] else None,
            "b_wr": (p["b_score"] / p["games"]) if p["games"] else None,
            "a_wilson_lo": wilson_lower(p["a_score"], p["games"]) if p["games"] else None,
            "b_wilson_lo": wilson_lower(p["b_score"], p["games"]) if p["games"] else None,
            "a_seat0_games": p.get("a_seat0_games", 0),
            "a_seat0_score": p.get("a_seat0_score", 0.0),
        }

    full = {
        "schema": "poke_bot.baseline_rank.v1",
        "matchup_convention": (
            "Unordered pair {A,B}: N games with seat-swap "
            "(N/2 A seat0, N/2 B seat0). Not 100 each direction."
        ),
        "ranking_metric": ranking_metric,
        "elo": {"init": ELO_INIT, "k": ELO_K, "epochs": ELO_EPOCHS},
        "meta": meta,
        "agents": agent_ids,
        "ranked": ranked,
        "tiers": tiers,
        "elo_by_agent": {k: round(v, 2) for k, v in elo.items()},
        "pairs": pairs_out,
        "wr_matrix": matrix,
    }
    tiers_art = _build_tiers_artifact(ranked, tiers, ranking_metric=ranking_metric)

    _atomic_write_json(OUT_JSON, full)
    _atomic_write_json(OUT_TIERS, tiers_art)
    _write_md(
        OUT_MD,
        ranked=ranked,
        tiers=tiers,
        agent_ids=agent_ids,
        pairs=pairs,
        meta={**meta, "ranking_metric": ranking_metric},
    )
    return {"ranked": ranked, "tiers": tiers, "full": full, "tiers_artifact": tiers_art}


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def _load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT.is_file():
        return {"pairs": {}, "skipped": {}, "completed_seeds": {}, "meta": {}}
    try:
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    except Exception:
        return {"pairs": {}, "skipped": {}, "completed_seeds": {}, "meta": {}}


def _save_checkpoint(state: dict[str, Any]) -> None:
    # Strip per-result lists periodically? Keep them for Elo fidelity but they
    # can get large (32k × 3 ints). Fine for now (~few MB).
    _atomic_write_json(CHECKPOINT, state)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--games-per-pair",
        type=int,
        default=100,
        help="Games per unordered pair (even; half each seat). Default 100.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CPU workers (default 8 — leave room for live hammer RR).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="+", help="Subset of baseline ids")
    p.add_argument(
        "--timeout-s",
        type=int,
        default=int(os.environ.get("POKEBOT_GAME_TIMEOUT_S", "180")),
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Persist checkpoint every N completed games.",
    )
    p.add_argument(
        "--emit-only",
        action="store_true",
        help="Re-emit JSON/MD/tiers from checkpoint without playing games.",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and start clean.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (paths.OUTPUTS_DIR / "logs").mkdir(parents=True, exist_ok=True)

    if args.games_per_pair < 2 or args.games_per_pair % 2 != 0:
        print("ERROR: --games-per-pair must be even and ≥2", file=sys.stderr)
        return 2

    # CPU-only workers: no per-worker CUDA contexts.
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    specs = ensure_baselines_installed(load_manifest())
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.id in wanted]
    specs, dropped = filter_loadable_baselines(specs)

    broken: dict[str, Any] = {}
    if BROKEN_PATH.is_file():
        try:
            broken = dict(json.loads(BROKEN_PATH.read_text(encoding="utf-8")))
        except Exception:
            broken = {}
    # Drop stale resource/timeout entries (same heal as RR).
    restored = [sid for sid, e in broken.items() if _entry_is_resource(e)]
    for sid in restored:
        broken.pop(sid, None)
    if restored:
        _atomic_write_json(BROKEN_PATH, broken)
        print(f"[rank] restored {len(restored)} infra-blacklisted baselines", flush=True)

    for sid, err in dropped:
        if sid not in broken:
            broken[sid] = {
                "error": f"import: {err}",
                "kind": "import",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "deleted": False,
                "source": "rank_baselines (import filter)",
            }
    specs = [s for s in specs if s.id not in broken]
    if not specs:
        print("ERROR: no loadable baselines", file=sys.stderr)
        return 2

    agent_ids = [s.id for s in specs]
    spec_by_id = {s.id: s for s in specs}
    all_pairs = list(combinations(sorted(agent_ids), 2))
    n_pairs = len(all_pairs)
    total_games = n_pairs * args.games_per_pair

    print(
        f"== rank_baselines agents={len(agent_ids)} pairs={n_pairs} "
        f"games/pair={args.games_per_pair} (50/50 seat-swap) "
        f"total_games={total_games} workers={args.workers}",
        flush=True,
    )
    print(f"   field: {agent_ids}", flush=True)
    print(
        f"   artifacts → {OUT_JSON.name}, {OUT_MD.name}, {OUT_TIERS.name}; "
        f"ckpt={CHECKPOINT.name}; log hint={LOG_HINT}",
        flush=True,
    )

    if args.fresh and CHECKPOINT.is_file():
        CHECKPOINT.unlink()
        print("[rank] wiped checkpoint (--fresh)", flush=True)

    state = _load_checkpoint()
    pairs: dict[str, dict] = state.get("pairs") or {}
    # Rehydrate missing pair shells.
    for a, b in all_pairs:
        k = _pair_key(a, b)
        if k not in pairs:
            pairs[k] = _empty_pair(a, b)
        else:
            # Ensure required keys exist.
            base = _empty_pair(a, b)
            for kk, vv in base.items():
                pairs[k].setdefault(kk, vv)

    # Seeds already done: set of (pair_key, seed).
    done_seeds: set[tuple[str, int]] = set()
    for k, p in pairs.items():
        for a_seat, winner, seed in p.get("results") or []:
            done_seeds.add((k, int(seed)))

    meta = {
        "agents": agent_ids,
        "n_agents": len(agent_ids),
        "n_pairs": n_pairs,
        "games_per_pair": args.games_per_pair,
        "workers": args.workers,
        "seed_base": args.seed,
        "matchup_convention": "unordered_pair_seat_swap_50_50",
        "started_at": state.get("meta", {}).get("started_at")
        or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if args.emit_only:
        done_pairs = sum(1 for p in pairs.values() if p["games"] >= args.games_per_pair)
        meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta["completed_pairs"] = done_pairs
        meta["status"] = "emit_only"
        out = emit_artifacts(agent_ids, pairs, meta=meta)
        print(f">> wrote {OUT_JSON}, {OUT_MD}, {OUT_TIERS}", flush=True)
        for t in out["tiers"]:
            print(f"   Tier {t['tier']} ({t['label']}): {t['agents']}", flush=True)
        return 0

    # Build remaining jobs with deterministic seeds so resume is stable.
    # Seed layout: for each pair in sorted combinations order, half seat0 then
    # half seat1, sequential from --seed.
    jobs: list[dict] = []
    seed = args.seed
    half = args.games_per_pair // 2
    for a, b in all_pairs:
        k = _pair_key(a, b)
        if a in broken or b in broken:
            # Advance seed counter to keep later pairs' seeds stable, but do
            # not enqueue games for excluded agents.
            seed += args.games_per_pair
            continue
        for a_seat in (0, 1):
            for _i in range(half):
                slot_seed = seed
                seed += 1
                if (k, slot_seed) in done_seeds:
                    continue
                if int(pairs[k]["games"]) >= args.games_per_pair:
                    continue
                jobs.append(
                    {
                        "a_id": a,
                        "b_id": b,
                        "a_spec": _spec_payload(spec_by_id[a]),
                        "b_spec": _spec_payload(spec_by_id[b]),
                        "a_seat": a_seat,
                        "seed": slot_seed,
                        "timeout_s": args.timeout_s,
                        "pair_key": k,
                    }
                )

    # Count-based backfill if an older checkpoint used a different seed scheme
    # (pair short of N games but all scheduled seeds marked done).
    for a, b in all_pairs:
        if a in broken or b in broken:
            continue
        k = _pair_key(a, b)
        need = args.games_per_pair - int(pairs[k]["games"])
        # Subtract already-queued jobs for this pair.
        queued = sum(1 for j in jobs if j["pair_key"] == k)
        need -= queued
        for i in range(max(0, need)):
            slot_seed = args.seed + 50_000_000 + seed
            seed += 1
            jobs.append(
                {
                    "a_id": a,
                    "b_id": b,
                    "a_spec": _spec_payload(spec_by_id[a]),
                    "b_spec": _spec_payload(spec_by_id[b]),
                    "a_seat": i % 2,
                    "seed": slot_seed,
                    "timeout_s": args.timeout_s,
                    "pair_key": k,
                }
            )

    already = sum(int(p["games"]) for p in pairs.values())
    print(
        f"[rank] resume: {already} games on disk, {len(jobs)} remaining "
        f"(of {total_games})",
        flush=True,
    )

    # Blacklist bookkeeping for this run (append code-faults only).
    def _persist_broken() -> None:
        _atomic_write_json(BROKEN_PATH, broken)

    # Track live overall score for tqdm.
    live_score: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])  # score, n

    # Seed live from checkpoint.
    for p in pairs.values():
        if p["games"] <= 0:
            continue
        live_score[p["a"]][0] += p["a_score"]
        live_score[p["a"]][1] += p["games"]
        live_score[p["b"]][0] += p["b_score"]
        live_score[p["b"]][1] += p["games"]

    completed_this = 0
    skip_agents: set[str] = set(broken.keys())

    def _save() -> None:
        state_out = {
            "pairs": pairs,
            "skipped": {sid: broken[sid] for sid in broken if sid in agent_ids or True},
            "meta": meta,
        }
        _save_checkpoint(state_out)

    if jobs:
        with WorkerPool(num_workers=args.workers) as pool:
            bar = tqdm(total=len(jobs), desc="baseline RR", unit="game")
            for res in pool.imap_unordered(_game_job, jobs):
                a_id, b_id = res["a_id"], res["b_id"]
                # Drop games involving newly-broken agents.
                if a_id in skip_agents or b_id in skip_agents:
                    bar.update(1)
                    continue

                if res.get("resource_error"):
                    # Infra — skip this game, do not blacklist.
                    bar.update(1)
                    bar.set_postfix_str(f"infra-skip {str(res.get('error'))[:40]}")
                    continue

                failed = res.get("baseline_failed")
                if failed:
                    if failed not in broken:
                        broken[failed] = {
                            "error": res.get("error"),
                            "kind": "runtime",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "deleted": False,
                            "source": "rank_baselines (runtime skip)",
                        }
                        _persist_broken()
                        skip_agents.add(failed)
                        print(
                            f"\n[rank] blacklisted {failed}: {res.get('error')} "
                            f"(left on disk)",
                            flush=True,
                        )
                    # Still count the forfeit game into the matrix.
                elif res.get("error") and res.get("winner") == 2 and not failed:
                    # Ambiguous worker error without attribution — skip game.
                    bar.update(1)
                    continue

                k = _pair_key(a_id, b_id)
                _record_game(pairs[k], res)
                # Live WR: points for winner.
                winner = int(res["winner"])
                a_seat = int(res["a_seat"])
                if winner == 2:
                    live_score[a_id][0] += 0.5
                    live_score[b_id][0] += 0.5
                elif winner == a_seat:
                    live_score[a_id][0] += 1.0
                else:
                    live_score[b_id][0] += 1.0
                live_score[a_id][1] += 1
                live_score[b_id][1] += 1

                completed_this += 1
                bar.update(1)
                # Postfix: overall field WR spread (top Elo proxy via raw WR).
                if live_score:
                    best = max(
                        (
                            (sid, sc / n if n else 0.0)
                            for sid, (sc, n) in live_score.items()
                            if n >= 5
                        ),
                        key=lambda x: x[1],
                        default=(None, 0.0),
                    )
                    worst = min(
                        (
                            (sid, sc / n if n else 0.0)
                            for sid, (sc, n) in live_score.items()
                            if n >= 5
                        ),
                        key=lambda x: x[1],
                        default=(None, 0.0),
                    )
                    bar.set_postfix(
                        top=f"{(best[0] or '?')[:12]}={best[1]:.0%}",
                        low=f"{(worst[0] or '?')[:12]}={worst[1]:.0%}",
                        done=already + completed_this,
                    )

                if completed_this % max(1, args.checkpoint_every) == 0:
                    _save()
            bar.close()

    _save()

    # Drop pairs involving fully-skipped agents from ranking field? Keep them
    # but rank only agents not in skip_agents (or with games>0).
    active_ids = [aid for aid in agent_ids if aid not in skip_agents]
    if not active_ids:
        active_ids = agent_ids

    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["completed_games"] = sum(int(p["games"]) for p in pairs.values())
    meta["completed_pairs"] = sum(
        1 for p in pairs.values() if p["games"] >= args.games_per_pair
    )
    meta["blacklisted"] = sorted(skip_agents)
    meta["status"] = "complete" if meta["completed_pairs"] >= n_pairs else "partial"

    out = emit_artifacts(active_ids, pairs, meta=meta)
    print(f">> wrote {OUT_JSON}", flush=True)
    print(f">> wrote {OUT_MD}", flush=True)
    print(f">> wrote {OUT_TIERS}", flush=True)
    print(
        f">> status={meta['status']} pairs={meta['completed_pairs']}/{n_pairs} "
        f"games={meta['completed_games']}",
        flush=True,
    )
    print("== 5-tier ladder ==", flush=True)
    for t in out["tiers"]:
        print(
            f"  Tier {t['tier']} [{t['label']}] "
            f"sample={t['sampling_weight']} loss×={t['loss_multiplier']}:",
            flush=True,
        )
        for aid in t["agents"]:
            r = next(x for x in out["ranked"] if x["id"] == aid)
            print(
                f"    #{r['rank']:2d} {aid:32s} elo={r['elo']:.1f} wr={r['wr']:.1%}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
