"""Convert ladder episode JSON into per-seat training JSONL.

Hard rules (plan § Imperfect information):
  - One training sequence per seat, built from **that seat's own observations**.
  - Policy / value inputs must be the acting seat's information set only.
  - Opponent hidden hand, face-down prize contents, and deck order must NOT
    appear in ``observation`` used for features. If a raw log leaks them, they
    are stripped into ``aux_labels`` and the observation is re-masked.
  - Conversion asserts info-set integrity; violations are flagged / dropped.
"""

from __future__ import annotations

import copy
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import archetypes

# Archetypes that get their own bootstrap buckets (Hammer is signature-first).
BOOTSTRAP_ARCHETYPES: tuple[str, ...] = (
    "hammer-pult",
    "dragapult",
    "dragapult-dudunsparce",
    "dragapult-dusknoir",
    "dragapult-blaziken",
)


class InfoSetViolation(Exception):
    """Raised when opponent-private fields leak into a policy/value observation."""


@dataclass
class InfoSetReport:
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    remasked: bool = False

    def flag(self, msg: str) -> None:
        self.ok = False
        self.violations.append(msg)


def extract_setup_decks(payload: dict[str, Any]) -> list[Optional[list[int]]]:
    """Pull each seat's 60-card submission from early episode actions."""
    decks: list[Optional[list[int]]] = [None, None]
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for seat, entry in enumerate(step[:2]):
            if decks[seat] is not None:
                continue
            action = entry.get("action")
            if (
                isinstance(action, list)
                and len(action) == 60
                and all(isinstance(x, int) for x in action)
            ):
                decks[seat] = [int(x) for x in action]
        if decks[0] is not None and decks[1] is not None:
            break
    return decks


def classify_episode_seats(payload: dict[str, Any]) -> tuple[list[Optional[list[int]]], list[str]]:
    """Return (decks, archetype_ids) for seats 0/1."""
    decks = extract_setup_decks(payload)
    arches = [
        archetypes.classify_deck(d) if d is not None else archetypes.UNKNOWN for d in decks
    ]
    return decks, arches


def episode_matches_archetype(payload: dict[str, Any], archetype_id: str) -> tuple[bool, list[int]]:
    """True if either seat classifies as ``archetype_id``; return matching seats."""
    _, arches = classify_episode_seats(payload)
    seats = [i for i, a in enumerate(arches) if a == archetype_id]
    return bool(seats), seats


def _agent_names(payload: dict[str, Any]) -> tuple[str, str]:
    info = payload.get("info") or {}
    team_names = info.get("TeamNames") or []
    if len(team_names) >= 2:
        return str(team_names[0]), str(team_names[1])
    agents = info.get("Agents") or []
    if len(agents) >= 2:
        return (
            str((agents[0] or {}).get("Name") or "agent0"),
            str((agents[1] or {}).get("Name") or "agent1"),
        )
    return "agent0", "agent1"


def episode_id_of(payload: dict[str, Any], fallback: str = "") -> str:
    info = payload.get("info") or {}
    if info.get("EpisodeId") is not None:
        return str(info["EpisodeId"])
    if payload.get("id") is not None:
        return str(payload["id"])
    return fallback


def _final_winner(payload: dict[str, Any]) -> int:
    """Return 0 / 1 / 2 (draw) / -1 (unknown)."""
    steps = payload.get("steps") or []
    if steps:
        last = steps[-1]
        if isinstance(last, list):
            for entry in last[:2]:
                cur = ((entry.get("observation") or {}).get("current")) or {}
                result = cur.get("result")
                if result is not None and int(result) >= 0:
                    return int(result)
    rewards = payload.get("rewards") or []
    if len(rewards) >= 2:
        r0, r1 = float(rewards[0]), float(rewards[1])
        if r0 > r1:
            return 0
        if r1 > r0:
            return 1
        return 2
    return -1


def seat_value(winner: int, seat: int) -> float:
    if winner < 0 or winner == 2:
        return 0.0
    return 1.0 if winner == seat else -1.0


def _strip_opp_private(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], InfoSetReport]:
    """Return (masked_obs, aux_labels, report) for one seat observation.

    Privileged opponent fields (hand card list, revealed face-down prizes,
    deck order) are moved into ``aux_labels`` and cleared from the observation
    copy. Prefer self-play / sim dumps that retain both seats' private zones so
    remask can fill ``opp_hand`` / ``opp_deck_order`` / ``opp_prizes`` for
    belief-head multilabel training; ladder seat obs are often already
    info-set clean and yield absent labels (losses mask).
    """
    report = InfoSetReport()
    aux: dict[str, Any] = {
        "opp_hand": None,
        "opp_prizes": None,
        "opp_deck_order": None,
    }
    obs = copy.deepcopy(observation)
    current = obs.get("current")
    if not isinstance(current, dict):
        return obs, aux, report

    your_index = int(current.get("yourIndex", 0))
    players = current.get("players") or []
    if len(players) < 2:
        report.flag("observation.current.players missing both seats")
        return obs, aux, report

    opp_index = 1 - your_index
    opp = players[opp_index]
    if not isinstance(opp, dict):
        report.flag("opponent PlayerState missing")
        return obs, aux, report

    # --- hand (must be None for opponent) ---
    opp_hand = opp.get("hand")
    if opp_hand is not None:
        # Leak: store as aux label only, then mask.
        if isinstance(opp_hand, list):
            aux["opp_hand"] = opp_hand
        report.remasked = True
        report.flag("opponent hand present in raw observation (moved to aux_labels)")
        opp["hand"] = None

    # --- prizes ---
    # Face-down prizes are None; non-None entries are *visible* (looked at / taken
    # effects) and belong in the info set. We only reject a parallel privileged
    # prize-order dump if one appears under a non-standard key.
    for leak_key in ("prizeOrder", "prize_cards", "hiddenPrize", "truePrize"):
        if leak_key in opp and opp[leak_key] is not None:
            aux["opp_prizes"] = opp.pop(leak_key)
            report.remasked = True
            report.flag(f"opponent privileged prize field {leak_key!r} stripped to aux_labels")

    # --- deck order must never appear ---
    if opp.get("deck") is not None:
        aux["opp_deck_order"] = opp.get("deck")
        report.remasked = True
        report.flag("opponent deck order present (stripped to aux_labels)")
        opp.pop("deck", None)

    # Final hard check after remask: opp.hand must be None.
    if opp.get("hand") is not None:
        report.flag("opponent hand still not None after remask")

    # Remask success → OK for training (violation strings retained for logging).
    if opp.get("hand") is None:
        report.ok = True
    else:
        report.ok = False

    return obs, aux, report


def assert_info_set(observation: dict[str, Any], *, strict: bool = True) -> InfoSetReport:
    """Assert (and optionally remask) that ``observation`` is info-set clean.

    When ``strict`` is True and a leak cannot be remasked cleanly, raises
    :class:`InfoSetViolation`.
    """
    masked, _aux, report = _strip_opp_private(observation)
    # Re-check masked view without mutating caller's dict when already clean.
    current = (masked.get("current") or {}) if isinstance(masked, dict) else {}
    players = current.get("players") or []
    if len(players) >= 2:
        your_index = int(current.get("yourIndex", 0))
        opp = players[1 - your_index]
        if isinstance(opp, dict) and opp.get("hand") is not None:
            report.flag("assert failed: opp.hand is not None")
            report.ok = False
    if strict and not report.ok:
        raise InfoSetViolation("; ".join(report.violations) or "info-set violation")
    return report


def _is_option_index_action(action: Any, option_count: int) -> bool:
    if not isinstance(action, list) or not action:
        return False
    if option_count <= 0:
        return False
    return all(isinstance(v, int) and 0 <= v < option_count for v in action)


def convert_episode_to_records(
    payload: dict[str, Any],
    *,
    source: str = "",
    archetype_filter: Optional[str] = None,
    require_complete: bool = True,
    strict_info_set: bool = True,
) -> list[dict[str, Any]]:
    """Convert one episode into per-seat sequence records.

    Each record is one JSONL line: whole-game decision sequence for one seat
    that matches ``archetype_filter`` (or any bootstrap archetype if None).
    """
    decks, arches = classify_episode_seats(payload)
    winner = _final_winner(payload)
    if require_complete and winner < 0:
        return []

    agent_names = _agent_names(payload)
    ep_id = episode_id_of(payload)

    # Collect decision steps per seat.
    seat_steps: list[list[dict[str, Any]]] = [[], []]
    info_flags: list[list[str]] = [[], []]
    remasked_any = [False, False]

    steps = payload.get("steps") or []
    for env_step, step in enumerate(steps):
        if not isinstance(step, list):
            continue
        for seat, entry in enumerate(step[:2]):
            if not isinstance(entry, dict):
                continue
            observation = entry.get("observation") or {}
            select = observation.get("select")
            current = observation.get("current")
            action = entry.get("action") or []

            # Skip deck-submit / non-decision frames.
            if select is None or current is None:
                continue
            options = (select.get("option") or []) if isinstance(select, dict) else []
            if not _is_option_index_action(action, len(options)):
                continue

            # Only keep the seat's own view (entry index == acting yourIndex usually,
            # but always trust yourIndex for masking).
            your_index = int(current.get("yourIndex", seat))
            if your_index != seat:
                # Observation belongs to a different acting player — skip.
                # (Kaggle stores each seat's view in that seat's entry.)
                continue

            masked_obs, aux, report = _strip_opp_private(observation)
            if not report.ok and strict_info_set and not report.remasked:
                raise InfoSetViolation(
                    f"episode={ep_id} seat={seat} env_step={env_step}: "
                    + "; ".join(report.violations)
                )
            if report.violations:
                info_flags[seat].extend(report.violations)
            if report.remasked:
                remasked_any[seat] = True

            # Drop empty aux fields.
            aux_clean = {k: v for k, v in aux.items() if v is not None}
            # Always attach opponent archetype label for ID-net training.
            opp_seat = 1 - seat
            aux_clean["opp_archetype"] = arches[opp_seat]
            aux_clean["opp_agent"] = agent_names[opp_seat]

            seat_steps[seat].append(
                {
                    "env_step": env_step,
                    "observation": masked_obs,
                    "action": [int(x) for x in action],
                    "select_min_count": int(select.get("minCount", 0)),
                    "select_max_count": int(select.get("maxCount", 0)),
                    "legal_action_count": len(options),
                    "aux_labels": aux_clean,
                }
            )

    # Scope B labels (lethal / prize-race): public prize counts + post-hoc
    # prize-take from the seat trajectory. Harmless for Scope A; masked in
    # loss when absent. core_kernel keeps strategy loss weights at 0.
    from .blackwell_heads import attach_blackwell_strategy_labels

    for seat in (0, 1):
        attach_blackwell_strategy_labels(seat_steps[seat])

    records: list[dict[str, Any]] = []
    for seat in (0, 1):
        arch = arches[seat]
        if archetype_filter is not None and arch != archetype_filter:
            continue
        if archetype_filter is None and arch not in BOOTSTRAP_ARCHETYPES:
            continue
        if not seat_steps[seat]:
            continue
        if decks[seat] is None:
            continue

        opp = 1 - seat
        info_ok = not info_flags[seat] or remasked_any[seat]
        # Remasked leaks are OK; unrepaired flags fail.
        if info_flags[seat] and not remasked_any[seat]:
            info_ok = False
        if remasked_any[seat]:
            # remask succeeded → OK
            info_ok = True

        if strict_info_set and not info_ok:
            continue

        records.append(
            {
                "episode_id": ep_id,
                "source": source,
                "seat": seat,
                "agent": agent_names[seat],
                "opp_agent": agent_names[opp],
                "archetype": arch,
                "opp_archetype": arches[opp],
                "deck": list(decks[seat]),
                "opp_deck": list(decks[opp]) if decks[opp] is not None else None,
                "value": seat_value(winner, seat),
                "winner": winner,
                "steps": seat_steps[seat],
                "info_set_ok": info_ok,
                "info_set_flags": list(dict.fromkeys(info_flags[seat])),
                "info_set_remasked": remasked_any[seat],
                "n_decisions": len(seat_steps[seat]),
            }
        )
    return records


def load_episode_payload(path_or_ref: Path | str) -> dict[str, Any]:
    """Load episode JSON from a filesystem path or ``zip:<zip>::<member>`` ref."""
    ref = str(path_or_ref)
    if ref.startswith("zip:"):
        # zip:/path/to/file.zip::member.json
        body = ref[len("zip:") :]
        zip_path_s, _, member = body.partition("::")
        with zipfile.ZipFile(zip_path_s, "r") as zf:
            with zf.open(member) as fh:
                return json.loads(fh.read().decode("utf-8"))
    path = Path(ref)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _convert_job(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker entry: load + convert one episode."""
    try:
        payload = load_episode_payload(job["ref"])
        return convert_episode_to_records(
            payload,
            source=job.get("source", ""),
            archetype_filter=job.get("archetype_filter"),
            require_complete=bool(job.get("require_complete", True)),
            strict_info_set=bool(job.get("strict_info_set", True)),
        )
    except (InfoSetViolation, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return []


def convert_episodes_parallel(
    jobs: list[dict[str, Any]],
    *,
    workers: int = 28,
    desc: str = "convert",
) -> list[dict[str, Any]]:
    """Convert many episodes with a process pool; returns flat list of records.

    Shows a tqdm bar over completed jobs (process-pool friendly).
    """
    if not jobs:
        return []
    from tqdm.auto import tqdm

    workers = max(1, int(workers))
    if workers == 1 or len(jobs) == 1:
        out: list[dict[str, Any]] = []
        for job in tqdm(jobs, desc=desc, unit="ep"):
            out.extend(_convert_job(job))
        return out

    # Ensure child processes can ``import poke_bot`` (repo root on PYTHONPATH).
    import os
    from . import paths as _paths

    root = str(_paths.REPO_ROOT)
    prev = os.environ.get("PYTHONPATH", "")
    if root not in prev.split(os.pathsep):
        os.environ["PYTHONPATH"] = root + (os.pathsep + prev if prev else "")

    out: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_convert_job, job) for job in jobs]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=desc,
            unit="ep",
        ):
            try:
                out.extend(fut.result())
            except Exception:
                continue
    return out


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def append_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def filter_episode_quick(
    payload: dict[str, Any],
    archetype_id: str,
) -> bool:
    """Cheap pre-filter: True if either seat matches ``archetype_id``."""
    ok, _ = episode_matches_archetype(payload, archetype_id)
    return ok


def opponent_archetype_diversity(records: Iterable[dict[str, Any]]) -> set[str]:
    """Unique opposing labels across converted records.

    Classified archetypes are used when known. For ``unknown`` opponents we fall
    back to a stable deck fingerprint (so diversity is not stuck at 1 while only
    the Dragapult family is registered in :mod:`poke_bot.archetypes`).
    """
    labels: set[str] = set()
    for r in records:
        arch = str(r.get("opp_archetype") or archetypes.UNKNOWN)
        if arch != archetypes.UNKNOWN:
            labels.add(arch)
            continue
        deck = r.get("opp_deck")
        if isinstance(deck, list) and deck:
            # Multiset fingerprint — distinguishes lists without a named archetype.
            fp = tuple(sorted(int(x) for x in deck))
            labels.add(f"deck:{hash(fp)}")
        else:
            labels.add(f"agent:{r.get('opp_agent') or 'unknown'}")
    return labels
