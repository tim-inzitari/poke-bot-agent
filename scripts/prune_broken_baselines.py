#!/usr/bin/env python3
"""One-time cleanup: DELETE hard-crashing baselines from disk + manifest.

This is a *one-shot* maintenance pass, NOT the ongoing round-robin policy. The
round-robin loop only ever SKIPS crashers at runtime (persistent blacklist in
``outputs/eval/broken_baselines.json`` + field exclusion) and never deletes.

What this script does:
  1. Pre-scans each baseline for a *reproducible code fault*:
       - import/load failure (hardcoded Kaggle paths, syntax, missing deck), or
       - an exception / illegal-move-that-errors during a few quick smoke games
         vs a random-legal opponent (seat-swapped, short per-game timeout).
     Transient TIMEOUTS are NOT treated as faults (a slow game is a lost game),
     so we never over-delete a merely-slow agent.
  2. For each confirmed hard-crasher (default; use --dry-run to only report):
       - deletes its payload dirs under ``baselines/{official,community,roster,
         decks,kernels}/<dir>/`` (safe: only ever under the baselines dir), and
       - drops it from ``baselines/manifest.json`` (+ ``excluded_broken`` list)
         so ``scripts/download_baselines.sh`` won't re-fetch it.
  3. Records every removal in ``outputs/eval/broken_baselines.json`` with the
     error, kind, timestamp, and deletion status (durable audit trail).

Example:
    python scripts/prune_broken_baselines.py            # scan + delete crashers
    python scripts/prune_broken_baselines.py --dry-run  # scan + report only
    python scripts/prune_broken_baselines.py --only raging-bolt-ex
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import config, deck_pool, paths  # noqa: E402
from poke_bot.agent import _fail_closed_legal, install_quiet_stdout, play_game  # noqa: E402
from poke_bot.baselines_runtime import (  # noqa: E402
    BaselineSpec,
    delete_baseline_payload,
    load_baseline_agent,
    load_manifest,
    remove_from_manifest,
)

BROKEN_PATH = paths.OUTPUTS_DIR / "eval" / "broken_baselines.json"

# Confirmed hard-crashers observed in real round-robin runs whose fault only
# triggers in specific mid-game states a random smoke opponent won't reliably
# force. These are deleted in the one-time pass regardless of the quick scan.
KNOWN_BROKEN: dict[str, tuple[str, str]] = {
    "raging-bolt-ex": (
        "exception",
        "IndexError: list index out of range "
        "(main.py:319 choose_discard_low_value; observed in round_robin_hammer.log)",
    ),
}


def _make_random_agent(deck: list[int], rng: random.Random):
    """A never-crashing opponent: legal random selects, deck on deck-request."""

    def _fn(obs: dict) -> list[int]:
        if obs is None or obs.get("select") is None:
            return list(deck)
        return _fail_closed_legal(obs, [], rng)

    return _fn


def _is_timeout(error: str | None) -> bool:
    if not error:
        return False
    e = error.lower()
    return "timeouterror" in e or "exceeded" in e and "s" in e


def _smoke_baseline(
    spec: BaselineSpec, opp_deck: list[int], *, games: int, timeout_s: int
) -> tuple[str, str | None]:
    """Return ``(kind, error)``: kind in {ok, import, exception}.

    ``timeout`` outcomes are ignored (reported as ``ok``) so a slow agent is
    never deleted — only reproducible code faults count.
    """
    try:
        opp_fn_deck = load_baseline_agent(spec)
    except Exception as exc:  # noqa: BLE001 - import/load fault = hard crash
        return "import", f"{type(exc).__name__}: {exc}"

    base_fn, base_deck = opp_fn_deck
    rng = random.Random(1234)
    had_alarm = hasattr(signal, "SIGALRM")

    def _on_timeout(signum, frame):
        raise TimeoutError(f"game exceeded {timeout_s}s")

    if had_alarm:
        signal.signal(signal.SIGALRM, _on_timeout)

    for g in range(games):
        opp = _make_random_agent(opp_deck, random.Random(9000 + g))
        base_seat = g % 2  # seat-swap across games
        if had_alarm:
            signal.alarm(timeout_s)
        try:
            if base_seat == 0:
                res = play_game(base_fn, opp, base_deck, opp_deck)
            else:
                res = play_game(opp, base_fn, opp_deck, base_deck)
        except BaseException as exc:  # noqa: BLE001 - be defensive
            res = {"failed_seat": base_seat, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if had_alarm:
                signal.alarm(0)

        fseat = res.get("failed_seat")
        if fseat is None:
            continue
        err = res.get("error")
        if fseat == base_seat and not _is_timeout(err):
            return "exception", err  # reproducible code fault attributed to baseline
        # Either our random opponent (never crashes) or a timeout → not a fault.
    return "ok", None


def _load_broken() -> dict:
    if BROKEN_PATH.is_file():
        try:
            return dict(json.loads(BROKEN_PATH.read_text(encoding="utf-8")))
        except Exception:
            return {}
    return {}


def _persist_broken(broken: dict) -> None:
    BROKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BROKEN_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(broken, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(BROKEN_PATH)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", help="Only scan these agent ids.")
    ap.add_argument("--games", type=int, default=8, help="Smoke games per baseline.")
    ap.add_argument("--timeout", type=int, default=45, help="Per-game timeout (s).")
    ap.add_argument(
        "--dry-run", action="store_true", help="Scan + report; do NOT delete."
    )
    ap.add_argument(
        "--agent-verbose",
        action="store_true",
        help="Allow baseline print() spam through (default: suppressed).",
    )
    args = ap.parse_args(argv)

    if not args.agent_verbose and not config.agent_verbose():
        install_quiet_stdout(False)  # keep the scan log clean (tqdm-style only)

    specs = load_manifest()
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.id in wanted]

    opp_deck = deck_pool.primary_deck()
    broken = _load_broken()

    print(
        f"[prune] one-time scan of {len(specs)} baseline(s) "
        f"(games={args.games}, timeout={args.timeout}s, "
        f"{'DRY-RUN' if args.dry_run else 'DELETE'})",
        file=sys.stderr,
        flush=True,
    )

    deleted: list[tuple[str, str, str]] = []  # (id, kind, error)
    healthy = 0
    for spec in specs:
        if spec.id in KNOWN_BROKEN:
            kind, err = KNOWN_BROKEN[spec.id]
            print(
                f"[prune]   known-broken {spec.id} (skipping smoke)",
                file=sys.stderr,
                flush=True,
            )
        else:
            kind, err = _smoke_baseline(
                spec, opp_deck, games=args.games, timeout_s=args.timeout
            )
        if kind == "ok":
            healthy += 1
            print(f"[prune]   ok       {spec.id}", file=sys.stderr, flush=True)
            continue

        # Confirmed hard crasher → record + (unless dry-run) delete.
        record = {
            "error": err,
            "kind": kind,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "deleted": False,
            "removed_from_manifest": False,
            "source": "prune_broken_baselines (one-time)",
        }
        removed_dirs: list[str] = []
        manifest_removed = False
        if not args.dry_run:
            removed_dirs = delete_baseline_payload(spec.dir_name)
            manifest_removed = remove_from_manifest(spec.id)
            record["deleted"] = True
            record["removed_dirs"] = removed_dirs
            record["removed_from_manifest"] = manifest_removed
        broken[spec.id] = record
        deleted.append((spec.id, kind, err or ""))

        verb = "WOULD DELETE" if args.dry_run else "DELETED"
        where = (
            "disk + manifest"
            if manifest_removed or removed_dirs
            else "manifest+disk (already gone)"
        )
        print(
            f"[prune] {verb} broken baseline {spec.id}: {kind}: {err} "
            f"(removed from {where})",
            file=sys.stderr,
            flush=True,
        )

    _persist_broken(broken)

    print("\n[prune] summary:", file=sys.stderr, flush=True)
    print(f"[prune]   healthy: {healthy}", file=sys.stderr, flush=True)
    print(
        f"[prune]   {'would delete' if args.dry_run else 'deleted'}: "
        f"{len(deleted)}",
        file=sys.stderr,
        flush=True,
    )
    for sid, kind, err in deleted:
        print(f"[prune]     - {sid} [{kind}]: {err[:120]}", file=sys.stderr, flush=True)
    print(f"[prune]   audit trail: {BROKEN_PATH}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
