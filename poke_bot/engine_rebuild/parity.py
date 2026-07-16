"""Parity helpers: fingerprint selects and hash transitions vs official libcg.

When ``ptcg_engine`` / ``libcg`` are available, record golden games from the
official binary and compare fork transitions with :func:`transition_hash`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TransitionRecord:
    """One decision → next observation fingerprint."""

    seed: int
    step_index: int
    select_fingerprint: str
    action: list[int]
    next_select_fingerprint: str
    result: int  # -1 unfinished, else winner / draw code
    notes: str = ""


def fingerprint_select(obs: dict) -> str:
    """Stable fingerprint of the legal-action surface (not full hidden state).

    Uses option ids / areas when present; falls back to option count. Enough to
    catch “wrong legal set” forks without requiring god-view equality on every
    field.
    """
    sel = (obs or {}).get("select")
    if sel is None:
        cur = (obs or {}).get("current") or {}
        return f"terminal:result={cur.get('result', -1)}"
    options = sel.get("option") or []
    parts: list[Any] = [
        sel.get("type"),
        sel.get("minCount"),
        sel.get("maxCount"),
        len(options),
    ]
    for opt in options:
        if isinstance(opt, dict):
            parts.append(
                (
                    opt.get("id"),
                    opt.get("area"),
                    opt.get("cardId"),
                    opt.get("optionType"),
                )
            )
        else:
            parts.append(repr(opt))
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def transition_hash(records: list[TransitionRecord]) -> str:
    """Hash a full episode (or suite) of transitions for golden comparison."""
    payload = [asdict(r) for r in records]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def record_episode(
    *,
    seed: int,
    steps: list[tuple[dict, list[int], dict]],
) -> list[TransitionRecord]:
    """Build records from ``(obs_before, action, obs_after)`` triples."""
    out: list[TransitionRecord] = []
    for i, (before, action, after) in enumerate(steps):
        cur = (after or {}).get("current") or {}
        result = int(cur.get("result", -1))
        out.append(
            TransitionRecord(
                seed=seed,
                step_index=i,
                select_fingerprint=fingerprint_select(before),
                action=list(action),
                next_select_fingerprint=fingerprint_select(after),
                result=result,
            )
        )
    return out


def assert_parity(
    official: list[TransitionRecord],
    candidate: list[TransitionRecord],
    *,
    label: str = "fork",
) -> None:
    """Raise ``AssertionError`` with a clear diff if hashes diverge."""
    h_off = transition_hash(official)
    h_cand = transition_hash(candidate)
    if h_off == h_cand:
        return
    n = min(len(official), len(candidate))
    first: Optional[int] = None
    for i in range(n):
        if official[i] != candidate[i]:
            first = i
            break
    if first is None and len(official) != len(candidate):
        first = n
    raise AssertionError(
        f"parity fail ({label}): official={h_off[:12]}… {label}={h_cand[:12]}… "
        f"first_diff_step={first} len_off={len(official)} len_cand={len(candidate)}"
    )
