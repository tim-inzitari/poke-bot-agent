"""Stage C — critical-node detection and belief-aware search receipts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from poke_bot.slowking_reverse_engineered_policy import (
    ACADEMY_AT_NIGHT,
    CTX_TOP_DECK,
    NIGHT_STRETCHER,
    ULTRA_BALL,
)

from .authority import RESEARCH_ONLY, SEARCH_RECEIPT_SCHEMA


SearchFn = Callable[[dict[str, Any], Sequence[Sequence[int]]], dict[str, Any]]


CRITICAL_STAGE_CLASSES = frozenset(
    {
        "academy_seek_topdeck",
        "night_stretcher_recovery",
        "ultra_ball_search",
        "poke_pad_search",
        "opening_active",
    }
)


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def policy_entropy(logits: Sequence[float]) -> float:
    finite = [float(x) for x in logits if math.isfinite(float(x))]
    if not finite:
        return 0.0
    m = max(finite)
    exps = [math.exp(x - m) for x in finite]
    z = sum(exps)
    probs = [e / z for e in exps]
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def is_critical_decision(
    row: dict[str, Any],
    *,
    policy_logits: Optional[Sequence[float]] = None,
    entropy_threshold: float = 1.2,
    critic_disagreement: float = 0.0,
    disagreement_threshold: float = 0.25,
) -> tuple[bool, str]:
    """Gate Stage-C search. Heuristic-covered and high-entropy prompts qualify."""
    heuristic = row.get("heuristic") or {}
    stage = str(heuristic.get("stage_class") or "")
    if stage in CRITICAL_STAGE_CLASSES:
        return True, f"heuristic_stage:{stage}"

    obs = row.get("observation") or {}
    select = obs.get("select") or {}
    context = _int(select.get("context"))
    effect = select.get("effect") if isinstance(select.get("effect"), dict) else {}
    effect_id = _int(effect.get("id")) if effect else None
    if context == CTX_TOP_DECK and effect_id == ACADEMY_AT_NIGHT:
        return True, "academy_topdeck"
    if effect_id in {NIGHT_STRETCHER, ULTRA_BALL}:
        return True, f"effect:{effect_id}"

    if policy_logits is not None and policy_entropy(policy_logits) >= entropy_threshold:
        return True, "policy_entropy"
    if critic_disagreement >= disagreement_threshold:
        return True, "critic_disagreement"
    return False, "not_critical"


@dataclass
class SearchReceipt:
    payload: dict[str, Any]

    def digest(self) -> str:
        blob = json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(blob).hexdigest()


def build_search_receipt(
    row: dict[str, Any],
    *,
    reason: str,
    candidate_actions: Sequence[Sequence[int]],
    visit_counts: Sequence[float],
    q_values: Sequence[float],
    chosen_action: Sequence[int],
    belief_particles: Optional[list[dict[str, Any]]] = None,
    simulator_version: str = "research_exact_cabt",
    max_sims: int = 0,
) -> SearchReceipt:
    payload = {
        "schema": SEARCH_RECEIPT_SCHEMA,
        "research_only": RESEARCH_ONLY,
        "training_authority": False,
        "game_id": row.get("game_id"),
        "env_step": row.get("env_step"),
        "reason": reason,
        "state_identity": {
            "source_date": row.get("source_date"),
            "episode_id": row.get("episode_id"),
            "seat": row.get("seat"),
            "observation_turn": ((row.get("observation") or {}).get("current") or {}).get(
                "turn"
            ),
        },
        "belief_particles": list(belief_particles or []),
        "candidate_actions": [list(a) for a in candidate_actions],
        "visit_counts": [float(v) for v in visit_counts],
        "q_values": [float(q) for q in q_values],
        "chosen_action": list(chosen_action),
        "simulator_version": simulator_version,
        "max_sims": int(max_sims),
    }
    receipt = SearchReceipt(payload=payload)
    payload["receipt_sha256"] = receipt.digest()
    return receipt


def mock_top_k_search(
    row: dict[str, Any],
    *,
    top_k: int = 4,
) -> SearchReceipt:
    """Deterministic research stub when BeliefMCTS/cg is unavailable.

    Uses teacher action + uniform visits over top-K legal actions as a
    receipt-shaped placeholder. Replace with exact BeliefMCTS on host.
    """
    legal = [list(c) for c in (row.get("legal_action_combos") or [])]
    if not legal:
        legal = [list(row.get("action") or [])]
    k = min(top_k, len(legal))
    candidates = legal[:k]
    visits = [1.0] * k
    selected = list(row.get("action") or candidates[0])
    if selected in candidates:
        visits[candidates.index(selected)] = float(k)
    total = sum(visits)
    q_values = [v / total for v in visits]
    critical, reason = is_critical_decision(row)
    if not critical:
        reason = "forced_mock_noncritical"
    return build_search_receipt(
        row,
        reason=reason,
        candidate_actions=candidates,
        visit_counts=visits,
        q_values=q_values,
        chosen_action=selected,
        belief_particles=[{"kind": "mock_public_consistent", "weight": 1.0}],
        max_sims=0,
    )


def run_stage_c_search(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    search_fn: Optional[SearchFn] = None,
    max_critical: int = 0,
) -> dict[str, Any]:
    """Emit immutable search receipts for critical decisions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "stage_c_search_receipts.jsonl"
    n_critical = 0
    n_written = 0
    with receipt_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            critical, reason = is_critical_decision(row)
            if not critical:
                continue
            n_critical += 1
            if max_critical and n_written >= max_critical:
                break
            if search_fn is None:
                receipt = mock_top_k_search(row)
                receipt.payload["reason"] = reason
            else:
                result = search_fn(row["observation"], row.get("legal_action_combos") or [])
                receipt = build_search_receipt(
                    row,
                    reason=reason,
                    candidate_actions=result.get("candidate_actions") or [],
                    visit_counts=result.get("visit_counts") or [],
                    q_values=result.get("q_values") or [],
                    chosen_action=result.get("chosen_action") or row.get("action") or [],
                    belief_particles=result.get("belief_particles"),
                    simulator_version=str(result.get("simulator_version") or "exact"),
                    max_sims=int(result.get("max_sims") or 0),
                )
            handle.write(json.dumps(receipt.payload, separators=(",", ":")) + "\n")
            n_written += 1
    summary = {
        "schema": "poke_bot.slowking_distill.stage_c_summary/v1",
        "research_only": RESEARCH_ONLY,
        "n_rows": len(rows),
        "n_critical": n_critical,
        "n_receipts": n_written,
        "receipts_path": str(receipt_path),
        "search_backend": "custom" if search_fn is not None else "mock_top_k",
    }
    (out_dir / "stage_c_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
