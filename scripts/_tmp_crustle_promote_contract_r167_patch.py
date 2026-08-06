"""Normalize replacement promote contracts for dual-Marnie specialist ids (r167).

Live symptom: public_mix_refill stages valid spares for missing cells, but
_promote_replacement_spool rejects them because result opp_archetype is the
baseline specialist id while the primary job contract uses the true archetype
(marnie-s-grimmsnarl-ex). Exact match promotes; archetype mismatch promotes
nothing and loops until the 32-round fail-closed.

Also: train_pure_rl is launched as a script (__main__), so patching only
scripts.train_pure_rl is insufficient; apply across every loaded train_pure_rl
module file, including __main__.
"""
from __future__ import annotations

from typing import Any


def _normalize_opp_archetype_for_contract(
    *,
    record: dict[str, Any],
    runtime_audit_row: dict[str, Any],
) -> str:
    opponent_id = str(
        runtime_audit_row.get("opponent_id")
        or (record.get("target_provenance") or {}).get("opponent_id")
        or ""
    )
    raw = str(record.get("opp_archetype") or "")
    prov = dict(record.get("target_provenance") or {})
    prov_arch = str(prov.get("opponent_archetype_id") or "")
    audit = dict(prov.get("matchup_runtime_audit") or {})
    active = str(audit.get("active_archetype_id") or "")
    # Prefer a true archetype over a specialist baseline id echo.
    for candidate in (active, raw, prov_arch):
        if candidate and candidate != opponent_id and not candidate.startswith("specialist-"):
            return candidate
    if active:
        return active
    if raw:
        return raw
    return prov_arch


def replacement_schedule_contract_from_result(
    runtime_audit_row: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("replacement result lacks a schedule record")
    record = dict(records[0])
    provenance = dict(record.get("target_provenance") or {})
    return {
        "our_seat": int(
            runtime_audit_row.get("our_seat")
            if runtime_audit_row.get("our_seat") is not None
            else -1
        ),
        "opponent_id": str(runtime_audit_row.get("opponent_id") or ""),
        "archetype": str(record.get("archetype") or runtime_audit_row.get("archetype") or ""),
        "opp_archetype": _normalize_opp_archetype_for_contract(
            record=record,
            runtime_audit_row=runtime_audit_row,
        ),
        "opponent_checkpoint_digest": str(
            provenance.get("opponent_checkpoint_digest") or ""
        ),
        "opponent_content_digest": str(
            provenance.get("opponent_content_digest") or ""
        ),
        "opponent_training_group": str(
            provenance.get("opponent_training_group") or ""
        ),
    }


def apply_train_pure_rl_promote_patch() -> None:
    import sys

    for _name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        file = str(getattr(mod, "__file__", "") or "")
        if "train_pure_rl.py" not in file.replace("\\", "/"):
            continue
        if hasattr(mod, "_replacement_schedule_contract_from_result"):
            mod._replacement_schedule_contract_from_result = (  # type: ignore[attr-defined]
                replacement_schedule_contract_from_result
            )
    try:
        import scripts.train_pure_rl as train
    except Exception:
        return
    train._replacement_schedule_contract_from_result = (  # type: ignore[attr-defined]
        replacement_schedule_contract_from_result
    )
