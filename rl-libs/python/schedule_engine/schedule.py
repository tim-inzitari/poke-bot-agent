"""Opaque staged schedules: head ids / weights / epoch stages from a contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    stage_index: int
    enabled: tuple[str, ...]
    weights: dict[str, float]
    digest: str
    schema: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "stage_index": self.stage_index,
            "enabled": list(self.enabled),
            "weights": dict(self.weights),
            "digest": self.digest,
            "schema": self.schema,
        }


def _canonical_ids(raw_ids: Sequence[str], aliases: Mapping[str, str] | None = None) -> tuple[str, ...]:
    alias = dict(aliases or {})
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_ids:
        name = alias.get(str(raw), str(raw))
        if name in seen:
            raise ValueError(f"duplicate id: {name}")
        seen.add(name)
        out.append(name)
    return tuple(out)


def validate_schedule(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a competition-agnostic staged schedule contract.

    Expected shape::

        {
          "schema": "...",
          "ids": ["a", "b", ...],
          "weights": {"a": 1.0, ...},
          "total_epochs": 10,
          "stages": [
            {"epochs": [1, 3], "enable": ["a"]},
            {"epochs": [4, 10], "add": ["b"]}
          ],
          "aliases": {"A": "a"}  # optional
        }
    """
    schema = str(raw.get("schema") or "")
    if not schema:
        raise ValueError("schedule schema required")
    aliases = raw.get("aliases") if isinstance(raw.get("aliases"), Mapping) else {}
    ids = _canonical_ids(list(raw.get("ids") or []), aliases)
    if not ids:
        raise ValueError("schedule ids required")
    weights_in = raw.get("weights")
    if not isinstance(weights_in, Mapping):
        raise ValueError("schedule weights required")
    weights: dict[str, float] = {}
    for raw_name, value in weights_in.items():
        name = aliases.get(str(raw_name), str(raw_name)) if aliases else str(raw_name)
        w = float(value)
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"invalid weight for {name}")
        weights[name] = w
    if set(weights) != set(ids):
        raise ValueError("weights must cover exactly the declared ids")
    total_epochs = int(raw.get("total_epochs", -1))
    if total_epochs < 1:
        raise ValueError("total_epochs must be >= 1")
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages required")
    stages: list[dict[str, Any]] = []
    covered: list[int] = []
    cumulative: list[str] = []
    for index, row in enumerate(raw_stages, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("stage must be a mapping")
        bounds = row.get("epochs")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or int(bounds[0]) > int(bounds[1])
        ):
            raise ValueError("stage has invalid epoch bounds")
        first, last = int(bounds[0]), int(bounds[1])
        covered.extend(range(first, last + 1))
        if row.get("enable_all") is True:
            cumulative = list(ids)
        else:
            source = row.get("enable") if index == 1 else row.get("add")
            if not isinstance(source, list) or not source:
                raise ValueError("stage must enable or add named ids")
            for raw_name in source:
                name = aliases.get(str(raw_name), str(raw_name)) if aliases else str(raw_name)
                if name not in ids:
                    raise ValueError(f"unknown id in stage: {name}")
                if name not in cumulative:
                    cumulative.append(name)
        stages.append(
            {
                "index": index,
                "epochs": [first, last],
                "enabled": [name for name in ids if name in cumulative],
            }
        )
    if covered != list(range(1, total_epochs + 1)):
        raise ValueError("stages must cover epochs 1..N exactly once")
    if set(stages[-1]["enabled"]) != set(ids):
        raise ValueError("final stage must enable all ids")
    return {
        "schema": schema,
        "ids": list(ids),
        "weights": {name: weights[name] for name in ids},
        "total_epochs": total_epochs,
        "stages": stages,
    }


def schedule_digest(raw: Mapping[str, Any]) -> str:
    contract = validate_schedule(raw)
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def epoch_plan(raw: Mapping[str, Any], epoch: int) -> EpochPlan:
    contract = validate_schedule(raw)
    epoch = int(epoch)
    if not 1 <= epoch <= int(contract["total_epochs"]):
        raise ValueError(f"epoch outside 1..{contract['total_epochs']}: {epoch}")
    stage = next(
        row
        for row in contract["stages"]
        if int(row["epochs"][0]) <= epoch <= int(row["epochs"][1])
    )
    enabled = tuple(str(name) for name in stage["enabled"])
    weights = {
        name: (float(contract["weights"][name]) if name in enabled else 0.0)
        for name in contract["ids"]
    }
    return EpochPlan(
        epoch=epoch,
        stage_index=int(stage["index"]),
        enabled=enabled,
        weights=weights,
        digest=schedule_digest(raw),
        schema=str(contract["schema"]),
    )
