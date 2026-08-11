"""Fixed-capacity Matchup Adapter V6 and explicit V5 compatibility tools.

V6 separates physical tensor identity from the mutable logical roster. The
first 18 slots are the exact V5 route order; later roster edits allocate or
retire registry slots without changing the checkpoint architecture.

This module is staged alongside the live V5 implementation. Callers must not
select a V6 checkpoint until a receipt-backed compatibility canary passes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .matchup_adapters import (
    ADAPTER_CHECKPOINT_FORMAT as V5_ADAPTER_CHECKPOINT_FORMAT,
)
from .matchup_adapters import (
    BOTTLENECK_DIM,
    HIDDEN_DIM,
    RESIDUAL_SCALE,
    UNKNOWN_ROUTE,
)
from .matchup_adapters import (
    EXPERT_IDS as V5_EXPERT_IDS,
)
from .matchup_adapters import (
    MatchupAdapterBank as MatchupAdapterBankV5,
)

ADAPTER_CHECKPOINT_FORMAT = "poke-bot-matchup-adapter-bank-v6"
SLOT_CAPACITY = 64
LEGACY_V5_PREFIX_LENGTH = 18
REGISTRY_SCHEMA = "poke_bot.matchup_adapter_roster/v1"
SLOT_REGISTRY_SCHEMA = "poke_bot.matchup_adapter_slot_registry/v1"
MIGRATION_SCHEMA = "poke_bot.matchup_adapter_v5_to_v6/v1"
PARAMETERS_PER_SLOT = 4
ROUTABLE_STATUSES = frozenset({"active", "dormant"})
ALLOWED_STATUSES = frozenset({"active", "dormant", "retired", "unused"})
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "state" / "matchup_adapter_roster.json"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def registry_digest(registry: Mapping[str, Any]) -> str:
    """Return the checksum pinned into every V6 checkpoint contract."""

    return "sha256:" + hashlib.sha256(_canonical_json(dict(registry))).hexdigest()


def load_slot_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Load and fail-closed validate the mutable logical slot registry."""

    resolved = Path(path).expanduser().resolve()
    registry = json.loads(resolved.read_text(encoding="utf-8"))
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"unsupported matchup slot registry schema: {resolved}")
    if registry.get("slot_schema") != SLOT_REGISTRY_SCHEMA:
        raise ValueError(f"unsupported V6 slot extension schema: {resolved}")
    if int(registry.get("slot_capacity", -1)) != SLOT_CAPACITY:
        raise ValueError("V6 registry slot_capacity must be exactly 64")
    if int(registry.get("legacy_v5_prefix_length", -1)) != LEGACY_V5_PREFIX_LENGTH:
        raise ValueError("V6 registry legacy prefix must be exactly 18")
    if registry.get("checkpoint_format") != ADAPTER_CHECKPOINT_FORMAT:
        raise ValueError("V6 registry checkpoint format mismatch")

    slots = list(registry.get("slots") or [])
    if len(slots) != SLOT_CAPACITY:
        raise ValueError("V6 registry must contain exactly 64 physical slots")
    seen_ids: set[str] = set()
    for expected_slot, row in enumerate(slots):
        if int(row.get("slot", -1)) != expected_slot:
            raise ValueError("V6 slot rows must be complete and index ordered")
        status = str(row.get("status") or "")
        archetype_id = row.get("archetype_id")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid V6 slot status at {expected_slot}: {status}")
        if status == "unused" and archetype_id is not None:
            raise ValueError("unused V6 slots cannot own an archetype identity")
        if status != "unused":
            if not isinstance(archetype_id, str) or not archetype_id:
                raise ValueError("allocated V6 slots require an archetype identity")
            if archetype_id in seen_ids:
                raise ValueError(f"duplicate V6 archetype identity: {archetype_id}")
            seen_ids.add(archetype_id)

    prefix = tuple(slots[index]["archetype_id"] for index in range(18))
    if prefix != V5_EXPERT_IDS:
        raise ValueError("V6 slots 0..17 must preserve the exact V5 route order")
    active = tuple(registry.get("active_expert_ids") or ())
    if tuple(registry.get("expert_ids") or ()) != active:
        raise ValueError("legacy expert_ids compatibility view differs from active roster")
    routable = tuple(
        row["archetype_id"]
        for row in slots
        if row["status"] in ROUTABLE_STATUSES
    )
    if active != routable:
        raise ValueError(
            "active_expert_ids must exactly match routable slots in slot order"
        )
    if int(registry.get("required_specialist_count", -1)) != len(routable):
        raise ValueError("V6 required specialist count differs from active roster")

    aliases = dict(registry.get("logical_aliases") or {})
    for alias, target in aliases.items():
        if alias in seen_ids or target not in seen_ids:
            raise ValueError(f"invalid V6 logical alias {alias!r} -> {target!r}")
    return registry


def slot_map(registry: Mapping[str, Any]) -> dict[str, int]:
    """Return canonical and alias identities mapped to stable physical slots."""

    result = {
        str(row["archetype_id"]): int(row["slot"])
        for row in registry["slots"]
        if row["status"] in ROUTABLE_STATUSES
    }
    for alias, target in dict(registry.get("logical_aliases") or {}).items():
        if target in result:
            result[str(alias)] = result[str(target)]
    return result


def route_for_archetype(
    archetype_id: str | None,
    *,
    registry: Mapping[str, Any] | None = None,
) -> int:
    if archetype_id is None:
        return UNKNOWN_ROUTE
    resolved = dict(registry) if registry is not None else load_slot_registry()
    return slot_map(resolved).get(archetype_id, UNKNOWN_ROUTE)


def archetype_for_route(
    route: int,
    *,
    registry: Mapping[str, Any] | None = None,
) -> str | None:
    route = int(route)
    if route == UNKNOWN_ROUTE:
        return None
    if route < 0 or route >= SLOT_CAPACITY:
        raise ValueError(f"route must be -1 or in [0, 63], got {route}")
    resolved = dict(registry) if registry is not None else load_slot_registry()
    row = resolved["slots"][route]
    if row["status"] not in ROUTABLE_STATUSES:
        return None
    return str(row["archetype_id"])


def allocate_archetype(
    registry: Mapping[str, Any],
    archetype_id: str,
    *,
    status: str = "dormant",
    lineage: str = "registry-allocation",
) -> dict[str, Any]:
    """Return a revised registry with one identity in the lowest unused slot."""

    if status not in ROUTABLE_STATUSES:
        raise ValueError("new V6 archetypes must start active or dormant")
    revised = copy.deepcopy(dict(registry))
    load_slot_registry_dict(revised)
    if archetype_id in {
        row.get("archetype_id") for row in revised["slots"]
    } or archetype_id in dict(revised.get("logical_aliases") or {}):
        raise ValueError(f"V6 archetype identity already exists: {archetype_id}")
    target = next(
        (row for row in revised["slots"] if row["status"] == "unused"),
        None,
    )
    if target is None:
        raise RuntimeError("V6 slot capacity is exhausted; major migration required")
    target.update(
        archetype_id=archetype_id,
        status=status,
        lineage=lineage,
    )
    revised["active_expert_ids"] = [
        row["archetype_id"]
        for row in revised["slots"]
        if row["status"] in ROUTABLE_STATUSES
    ]
    revised["expert_ids"] = list(revised["active_expert_ids"])
    revised["required_specialist_count"] = len(revised["active_expert_ids"])
    revised["revision"] = int(revised.get("revision", 0)) + 1
    load_slot_registry_dict(revised)
    return revised


def retire_archetype(
    registry: Mapping[str, Any],
    archetype_id: str,
) -> dict[str, Any]:
    """Tombstone an identity without deleting, reindexing, or recycling it."""

    revised = copy.deepcopy(dict(registry))
    load_slot_registry_dict(revised)
    row = next(
        (
            slot
            for slot in revised["slots"]
            if slot.get("archetype_id") == archetype_id
        ),
        None,
    )
    if row is None:
        raise KeyError(archetype_id)
    if row["status"] == "retired":
        return revised
    row["status"] = "retired"
    revised["active_expert_ids"] = [
        slot["archetype_id"]
        for slot in revised["slots"]
        if slot["status"] in ROUTABLE_STATUSES
    ]
    revised["expert_ids"] = list(revised["active_expert_ids"])
    revised["required_specialist_count"] = len(revised["active_expert_ids"])
    revised["revision"] = int(revised.get("revision", 0)) + 1
    load_slot_registry_dict(revised)
    return revised


def load_slot_registry_dict(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an in-memory registry with the same rules as the file loader."""

    # Keep one validator by writing no temporary files: duplicate the small
    # structural checks through a deterministic JSON round trip and helper.
    value = copy.deepcopy(dict(registry))
    if value.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported matchup slot registry schema")
    if value.get("slot_schema") != SLOT_REGISTRY_SCHEMA:
        raise ValueError("unsupported V6 slot extension schema")
    if int(value.get("slot_capacity", -1)) != SLOT_CAPACITY:
        raise ValueError("V6 registry slot_capacity must be exactly 64")
    if int(value.get("legacy_v5_prefix_length", -1)) != LEGACY_V5_PREFIX_LENGTH:
        raise ValueError("V6 registry legacy prefix must be exactly 18")
    if value.get("checkpoint_format") != ADAPTER_CHECKPOINT_FORMAT:
        raise ValueError("V6 registry checkpoint format mismatch")
    slots = list(value.get("slots") or [])
    if len(slots) != SLOT_CAPACITY:
        raise ValueError("V6 registry must contain exactly 64 physical slots")
    seen: set[str] = set()
    for index, row in enumerate(slots):
        if row.get("slot") != index or row.get("status") not in ALLOWED_STATUSES:
            raise ValueError("invalid V6 slot row")
        identity = row.get("archetype_id")
        if row["status"] == "unused":
            if identity is not None:
                raise ValueError("unused V6 slot has an identity")
        elif not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("allocated V6 slots require unique identities")
        else:
            seen.add(identity)
    if tuple(slots[index]["archetype_id"] for index in range(18)) != V5_EXPERT_IDS:
        raise ValueError("V6 registry changed the V5 prefix")
    expected_active = [
        row["archetype_id"]
        for row in slots
        if row["status"] in ROUTABLE_STATUSES
    ]
    if value.get("active_expert_ids") != expected_active:
        raise ValueError("V6 active roster differs from routable slots")
    if value.get("expert_ids") != expected_active:
        raise ValueError("V6 legacy expert_ids view differs from active roster")
    if int(value.get("required_specialist_count", -1)) != len(expected_active):
        raise ValueError("V6 required specialist count differs from active roster")
    aliases = dict(value.get("logical_aliases") or {})
    for alias, target in aliases.items():
        if alias in seen or target not in seen:
            raise ValueError(f"invalid V6 logical alias {alias!r} -> {target!r}")
    return value


def validate_append_only_registry_extension(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    require_new_slots_dormant: bool = True,
    require_exact_source_identities: bool = True,
) -> dict[str, Any]:
    """Validate one append-only Router Format 6 roster extension.

    This is deliberately narrower than the general operator add/retire helpers.
    It is for a sealed source refresh that may *only* allocate previously unused
    physical slots.  Every previously allocated slot, its crosswalk row, and
    every existing alias stay byte-for-byte equivalent at the JSON-value level.
    The caller remains responsible for proving checkpoint tensor equality; this
    function only proves that the logical registry cannot ask an old tensor row
    to represent a new identity.

    New source identities default to exact, dormant routes.  A caller that is
    validating a later, separately receipted activation can opt out of the
    dormant requirement, but it still cannot reindex, recycle, or rename a
    prior route.
    """

    original = load_slot_registry_dict(baseline)
    revised = load_slot_registry_dict(candidate)

    immutable_top_level = (
        "schema",
        "slot_schema",
        "checkpoint_format",
        "slot_capacity",
        "legacy_v5_prefix_length",
        "physical_checkpoint_rows",
        "v6_physical_checkpoint_slots",
        "slot_reuse_policy",
        "allowed_slot_statuses",
        "routable_slot_statuses",
        "v5_migration",
        "migration_from_v4",
    )
    for field in immutable_top_level:
        if revised.get(field) != original.get(field):
            raise ValueError(f"append-only V6 migration changed {field}")

    original_slots = list(original["slots"])
    revised_slots = list(revised["slots"])
    original_ids = {
        str(row["archetype_id"])
        for row in original_slots
        if row["status"] != "unused"
    }
    added_slots: list[int] = []
    added_ids: list[str] = []
    for slot, (before, after) in enumerate(
        zip(original_slots, revised_slots, strict=True)
    ):
        if before["status"] != "unused":
            if after != before:
                raise ValueError(
                    f"append-only V6 migration changed existing slot {slot}"
                )
            continue
        if after == before:
            continue
        if (
            after.get("slot") != slot
            or after.get("status") not in ROUTABLE_STATUSES
            or not isinstance(after.get("archetype_id"), str)
            or not str(after["archetype_id"])
        ):
            raise ValueError(
                f"append-only V6 migration allocated invalid new slot {slot}"
            )
        if require_new_slots_dormant and after["status"] != "dormant":
            raise ValueError(
                f"append-only V6 migration must start slot {slot} dormant"
            )
        archetype_id = str(after["archetype_id"])
        if archetype_id in original_ids or archetype_id in added_ids:
            raise ValueError(
                f"append-only V6 migration reused identity {archetype_id!r}"
            )
        added_slots.append(slot)
        added_ids.append(archetype_id)

    unused_slots = [
        slot for slot, row in enumerate(original_slots) if row["status"] == "unused"
    ]
    if added_slots != unused_slots[: len(added_slots)]:
        raise ValueError("append-only V6 migration skipped a lower never-used slot")

    original_aliases = dict(original.get("logical_aliases") or {})
    if dict(revised.get("logical_aliases") or {}) != original_aliases:
        raise ValueError("append-only V6 migration changed logical aliases")

    original_names = dict(original.get("canonical_display_names") or {})
    revised_names = dict(revised.get("canonical_display_names") or {})
    if any(revised_names.get(key) != value for key, value in original_names.items()):
        raise ValueError("append-only V6 migration changed canonical display names")
    if set(revised_names) - set(original_names) - set(added_ids):
        raise ValueError("append-only V6 migration added unrelated display names")

    original_priority = list(original.get("specialist_priority") or ())
    revised_priority = list(revised.get("specialist_priority") or ())
    if [value for value in revised_priority if value not in added_ids] != original_priority:
        raise ValueError("append-only V6 migration reordered existing priority")
    if len(revised_priority) != len(set(revised_priority)):
        raise ValueError("append-only V6 migration has duplicate priority identities")

    original_meta = dict(original.get("meta_analysis_source") or {})
    revised_meta = dict(revised.get("meta_analysis_source") or {})
    original_crosswalk = dict(original_meta.pop("crosswalk", {}) or {})
    revised_crosswalk = dict(revised_meta.pop("crosswalk", {}) or {})
    # The snapshot itself is expected to advance with an authenticated ingest.
    # Its old crosswalk rows and all source-policy fields are immutable.
    original_snapshot = original_meta.pop("source_snapshot", None)
    revised_snapshot = revised_meta.pop("source_snapshot", None)
    if revised_meta != original_meta:
        raise ValueError("append-only V6 migration changed meta source policy")
    if not isinstance(revised_snapshot, Mapping):
        raise ValueError("append-only V6 migration lacks a source snapshot")
    if any(
        revised_crosswalk.get(key) != value
        for key, value in original_crosswalk.items()
    ):
        raise ValueError("append-only V6 migration changed an existing crosswalk row")
    if set(revised_crosswalk) - set(original_crosswalk) != set(added_ids):
        raise ValueError("append-only V6 migration crosswalk does not match new slots")

    exact_pairs = {
        (row.get("source_id"), row.get("source_name"))
        for row in revised_crosswalk.values()
        if isinstance(row, Mapping) and row.get("status") == "exact"
    }
    if len(exact_pairs) != sum(
        1
        for row in revised_crosswalk.values()
        if isinstance(row, Mapping) and row.get("status") == "exact"
    ):
        raise ValueError("append-only V6 migration duplicates an exact source identity")
    for archetype_id in added_ids:
        row = revised_crosswalk.get(archetype_id)
        if not isinstance(row, Mapping):
            raise ValueError(
                f"append-only V6 migration lacks crosswalk row for {archetype_id!r}"
            )
        if require_exact_source_identities and row.get("status") != "exact":
            raise ValueError(
                f"append-only V6 migration requires an exact source identity for {archetype_id!r}"
            )
        source_id = row.get("source_id")
        source_name = row.get("source_name")
        if (
            isinstance(source_id, bool)
            or not isinstance(source_id, int)
            or source_id < 0
            or not isinstance(source_name, str)
            or not source_name
        ):
            raise ValueError(
                f"append-only V6 migration has invalid source identity for {archetype_id!r}"
            )

    return {
        "baseline_registry_digest": registry_digest(original),
        "candidate_registry_digest": registry_digest(revised),
        "added_slots": added_slots,
        "added_archetype_ids": added_ids,
        "existing_slots_unchanged": True,
        "new_slots_dormant": all(
            revised_slots[slot]["status"] == "dormant" for slot in added_slots
        ),
        "source_snapshot_advanced": revised_snapshot != original_snapshot,
    }


class _MatchupExpertV6(nn.Module):
    def __init__(self, *, exact_zero: bool) -> None:
        super().__init__()
        self.down = nn.Linear(HIDDEN_DIM, BOTTLENECK_DIM, bias=True)
        self.up = nn.Linear(BOTTLENECK_DIM, HIDDEN_DIM, bias=True)
        if exact_zero:
            for parameter in self.parameters():
                nn.init.zeros_(parameter)
        else:
            nn.init.xavier_uniform_(self.down.weight)
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, state: Tensor) -> Tensor:
        normalized = F.layer_norm(
            state,
            normalized_shape=(HIDDEN_DIM,),
            weight=None,
            bias=None,
        )
        return self.up(F.gelu(self.down(normalized)))


class MatchupAdapterBankV6(nn.Module):
    """Fixed 64-slot sparse residual bank."""

    hidden_dim = HIDDEN_DIM
    bottleneck_dim = BOTTLENECK_DIM
    residual_scale = RESIDUAL_SCALE
    slot_capacity = SLOT_CAPACITY

    def __init__(
        self,
        *,
        enabled: bool = False,
        registry: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.registry = (
            load_slot_registry()
            if registry is None
            else load_slot_registry_dict(registry)
        )
        self.expert_ids = tuple(self.registry["active_expert_ids"])
        self.experts = nn.ModuleList(
            _MatchupExpertV6(exact_zero=index >= LEGACY_V5_PREFIX_LENGTH)
            for index in range(SLOT_CAPACITY)
        )
        self.requires_grad_(False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Materialize a wholly absent V6 bank but reject partial state."""

        has_adapter_state = any(key.startswith(prefix) for key in state_dict)
        if not has_adapter_state:
            for key, value in self.state_dict().items():
                state_dict[prefix + key] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def config_dict(self) -> dict[str, Any]:
        return {
            "format": ADAPTER_CHECKPOINT_FORMAT,
            "hidden_dim": HIDDEN_DIM,
            "bottleneck_dim": BOTTLENECK_DIM,
            "residual_scale": RESIDUAL_SCALE,
            "slot_capacity": SLOT_CAPACITY,
            "unknown_route": UNKNOWN_ROUTE,
            "normalization": "parameter-free-layer-norm",
            "activation": "gelu",
            "residual_activation": "tanh",
            "slot_registry_digest": registry_digest(self.registry),
            "slot_registry": copy.deepcopy(self.registry),
        }

    def authorize_slots_for_training(
        self,
        archetype_ids: Sequence[str],
    ) -> tuple[int, ...]:
        """Enable gradients only for explicitly authorized logical routes.

        A newly allocated all-zero slot receives a down-projection
        initialization while retaining an exactly zero output projection. No
        other slot changes.
        """

        mapping = slot_map(self.registry)
        requested = tuple(archetype_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate V6 training authorization")
        routes: list[int] = []
        for archetype_id in requested:
            route = mapping.get(archetype_id)
            if route is None:
                raise ValueError(f"V6 route is not trainable/routable: {archetype_id}")
            routes.append(route)
        self.requires_grad_(False)
        for route in routes:
            expert = self.experts[route]
            if all(
                int(parameter.detach().count_nonzero().item()) == 0
                for parameter in expert.parameters()
            ):
                nn.init.xavier_uniform_(expert.down.weight)
                nn.init.zeros_(expert.down.bias)
            expert.requires_grad_(True)
        return tuple(routes)

    def forward(
        self,
        state: Tensor,
        routes: Tensor | Sequence[int],
        *,
        enabled: bool | None = None,
    ) -> Tensor:
        active = self.enabled if enabled is None else bool(enabled)
        if not active:
            return state
        if state.ndim < 1 or state.shape[-1] != HIDDEN_DIM:
            raise ValueError("state has the wrong hidden dimension")
        route_tensor = torch.as_tensor(routes, dtype=torch.long, device=state.device)
        if route_tensor.shape != state.shape[:-1]:
            raise ValueError("routes shape must equal state.shape[:-1]")
        flat_routes = route_tensor.reshape(-1)
        if flat_routes.numel() == 0:
            return state
        invalid = (flat_routes < UNKNOWN_ROUTE) | (flat_routes >= SLOT_CAPACITY)
        if bool(torch.any(invalid).item()):
            raise ValueError("routes contain an invalid V6 slot")
        routable = set(slot_map(self.registry).values())
        selected_routes = {
            int(route)
            for route in torch.unique(flat_routes[flat_routes >= 0])
            .detach()
            .cpu()
            .tolist()
            if int(route) in routable
        }
        if not selected_routes:
            return state
        flat_state = state.reshape(-1, HIDDEN_DIM)
        adapted = flat_state
        for route in sorted(selected_routes):
            rows = torch.nonzero(flat_routes == route, as_tuple=False).flatten()
            selected = flat_state.index_select(0, rows)
            residual = self.experts[route](selected)
            adapted = adapted.index_copy(
                0,
                rows,
                selected + RESIDUAL_SCALE * torch.tanh(residual),
            )
        return adapted.reshape_as(state)


def _expected_v5_config() -> dict[str, Any]:
    config = MatchupAdapterBankV5.config_dict()
    if (
        config.get("format") != V5_ADAPTER_CHECKPOINT_FORMAT
        or tuple(config.get("expert_ids") or ()) != V5_EXPERT_IDS
    ):
        raise RuntimeError("installed V5 adapter contract is not the known roster18")
    return config


def migrate_v5_adapter_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    prefix: str = "",
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Tensor]:
    """Copy all V5 rows bit-for-bit and append 46 exact-zero slots."""

    resolved = load_slot_registry() if registry is None else load_slot_registry_dict(registry)
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        target = MatchupAdapterBankV6(enabled=False, registry=resolved).state_dict()
    finally:
        torch.random.set_rng_state(rng_state)
    expected_v5_names = {
        name
        for name in target
        if int(name.split(".")[1]) < LEGACY_V5_PREFIX_LENGTH
    }
    supplied = {
        name.removeprefix(prefix): value
        for name, value in state_dict.items()
        if name.startswith(prefix + "experts.")
    }
    if set(supplied) != expected_v5_names:
        raise ValueError("V5 adapter state is incomplete or has unknown rows")
    migrated = {
        prefix + name: value.detach().clone()
        for name, value in target.items()
    }
    for name, value in supplied.items():
        if value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f"V5 adapter tensor contract mismatch: {name}")
        migrated[prefix + name] = value.detach().clone()
    return migrated


def project_v6_adapter_state_to_v5(
    state_dict: Mapping[str, Tensor],
    *,
    registry: Mapping[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, Tensor]:
    """Guarded projection for a V5-representable V6 bank."""

    resolved = load_slot_registry() if registry is None else load_slot_registry_dict(registry)
    for row in resolved["slots"][LEGACY_V5_PREFIX_LENGTH:]:
        if row["status"] in ROUTABLE_STATUSES:
            raise ValueError("cannot project an active V6-only slot to V5")
    projected: dict[str, Tensor] = {}
    for name, value in state_dict.items():
        if not name.startswith(prefix + "experts."):
            continue
        route = int(name.removeprefix(prefix).split(".")[1])
        if route < LEGACY_V5_PREFIX_LENGTH:
            projected[name] = value.detach().clone()
        elif int(value.detach().count_nonzero().item()) != 0:
            raise ValueError("cannot project a trained V6-only slot to V5")
    expected = 4 * LEGACY_V5_PREFIX_LENGTH
    if len(projected) != expected:
        raise ValueError("V6 adapter state is incomplete")
    return projected


def migrate_v5_named_optimizer_state(
    optimizer_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve name-keyed V5 moments; V6-only slots intentionally get none."""

    migrated = copy.deepcopy(dict(optimizer_state))
    unknown = [
        name
        for name in migrated
        if name.startswith("experts.")
        and int(name.split(".")[1]) >= LEGACY_V5_PREFIX_LENGTH
    ]
    if unknown:
        raise ValueError("V5 named optimizer state contains non-V5 adapter rows")
    return migrated


def migrate_v5_positional_optimizer_state(
    optimizer_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand an exact adapter-only V5 optimizer without losing its moments.

    The dormant adapter optimizer is intentionally separate from the model's
    main optimizer and contains one deterministic parameter group in module
    order. V5 therefore owns IDs 0..71 (four parameters for each of 18
    experts). V6 preserves those IDs and appends IDs 72..255 with no optimizer
    moments, matching a freshly constructed 64-slot adapter optimizer.
    """

    migrated = copy.deepcopy(dict(optimizer_state))
    groups = list(migrated.get("param_groups") or [])
    if len(groups) != 1:
        raise ValueError("V5 adapter optimizer must contain exactly one group")
    saved_params = list(groups[0].get("params") or [])
    expected_v5_count = LEGACY_V5_PREFIX_LENGTH * PARAMETERS_PER_SLOT
    expected_v5_params = list(range(expected_v5_count))
    if saved_params != expected_v5_params:
        raise ValueError(
            "V5 adapter optimizer parameter IDs are not the exact canonical order"
        )
    saved_state = dict(migrated.get("state") or {})
    if not set(saved_state).issubset(set(expected_v5_params)):
        raise ValueError("V5 adapter optimizer contains unknown parameter state")
    groups[0]["params"] = list(range(SLOT_CAPACITY * PARAMETERS_PER_SLOT))
    migrated["param_groups"] = groups
    migrated["state"] = saved_state
    return migrated


def v5_optimizer_parameter_name_map() -> dict[str, str]:
    """Return the verified canonical positional-ID to parameter-name map."""

    names = [name for name, _ in MatchupAdapterBankV5().named_parameters()]
    expected = LEGACY_V5_PREFIX_LENGTH * PARAMETERS_PER_SLOT
    if len(names) != expected:
        raise RuntimeError("installed V5 bank has an unexpected parameter layout")
    return {str(index): name for index, name in enumerate(names)}


def migrate_v5_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a V6 derivative without mutating or replacing the V5 payload."""

    migrated = copy.deepcopy(dict(payload))
    extra = dict(migrated.get("extra") or {})
    if extra.get("matchup_adapter_config") != _expected_v5_config():
        raise ValueError("checkpoint is not the exact known V5 roster18 contract")
    if (
        extra.get("matchup_adapter_optimizer_included") is True
        and "matchup_adapter_named_optimizer_state" not in extra
    ):
        raise ValueError(
            "adapter optimizer is positional; provide name-keyed state before migration"
        )
    state = dict(migrated.get("model_state_dict") or {})
    adapter_state = migrate_v5_adapter_state_dict(
        state,
        prefix="matchup_adapter_bank.",
        registry=registry,
    )
    for name in list(state):
        if name.startswith("matchup_adapter_bank."):
            del state[name]
    state.update(adapter_state)
    migrated["model_state_dict"] = state
    resolved = load_slot_registry() if registry is None else load_slot_registry_dict(registry)
    bank = MatchupAdapterBankV6(enabled=False, registry=resolved)
    v6_config = bank.config_dict()
    extra["matchup_adapter_config"] = v6_config
    if "matchup_adapter_named_optimizer_state" in extra:
        extra["matchup_adapter_named_optimizer_state"] = (
            migrate_v5_named_optimizer_state(
                extra["matchup_adapter_named_optimizer_state"]
            )
        )
    positional_optimizer = extra.get("dormant_matchup_adapter_optimizer_state")
    if positional_optimizer:
        extra["dormant_matchup_adapter_optimizer_state"] = (
            migrate_v5_positional_optimizer_state(positional_optimizer)
        )
        extra["dormant_matchup_adapter_optimizer_parameter_names"] = (
            v5_optimizer_parameter_name_map()
        )
    dormant_bank = extra.get("dormant_matchup_adapter_bank")
    if isinstance(dormant_bank, Mapping):
        migrated_dormant_bank = copy.deepcopy(dict(dormant_bank))
        migrated_dormant_bank["adapter_config"] = copy.deepcopy(v6_config)
        migrated_dormant_bank["parameter_count"] = sum(
            int(value.numel()) for value in bank.state_dict().values()
        )
        extra["dormant_matchup_adapter_bank"] = migrated_dormant_bank
    extra["matchup_adapter_v6_migration"] = {
        "schema": MIGRATION_SCHEMA,
        "source_format": V5_ADAPTER_CHECKPOINT_FORMAT,
        "target_format": ADAPTER_CHECKPOINT_FORMAT,
        "source_rows": LEGACY_V5_PREFIX_LENGTH,
        "target_slots": SLOT_CAPACITY,
        "retained_rows_byte_identical": True,
        "unused_slots_exact_zero": True,
        "dormant_optimizer_v5_parameter_count": (
            LEGACY_V5_PREFIX_LENGTH * PARAMETERS_PER_SLOT
            if positional_optimizer
            else 0
        ),
        "dormant_optimizer_v6_parameter_count": (
            SLOT_CAPACITY * PARAMETERS_PER_SLOT if positional_optimizer else 0
        ),
        "dormant_optimizer_moments_preserved": bool(positional_optimizer),
        "slot_registry_digest": registry_digest(resolved),
    }
    migrated["extra"] = extra
    model_config = dict(migrated.get("model_config") or {})
    model_config["matchup_adapter_format"] = ADAPTER_CHECKPOINT_FORMAT
    model_config["matchup_adapter_registry"] = copy.deepcopy(resolved)
    migrated["model_config"] = model_config
    return migrated


def resolve_ptcgreplay_mapping(
    *,
    source_id: int,
    source_name: str,
    card_ids: Iterable[int] | None = None,
    classifier: Callable[[Iterable[int]], str] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an exact or card-signature mapping without fuzzy names."""

    resolved = load_slot_registry() if registry is None else load_slot_registry_dict(registry)
    crosswalk = dict(resolved["meta_analysis_source"]["crosswalk"])
    exact_matches = [
        archetype_id
        for archetype_id, row in crosswalk.items()
        if row.get("status") == "exact"
        and row.get("source_id") == int(source_id)
        and row.get("source_name") == source_name
    ]
    if len(exact_matches) == 1:
        return {"status": "exact", "archetype_id": exact_matches[0]}
    if exact_matches:
        return {"status": "ambiguous", "archetype_id": None}
    if card_ids is None or classifier is None:
        return {"status": "missing", "archetype_id": None}
    classified = classifier(card_ids)
    row = dict(crosswalk.get(classified) or {})
    allowed = set(row.get("allowed_source_family_ids") or ())
    if (
        row.get("status") == "card_signature"
        and int(source_id) in allowed
        and row.get("source_name") == source_name
    ):
        return {"status": "card_signature", "archetype_id": classified}
    return {"status": "ambiguous", "archetype_id": None}


__all__ = [
    "ADAPTER_CHECKPOINT_FORMAT",
    "LEGACY_V5_PREFIX_LENGTH",
    "SLOT_CAPACITY",
    "MatchupAdapterBankV6",
    "allocate_archetype",
    "archetype_for_route",
    "load_slot_registry",
    "migrate_v5_adapter_state_dict",
    "migrate_v5_checkpoint_payload",
    "migrate_v5_named_optimizer_state",
    "migrate_v5_positional_optimizer_state",
    "project_v6_adapter_state_to_v5",
    "registry_digest",
    "resolve_ptcgreplay_mapping",
    "retire_archetype",
    "route_for_archetype",
    "validate_append_only_registry_extension",
    "v5_optimizer_parameter_name_map",
]
