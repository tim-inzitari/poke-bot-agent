from __future__ import annotations

import pytest

from poke_bot.matchup_adapter_routes import (
    require_runtime_route_binding,
    resolve_matchup_adapter_route_contract,
)
from poke_bot.matchup_adapters import (
    ADAPTER_CHECKPOINT_FORMAT as V5_ADAPTER_CHECKPOINT_FORMAT,
)
from poke_bot.matchup_adapters import EXPERT_IDS
from poke_bot.matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
)
from poke_bot.matchup_adapters_v6 import (
    SLOT_CAPACITY,
    load_slot_registry,
    registry_digest,
)


def test_v5_route_contract_preserves_the_contiguous_roster18_slots() -> None:
    contract = resolve_matchup_adapter_route_contract(
        {
            "format": V5_ADAPTER_CHECKPOINT_FORMAT,
            "expert_ids": list(EXPERT_IDS),
        }
    )

    assert contract.target_ids == EXPERT_IDS
    assert contract.physical_slots == tuple(range(len(EXPERT_IDS)))
    assert contract.slot_capacity == len(EXPERT_IDS)
    assert contract.slot_registry_digest is None


def test_v6_route_contract_binds_the_embedded_registry_and_physical_slots() -> None:
    registry = load_slot_registry()
    contract = resolve_matchup_adapter_route_contract(
        {
            "format": V6_ADAPTER_CHECKPOINT_FORMAT,
            "slot_capacity": SLOT_CAPACITY,
            "slot_registry_digest": registry_digest(registry),
            "slot_registry": registry,
        }
    )

    assert contract.target_ids[-1] == "teal-mask-ogerpon-ex"
    assert contract.physical_slots[-1] == 18
    assert contract.slot_capacity == SLOT_CAPACITY
    assert contract.slot_registry_digest == registry_digest(registry)
    require_runtime_route_binding(
        contract.runtime_binding(),
        contract,
    )


def test_v6_route_contract_rejects_a_changed_registry_digest() -> None:
    registry = load_slot_registry()

    with pytest.raises(ValueError, match="registry digest changed"):
        resolve_matchup_adapter_route_contract(
            {
                "format": V6_ADAPTER_CHECKPOINT_FORMAT,
                "slot_capacity": SLOT_CAPACITY,
                "slot_registry_digest": "sha256:" + "0" * 64,
                "slot_registry": registry,
            }
        )
