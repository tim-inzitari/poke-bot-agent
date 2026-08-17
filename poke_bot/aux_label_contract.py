"""Fail-closed validation for privileged auxiliary training labels.

These labels never belong in policy/value observations.  They are produced by
the simulator or authoritative visual-trace reconstruction and consumed only
as supervised targets.  A malformed *present* label must therefore raise: if
it were silently dropped, an unknown card could become a false all-absent
target for the opponent-hand/remainder heads.
"""

from __future__ import annotations

from typing import Any, Optional


class AuxLabelContractError(ValueError):
    """A present privileged label does not satisfy the target schema."""


def card_ids_from_aux_field(
    value: Any,
    *,
    field_name: str,
) -> Optional[list[int]]:
    """Flatten one exact-card auxiliary field, preserving absent vs empty.

    ``None`` means the target is unavailable and should be masked.  ``[]`` is
    an available, exact empty-zone target.  Supported card entries are integer
    IDs or serialized card dictionaries containing an integer ``id``; nested
    lists are accepted because older replay imports may retain zone grouping.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise AuxLabelContractError(f"{field_name} contains boolean card id")
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, dict):
        if "id" not in value or value["id"] is None:
            raise AuxLabelContractError(
                f"{field_name} card object is missing a non-null id"
            )
        card_id = value["id"]
        if isinstance(card_id, bool) or not isinstance(card_id, int):
            raise AuxLabelContractError(
                f"{field_name} card id must be an integer: {card_id!r}"
            )
        return [int(card_id)]
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for index, item in enumerate(value):
            nested = card_ids_from_aux_field(
                item,
                field_name=f"{field_name}[{index}]",
            )
            # ``item`` cannot be top-level absent: None inside a present exact
            # zone is ambiguous/masked data and must not become a false zero.
            if nested is None:
                raise AuxLabelContractError(
                    f"{field_name}[{index}] is null inside a present exact label"
                )
            result.extend(nested)
        return result
    raise AuxLabelContractError(
        f"{field_name} has unsupported card-label value {type(value).__name__}"
    )


def validated_unique_card_ids(
    value: Any,
    card_vocab: int,
    *,
    field_name: str,
) -> Optional[list[int]]:
    """Return sorted unique IDs after exact engine-vocabulary validation."""
    vocab = int(card_vocab)
    if vocab <= 1:
        raise AuxLabelContractError(f"invalid card vocabulary size: {card_vocab}")
    raw = card_ids_from_aux_field(value, field_name=field_name)
    if raw is None:
        return None
    for card_id in raw:
        # Engine card row zero is reserved and is never a physical hidden card.
        if not 1 <= int(card_id) < vocab:
            raise AuxLabelContractError(
                f"{field_name} card id {card_id} is outside 1..{vocab - 1}"
            )
    return sorted(set(raw))
