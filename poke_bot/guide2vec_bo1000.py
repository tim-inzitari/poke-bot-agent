"""Strict no-search BO1000 schedule and receipt compiler for Guide2Vec r212.

This module deliberately does not run a game, load a checkpoint, or import any
planner/runtime implementation.  It owns only the immutable 500-pair schedule
shape and the fail-closed compilation of terminal game and Guide2Vec decision
receipts for the isolated r212 experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

GUIDE2VEC_BO1000_PAIR_COUNT = 500
GUIDE2VEC_BO1000_GAME_COUNT = 1_000
GUIDE2VEC_EVALUATION_ID = "alakazam-r212-guide2vec-no-mcts-bo1000"
GUIDE2VEC_ARM = "frozen_r195_direct_policy_plus_frozen_guide2vec_bounded_logit_bonus"
CONTROL_ARM = "frozen_r195_no_rtp_direct_policy"
GUIDE2VEC_BO1000_REPORT_SCHEMA = (
    "poke_bot.alakazam_guide2vec_no_mcts_bo1000_r212_report/v1"
)
R212_CONTRACT_SHA256 = (
    "sha256:aa9c7b8158c91d183c092b92bab3047c7bd7af705d539c68cdd3e9c206c0c2b9"
)

R195_SUBMISSION_ID = 55378392
R195_SUBMISSION_MESSAGE = (
    "alakazam training milestone iter 21 copy 1/2 first 261d367e131e NO RTP"
)
R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_CHECKPOINT_BYTES = 127_914_385
R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R195_DECK_ID = "alakazam-owner-rtp-pilot-r175"
R195_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
GUIDE2VEC_MAX_LOGIT_BONUS = 0.05
GUIDE2VEC_PARAMETER_COUNT_MIN = 100_000
GUIDE2VEC_PARAMETER_COUNT_MAX = 500_000

_PAIR_SCHEMA = "poke_bot.alakazam_guide2vec_bo1000_pair_request/v1"
_RNG_SCHEMA = "poke_bot.alakazam_guide2vec_bo1000_sealed_rng/v1"
_GAME_SCHEMA = "poke_bot.alakazam_guide2vec_bo1000_game_request/v1"
_IS_FIRST_SCHEMA = "poke_bot.alakazam_guide2vec_bo1000_is_first_attestation/v1"
_CONTROL_ABSENCE_SCHEMA = (
    "poke_bot.alakazam_guide2vec_bo1000_control_graph_absence_attestation/v1"
)
_MATCHUP_ADAPTER_PARITY_SCHEMA = (
    "poke_bot.alakazam_guide2vec_bo1000_matchup_adapter_parity_attestation/v1"
)
CONTROL_GUIDE2VEC_PRESENCE = "absent_from_runtime_graph"
CANDIDATE_GUIDE2VEC_PRESENCE = "loaded_frozen_component"


class Guide2VecBO1000Error(ValueError):
    """Raised when r212 schedule or receipt evidence is incomplete or invalid."""


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise Guide2VecBO1000Error(f"{name} must be a sha256 digest")
    suffix = value.removeprefix("sha256:")
    if len(suffix) != 64 or any(char not in "0123456789abcdef" for char in suffix):
        raise Guide2VecBO1000Error(f"{name} must be a lowercase sha256 digest")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Guide2VecBO1000Error(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str, minimum: float = 0.0) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise Guide2VecBO1000Error(f"{name} must be finite")
    result = float(value)
    if result < minimum:
        raise Guide2VecBO1000Error(f"{name} must be >= {minimum}")
    return result


def _exact_fields(
    raw: object,
    fields: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise Guide2VecBO1000Error(f"{label} must be an object")
    actual = set(raw)
    if actual != fields:
        unknown = sorted(actual - fields)
        missing = sorted(fields - actual)
        raise Guide2VecBO1000Error(
            f"{label} fields are invalid (unknown={unknown}, missing={missing})"
        )
    return dict(raw)


def _expected_pair_rng(pair_nonce_sha256: str) -> str:
    return _canonical_sha256(
        {
            "schema": _RNG_SCHEMA,
            "pair_nonce_sha256": pair_nonce_sha256,
            "kind": "initial",
        }
    )


def _expected_deck_order_rng(pair_nonce_sha256: str) -> str:
    return _canonical_sha256(
        {
            "schema": _RNG_SCHEMA,
            "pair_nonce_sha256": pair_nonce_sha256,
            "kind": "deck_order",
        }
    )


def _expected_first_actor_seat(pair_nonce_sha256: str) -> int:
    _digest(pair_nonce_sha256, name="pair_nonce_sha256")
    return int(pair_nonce_sha256[-1], 16) % 2


def _expected_game_nonce(
    pair_nonce_sha256: str,
    game_index: int,
    guide2vec_seat: int,
    control_seat: int,
    sealed_initial_first_actor_seat: int,
) -> str:
    return _canonical_sha256(
        {
            "schema": _GAME_SCHEMA,
            "pair_nonce_sha256": pair_nonce_sha256,
            "game_index": game_index,
            "guide2vec_seat": guide2vec_seat,
            "control_seat": control_seat,
            "sealed_initial_first_actor_seat": sealed_initial_first_actor_seat,
        }
    )


def expected_is_first_attestation(
    *,
    game_nonce_sha256: str,
    observed_first_actor_seat: int,
    guide2vec_seat: int,
    control_seat: int,
    observed_first_actor_arm: str,
    guide2vec_is_first: bool,
    control_is_first: bool,
) -> str:
    """Return the canonical digest for the explicit engine IsFirst assertion."""

    return _canonical_sha256(
        {
            "schema": _IS_FIRST_SCHEMA,
            "game_nonce_sha256": game_nonce_sha256,
            "observed_first_actor_seat": observed_first_actor_seat,
            "guide2vec_seat": guide2vec_seat,
            "control_seat": control_seat,
            "observed_first_actor_arm": observed_first_actor_arm,
            "guide2vec_is_first": guide2vec_is_first,
            "control_is_first": control_is_first,
        }
    )


def expected_control_guide2vec_absence_attestation(
    *,
    game_nonce_sha256: str,
    control_runtime_graph_sha256: str,
    control_runtime_graph_observation_sha256: str,
) -> str:
    """Bind proof that the control graph contains no Guide2Vec object or hook."""

    _digest(game_nonce_sha256, name="game_nonce_sha256")
    _digest(control_runtime_graph_sha256, name="control_runtime_graph_sha256")
    _digest(
        control_runtime_graph_observation_sha256,
        name="control_runtime_graph_observation_sha256",
    )
    return _canonical_sha256(
        {
            "schema": _CONTROL_ABSENCE_SCHEMA,
            "game_nonce_sha256": game_nonce_sha256,
            "control_runtime_graph_sha256": control_runtime_graph_sha256,
            "control_runtime_graph_observation_sha256": (
                control_runtime_graph_observation_sha256
            ),
            "control_guide2vec_presence": CONTROL_GUIDE2VEC_PRESENCE,
            "control_guide2vec_module_instance_count": 0,
            "control_guide2vec_parameter_count": 0,
            "control_guide2vec_state_dict_key_count": 0,
            "control_guide2vec_forward_hook_count": 0,
            "control_guide2vec_linear_transform_count": 0,
            "control_guide2vec_disabled_or_zeroed": False,
        }
    )


def expected_matchup_adapter_parity_attestation(
    *,
    game_nonce_sha256: str,
    candidate_matchup_tree_sha256: str,
    control_matchup_tree_sha256: str,
    candidate_matchup_adapter_bank_sha256: str,
    control_matchup_adapter_bank_sha256: str,
    candidate_matchup_adapter_training_receipt_sha256: str,
    control_matchup_adapter_training_receipt_sha256: str,
    candidate_matchup_adapter_runtime_graph_sha256: str,
    control_matchup_adapter_runtime_graph_sha256: str,
    candidate_matchup_adapter_enabled: bool,
    control_matchup_adapter_enabled: bool,
    candidate_matchup_adapter_trained: bool,
    control_matchup_adapter_trained: bool,
    candidate_matchup_adapter_frozen: bool,
    control_matchup_adapter_frozen: bool,
) -> str:
    """Bind the identical frozen r195 matchup-adapter graph on both arms."""

    for name, value in (
        ("game_nonce_sha256", game_nonce_sha256),
        ("candidate_matchup_tree_sha256", candidate_matchup_tree_sha256),
        ("control_matchup_tree_sha256", control_matchup_tree_sha256),
        (
            "candidate_matchup_adapter_bank_sha256",
            candidate_matchup_adapter_bank_sha256,
        ),
        ("control_matchup_adapter_bank_sha256", control_matchup_adapter_bank_sha256),
        (
            "candidate_matchup_adapter_training_receipt_sha256",
            candidate_matchup_adapter_training_receipt_sha256,
        ),
        (
            "control_matchup_adapter_training_receipt_sha256",
            control_matchup_adapter_training_receipt_sha256,
        ),
        (
            "candidate_matchup_adapter_runtime_graph_sha256",
            candidate_matchup_adapter_runtime_graph_sha256,
        ),
        (
            "control_matchup_adapter_runtime_graph_sha256",
            control_matchup_adapter_runtime_graph_sha256,
        ),
    ):
        _digest(value, name=name)
    return _canonical_sha256(
        {
            "schema": _MATCHUP_ADAPTER_PARITY_SCHEMA,
            "game_nonce_sha256": game_nonce_sha256,
            "candidate_matchup_tree_sha256": candidate_matchup_tree_sha256,
            "control_matchup_tree_sha256": control_matchup_tree_sha256,
            "candidate_matchup_adapter_bank_sha256": (
                candidate_matchup_adapter_bank_sha256
            ),
            "control_matchup_adapter_bank_sha256": (
                control_matchup_adapter_bank_sha256
            ),
            "candidate_matchup_adapter_training_receipt_sha256": (
                candidate_matchup_adapter_training_receipt_sha256
            ),
            "control_matchup_adapter_training_receipt_sha256": (
                control_matchup_adapter_training_receipt_sha256
            ),
            "candidate_matchup_adapter_runtime_graph_sha256": (
                candidate_matchup_adapter_runtime_graph_sha256
            ),
            "control_matchup_adapter_runtime_graph_sha256": (
                control_matchup_adapter_runtime_graph_sha256
            ),
            "candidate_matchup_adapter_enabled": candidate_matchup_adapter_enabled,
            "control_matchup_adapter_enabled": control_matchup_adapter_enabled,
            "candidate_matchup_adapter_trained": candidate_matchup_adapter_trained,
            "control_matchup_adapter_trained": control_matchup_adapter_trained,
            "candidate_matchup_adapter_frozen": candidate_matchup_adapter_frozen,
            "control_matchup_adapter_frozen": control_matchup_adapter_frozen,
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenR195RuntimeIdentity:
    """The non-Guide2Vec runtime identity shared byte-for-byte by both arms."""

    model_config_sha256: str
    matchup_tree_sha256: str
    matchup_adapter_bank_sha256: str
    matchup_adapter_training_receipt_sha256: str
    matchup_adapter_runtime_graph_sha256: str
    matchup_adapter_enabled: bool
    matchup_adapter_trained: bool
    matchup_adapter_frozen: bool
    direct_runtime_graph_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "model_config_sha256",
            "matchup_tree_sha256",
            "matchup_adapter_bank_sha256",
            "matchup_adapter_training_receipt_sha256",
            "matchup_adapter_runtime_graph_sha256",
            "direct_runtime_graph_sha256",
        ):
            _digest(getattr(self, field), name=field)
        if self.matchup_tree_sha256 != R195_MATCHUP_TREE_SHA256:
            raise Guide2VecBO1000Error(
                "matchup_tree_sha256 must bind the exact r195 NO-RTP package"
            )
        for field in (
            "matchup_adapter_enabled",
            "matchup_adapter_trained",
            "matchup_adapter_frozen",
        ):
            value = getattr(self, field)
            if type(value) is not bool:
                raise Guide2VecBO1000Error(f"{field} must be boolean")
            if not value:
                raise Guide2VecBO1000Error(
                    "the exact r195 matchup adapter must be enabled, trained, and frozen"
                )

    def as_payload(self) -> dict[str, object]:
        return {
            "submission_id": R195_SUBMISSION_ID,
            "submission_message": R195_SUBMISSION_MESSAGE,
            "checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "checkpoint_bytes": R195_CHECKPOINT_BYTES,
            "bundle_sha256": R195_BUNDLE_SHA256,
            "deck_id": R195_DECK_ID,
            "deck_cards_sha256": R195_DECK_CARDS_SHA256,
            "model_config_sha256": self.model_config_sha256,
            "matchup_tree_sha256": self.matchup_tree_sha256,
            "matchup_adapter_bank_sha256": self.matchup_adapter_bank_sha256,
            "matchup_adapter_training_receipt_sha256": (
                self.matchup_adapter_training_receipt_sha256
            ),
            "matchup_adapter_runtime_graph_sha256": (
                self.matchup_adapter_runtime_graph_sha256
            ),
            "matchup_adapter_enabled": self.matchup_adapter_enabled,
            "matchup_adapter_trained": self.matchup_adapter_trained,
            "matchup_adapter_frozen": self.matchup_adapter_frozen,
            "direct_runtime_graph_sha256": self.direct_runtime_graph_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "poke_bot.alakazam_r195_no_rtp_runtime_identity/v1",
                **self.as_payload(),
            }
        )


@dataclass(frozen=True, slots=True)
class Guide2VecExperimentIdentity:
    """Immutable r212 candidate, base, source, and isolated-output identity."""

    base_runtime: FrozenR195RuntimeIdentity
    guide2vec_checkpoint_sha256: str
    guide2vec_training_receipt_sha256: str
    guide2vec_runtime_config_sha256: str
    guide2vec_parameter_count: int
    candidate_runtime_graph_sha256: str
    control_runtime_graph_sha256: str
    candidate_guide2vec_component_graph_sha256: str
    runtime_graph_difference_receipt_sha256: str
    source_snapshot_sha256: str
    evaluation_output_identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.base_runtime, FrozenR195RuntimeIdentity):
            raise Guide2VecBO1000Error("base_runtime must be FrozenR195RuntimeIdentity")
        for field in (
            "guide2vec_checkpoint_sha256",
            "guide2vec_training_receipt_sha256",
            "guide2vec_runtime_config_sha256",
            "candidate_runtime_graph_sha256",
            "control_runtime_graph_sha256",
            "candidate_guide2vec_component_graph_sha256",
            "runtime_graph_difference_receipt_sha256",
            "source_snapshot_sha256",
            "evaluation_output_identity_sha256",
        ):
            _digest(getattr(self, field), name=field)
        _exact_int(
            self.guide2vec_parameter_count,
            name="guide2vec_parameter_count",
            minimum=GUIDE2VEC_PARAMETER_COUNT_MIN,
        )
        if self.guide2vec_parameter_count > GUIDE2VEC_PARAMETER_COUNT_MAX:
            raise Guide2VecBO1000Error(
                "guide2vec_parameter_count exceeds the r212 500k hard maximum"
            )
        if (
            self.control_runtime_graph_sha256
            != self.base_runtime.direct_runtime_graph_sha256
        ):
            raise Guide2VecBO1000Error(
                "control runtime graph must be the exact r195 direct graph"
            )
        if self.candidate_runtime_graph_sha256 == self.control_runtime_graph_sha256:
            raise Guide2VecBO1000Error(
                "candidate and control runtime graphs must be separately attested"
            )
        if (
            self.candidate_guide2vec_component_graph_sha256
            == self.control_runtime_graph_sha256
        ):
            raise Guide2VecBO1000Error(
                "candidate Guide2Vec component graph cannot be the control graph"
            )
        protected = {
            self.base_runtime.identity_sha256,
            self.guide2vec_checkpoint_sha256,
            self.guide2vec_training_receipt_sha256,
            self.guide2vec_runtime_config_sha256,
            self.candidate_runtime_graph_sha256,
            self.control_runtime_graph_sha256,
            self.candidate_guide2vec_component_graph_sha256,
            self.runtime_graph_difference_receipt_sha256,
            self.source_snapshot_sha256,
        }
        if self.evaluation_output_identity_sha256 in protected:
            raise Guide2VecBO1000Error(
                "evaluation_output_identity_sha256 must be isolated from model/source identities"
            )

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": "poke_bot.alakazam_guide2vec_bo1000_identity/v1",
            "r212_contract_sha256": R212_CONTRACT_SHA256,
            "evaluation_id": GUIDE2VEC_EVALUATION_ID,
            "base_runtime": self.base_runtime.as_payload(),
            "guide2vec_checkpoint_sha256": self.guide2vec_checkpoint_sha256,
            "guide2vec_training_receipt_sha256": self.guide2vec_training_receipt_sha256,
            "guide2vec_runtime_config_sha256": self.guide2vec_runtime_config_sha256,
            "guide2vec_parameter_count": self.guide2vec_parameter_count,
            "candidate_runtime_graph_sha256": self.candidate_runtime_graph_sha256,
            "control_runtime_graph_sha256": self.control_runtime_graph_sha256,
            "candidate_guide2vec_component_graph_sha256": (
                self.candidate_guide2vec_component_graph_sha256
            ),
            "runtime_graph_difference_receipt_sha256": (
                self.runtime_graph_difference_receipt_sha256
            ),
            "control_guide2vec_presence": CONTROL_GUIDE2VEC_PRESENCE,
            "candidate_guide2vec_presence": CANDIDATE_GUIDE2VEC_PRESENCE,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "evaluation_output_identity_sha256": self.evaluation_output_identity_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.as_payload())


@dataclass(frozen=True, slots=True)
class Guide2VecBO1000GameSpec:
    """One immutable request in the r212 paired schedule."""

    pair_index: int
    pair_id: str
    pair_nonce_sha256: str
    pair_initial_rng_sha256: str
    pair_deck_order_rng_sha256: str
    sealed_initial_first_actor_seat: int
    game_index: int
    game_nonce_sha256: str
    guide2vec_seat: int
    control_seat: int
    experiment_identity_sha256: str
    evaluation_output_identity_sha256: str

    def __post_init__(self) -> None:
        _exact_int(self.pair_index, name="pair_index")
        if not self.pair_id:
            raise Guide2VecBO1000Error("pair_id must be nonempty")
        for field in (
            "pair_nonce_sha256",
            "pair_initial_rng_sha256",
            "pair_deck_order_rng_sha256",
            "game_nonce_sha256",
            "experiment_identity_sha256",
            "evaluation_output_identity_sha256",
        ):
            _digest(getattr(self, field), name=field)
        if self.game_index not in {0, 1}:
            raise Guide2VecBO1000Error("game_index must be 0 or 1")
        if (
            self.guide2vec_seat not in {0, 1}
            or self.control_seat != 1 - self.guide2vec_seat
        ):
            raise Guide2VecBO1000Error("game arms must occupy exact opposite seats")
        if self.sealed_initial_first_actor_seat not in {0, 1}:
            raise Guide2VecBO1000Error("sealed_initial_first_actor_seat must be 0 or 1")
        if self.pair_initial_rng_sha256 != _expected_pair_rng(self.pair_nonce_sha256):
            raise Guide2VecBO1000Error(
                "pair_initial_rng_sha256 must be sealed from the pair nonce"
            )
        if self.pair_deck_order_rng_sha256 != _expected_deck_order_rng(
            self.pair_nonce_sha256
        ):
            raise Guide2VecBO1000Error(
                "pair_deck_order_rng_sha256 must be sealed from the pair nonce"
            )
        if self.sealed_initial_first_actor_seat != _expected_first_actor_seat(
            self.pair_nonce_sha256
        ):
            raise Guide2VecBO1000Error(
                "sealed_initial_first_actor_seat must be sealed pair material"
            )
        if self.game_nonce_sha256 != _expected_game_nonce(
            self.pair_nonce_sha256,
            self.game_index,
            self.guide2vec_seat,
            self.control_seat,
            self.sealed_initial_first_actor_seat,
        ):
            raise Guide2VecBO1000Error(
                "game_nonce_sha256 must bind pair RNG, seat swap, and initial actor"
            )

    def as_payload(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, raw: object) -> Guide2VecBO1000GameSpec:
        values = _exact_fields(
            raw, set(cls.__dataclass_fields__), label="Guide2VecBO1000GameSpec"
        )
        try:
            return cls(**values)
        except (TypeError, Guide2VecBO1000Error) as exc:
            raise Guide2VecBO1000Error("game spec payload is invalid") from exc


def build_guide2vec_bo1000_schedule(
    seed_identity_sha256: str,
    experiment: Guide2VecExperimentIdentity,
) -> tuple[Guide2VecBO1000GameSpec, ...]:
    """Build the fixed 500-pair/1,000-game candidate-seat-swapped schedule."""

    _digest(seed_identity_sha256, name="seed_identity_sha256")
    if not isinstance(experiment, Guide2VecExperimentIdentity):
        raise Guide2VecBO1000Error("experiment must be Guide2VecExperimentIdentity")
    schedule: list[Guide2VecBO1000GameSpec] = []
    for pair_index in range(GUIDE2VEC_BO1000_PAIR_COUNT):
        pair_nonce = _canonical_sha256(
            {
                "schema": _PAIR_SCHEMA,
                "seed_identity_sha256": seed_identity_sha256,
                "experiment_identity_sha256": experiment.identity_sha256,
                "pair_index": pair_index,
            }
        )
        pair_id = f"r212-guide2vec-pair-{pair_index:06d}-{pair_nonce[7:19]}"
        initial_actor = _expected_first_actor_seat(pair_nonce)
        for game_index in (0, 1):
            guide2vec_seat = game_index
            control_seat = 1 - guide2vec_seat
            schedule.append(
                Guide2VecBO1000GameSpec(
                    pair_index=pair_index,
                    pair_id=pair_id,
                    pair_nonce_sha256=pair_nonce,
                    pair_initial_rng_sha256=_expected_pair_rng(pair_nonce),
                    pair_deck_order_rng_sha256=_expected_deck_order_rng(pair_nonce),
                    sealed_initial_first_actor_seat=initial_actor,
                    game_index=game_index,
                    game_nonce_sha256=_expected_game_nonce(
                        pair_nonce,
                        game_index,
                        guide2vec_seat,
                        control_seat,
                        initial_actor,
                    ),
                    guide2vec_seat=guide2vec_seat,
                    control_seat=control_seat,
                    experiment_identity_sha256=experiment.identity_sha256,
                    evaluation_output_identity_sha256=(
                        experiment.evaluation_output_identity_sha256
                    ),
                )
            )
    return tuple(schedule)


@dataclass(frozen=True, slots=True)
class Guide2VecDecisionReceipt:
    """One candidate-side decision outcome; there is no search telemetry."""

    game_nonce_sha256: str
    decision_index: int
    acting_seat: int
    legal_option_count: int
    eligible: bool
    abstained: bool
    bonus_applied: bool
    action_changed_from_direct_policy: bool
    max_applied_logit_bonus: float
    direct_action_sha256: str
    final_action_sha256: str
    legal_options_sha256: str
    guide2vec_scores_sha256: str
    guide2vec_action_latency_seconds: float
    total_action_latency_seconds: float

    def __post_init__(self) -> None:
        _digest(self.game_nonce_sha256, name="decision game_nonce_sha256")
        _exact_int(self.decision_index, name="decision_index")
        if self.acting_seat not in {0, 1}:
            raise Guide2VecBO1000Error("decision acting_seat must be 0 or 1")
        _exact_int(self.legal_option_count, name="legal_option_count", minimum=1)
        for field in (
            "eligible",
            "abstained",
            "bonus_applied",
            "action_changed_from_direct_policy",
        ):
            if type(getattr(self, field)) is not bool:
                raise Guide2VecBO1000Error(f"{field} must be boolean")
        bonus = _finite(self.max_applied_logit_bonus, name="max_applied_logit_bonus")
        if bonus > GUIDE2VEC_MAX_LOGIT_BONUS:
            raise Guide2VecBO1000Error(
                "max_applied_logit_bonus exceeds the r212 0.05 hard cap"
            )
        for field in (
            "direct_action_sha256",
            "final_action_sha256",
            "legal_options_sha256",
            "guide2vec_scores_sha256",
        ):
            _digest(getattr(self, field), name=field)
        guide_latency = _finite(
            self.guide2vec_action_latency_seconds,
            name="guide2vec_action_latency_seconds",
        )
        total_latency = _finite(
            self.total_action_latency_seconds,
            name="total_action_latency_seconds",
        )
        if guide_latency > total_latency:
            raise Guide2VecBO1000Error(
                "guide2vec action latency cannot exceed total action latency"
            )
        if not self.eligible and (
            not self.abstained
            or self.bonus_applied
            or self.action_changed_from_direct_policy
            or bonus != 0.0
        ):
            raise Guide2VecBO1000Error(
                "an ineligible stage must be an exact direct-policy abstention"
            )
        if self.abstained and (
            self.bonus_applied or self.action_changed_from_direct_policy or bonus != 0.0
        ):
            raise Guide2VecBO1000Error(
                "an abstention must preserve the exact direct-policy action"
            )
        if self.bonus_applied and (not self.eligible or self.abstained or bonus <= 0.0):
            raise Guide2VecBO1000Error(
                "a Guide2Vec bonus requires an eligible non-abstaining stage"
            )
        if not self.bonus_applied and bonus != 0.0:
            raise Guide2VecBO1000Error(
                "a non-applied Guide2Vec bonus must have zero magnitude"
            )
        if self.action_changed_from_direct_policy != (
            self.direct_action_sha256 != self.final_action_sha256
        ):
            raise Guide2VecBO1000Error(
                "action_changed_from_direct_policy must match action identities"
            )
        if self.action_changed_from_direct_policy and not self.bonus_applied:
            raise Guide2VecBO1000Error(
                "only an applied bounded Guide2Vec bonus may change an action"
            )

    def as_payload(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, raw: object) -> Guide2VecDecisionReceipt:
        values = _exact_fields(
            raw, set(cls.__dataclass_fields__), label="Guide2VecDecisionReceipt"
        )
        try:
            return cls(**values)
        except (TypeError, Guide2VecBO1000Error) as exc:
            raise Guide2VecBO1000Error("Guide2Vec decision payload is invalid") from exc


@dataclass(frozen=True, slots=True)
class Guide2VecBO1000GameReceipt:
    """Terminal evidence for both arms of one scheduled r212 game."""

    game_nonce_sha256: str
    pair_id: str
    game_index: int
    guide2vec_seat: int
    control_seat: int
    pair_initial_rng_sha256: str
    pair_deck_order_rng_sha256: str
    sealed_initial_first_actor_seat: int
    observed_first_actor_seat: int
    observed_first_actor_arm: str
    guide2vec_is_first: bool
    control_is_first: bool
    is_first_attestation_sha256: str
    turn_order_observation_sha256: str
    base_runtime_identity_sha256: str
    experiment_identity_sha256: str
    evaluation_output_identity_sha256: str
    candidate_runtime_graph_sha256: str
    candidate_base_runtime_graph_sha256: str
    control_runtime_graph_sha256: str
    candidate_matchup_tree_sha256: str
    control_matchup_tree_sha256: str
    candidate_matchup_adapter_bank_sha256: str
    control_matchup_adapter_bank_sha256: str
    candidate_matchup_adapter_training_receipt_sha256: str
    control_matchup_adapter_training_receipt_sha256: str
    candidate_matchup_adapter_runtime_graph_sha256: str
    control_matchup_adapter_runtime_graph_sha256: str
    candidate_matchup_adapter_enabled: bool
    control_matchup_adapter_enabled: bool
    candidate_matchup_adapter_trained: bool
    control_matchup_adapter_trained: bool
    candidate_matchup_adapter_frozen: bool
    control_matchup_adapter_frozen: bool
    matchup_adapter_parity_attestation_sha256: str
    candidate_guide2vec_component_graph_sha256: str
    runtime_graph_difference_receipt_sha256: str
    candidate_guide2vec_checkpoint_sha256: str
    candidate_guide2vec_training_receipt_sha256: str
    candidate_guide2vec_runtime_config_sha256: str
    candidate_guide2vec_presence: str
    candidate_guide2vec_module_instance_count: int
    candidate_guide2vec_parameter_count: int
    candidate_guide2vec_frozen: bool
    control_guide2vec_presence: str
    control_guide2vec_module_instance_count: int
    control_guide2vec_parameter_count: int
    control_guide2vec_state_dict_key_count: int
    control_guide2vec_forward_hook_count: int
    control_guide2vec_linear_transform_count: int
    control_guide2vec_disabled_or_zeroed: bool
    control_runtime_graph_observation_sha256: str
    control_guide2vec_absence_attestation_sha256: str
    guide2vec_execution_mode: str
    control_execution_mode: str
    terminal_status: str
    winner_seat: int | None
    illegal_action_count: int
    forfeit_count: int
    crash_count: int
    timeout_count: int
    guide2vec_decisions: tuple[Guide2VecDecisionReceipt, ...]

    def __post_init__(self) -> None:
        _digest(self.game_nonce_sha256, name="game_nonce_sha256")
        if not self.pair_id:
            raise Guide2VecBO1000Error("pair_id must be nonempty")
        if self.game_index not in {0, 1}:
            raise Guide2VecBO1000Error("game_index must be 0 or 1")
        if (
            self.guide2vec_seat not in {0, 1}
            or self.control_seat != 1 - self.guide2vec_seat
        ):
            raise Guide2VecBO1000Error("receipt arms must occupy exact opposite seats")
        for field in (
            "pair_initial_rng_sha256",
            "pair_deck_order_rng_sha256",
            "is_first_attestation_sha256",
            "turn_order_observation_sha256",
            "base_runtime_identity_sha256",
            "experiment_identity_sha256",
            "evaluation_output_identity_sha256",
            "candidate_runtime_graph_sha256",
            "candidate_base_runtime_graph_sha256",
            "control_runtime_graph_sha256",
            "candidate_matchup_tree_sha256",
            "control_matchup_tree_sha256",
            "candidate_matchup_adapter_bank_sha256",
            "control_matchup_adapter_bank_sha256",
            "candidate_matchup_adapter_training_receipt_sha256",
            "control_matchup_adapter_training_receipt_sha256",
            "candidate_matchup_adapter_runtime_graph_sha256",
            "control_matchup_adapter_runtime_graph_sha256",
            "matchup_adapter_parity_attestation_sha256",
            "candidate_guide2vec_component_graph_sha256",
            "runtime_graph_difference_receipt_sha256",
            "candidate_guide2vec_checkpoint_sha256",
            "candidate_guide2vec_training_receipt_sha256",
            "candidate_guide2vec_runtime_config_sha256",
            "control_runtime_graph_observation_sha256",
            "control_guide2vec_absence_attestation_sha256",
        ):
            _digest(getattr(self, field), name=field)
        if (
            self.candidate_base_runtime_graph_sha256
            != self.control_runtime_graph_sha256
        ):
            raise Guide2VecBO1000Error(
                "candidate base graph and control direct graph must be identical"
            )
        if (
            self.candidate_matchup_tree_sha256 != R195_MATCHUP_TREE_SHA256
            or self.control_matchup_tree_sha256 != R195_MATCHUP_TREE_SHA256
        ):
            raise Guide2VecBO1000Error(
                "both arms must use the exact public r195 matchup tree"
            )
        for candidate_field, control_field, label in (
            (
                "candidate_matchup_adapter_bank_sha256",
                "control_matchup_adapter_bank_sha256",
                "adapter bank",
            ),
            (
                "candidate_matchup_adapter_training_receipt_sha256",
                "control_matchup_adapter_training_receipt_sha256",
                "adapter training receipt",
            ),
            (
                "candidate_matchup_adapter_runtime_graph_sha256",
                "control_matchup_adapter_runtime_graph_sha256",
                "adapter runtime graph",
            ),
        ):
            if getattr(self, candidate_field) != getattr(self, control_field):
                raise Guide2VecBO1000Error(
                    f"candidate and control {label} digests must be identical"
                )
        for field in (
            "candidate_matchup_adapter_enabled",
            "control_matchup_adapter_enabled",
            "candidate_matchup_adapter_trained",
            "control_matchup_adapter_trained",
            "candidate_matchup_adapter_frozen",
            "control_matchup_adapter_frozen",
        ):
            value = getattr(self, field)
            if type(value) is not bool:
                raise Guide2VecBO1000Error(f"{field} must be boolean")
            if not value:
                raise Guide2VecBO1000Error(
                    "both arms must enable the same trained, frozen r195 matchup adapter"
                )
        expected_adapter_parity = expected_matchup_adapter_parity_attestation(
            game_nonce_sha256=self.game_nonce_sha256,
            candidate_matchup_tree_sha256=self.candidate_matchup_tree_sha256,
            control_matchup_tree_sha256=self.control_matchup_tree_sha256,
            candidate_matchup_adapter_bank_sha256=(
                self.candidate_matchup_adapter_bank_sha256
            ),
            control_matchup_adapter_bank_sha256=(
                self.control_matchup_adapter_bank_sha256
            ),
            candidate_matchup_adapter_training_receipt_sha256=(
                self.candidate_matchup_adapter_training_receipt_sha256
            ),
            control_matchup_adapter_training_receipt_sha256=(
                self.control_matchup_adapter_training_receipt_sha256
            ),
            candidate_matchup_adapter_runtime_graph_sha256=(
                self.candidate_matchup_adapter_runtime_graph_sha256
            ),
            control_matchup_adapter_runtime_graph_sha256=(
                self.control_matchup_adapter_runtime_graph_sha256
            ),
            candidate_matchup_adapter_enabled=(self.candidate_matchup_adapter_enabled),
            control_matchup_adapter_enabled=self.control_matchup_adapter_enabled,
            candidate_matchup_adapter_trained=(self.candidate_matchup_adapter_trained),
            control_matchup_adapter_trained=self.control_matchup_adapter_trained,
            candidate_matchup_adapter_frozen=(self.candidate_matchup_adapter_frozen),
            control_matchup_adapter_frozen=self.control_matchup_adapter_frozen,
        )
        if self.matchup_adapter_parity_attestation_sha256 != expected_adapter_parity:
            raise Guide2VecBO1000Error(
                "matchup adapter parity attestation must bind both frozen r195 graphs"
            )
        if self.candidate_runtime_graph_sha256 == self.control_runtime_graph_sha256:
            raise Guide2VecBO1000Error(
                "candidate full graph must differ by its frozen Guide2Vec component"
            )
        if self.candidate_guide2vec_presence != CANDIDATE_GUIDE2VEC_PRESENCE:
            raise Guide2VecBO1000Error(
                "candidate alone must load one frozen Guide2Vec component"
            )
        _exact_int(
            self.candidate_guide2vec_module_instance_count,
            name="candidate_guide2vec_module_instance_count",
        )
        if self.candidate_guide2vec_module_instance_count != 1:
            raise Guide2VecBO1000Error(
                "candidate must load exactly one Guide2Vec module instance"
            )
        _exact_int(
            self.candidate_guide2vec_parameter_count,
            name="candidate_guide2vec_parameter_count",
            minimum=GUIDE2VEC_PARAMETER_COUNT_MIN,
        )
        if self.candidate_guide2vec_parameter_count > GUIDE2VEC_PARAMETER_COUNT_MAX:
            raise Guide2VecBO1000Error(
                "candidate Guide2Vec parameter count exceeds the r212 maximum"
            )
        if type(self.candidate_guide2vec_frozen) is not bool:
            raise Guide2VecBO1000Error("candidate_guide2vec_frozen must be boolean")
        if not self.candidate_guide2vec_frozen:
            raise Guide2VecBO1000Error(
                "candidate Guide2Vec must be frozen for the mirror"
            )
        if self.control_guide2vec_presence != CONTROL_GUIDE2VEC_PRESENCE:
            raise Guide2VecBO1000Error(
                "control Guide2Vec component must be absent, not disabled or zeroed"
            )
        for field in (
            "control_guide2vec_module_instance_count",
            "control_guide2vec_parameter_count",
            "control_guide2vec_state_dict_key_count",
            "control_guide2vec_forward_hook_count",
            "control_guide2vec_linear_transform_count",
        ):
            value = _exact_int(getattr(self, field), name=field)
            if value != 0:
                raise Guide2VecBO1000Error(
                    f"{field} must be zero because Guide2Vec is absent from control"
                )
        if type(self.control_guide2vec_disabled_or_zeroed) is not bool:
            raise Guide2VecBO1000Error(
                "control_guide2vec_disabled_or_zeroed must be boolean"
            )
        if self.control_guide2vec_disabled_or_zeroed:
            raise Guide2VecBO1000Error(
                "control cannot instantiate a disabled or zeroed Guide2Vec path"
            )
        expected_control_absence = expected_control_guide2vec_absence_attestation(
            game_nonce_sha256=self.game_nonce_sha256,
            control_runtime_graph_sha256=self.control_runtime_graph_sha256,
            control_runtime_graph_observation_sha256=(
                self.control_runtime_graph_observation_sha256
            ),
        )
        if (
            self.control_guide2vec_absence_attestation_sha256
            != expected_control_absence
        ):
            raise Guide2VecBO1000Error(
                "control absence attestation must bind the exact Guide2Vec-free graph"
            )
        if self.sealed_initial_first_actor_seat not in {0, 1}:
            raise Guide2VecBO1000Error("sealed_initial_first_actor_seat must be 0 or 1")
        if self.observed_first_actor_seat not in {0, 1}:
            raise Guide2VecBO1000Error(
                "observed_first_actor_seat must be explicitly 0 or 1"
            )
        if self.observed_first_actor_arm not in {GUIDE2VEC_ARM, CONTROL_ARM}:
            raise Guide2VecBO1000Error("observed_first_actor_arm is invalid")
        if (
            type(self.guide2vec_is_first) is not bool
            or type(self.control_is_first) is not bool
        ):
            raise Guide2VecBO1000Error("IsFirst attestation values must be boolean")
        expected_candidate_is_first = (
            self.observed_first_actor_seat == self.guide2vec_seat
        )
        expected_first_arm = (
            GUIDE2VEC_ARM if expected_candidate_is_first else CONTROL_ARM
        )
        if self.observed_first_actor_arm != expected_first_arm:
            raise Guide2VecBO1000Error(
                "observed_first_actor_arm must agree with the observed actor seat"
            )
        if (
            self.guide2vec_is_first != expected_candidate_is_first
            or self.control_is_first == expected_candidate_is_first
        ):
            raise Guide2VecBO1000Error(
                "explicit IsFirst attestations must describe exactly one first arm"
            )
        expected_attestation = expected_is_first_attestation(
            game_nonce_sha256=self.game_nonce_sha256,
            observed_first_actor_seat=self.observed_first_actor_seat,
            guide2vec_seat=self.guide2vec_seat,
            control_seat=self.control_seat,
            observed_first_actor_arm=self.observed_first_actor_arm,
            guide2vec_is_first=self.guide2vec_is_first,
            control_is_first=self.control_is_first,
        )
        if self.is_first_attestation_sha256 != expected_attestation:
            raise Guide2VecBO1000Error(
                "is_first_attestation_sha256 must bind the observed engine turn order"
            )
        if self.guide2vec_execution_mode != "bounded_guide_logit_bonus":
            raise Guide2VecBO1000Error(
                "Guide2Vec execution mode must be the bounded direct-policy bonus"
            )
        if self.control_execution_mode != "frozen_r195_no_rtp_direct_policy":
            raise Guide2VecBO1000Error(
                "control execution mode must be the exact frozen direct policy"
            )
        if self.terminal_status not in {"completed", "failed_closed"}:
            raise Guide2VecBO1000Error(
                "terminal_status must be completed or failed_closed"
            )
        if self.winner_seat not in {None, 0, 1}:
            raise Guide2VecBO1000Error("winner_seat must be 0, 1, or None")
        if self.terminal_status == "failed_closed" and self.winner_seat is not None:
            raise Guide2VecBO1000Error("a failed-closed game cannot claim a winner")
        for field in (
            "illegal_action_count",
            "forfeit_count",
            "crash_count",
            "timeout_count",
        ):
            _exact_int(getattr(self, field), name=field)
        if not isinstance(self.guide2vec_decisions, tuple) or any(
            not isinstance(decision, Guide2VecDecisionReceipt)
            for decision in self.guide2vec_decisions
        ):
            raise Guide2VecBO1000Error(
                "guide2vec_decisions must be a tuple of Guide2VecDecisionReceipt"
            )
        seen_indices: set[int] = set()
        for decision in self.guide2vec_decisions:
            if decision.game_nonce_sha256 != self.game_nonce_sha256:
                raise Guide2VecBO1000Error("decision/game nonce binding mismatch")
            if decision.acting_seat != self.guide2vec_seat:
                raise Guide2VecBO1000Error(
                    "only the Guide2Vec arm may emit Guide2Vec decision evidence"
                )
            if decision.decision_index in seen_indices:
                raise Guide2VecBO1000Error("duplicate Guide2Vec decision_index in game")
            seen_indices.add(decision.decision_index)
        if self.terminal_status == "completed" and not self.guide2vec_decisions:
            raise Guide2VecBO1000Error(
                "a completed candidate game needs Guide2Vec decision evidence"
            )

    def as_payload(self) -> dict[str, object]:
        payload = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "guide2vec_decisions"
        }
        payload["guide2vec_decisions"] = [
            decision.as_payload() for decision in self.guide2vec_decisions
        ]
        return payload

    @classmethod
    def from_payload(cls, raw: object) -> Guide2VecBO1000GameReceipt:
        values = _exact_fields(
            raw, set(cls.__dataclass_fields__), label="Guide2VecBO1000GameReceipt"
        )
        decisions = values.get("guide2vec_decisions")
        if not isinstance(decisions, list):
            raise Guide2VecBO1000Error("guide2vec_decisions must be a list")
        values["guide2vec_decisions"] = tuple(
            Guide2VecDecisionReceipt.from_payload(item) for item in decisions
        )
        try:
            return cls(**values)
        except (TypeError, Guide2VecBO1000Error) as exc:
            raise Guide2VecBO1000Error(
                f"game receipt payload is invalid: {exc}"
            ) from exc


def _distribution(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "minimum": None,
            "maximum": None,
        }
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    return {
        "count": count,
        "mean": sum(ordered) / count,
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * count) - 1)],
        "p99": ordered[max(0, math.ceil(0.99 * count) - 1)],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _candidate_outcome(receipt: Guide2VecBO1000GameReceipt) -> str:
    if receipt.terminal_status != "completed":
        return "failed_closed"
    if receipt.winner_seat is None:
        return "draw"
    return "win" if receipt.winner_seat == receipt.guide2vec_seat else "loss"


def _control_outcome(receipt: Guide2VecBO1000GameReceipt) -> str:
    if receipt.terminal_status != "completed":
        return "failed_closed"
    if receipt.winner_seat is None:
        return "draw"
    return "win" if receipt.winner_seat == receipt.control_seat else "loss"


def _outcome_counts(outcomes: Iterable[str]) -> dict[str, int]:
    materialized = tuple(outcomes)
    return {
        label: materialized.count(label)
        for label in ("win", "draw", "loss", "failed_closed")
    }


def _score(outcome: str) -> float:
    return 1.0 if outcome == "win" else 0.5 if outcome == "draw" else 0.0


def _validate_schedule(
    schedule: Sequence[Guide2VecBO1000GameSpec],
    *,
    experiment: Guide2VecExperimentIdentity,
) -> tuple[
    dict[str, Guide2VecBO1000GameSpec],
    dict[str, list[Guide2VecBO1000GameSpec]],
]:
    if len(schedule) != GUIDE2VEC_BO1000_GAME_COUNT:
        raise Guide2VecBO1000Error("schedule must contain exactly 1000 games")
    by_nonce: dict[str, Guide2VecBO1000GameSpec] = {}
    by_pair: dict[str, list[Guide2VecBO1000GameSpec]] = {}
    for spec in schedule:
        if not isinstance(spec, Guide2VecBO1000GameSpec):
            raise Guide2VecBO1000Error(
                "schedule must contain Guide2VecBO1000GameSpec values"
            )
        if spec.experiment_identity_sha256 != experiment.identity_sha256:
            raise Guide2VecBO1000Error(
                "schedule experiment identity does not match the frozen candidate"
            )
        if (
            spec.evaluation_output_identity_sha256
            != experiment.evaluation_output_identity_sha256
        ):
            raise Guide2VecBO1000Error(
                "schedule output identity is not the isolated r212 output"
            )
        if spec.game_nonce_sha256 in by_nonce:
            raise Guide2VecBO1000Error("duplicate game nonce in schedule")
        by_nonce[spec.game_nonce_sha256] = spec
        by_pair.setdefault(spec.pair_id, []).append(spec)
    if len(by_pair) != GUIDE2VEC_BO1000_PAIR_COUNT:
        raise Guide2VecBO1000Error("schedule must contain exactly 500 unique pairs")
    if {spec.pair_index for spec in schedule} != set(
        range(GUIDE2VEC_BO1000_PAIR_COUNT)
    ):
        raise Guide2VecBO1000Error(
            "schedule pair indices must be exactly 0 through 499"
        )
    pair_nonce_to_id: dict[str, str] = {}
    for pair_id, specs in by_pair.items():
        if len(specs) != 2:
            raise Guide2VecBO1000Error("every pair must contain exactly two games")
        ordered = sorted(specs, key=lambda item: item.game_index)
        if [
            (spec.game_index, spec.guide2vec_seat, spec.control_seat)
            for spec in ordered
        ] != [(0, 0, 1), (1, 1, 0)]:
            raise Guide2VecBO1000Error(
                "every pair must swap the Guide2Vec candidate between seats"
            )
        if len({spec.pair_index for spec in specs}) != 1:
            raise Guide2VecBO1000Error("pair index mismatch inside scheduled pair")
        if len({spec.pair_nonce_sha256 for spec in specs}) != 1:
            raise Guide2VecBO1000Error("pair nonce mismatch inside scheduled pair")
        if len({spec.pair_initial_rng_sha256 for spec in specs}) != 1:
            raise Guide2VecBO1000Error("pair initial RNG mismatch inside schedule")
        if len({spec.pair_deck_order_rng_sha256 for spec in specs}) != 1:
            raise Guide2VecBO1000Error("pair deck-order RNG mismatch inside schedule")
        if len({spec.sealed_initial_first_actor_seat for spec in specs}) != 1:
            raise Guide2VecBO1000Error(
                "pair initial actor must be explicit sealed material shared by both games"
            )
        pair_nonce = specs[0].pair_nonce_sha256
        expected_pair_id = (
            f"r212-guide2vec-pair-{specs[0].pair_index:06d}-{pair_nonce[7:19]}"
        )
        if pair_id != expected_pair_id:
            raise Guide2VecBO1000Error("pair_id must bind r212 pair index and nonce")
        known_pair_id = pair_nonce_to_id.setdefault(pair_nonce, pair_id)
        if known_pair_id != pair_id:
            raise Guide2VecBO1000Error("pair nonce is reused across pairs")
        first_seat = specs[0].sealed_initial_first_actor_seat
        if sum(spec.guide2vec_seat == first_seat for spec in specs) != 1:
            raise Guide2VecBO1000Error(
                "each sealed pair must schedule exactly one candidate first game"
            )
    return by_nonce, by_pair


def _revalidate_receipt(receipt: Guide2VecBO1000GameReceipt) -> None:
    """Reconstruct the frozen dataclass before trusting potentially mutated input."""

    try:
        Guide2VecBO1000GameReceipt(
            **{
                field: getattr(receipt, field)
                for field in Guide2VecBO1000GameReceipt.__dataclass_fields__
            }
        )
    except (TypeError, Guide2VecBO1000Error) as exc:
        raise Guide2VecBO1000Error(
            "game receipt violates the canonical r212 direct-policy contract"
        ) from exc


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def compile_guide2vec_bo1000_report(
    schedule: Sequence[Guide2VecBO1000GameSpec],
    receipts: Iterable[Guide2VecBO1000GameReceipt],
    *,
    experiment: Guide2VecExperimentIdentity,
) -> dict[str, object]:
    """Compile all 1,000 receipts or fail closed without a partial BO1000 result."""

    if not isinstance(experiment, Guide2VecExperimentIdentity):
        raise Guide2VecBO1000Error("experiment must be Guide2VecExperimentIdentity")
    by_nonce, _by_pair = _validate_schedule(schedule, experiment=experiment)
    observed: dict[str, Guide2VecBO1000GameReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, Guide2VecBO1000GameReceipt):
            raise Guide2VecBO1000Error(
                "receipts must contain Guide2VecBO1000GameReceipt values"
            )
        _revalidate_receipt(receipt)
        if receipt.game_nonce_sha256 in observed:
            raise Guide2VecBO1000Error("duplicate game receipt")
        spec = by_nonce.get(receipt.game_nonce_sha256)
        if spec is None:
            raise Guide2VecBO1000Error("receipt is not in the immutable schedule")
        if (
            receipt.pair_id != spec.pair_id
            or receipt.game_index != spec.game_index
            or receipt.guide2vec_seat != spec.guide2vec_seat
            or receipt.control_seat != spec.control_seat
        ):
            raise Guide2VecBO1000Error("receipt schedule identity mismatch")
        if (
            receipt.pair_initial_rng_sha256 != spec.pair_initial_rng_sha256
            or receipt.pair_deck_order_rng_sha256 != spec.pair_deck_order_rng_sha256
        ):
            raise Guide2VecBO1000Error(
                "receipt does not match the sealed initial RNG/deck-order pair"
            )
        if (
            receipt.sealed_initial_first_actor_seat
            != spec.sealed_initial_first_actor_seat
            or receipt.observed_first_actor_seat != spec.sealed_initial_first_actor_seat
        ):
            raise Guide2VecBO1000Error(
                "observed first actor must match the explicit sealed pair material"
            )
        if (
            receipt.base_runtime_identity_sha256
            != experiment.base_runtime.identity_sha256
            or receipt.experiment_identity_sha256 != experiment.identity_sha256
            or receipt.evaluation_output_identity_sha256
            != experiment.evaluation_output_identity_sha256
        ):
            raise Guide2VecBO1000Error(
                "receipt base/candidate/output identity does not match the isolated experiment"
            )
        if (
            receipt.candidate_runtime_graph_sha256
            != experiment.candidate_runtime_graph_sha256
            or receipt.candidate_base_runtime_graph_sha256
            != experiment.control_runtime_graph_sha256
            or receipt.control_runtime_graph_sha256
            != experiment.control_runtime_graph_sha256
            or receipt.candidate_guide2vec_component_graph_sha256
            != experiment.candidate_guide2vec_component_graph_sha256
            or receipt.runtime_graph_difference_receipt_sha256
            != experiment.runtime_graph_difference_receipt_sha256
        ):
            raise Guide2VecBO1000Error(
                "receipt runtime graphs do not match the candidate/control graph split"
            )
        base_runtime = experiment.base_runtime
        if (
            receipt.candidate_matchup_tree_sha256 != base_runtime.matchup_tree_sha256
            or receipt.control_matchup_tree_sha256 != base_runtime.matchup_tree_sha256
            or receipt.candidate_matchup_adapter_bank_sha256
            != base_runtime.matchup_adapter_bank_sha256
            or receipt.control_matchup_adapter_bank_sha256
            != base_runtime.matchup_adapter_bank_sha256
            or receipt.candidate_matchup_adapter_training_receipt_sha256
            != base_runtime.matchup_adapter_training_receipt_sha256
            or receipt.control_matchup_adapter_training_receipt_sha256
            != base_runtime.matchup_adapter_training_receipt_sha256
            or receipt.candidate_matchup_adapter_runtime_graph_sha256
            != base_runtime.matchup_adapter_runtime_graph_sha256
            or receipt.control_matchup_adapter_runtime_graph_sha256
            != base_runtime.matchup_adapter_runtime_graph_sha256
        ):
            raise Guide2VecBO1000Error(
                "receipt matchup adapter identity does not match the frozen r195 adapter"
            )
        if (
            receipt.candidate_guide2vec_checkpoint_sha256
            != experiment.guide2vec_checkpoint_sha256
            or receipt.candidate_guide2vec_training_receipt_sha256
            != experiment.guide2vec_training_receipt_sha256
            or receipt.candidate_guide2vec_runtime_config_sha256
            != experiment.guide2vec_runtime_config_sha256
            or receipt.candidate_guide2vec_parameter_count
            != experiment.guide2vec_parameter_count
        ):
            raise Guide2VecBO1000Error(
                "receipt Guide2Vec identity does not match the frozen candidate"
            )
        observed[receipt.game_nonce_sha256] = receipt

    missing = sorted(set(by_nonce) - set(observed))
    if missing:
        raise Guide2VecBO1000Error(
            f"missing {len(missing)} scheduled terminal game receipts"
        )

    games = [observed[spec.game_nonce_sha256] for spec in schedule]
    pair_receipts: dict[str, list[Guide2VecBO1000GameReceipt]] = {}
    for game in games:
        pair_receipts.setdefault(game.pair_id, []).append(game)
    for pair_id, pair in pair_receipts.items():
        if len(pair) != 2:
            raise Guide2VecBO1000Error(
                f"pair {pair_id} must contain exactly two terminal receipts"
            )
        if len({game.pair_initial_rng_sha256 for game in pair}) != 1:
            raise Guide2VecBO1000Error(f"pair {pair_id} initial RNG mismatch")
        if len({game.pair_deck_order_rng_sha256 for game in pair}) != 1:
            raise Guide2VecBO1000Error(f"pair {pair_id} deck-order RNG mismatch")
        if sum(game.guide2vec_is_first for game in pair) != 1:
            raise Guide2VecBO1000Error(
                f"pair {pair_id} must attest exactly one Guide2Vec-first game"
            )
        if sum(game.control_is_first for game in pair) != 1:
            raise Guide2VecBO1000Error(
                f"pair {pair_id} must attest exactly one direct-policy-first game"
            )

    guide_first = sum(game.guide2vec_is_first for game in games)
    guide_second = len(games) - guide_first
    control_first = sum(game.control_is_first for game in games)
    control_second = len(games) - control_first
    if (
        guide_first != GUIDE2VEC_BO1000_PAIR_COUNT
        or guide_second != GUIDE2VEC_BO1000_PAIR_COUNT
        or control_first != GUIDE2VEC_BO1000_PAIR_COUNT
        or control_second != GUIDE2VEC_BO1000_PAIR_COUNT
    ):
        raise Guide2VecBO1000Error(
            "actual first/second IsFirst attestations are not exactly 500/500"
        )

    candidate_outcomes = [_candidate_outcome(game) for game in games]
    control_outcomes = [_control_outcome(game) for game in games]
    candidate_counts = _outcome_counts(candidate_outcomes)
    control_counts = _outcome_counts(control_outcomes)
    candidate_by_seat = {
        str(seat): _outcome_counts(
            _candidate_outcome(game) for game in games if game.guide2vec_seat == seat
        )
        for seat in (0, 1)
    }
    candidate_by_turn_order = {
        "first": _outcome_counts(
            _candidate_outcome(game) for game in games if game.guide2vec_is_first
        ),
        "second": _outcome_counts(
            _candidate_outcome(game) for game in games if not game.guide2vec_is_first
        ),
    }
    control_by_turn_order = {
        "first": _outcome_counts(
            _control_outcome(game) for game in games if game.control_is_first
        ),
        "second": _outcome_counts(
            _control_outcome(game) for game in games if not game.control_is_first
        ),
    }

    pair_matrix: dict[str, int] = {}
    pair_scores: list[float] = []
    failed_pair_ids: list[str] = []
    for pair_id in sorted(pair_receipts):
        pair = sorted(pair_receipts[pair_id], key=lambda game: game.game_index)
        outcomes = tuple(_candidate_outcome(game) for game in pair)
        key = f"{outcomes[0]}__{outcomes[1]}"
        pair_matrix[key] = pair_matrix.get(key, 0) + 1
        if "failed_closed" in outcomes:
            failed_pair_ids.append(pair_id)
        else:
            pair_scores.append(sum(_score(outcome) for outcome in outcomes) / 2.0)
    paired_score = None if not pair_scores else sum(pair_scores) / len(pair_scores)
    paired_difference = None if paired_score is None else paired_score - 0.5
    if len(pair_scores) >= 2:
        standard_error = statistics.stdev(pair_scores) / math.sqrt(len(pair_scores))
        paired_ci: list[float] | None = [
            paired_difference - 1.96 * standard_error,  # type: ignore[operator]
            paired_difference + 1.96 * standard_error,  # type: ignore[operator]
        ]
    else:
        paired_ci = None

    decisions = [decision for game in games for decision in game.guide2vec_decisions]
    eligible = [decision for decision in decisions if decision.eligible]
    abstentions = [decision for decision in decisions if decision.abstained]
    applied = [decision for decision in decisions if decision.bonus_applied]
    changed = [
        decision for decision in decisions if decision.action_changed_from_direct_policy
    ]
    runtime_failures = (
        candidate_counts["failed_closed"] + control_counts["failed_closed"]
    )
    report: dict[str, object] = {
        "schema": GUIDE2VEC_BO1000_REPORT_SCHEMA,
        "status": (
            "complete"
            if runtime_failures == 0
            else "failed_closed_complete_runtime_evidence"
        ),
        "support": {
            "scheduled_games": GUIDE2VEC_BO1000_GAME_COUNT,
            "observed_terminal_game_receipts": len(games),
            "rng_matched_pairs": len(pair_receipts),
            "guide2vec_as_seat_0": sum(game.guide2vec_seat == 0 for game in games),
            "guide2vec_as_seat_1": sum(game.guide2vec_seat == 1 for game in games),
            "guide2vec_actual_first": guide_first,
            "guide2vec_actual_second": guide_second,
            "no_rtp_actual_first": control_first,
            "no_rtp_actual_second": control_second,
            "paired_analysis_eligible_pairs": len(pair_scores),
            "paired_analysis_excluded_failed_closed_pairs": len(failed_pair_ids),
        },
        "identities": {
            "evaluation_id": GUIDE2VEC_EVALUATION_ID,
            "r212_contract_sha256": R212_CONTRACT_SHA256,
            "candidate_arm": GUIDE2VEC_ARM,
            "control_arm": CONTROL_ARM,
            "frozen_r195_base_runtime": experiment.base_runtime.as_payload(),
            "frozen_r195_base_runtime_identity_sha256": (
                experiment.base_runtime.identity_sha256
            ),
            "candidate_guide2vec_checkpoint_sha256": (
                experiment.guide2vec_checkpoint_sha256
            ),
            "candidate_guide2vec_training_receipt_sha256": (
                experiment.guide2vec_training_receipt_sha256
            ),
            "candidate_guide2vec_runtime_config_sha256": (
                experiment.guide2vec_runtime_config_sha256
            ),
            "candidate_guide2vec_parameter_count": (
                experiment.guide2vec_parameter_count
            ),
            "matchup_adapter": {
                "public_matchup_tree_sha256": (
                    experiment.base_runtime.matchup_tree_sha256
                ),
                "frozen_trained_adapter_bank_sha256": (
                    experiment.base_runtime.matchup_adapter_bank_sha256
                ),
                "training_receipt_sha256": (
                    experiment.base_runtime.matchup_adapter_training_receipt_sha256
                ),
                "shared_adapter_runtime_graph_sha256": (
                    experiment.base_runtime.matchup_adapter_runtime_graph_sha256
                ),
                "candidate_enabled": True,
                "control_enabled": True,
                "candidate_trained": True,
                "control_trained": True,
                "candidate_frozen": True,
                "control_frozen": True,
                "parity_attested_games": len(games),
            },
            "runtime_graphs": {
                "candidate_full_graph_sha256": (
                    experiment.candidate_runtime_graph_sha256
                ),
                "candidate_base_graph_sha256": (
                    experiment.control_runtime_graph_sha256
                ),
                "control_direct_graph_sha256": (
                    experiment.control_runtime_graph_sha256
                ),
                "candidate_guide2vec_component_graph_sha256": (
                    experiment.candidate_guide2vec_component_graph_sha256
                ),
                "runtime_graph_difference_receipt_sha256": (
                    experiment.runtime_graph_difference_receipt_sha256
                ),
                "candidate_guide2vec_presence": CANDIDATE_GUIDE2VEC_PRESENCE,
                "control_guide2vec_presence": CONTROL_GUIDE2VEC_PRESENCE,
                "control_guide2vec_module_instance_count": 0,
                "control_guide2vec_parameter_count": 0,
                "control_guide2vec_state_dict_key_count": 0,
                "control_guide2vec_forward_hook_count": 0,
                "control_guide2vec_linear_transform_count": 0,
                "control_guide2vec_disabled_or_zeroed": False,
                "control_absence_attested_games": len(games),
            },
            "source_snapshot_sha256": experiment.source_snapshot_sha256,
            "experiment_identity_sha256": experiment.identity_sha256,
            "evaluation_output_identity_sha256": (
                experiment.evaluation_output_identity_sha256
            ),
            "schedule_sha256": _canonical_sha256(
                [spec.as_payload() for spec in schedule]
            ),
        },
        "game_outcomes": {
            "by_arm": {
                GUIDE2VEC_ARM: candidate_counts,
                CONTROL_ARM: control_counts,
            },
            "guide2vec_by_seat": candidate_by_seat,
            "guide2vec_by_actual_turn_order": candidate_by_turn_order,
            "no_rtp_by_actual_turn_order": control_by_turn_order,
            "paired_outcome_matrix": dict(sorted(pair_matrix.items())),
            "paired_analysis": {
                "eligible_completed_pairs": len(pair_scores),
                "excluded_failed_closed_pairs": len(failed_pair_ids),
                "excluded_failed_closed_pair_ids_sha256": _canonical_sha256(
                    failed_pair_ids
                ),
                "imputation_used": False,
            },
            "paired_guide2vec_score": (paired_score if runtime_failures == 0 else None),
            "paired_win_rate_difference": (
                paired_difference if runtime_failures == 0 else None
            ),
            "paired_confidence_interval": (
                paired_ci if runtime_failures == 0 else None
            ),
            "paired_ci_method": (
                "pair-clustered normal approximation over two-game mean scores"
            ),
            "illegal_action_count": sum(game.illegal_action_count for game in games),
            "forfeit_count": sum(game.forfeit_count for game in games),
            "crash_count": sum(game.crash_count for game in games),
            "timeout_count": sum(game.timeout_count for game in games),
        },
        "guide2vec_decisions": {
            "candidate_decision_count": len(decisions),
            "eligible_stage_count": len(eligible),
            "abstain_stage_count": len(abstentions),
            "bonus_applied_stage_count": len(applied),
            "action_change_count": len(changed),
            "eligibility_rate": _rate(len(eligible), len(decisions)),
            "abstain_rate_over_all_stages": _rate(len(abstentions), len(decisions)),
            "abstain_rate_over_eligible_stages": _rate(
                sum(decision.abstained for decision in eligible), len(eligible)
            ),
            "bonus_applied_rate_over_eligible_stages": _rate(
                len(applied), len(eligible)
            ),
            "action_change_rate_over_eligible_stages": _rate(
                len(changed), len(eligible)
            ),
        },
        "latency": {
            "guide2vec_action_latency_seconds": _distribution(
                [decision.guide2vec_action_latency_seconds for decision in decisions]
            ),
            "total_action_latency_seconds": _distribution(
                [decision.total_action_latency_seconds for decision in decisions]
            ),
        },
        "authority": {
            "training_eligible": False,
            "serving_eligible": False,
            "production_action_authority_enabled": False,
            "selector_change_authorized": False,
            "kaggle_submission_authorized": False,
            "promotion_authorized": False,
        },
        "receipt_integrity": {
            "raw_per_game_and_per_decision_receipts_preserved": True,
            "missing_receipts_may_be_imputed": False,
            "paired_failed_closed_outcomes_imputed": False,
            "only_candidate_delta_is_frozen_bounded_guide2vec_bonus": True,
            "candidate_and_control_runtime_graphs_separately_bound": True,
            "frozen_r195_matchup_adapter_enabled_on_both_arms": True,
            "candidate_and_control_matchup_adapter_graph_identical": True,
            "control_guide2vec_module_absent_in_all_receipts": True,
            "control_guide2vec_parameter_or_state_key_count": 0,
            "control_guide2vec_hook_or_linear_transform_count": 0,
            "control_disabled_or_zeroed_substitute_allowed": False,
            "initial_actor_is_explicit_sealed_pair_material_not_inferred_from_seat": True,
            "all_1000_terminal_receipts_required": True,
        },
    }
    report["canonical_sha256"] = _canonical_sha256(report)
    return report


__all__ = [
    "CANDIDATE_GUIDE2VEC_PRESENCE",
    "CONTROL_ARM",
    "CONTROL_GUIDE2VEC_PRESENCE",
    "GUIDE2VEC_ARM",
    "GUIDE2VEC_BO1000_GAME_COUNT",
    "GUIDE2VEC_BO1000_PAIR_COUNT",
    "GUIDE2VEC_BO1000_REPORT_SCHEMA",
    "GUIDE2VEC_EVALUATION_ID",
    "GUIDE2VEC_MAX_LOGIT_BONUS",
    "R195_BUNDLE_SHA256",
    "R195_CHECKPOINT_BYTES",
    "R195_CHECKPOINT_SHA256",
    "R195_DECK_CARDS_SHA256",
    "R195_DECK_ID",
    "R195_MATCHUP_TREE_SHA256",
    "R195_SUBMISSION_ID",
    "R195_SUBMISSION_MESSAGE",
    "R212_CONTRACT_SHA256",
    "FrozenR195RuntimeIdentity",
    "Guide2VecBO1000Error",
    "Guide2VecBO1000GameReceipt",
    "Guide2VecBO1000GameSpec",
    "Guide2VecDecisionReceipt",
    "Guide2VecExperimentIdentity",
    "build_guide2vec_bo1000_schedule",
    "compile_guide2vec_bo1000_report",
    "expected_control_guide2vec_absence_attestation",
    "expected_is_first_attestation",
    "expected_matchup_adapter_parity_attestation",
]
