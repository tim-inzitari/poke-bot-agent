"""Validated, deterministic deck-family mixtures from official ladder episodes.

The artifact consumed here records *observed* ladder prevalence separately from
the deliberate training distribution.  It schedules stable deck-family IDs;
the trainer binds those IDs to its locally available 60-card representative
lists with :meth:`LadderDeckMix.bind_catalog`.

Keeping family prevalence independent from concrete deck files is intentional:
the official daily report establishes family counts, while representative lists
remain executable-agent assets that can change independently.  A run should pin
both this artifact digest and the hashes of the bound 60-card lists.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "poke_bot.ladder_deck_mix/v1"
REPRESENTATIVE_SCHEMA = "poke_bot.ladder_deck_representatives/v1"
_DIGEST_FIELD = "artifact_sha256"
_WEIGHT_TOLERANCE = 1e-9


class LadderDeckMixError(ValueError):
    """The ladder mixture or a bound representative deck is invalid."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_payload_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 of canonical JSON, excluding the self-declared digest field."""
    content = dict(payload)
    content.pop(_DIGEST_FIELD, None)
    return "sha256:" + hashlib.sha256(_canonical_json(content)).hexdigest()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LadderDeckMixError(f"{name} must be an integer")
    if value < minimum:
        raise LadderDeckMixError(f"{name} must be >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LadderDeckMixError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise LadderDeckMixError(f"{name} must be finite and >= {minimum}")
    return result


@dataclass(frozen=True)
class LadderDeck:
    """One ranked deck-family bucket in an official episode census."""

    source_rank: int
    deck_id: str
    observed_count: int
    observed_weight: float
    known_conditional_weight: float
    train_weight: float
    games_featuring: int
    game_share: float
    win_rate: float
    wilson_95: tuple[float, float]
    classification_method: str
    signature_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BoundLadderDeck:
    """A scheduled family bound to one immutable 60-card representative."""

    bucket: LadderDeck
    card_ids: tuple[int, ...]
    canonical_multiset_sha256: str


@dataclass(frozen=True)
class LadderDeckRepresentatives:
    """Checksummed modal 60-card representatives from the source episodes."""

    schema: str
    artifact_sha256: str
    source_mix_sha256: str
    source_dataset: str
    selection: str
    decks: Mapping[str, Mapping[str, Any]]

    def bind(self, mix: "LadderDeckMix") -> tuple[BoundLadderDeck, ...]:
        if self.source_mix_sha256 != mix.artifact_sha256:
            raise LadderDeckMixError(
                "representatives were derived from a different ladder mix: "
                f"representatives={self.source_mix_sha256} "
                f"mix={mix.artifact_sha256}"
            )
        expected = {deck.deck_id for deck in mix.decks}
        actual = set(self.decks)
        if actual != expected:
            raise LadderDeckMixError(
                "representative family IDs do not exactly match the ladder mix: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        catalog = {
            deck_id: tuple(int(card) for card in row["card_ids"])
            for deck_id, row in self.decks.items()
        }
        return mix.bind_catalog(catalog)

    def contract(self, mix: "LadderDeckMix") -> dict[str, Any]:
        """Immutable lineage fields, including every bound list multiset hash."""
        bound = self.bind(mix)
        return {
            "mix_id": mix.mix_id,
            "mix_artifact_sha256": mix.artifact_sha256,
            "representatives_artifact_sha256": self.artifact_sha256,
            "source_dataset": self.source_dataset,
            "selection": self.selection,
            "weights": mix.weights("train"),
            "representatives": [
                {
                    "deck_id": item.bucket.deck_id,
                    "source_rank": item.bucket.source_rank,
                    "observed_count": item.bucket.observed_count,
                    "train_weight": item.bucket.train_weight,
                    "card_count": len(item.card_ids),
                    "canonical_multiset_sha256": item.canonical_multiset_sha256,
                    "modal_seat_count": int(
                        self.decks[item.bucket.deck_id]["modal_seat_count"]
                    ),
                    "labeled_seat_count": int(
                        self.decks[item.bucket.deck_id]["labeled_seat_count"]
                    ),
                }
                for item in bound
            ],
        }


@dataclass(frozen=True)
class LadderDeckMix:
    """An immutable, evidence-bearing ladder distribution."""

    schema: str
    mix_id: str
    artifact_sha256: str
    source: Mapping[str, Any]
    coverage: Mapping[str, Any]
    weight_policy: Mapping[str, Any]
    decks: tuple[LadderDeck, ...]
    excluded: tuple[Mapping[str, Any], ...]

    def weights(self, basis: str = "train") -> dict[str, float]:
        """Return normalized active weights for training or observed-known play."""
        if basis == "train":
            values = {deck.deck_id: deck.train_weight for deck in self.decks}
        elif basis in {"observed", "known_observed"}:
            values = {
                deck.deck_id: deck.known_conditional_weight for deck in self.decks
            }
        else:
            raise LadderDeckMixError(f"unknown weight basis: {basis!r}")
        total = sum(values.values())
        if total <= 0.0:
            raise LadderDeckMixError(f"{basis} weights have zero mass")
        return {key: value / total for key, value in values.items()}

    def quotas(self, total: int, basis: str = "train") -> dict[str, int]:
        """Allocate exact integer counts with Hamilton/largest remainder."""
        return largest_remainder_quotas(self.weights(basis), total)

    def schedule_ids(
        self,
        total: int,
        *,
        seed: int,
        iteration: int = 0,
        stream: str = "self_play_our",
        basis: str = "train",
    ) -> tuple[str, ...]:
        """Return a cross-version deterministic permutation of exact quotas."""
        quotas = self.quotas(total, basis)
        tokens: list[tuple[str, int]] = []
        for deck_id in sorted(quotas):
            tokens.extend((deck_id, ordinal) for ordinal in range(quotas[deck_id]))
        prefix = (
            f"{self.artifact_sha256}|{int(seed)}|{int(iteration)}|{stream}|{basis}|"
        )
        tokens.sort(
            key=lambda token: hashlib.sha256(
                f"{prefix}{token[0]}|{token[1]}".encode("utf-8")
            ).digest()
        )
        return tuple(deck_id for deck_id, _ordinal in tokens)

    def schedule_provenance(
        self,
        total: int,
        *,
        seed: int,
        iteration: int = 0,
        stream: str = "self_play_our",
        basis: str = "train",
    ) -> dict[str, Any]:
        """Auditable metadata to attach to a run manifest or shard."""
        return {
            "mix_id": self.mix_id,
            "artifact_sha256": self.artifact_sha256,
            "dataset_slug": self.source["dataset_slug"],
            "window_start_utc": self.source["window_start_utc"],
            "window_end_utc_exclusive": self.source["window_end_utc_exclusive"],
            "basis": basis,
            "seed": int(seed),
            "iteration": int(iteration),
            "stream": stream,
            "total": int(total),
            "quotas": self.quotas(total, basis),
        }

    def bind_catalog(
        self, catalog: Mapping[str, Sequence[int]]
    ) -> tuple[BoundLadderDeck, ...]:
        """Fail closed unless every active family has a valid representative.

        Signature groups are AND-of-ORs: at least one card from every group must
        appear.  Buckets labeled only through played ace names have no signature
        groups and are therefore validated by ID and 60-card shape alone.
        """
        bound: list[BoundLadderDeck] = []
        for bucket in self.decks:
            if bucket.deck_id not in catalog:
                raise LadderDeckMixError(
                    f"missing representative deck for {bucket.deck_id!r}"
                )
            raw_cards = catalog[bucket.deck_id]
            if isinstance(raw_cards, (str, bytes)):
                raise LadderDeckMixError(
                    f"representative {bucket.deck_id!r} must be a card sequence"
                )
            cards: list[int] = []
            for value in raw_cards:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise LadderDeckMixError(
                        f"representative {bucket.deck_id!r} has non-integer card ID"
                    )
                cards.append(value)
            if len(cards) != 60:
                raise LadderDeckMixError(
                    f"representative {bucket.deck_id!r} has {len(cards)} cards; "
                    "expected 60"
                )
            present = set(cards)
            for group in bucket.signature_groups:
                if not present.intersection(group):
                    raise LadderDeckMixError(
                        f"representative {bucket.deck_id!r} misses signature "
                        f"group {list(group)!r}"
                    )
            canonical = ",".join(str(card_id) for card_id in sorted(cards)).encode(
                "ascii"
            )
            bound.append(
                BoundLadderDeck(
                    bucket=bucket,
                    card_ids=tuple(cards),
                    canonical_multiset_sha256=(
                        "sha256:" + hashlib.sha256(canonical).hexdigest()
                    ),
                )
            )
        return tuple(bound)


def largest_remainder_quotas(
    weights: Mapping[str, float], total: int
) -> dict[str, int]:
    """Convert normalized weights into exact deterministic integer quotas."""
    total = _integer(total, "total", minimum=0)
    if not weights:
        if total == 0:
            return {}
        raise LadderDeckMixError("cannot allocate a positive total without weights")
    cleaned = {
        str(key): _number(value, f"weight[{key!r}]") for key, value in weights.items()
    }
    if len(cleaned) != len(weights) or any(not key for key in cleaned):
        raise LadderDeckMixError("weight IDs must be unique non-empty strings")
    mass = sum(cleaned.values())
    if not math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=_WEIGHT_TOLERANCE):
        raise LadderDeckMixError(f"weights sum to {mass:.12f}, expected 1")
    raw = {key: total * weight for key, weight in cleaned.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(result.values())
    ranked = sorted(cleaned, key=lambda key: (-(raw[key] - result[key]), key))
    for key in ranked[:remaining]:
        result[key] += 1
    return result


def _parse_deck(raw: Mapping[str, Any], index: int) -> LadderDeck:
    prefix = f"decks[{index}]"
    deck_id = raw.get("deck_id")
    if not isinstance(deck_id, str) or not deck_id.strip():
        raise LadderDeckMixError(f"{prefix}.deck_id must be a non-empty string")
    groups_raw = raw.get("signature_groups", [])
    if not isinstance(groups_raw, list):
        raise LadderDeckMixError(f"{prefix}.signature_groups must be a list")
    groups: list[tuple[int, ...]] = []
    for group_index, group_raw in enumerate(groups_raw):
        if not isinstance(group_raw, list) or not group_raw:
            raise LadderDeckMixError(
                f"{prefix}.signature_groups[{group_index}] must be non-empty"
            )
        group = tuple(
            _integer(card_id, f"{prefix}.signature_groups[{group_index}]", minimum=1)
            for card_id in group_raw
        )
        if len(set(group)) != len(group):
            raise LadderDeckMixError(
                f"{prefix}.signature_groups[{group_index}] has duplicate IDs"
            )
        groups.append(group)
    wilson = raw.get("wilson_95")
    if not isinstance(wilson, list) or len(wilson) != 2:
        raise LadderDeckMixError(f"{prefix}.wilson_95 must contain [low, high]")
    method = raw.get("classification_method")
    if not isinstance(method, str) or not method:
        raise LadderDeckMixError(f"{prefix}.classification_method is required")
    return LadderDeck(
        source_rank=_integer(raw.get("source_rank"), f"{prefix}.source_rank", minimum=1),
        deck_id=deck_id,
        observed_count=_integer(
            raw.get("observed_count"), f"{prefix}.observed_count", minimum=1
        ),
        observed_weight=_number(raw.get("observed_weight"), f"{prefix}.observed_weight"),
        known_conditional_weight=_number(
            raw.get("known_conditional_weight"),
            f"{prefix}.known_conditional_weight",
        ),
        train_weight=_number(raw.get("train_weight"), f"{prefix}.train_weight"),
        games_featuring=_integer(
            raw.get("games_featuring"), f"{prefix}.games_featuring", minimum=1
        ),
        game_share=_number(raw.get("game_share"), f"{prefix}.game_share"),
        win_rate=_number(raw.get("win_rate"), f"{prefix}.win_rate"),
        wilson_95=(
            _number(wilson[0], f"{prefix}.wilson_95[0]"),
            _number(wilson[1], f"{prefix}.wilson_95[1]"),
        ),
        classification_method=method,
        signature_groups=tuple(groups),
    )


def load_ladder_deck_mix(path: str | Path) -> LadderDeckMix:
    """Load and fully validate a v1 ladder distribution artifact."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LadderDeckMixError(f"cannot load ladder mix {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LadderDeckMixError("ladder mix root must be an object")
    if payload.get("schema") != SCHEMA:
        raise LadderDeckMixError(
            f"unsupported ladder mix schema: {payload.get('schema')!r}"
        )
    declared_digest = payload.get(_DIGEST_FIELD)
    actual_digest = canonical_payload_digest(payload)
    if declared_digest != actual_digest:
        raise LadderDeckMixError(
            f"artifact digest mismatch: declared={declared_digest!r} "
            f"actual={actual_digest}"
        )
    mix_id = payload.get("mix_id")
    if not isinstance(mix_id, str) or not mix_id:
        raise LadderDeckMixError("mix_id must be a non-empty string")
    source = payload.get("source")
    coverage = payload.get("coverage")
    policy = payload.get("weight_policy")
    decks_raw = payload.get("decks")
    excluded = payload.get("excluded")
    if not isinstance(source, dict) or not isinstance(coverage, dict):
        raise LadderDeckMixError("source and coverage must be objects")
    if not isinstance(policy, dict):
        raise LadderDeckMixError("weight_policy must be an object")
    if not isinstance(decks_raw, list) or not decks_raw:
        raise LadderDeckMixError("decks must be a non-empty list")
    if not isinstance(excluded, list):
        raise LadderDeckMixError("excluded must be a list")
    required_source = (
        "dataset_slug",
        "dataset_url",
        "window_start_utc",
        "window_end_utc_exclusive",
    )
    for key in required_source:
        if not isinstance(source.get(key), str) or not source[key]:
            raise LadderDeckMixError(f"source.{key} is required")

    decks: list[LadderDeck] = []
    for index, raw in enumerate(decks_raw):
        if not isinstance(raw, dict):
            raise LadderDeckMixError(f"decks[{index}] must be an object")
        decks.append(_parse_deck(raw, index))
    ids = [deck.deck_id for deck in decks]
    if len(ids) != len(set(ids)):
        raise LadderDeckMixError("deck IDs must be unique")
    ranks = [deck.source_rank for deck in decks]
    if len(ranks) != len(set(ranks)) or ranks != sorted(ranks):
        raise LadderDeckMixError("source ranks must be unique and ascending")
    counts = [deck.observed_count for deck in decks]
    if counts != sorted(counts, reverse=True):
        raise LadderDeckMixError("decks must be ranked by observed count")

    total_seats = _integer(
        coverage.get("total_seat_appearances"),
        "coverage.total_seat_appearances",
        minimum=1,
    )
    recognized = _integer(
        coverage.get("recognized_seat_appearances"),
        "coverage.recognized_seat_appearances",
        minimum=1,
    )
    excluded_count = _integer(
        coverage.get("excluded_seat_appearances"),
        "coverage.excluded_seat_appearances",
        minimum=0,
    )
    if sum(counts) != recognized or recognized + excluded_count != total_seats:
        raise LadderDeckMixError("coverage counts do not match deck counts")

    empirical_mass = _number(policy.get("empirical_mass"), "empirical_mass")
    uniform_mass = _number(policy.get("uniform_coverage_mass"), "uniform_coverage_mass")
    if not math.isclose(
        empirical_mass + uniform_mass,
        1.0,
        rel_tol=0.0,
        abs_tol=_WEIGHT_TOLERANCE,
    ):
        raise LadderDeckMixError("training policy masses must sum to 1")
    for deck in decks:
        expected_observed = deck.observed_count / total_seats
        expected_known = deck.observed_count / recognized
        expected_train = empirical_mass * expected_known + uniform_mass / len(decks)
        if not math.isclose(
            deck.observed_weight,
            expected_observed,
            rel_tol=0.0,
            abs_tol=_WEIGHT_TOLERANCE,
        ):
            raise LadderDeckMixError(
                f"observed weight mismatch for {deck.deck_id!r}"
            )
        if not math.isclose(
            deck.known_conditional_weight,
            expected_known,
            rel_tol=0.0,
            abs_tol=_WEIGHT_TOLERANCE,
        ):
            raise LadderDeckMixError(
                f"known conditional weight mismatch for {deck.deck_id!r}"
            )
        if not math.isclose(
            deck.train_weight,
            expected_train,
            rel_tol=0.0,
            abs_tol=_WEIGHT_TOLERANCE,
        ):
            raise LadderDeckMixError(f"training weight mismatch for {deck.deck_id!r}")
        if not (0.0 <= deck.win_rate <= 1.0):
            raise LadderDeckMixError(f"invalid win rate for {deck.deck_id!r}")
        if not (0.0 <= deck.wilson_95[0] <= deck.wilson_95[1] <= 1.0):
            raise LadderDeckMixError(f"invalid Wilson interval for {deck.deck_id!r}")

    if not math.isclose(
        sum(deck.train_weight for deck in decks),
        1.0,
        rel_tol=0.0,
        abs_tol=_WEIGHT_TOLERANCE,
    ):
        raise LadderDeckMixError("training weights must sum to 1")
    if not math.isclose(
        sum(deck.known_conditional_weight for deck in decks),
        1.0,
        rel_tol=0.0,
        abs_tol=_WEIGHT_TOLERANCE,
    ):
        raise LadderDeckMixError("known observed weights must sum to 1")

    return LadderDeckMix(
        schema=SCHEMA,
        mix_id=mix_id,
        artifact_sha256=actual_digest,
        source=source,
        coverage=coverage,
        weight_policy=policy,
        decks=tuple(decks),
        excluded=tuple(excluded),
    )


def load_ladder_deck_representatives(
    path: str | Path,
) -> LadderDeckRepresentatives:
    """Load and validate the exact modal representative artifact."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LadderDeckMixError(
            f"cannot load ladder representatives {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise LadderDeckMixError("representatives root must be an object")
    if payload.get("schema") != REPRESENTATIVE_SCHEMA:
        raise LadderDeckMixError(
            f"unsupported representative schema: {payload.get('schema')!r}"
        )
    declared_digest = payload.get(_DIGEST_FIELD)
    actual_digest = canonical_payload_digest(payload)
    if declared_digest != actual_digest:
        raise LadderDeckMixError(
            "representative artifact digest mismatch: "
            f"declared={declared_digest!r} actual={actual_digest}"
        )
    for key in ("source_mix_sha256", "source_dataset", "selection"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise LadderDeckMixError(f"representatives.{key} is required")
    raw_decks = payload.get("decks")
    if not isinstance(raw_decks, dict) or not raw_decks:
        raise LadderDeckMixError("representatives.decks must be a non-empty object")
    decks: dict[str, Mapping[str, Any]] = {}
    for deck_id, raw in raw_decks.items():
        if not isinstance(deck_id, str) or not deck_id or not isinstance(raw, dict):
            raise LadderDeckMixError("invalid representative deck entry")
        cards = raw.get("card_ids")
        if not isinstance(cards, list) or len(cards) != 60:
            raise LadderDeckMixError(
                f"representative {deck_id!r} must contain exactly 60 card IDs"
            )
        for index, card in enumerate(cards):
            _integer(card, f"representatives.{deck_id}.card_ids[{index}]", minimum=1)
        modal = _integer(
            raw.get("modal_seat_count"),
            f"representatives.{deck_id}.modal_seat_count",
            minimum=1,
        )
        labeled = _integer(
            raw.get("labeled_seat_count"),
            f"representatives.{deck_id}.labeled_seat_count",
            minimum=1,
        )
        distinct = _integer(
            raw.get("distinct_lists"),
            f"representatives.{deck_id}.distinct_lists",
            minimum=1,
        )
        if modal > labeled:
            raise LadderDeckMixError(
                f"representative {deck_id!r} modal count exceeds labeled count"
            )
        decks[deck_id] = {
            "card_ids": tuple(int(card) for card in cards),
            "modal_seat_count": modal,
            "labeled_seat_count": labeled,
            "distinct_lists": distinct,
        }
    return LadderDeckRepresentatives(
        schema=REPRESENTATIVE_SCHEMA,
        artifact_sha256=actual_digest,
        source_mix_sha256=str(payload["source_mix_sha256"]),
        source_dataset=str(payload["source_dataset"]),
        selection=str(payload["selection"]),
        decks=decks,
    )
