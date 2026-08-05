"""Fail-closed observed-list specialist family contracts.

The family manifest is training metadata.  It does not create router identities,
change the singular package deck, or make evaluation rows training eligible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from .archetypes import classify_deck


SCHEMA = "poke_bot.specialist_archetype_families/v1"
SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
SPLITS = frozenset({"train", "dev", "locked"})


class ArchetypeFamilyError(ValueError):
    """The observed-list family contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class FamilyDeckEntry:
    deck_id: str


@dataclass(frozen=True)
class FamilyDeckMix:
    """Ladder-mix-compatible scheduler backed by one family manifest."""

    manifest: Mapping[str, Any]
    mix_id: str
    artifact_sha256: str
    decks: tuple[FamilyDeckEntry, ...]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "FamilyDeckMix":
        validated = validate_manifest(manifest, require_activation_ready=True)
        ids = tuple(sorted(family_probabilities(validated)))
        return cls(
            manifest=validated,
            mix_id=f"{SPECIALIST_ID}-observed-family-v1",
            artifact_sha256=str(validated["artifact_sha256"]),
            decks=tuple(FamilyDeckEntry(deck_id=value) for value in ids),
        )

    def weights(self, basis: str = "train") -> dict[str, float]:
        if basis != "train":
            raise ArchetypeFamilyError("family v1 exposes only train weights")
        return family_probabilities(self.manifest)

    def quotas(self, total: int, basis: str = "train") -> dict[str, int]:
        return hamilton_quotas(int(total), self.weights(basis))

    def schedule_ids(
        self,
        total: int,
        *,
        seed: int,
        iteration: int = 0,
        stream: str = "self_play_our",
        basis: str = "train",
    ) -> tuple[str, ...]:
        quotas = self.quotas(total, basis)
        tokens = [
            (variant_id, ordinal)
            for variant_id in sorted(quotas)
            for ordinal in range(quotas[variant_id])
        ]
        prefix = (
            f"{self.artifact_sha256}|{int(seed)}|{int(iteration)}|"
            f"{stream}|{basis}|"
        )
        tokens.sort(
            key=lambda token: hashlib.sha256(
                f"{prefix}{token[0]}|{token[1]}".encode("utf-8")
            ).digest()
        )
        return tuple(variant_id for variant_id, _ordinal in tokens)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def ordered_digest(cards: Sequence[int]) -> str:
    return digest_json([int(card) for card in cards])


def canonical_counts(cards: Sequence[int]) -> tuple[tuple[int, int], ...]:
    counts = Counter(int(card) for card in cards)
    return tuple(sorted(counts.items()))


def multiset_digest(cards: Sequence[int]) -> str:
    return digest_json([[card, count] for card, count in canonical_counts(cards)])


def swap_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Minimum single-card replacements between equal-size multisets."""
    if len(left) != len(right):
        raise ArchetypeFamilyError("swap distance requires equal-size lists")
    a, b = Counter(left), Counter(right)
    overlap = sum((a & b).values())
    return len(left) - overlap


def cluster_variants(
    variants: Sequence[Mapping[str, Any]], *, maximum_swap_distance: int = 2
) -> dict[str, str]:
    """Connected-component clustering under the locked <=2-swap relation."""
    parent = list(range(len(variants)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, row in enumerate(variants):
        for j in range(i):
            if swap_distance(row["card_ids"], variants[j]["card_ids"]) <= int(
                maximum_swap_distance
            ):
                union(i, j)
    groups: dict[int, list[str]] = defaultdict(list)
    for index, row in enumerate(variants):
        groups[find(index)].append(str(row["multiset_digest"]))
    cluster_id = {
        root: "cluster-" + hashlib.sha256(
            "\n".join(sorted(digests)).encode("ascii")
        ).hexdigest()[:16]
        for root, digests in groups.items()
    }
    return {
        str(row["variant_id"]): cluster_id[find(index)]
        for index, row in enumerate(variants)
    }


def split_clusters(
    cluster_ids: Iterable[str], *, package_cluster_id: str, seed: str
) -> dict[str, str]:
    """Deterministic cluster-stable 60/20/20 assignment.

    The package cluster is forced to train. Remaining clusters are ordered by
    a checksum-keyed hash and assigned using Hamilton integer quotas.
    """
    clusters = sorted(set(str(value) for value in cluster_ids))
    if package_cluster_id not in clusters:
        raise ArchetypeFamilyError("package cluster is absent")
    other = [value for value in clusters if value != package_cluster_id]
    quotas = hamilton_quotas(len(other), {"train": 0.60, "dev": 0.20, "locked": 0.20})
    ranked = sorted(
        other,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )
    result = {package_cluster_id: "train"}
    offset = 0
    for split in ("train", "dev", "locked"):
        for cluster in ranked[offset : offset + quotas[split]]:
            result[cluster] = split
        offset += quotas[split]
    return result


def hamilton_quotas(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    total = int(total)
    if total < 0 or not weights:
        raise ArchetypeFamilyError("invalid Hamilton allocation")
    normalized = {str(k): float(v) for k, v in weights.items()}
    if any(not math.isfinite(v) or v < 0 for v in normalized.values()):
        raise ArchetypeFamilyError("Hamilton weights must be finite and nonnegative")
    mass = sum(normalized.values())
    if mass <= 0:
        raise ArchetypeFamilyError("Hamilton weights have zero mass")
    exact = {key: total * value / mass for key, value in normalized.items()}
    result = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def family_probabilities(manifest: Mapping[str, Any]) -> dict[str, float]:
    """Exact 20% package / equal-cluster / equal-list training mass."""
    rows = [row for row in manifest["variants"] if row["split"] == "train"]
    package = [row for row in rows if row["package"]]
    if len(package) != 1:
        raise ArchetypeFamilyError("exactly one train package variant is required")
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["package"]:
            clusters[str(row["cluster_id"])].append(row)
    if not clusters:
        raise ArchetypeFamilyError("no non-package train clusters")
    result = {str(package[0]["variant_id"]): 0.20}
    cluster_mass = 0.80 / len(clusters)
    for cluster in sorted(clusters):
        per_list = cluster_mass / len(clusters[cluster])
        for row in clusters[cluster]:
            result[str(row["variant_id"])] = per_list
    return result


def schedule_variants(
    manifest: Mapping[str, Any], *, games: int, checksum_seed: str
) -> list[dict[str, Any]]:
    """Hamilton quotas, seeded permutation, seat balance and mirror derangement."""
    probabilities = family_probabilities(manifest)
    quotas = hamilton_quotas(int(games), probabilities)
    lookup = {str(row["variant_id"]): row for row in manifest["variants"]}
    scheduled: list[dict[str, Any]] = []
    for variant_id in sorted(quotas):
        row = lookup[variant_id]
        for ordinal in range(quotas[variant_id]):
            scheduled.append(
                {
                    "family_id": manifest["family_id"],
                    "variant_id": variant_id,
                    "manifest_digest": manifest["artifact_sha256"],
                    "ordered_digest": row["ordered_digest"],
                    "multiset_digest": row["multiset_digest"],
                    "seat": ordinal % 2,
                }
            )
    rng = random.Random(int(hashlib.sha256(checksum_seed.encode()).hexdigest(), 16))
    rng.shuffle(scheduled)
    # Choose the cyclic shift with the fewest mirrors.  Under the locked 20%
    # package cap and equal-cluster allocation a zero-mirror shift normally
    # exists; the fallback is still the deterministic minimum.
    ids = [row["variant_id"] for row in scheduled]
    positions = sorted(range(len(ids)), key=lambda index: (ids[index], index))
    ordered = [ids[index] for index in positions]
    max_count = max(Counter(ordered).values(), default=0)
    best = ordered[max_count:] + ordered[:max_count]
    opponent_by_position = {position: best[index] for index, position in enumerate(positions)}
    for index, row in enumerate(scheduled):
        row["opponent_variant_id"] = opponent_by_position[index]
    return scheduled


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    legality: Callable[[Sequence[int]], bool] | None = None,
    require_activation_ready: bool = False,
) -> dict[str, Any]:
    """Validate and return a defensive JSON-compatible copy."""
    data = json.loads(json.dumps(payload))
    if data.get("schema") != SCHEMA or data.get("family_id") != SPECIALIST_ID:
        raise ArchetypeFamilyError("wrong family schema or identity")
    rows = data.get("variants")
    if not isinstance(rows, list) or not rows:
        raise ArchetypeFamilyError("family contains no observed variants")
    seen_variant: set[str] = set()
    seen_multiset: set[str] = set()
    seen_provenance: set[str] = set()
    packages = 0
    measurements = 0
    for row in rows:
        if row.get("family_id") != SPECIALIST_ID:
            raise ArchetypeFamilyError("variant family identity mismatch")
        cards = row.get("card_ids")
        if not isinstance(cards, list) or len(cards) != 60:
            raise ArchetypeFamilyError("every variant must contain exactly 60 cards")
        if any(isinstance(card, bool) or not isinstance(card, int) or card <= 0 for card in cards):
            raise ArchetypeFamilyError("card IDs must be positive integers")
        counts = row.get("card_counts")
        expected_counts = [[card, count] for card, count in canonical_counts(cards)]
        if counts != expected_counts or sum(int(item[1]) for item in counts) != 60:
            raise ArchetypeFamilyError("card counts do not match the exact list")
        if row.get("ordered_digest") != ordered_digest(cards):
            raise ArchetypeFamilyError("ordered-list digest mismatch")
        if row.get("multiset_digest") != multiset_digest(cards):
            raise ArchetypeFamilyError("multiset digest mismatch")
        variant_id = str(row.get("variant_id", ""))
        provenance_digest = digest_json(row.get("provenance"))
        if not variant_id or variant_id in seen_variant:
            raise ArchetypeFamilyError("duplicate or missing variant ID")
        if row["multiset_digest"] in seen_multiset:
            raise ArchetypeFamilyError("duplicate canonical card multiset")
        if provenance_digest in seen_provenance:
            raise ArchetypeFamilyError("duplicate canonical provenance")
        seen_variant.add(variant_id)
        seen_multiset.add(row["multiset_digest"])
        seen_provenance.add(provenance_digest)
        legal = bool(legality(cards) if legality is not None else row.get("legality", {}).get("legal"))
        if not legal:
            raise ArchetypeFamilyError("variant failed existing legality rules")
        if classify_deck(cards) != SPECIALIST_ID or row.get("classification", {}).get("archetype_id") != SPECIALIST_ID:
            raise ArchetypeFamilyError("variant is outside the Marnie archetype")
        if row.get("split") not in SPLITS or not row.get("cluster_id"):
            raise ArchetypeFamilyError("invalid split or missing cluster")
        if not isinstance(row.get("capability_mask"), dict):
            raise ArchetypeFamilyError("capability mask is required")
        if set(row["capability_mask"]) != {
            "core_setup_continuity",
            "resource_attack_readiness",
            "long_horizon_prize_pressure",
        } or any(not isinstance(value, bool) for value in row["capability_mask"].values()):
            raise ArchetypeFamilyError("capability mask is incomplete or untyped")
        weight = row.get("training_weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise ArchetypeFamilyError("training weight must be finite and nonnegative")
        packages += bool(row.get("package"))
        measurements += bool(row.get("measurement"))
    if packages != 1 or measurements != 1:
        raise ArchetypeFamilyError("exactly one package and measurement variant required")
    package = next(row for row in rows if row["package"])
    if not package["measurement"] or package["split"] != "train":
        raise ArchetypeFamilyError("package variant must be the train measurement variant")
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cluster_splits[row["cluster_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in cluster_splits.values()):
        raise ArchetypeFamilyError("similarity cluster leaked across splits")
    recomputed = cluster_variants(rows)
    declared_partition = {
        frozenset(str(row["variant_id"]) for row in rows if row["cluster_id"] == cluster)
        for cluster in cluster_splits
    }
    recomputed_partition = {
        frozenset(variant for variant, value in recomputed.items() if value == cluster)
        for cluster in set(recomputed.values())
    }
    if declared_partition != recomputed_partition:
        raise ArchetypeFamilyError("declared similarity clusters do not match swap-distance clustering")
    non_package_clusters = set(cluster_splits) - {package["cluster_id"]}
    if require_activation_ready and len(non_package_clusters) < 12:
        raise ArchetypeFamilyError("activation requires at least 12 non-package clusters")
    declared = data.get("artifact_sha256")
    unsigned = dict(data)
    unsigned.pop("artifact_sha256", None)
    if declared != digest_json(unsigned):
        raise ArchetypeFamilyError("manifest artifact digest mismatch")
    return data


def legacy_row_variant(
    manifest: Mapping[str, Any], *, exact_deck_multiset_digest: str
) -> str | None:
    package = next(row for row in manifest["variants"] if row["package"])
    return (
        str(package["variant_id"])
        if exact_deck_multiset_digest == package["multiset_digest"]
        else None
    )


def validate_training_row(
    manifest: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate replay/rehearsal provenance and reject evaluation splits.

    Legacy rows are accepted only when their exact deck multiset matches the
    singular package variant.  The returned provenance is suitable for
    recording on a normalized replay row.
    """
    variants = {str(item["variant_id"]): item for item in manifest["variants"]}
    variant_id = str(row.get("variant_id") or "")
    if not variant_id:
        variant_id = legacy_row_variant(
            manifest,
            exact_deck_multiset_digest=str(row.get("multiset_digest") or ""),
        ) or ""
    variant = variants.get(variant_id)
    if variant is None:
        raise ArchetypeFamilyError("row has no exact admitted family variant")
    if variant["split"] != "train":
        raise ArchetypeFamilyError("development/locked variant entered training")
    required = {
        "family_id": manifest["family_id"],
        "variant_id": variant_id,
        "manifest_digest": manifest["artifact_sha256"],
        "ordered_digest": variant["ordered_digest"],
        "multiset_digest": variant["multiset_digest"],
    }
    for key, expected in required.items():
        observed = row.get(key)
        if observed not in (None, "", expected):
            raise ArchetypeFamilyError(f"row provenance mismatch: {key}")
    return required


def macro_replay_rows(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    total: int,
    checksum_seed: str,
) -> list[dict[str, Any]]:
    """Deterministically resample replay/rehearsal to family macro weights."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        provenance = validate_training_row(manifest, source)
        normalized = dict(source)
        normalized.update(provenance)
        buckets[provenance["variant_id"]].append(normalized)
    quotas = hamilton_quotas(int(total), family_probabilities(manifest))
    selected: list[dict[str, Any]] = []
    for variant_id, quota in sorted(quotas.items()):
        bucket = buckets.get(variant_id) or []
        if quota and not bucket:
            raise ArchetypeFamilyError(f"missing replay rows for train variant: {variant_id}")
        ranked = sorted(bucket, key=digest_json)
        rng = random.Random(int(hashlib.sha256(f"{checksum_seed}:{variant_id}".encode()).hexdigest(), 16))
        rng.shuffle(ranked)
        for index in range(quota):
            selected.append(dict(ranked[index % len(ranked)]))
    rng = random.Random(int(hashlib.sha256(f"{checksum_seed}:final".encode()).hexdigest(), 16))
    rng.shuffle(selected)
    return selected


def singular_package_variant(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only checksum-pinned package/measurement deck."""
    rows = [row for row in manifest["variants"] if row["package"] or row["measurement"]]
    if len(rows) != 1 or not rows[0]["package"] or not rows[0]["measurement"]:
        raise ArchetypeFamilyError("package materialization is not singular")
    return dict(rows[0])


__all__ = [
    "ArchetypeFamilyError",
    "FamilyDeckEntry",
    "FamilyDeckMix",
    "SCHEMA",
    "SPECIALIST_ID",
    "canonical_counts",
    "cluster_variants",
    "digest_json",
    "family_probabilities",
    "hamilton_quotas",
    "legacy_row_variant",
    "multiset_digest",
    "macro_replay_rows",
    "ordered_digest",
    "schedule_variants",
    "singular_package_variant",
    "split_clusters",
    "swap_distance",
    "validate_manifest",
    "validate_training_row",
]
