from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DeckVariant:
    slug: str
    variant: str
    cards: tuple[int, ...]
    path: Path


@dataclass
class ArchetypeRegistry:
    priors: dict[str, float]
    variants: list[DeckVariant] = field(default_factory=list)
    _by_slug: dict[str, list[DeckVariant]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        grouped: dict[str, list[DeckVariant]] = {}
        for variant in self.variants:
            grouped.setdefault(variant.slug, []).append(variant)
        self._by_slug = grouped

    def slugs(self) -> list[str]:
        return sorted(self.priors)

    def signature_cards(self, slug: str) -> Counter[int]:
        variants = self._by_slug.get(slug) or []
        if not variants:
            return Counter()
        counts: Counter[int] = Counter()
        for variant in variants:
            counts.update(variant.cards)
        return counts

    def deck_similarity(self, cards: list[int] | tuple[int, ...], slug: str) -> float:
        left = Counter(cards)
        right = self.signature_cards(slug)
        if not left or not right:
            return 0.0
        intersection = sum(min(left[card], right[card]) for card in set(left) | set(right))
        union = sum(max(left[card], right[card]) for card in set(left) | set(right))
        return intersection / union if union else 0.0

    def classify_deck(self, cards: list[int] | tuple[int, ...]) -> tuple[str, float]:
        if not cards:
            return ("unknown", 0.0)
        left = Counter(cards)
        best_slug = "unknown"
        best_score = 0.0
        for variant in self.variants:
            right = Counter(variant.cards)
            if not right:
                continue
            intersection = sum(min(left[card], right[card]) for card in set(left) | set(right))
            union = sum(max(left[card], right[card]) for card in set(left) | set(right))
            score = intersection / union if union else 0.0
            if score > best_score:
                best_slug = variant.slug
                best_score = score
        return best_slug, best_score

    def classify_with_prior(
        self,
        cards: list[int] | tuple[int, ...],
        *,
        visible_cards: Counter[int] | None = None,
    ) -> tuple[str, float]:
        slug, deck_score = self.classify_deck(cards)
        if visible_cards:
            visible_slug, visible_score = self._classify_visible(visible_cards)
            if visible_score > deck_score:
                slug, deck_score = visible_slug, visible_score

        prior = self.priors.get(slug, 0.0) / 100.0 if slug != "unknown" else 0.0
        confidence = min(1.0, deck_score * 0.85 + prior * 0.15)
        return slug, confidence

    def _classify_visible(self, visible: Counter[int]) -> tuple[str, float]:
        best_slug = "unknown"
        best_score = 0.0
        for slug in self.slugs():
            signature = self.signature_cards(slug)
            if not signature:
                continue
            overlap = sum(min(visible[card], signature[card]) for card in visible)
            score = overlap / max(1, sum(visible.values()))
            if score > best_score:
                best_slug = slug
                best_score = score
        return best_slug, best_score


def read_deck_csv(path: Path) -> list[int]:
    values: list[int] = []
    for raw_token in path.read_text(encoding="utf-8").replace(",", "\n").splitlines():
        token = raw_token.strip()
        if token:
            values.append(int(token))
    if len(values) != 60:
        raise ValueError(f"{path} must contain exactly 60 card IDs, got {len(values)}")
    return values


def parse_archetype_filename(stem: str) -> tuple[str, str]:
    match = re.match(r"^(?P<slug>.+?)(?P<variant>\d+)$", stem)
    if not match:
        return stem, "1"
    return match.group("slug"), match.group("variant")


def load_archetype_shares(path: Path) -> dict[str, float]:
    priors: dict[str, float] = {}
    if not path.is_file():
        return priors
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        priors[parts[0]] = float(parts[1])
    return priors


_REGISTRY_CACHE: dict[tuple[str, str], ArchetypeRegistry] = {}


def load_archetype_registry(
    root: Path,
    *,
    samples_dir: Path | None = None,
    shares_path: Path | None = None,
) -> ArchetypeRegistry:
    """Build the (read-only) archetype registry, memoized per process.

    The registry globs and parses every deck CSV, so callers that build it in a
    hot loop (e.g. once per generated episode) would otherwise cause heavy,
    contended disk I/O. The result is immutable and safe to share, so we cache
    it keyed by the resolved sample/shares paths.
    """
    samples_dir = samples_dir or (root / "decks/archetype-samples")
    shares_path = shares_path or (root / "decks/archetype-shares.txt")
    cache_key = (str(samples_dir.resolve()), str(shares_path.resolve()))
    cached = _REGISTRY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    priors = load_archetype_shares(shares_path)
    variants: list[DeckVariant] = []
    if samples_dir.is_dir():
        for path in sorted(samples_dir.glob("*.csv")):
            slug, variant = parse_archetype_filename(path.stem)
            variants.append(
                DeckVariant(
                    slug=slug,
                    variant=variant,
                    cards=tuple(read_deck_csv(path)),
                    path=path,
                )
            )
            priors.setdefault(slug, 0.0)
    registry = ArchetypeRegistry(priors=priors, variants=variants)
    _REGISTRY_CACHE[cache_key] = registry
    return registry


def slug_from_deck_name(name: str, registry: ArchetypeRegistry) -> str:
    stem = Path(name).stem if name else "unknown"
    if stem in registry.priors:
        return stem
    slug, _ = parse_archetype_filename(stem)
    if slug in registry.priors:
        return slug
    return stem


def weighted_deck_pool(
    root: Path,
    *,
    samples_dir: Path | None = None,
    shares_path: Path | None = None,
) -> list[tuple[str, list[int], float]]:
    """Return (deck_name, cards, weight) tuples for share-weighted matchups."""
    registry = load_archetype_registry(root, samples_dir=samples_dir, shares_path=shares_path)
    pool: list[tuple[str, list[int], float]] = []
    for variant in registry.variants:
        weight = float(registry.priors.get(variant.slug, 1.0))
        pool.append((variant.path.stem, list(variant.cards), weight))
    return pool
