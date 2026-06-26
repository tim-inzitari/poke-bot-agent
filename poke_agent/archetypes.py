from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from poke_agent.deck_pool import competitive_deck_archetype_slug, deck_matches_archetype_patterns

DEFAULT_COMPETITIVE_DIRS = (
    "decks/competitive/high_performing",
    "decks/competitive/the_rest",
)

# Slugs from competitive deck filenames that map to our heuristic archetype identities.
HEURISTIC_ARCHETYPE_SLUG_PATTERNS: dict[str, list[str]] = {
    "dragapult-ex": ["dragapult"],
    "mega-lucario-ex": ["mega-lucario", "lucario-hariyama"],
}


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
    _global_deck_presence: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _slug_deck_presence: dict[str, dict[int, float]] = field(default_factory=dict, init=False, repr=False)
    _distinctive_weights: dict[str, dict[int, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        grouped: dict[str, list[DeckVariant]] = {}
        for variant in self.variants:
            grouped.setdefault(variant.slug, []).append(variant)
        self._by_slug = grouped
        self._build_distinctive_signatures()

    def slugs(self) -> list[str]:
        return sorted(self._by_slug)

    def slug_deck_count(self, slug: str) -> int:
        return len(self._by_slug.get(slug) or [])

    def _build_distinctive_signatures(self) -> None:
        """Mine per-archetype signature weights from decklist corpus.

        Uses deck-presence rates (not raw 4-of counts) and downweights cards that
        appear across many archetypes so staples / overlapping tech do not dominate.
        """
        import math

        total_decks = len(self.variants)
        if total_decks == 0:
            return

        global_card_decks: Counter[int] = Counter()
        slug_card_decks: dict[str, Counter[int]] = {}

        for variant in self.variants:
            slug_card_decks.setdefault(variant.slug, Counter())
            for card in set(variant.cards):
                slug_card_decks[variant.slug][card] += 1
                global_card_decks[card] += 1

        self._global_deck_presence = {
            card: count / total_decks for card, count in global_card_decks.items()
        }

        for slug, card_decks in slug_card_decks.items():
            deck_count = self.slug_deck_count(slug)
            if deck_count == 0:
                continue
            presence = {card: hits / deck_count for card, hits in card_decks.items()}
            self._slug_deck_presence[slug] = presence

            weights: dict[int, float] = {}
            for card, slug_rate in presence.items():
                if slug_rate < 0.2:
                    continue
                global_rate = self._global_deck_presence.get(card, 0.0)
                distinctiveness = slug_rate / (global_rate + 1e-9)
                if distinctiveness < 1.1:
                    continue
                weights[card] = slug_rate * math.log1p(distinctiveness)
            self._distinctive_weights[slug] = weights

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

    def distinctive_card_weights(self, slug: str) -> dict[int, float]:
        return dict(self._distinctive_weights.get(slug) or {})

    def distinctive_visible_signatures(
        self,
        *,
        min_weight: float = 0.01,
        top_k: int = 40,
    ) -> dict[str, dict[int, float]]:
        """Export per-slug distinctive weights for inference bundles."""
        exported: dict[str, dict[int, float]] = {}
        for slug in self.slugs():
            ranked = sorted(
                self.distinctive_card_weights(slug).items(),
                key=lambda item: item[1],
                reverse=True,
            )
            kept = [(card, weight) for card, weight in ranked if weight >= min_weight][:top_k]
            if kept:
                exported[slug] = dict(kept)
        return exported

    def _weighted_visible_score(self, visible: Counter[int], weights: dict[int, float]) -> float:
        if not visible or not weights:
            return 0.0
        matched = [(card, weights.get(card, 0.0)) for card in visible if weights.get(card, 0.0) > 0.0]
        if not matched:
            return 0.0
        hit = sum(visible[card] * weight for card, weight in matched)
        match_count = len(matched)
        coverage = match_count / len(visible)
        strength = hit / match_count
        return coverage * strength

    def _classify_visible(self, visible: Counter[int]) -> tuple[str, float]:
        best_slug = "unknown"
        best_score = 0.0
        for slug in self.slugs():
            score = self._weighted_visible_score(visible, self.distinctive_card_weights(slug))
            if score > best_score:
                best_slug = slug
                best_score = score
        return best_slug, best_score

    def classify_visible_archetype(
        self,
        visible: Counter[int],
        *,
        min_score: float = 0.15,
    ) -> tuple[str, float]:
        """Predict the opponent's competitive archetype slug from visible board cards."""
        if not visible:
            return ("unknown", 0.0)
        slug, score = self._classify_visible(visible)
        if slug == "unknown" or score < min_score:
            return ("unknown", score)
        return slug, score

    def classify_visible_heuristic_archetype(
        self,
        visible: Counter[int],
        patterns_by_archetype: dict[str, list[str]] | None = None,
        *,
        min_score: float = 0.15,
    ) -> tuple[str, float]:
        """Map visible cards to one of our trained heuristic archetype families."""
        slug, score = self.classify_visible_archetype(visible, min_score=min_score)
        if slug == "unknown":
            return ("unknown", score)
        patterns_by_archetype = patterns_by_archetype or HEURISTIC_ARCHETYPE_SLUG_PATTERNS
        for archetype, patterns in patterns_by_archetype.items():
            if deck_matches_archetype_patterns(slug, patterns):
                return archetype, score
        return ("unknown", score)

    def heuristic_visible_signatures(
        self,
        patterns_by_archetype: dict[str, list[str]] | None = None,
    ) -> dict[str, Counter[int]]:
        """Deprecated aggregate; prefer :meth:`distinctive_visible_signatures`."""
        patterns_by_archetype = patterns_by_archetype or HEURISTIC_ARCHETYPE_SLUG_PATTERNS
        grouped: dict[str, Counter[int]] = {archetype: Counter() for archetype in patterns_by_archetype}
        for slug in self.slugs():
            for archetype, patterns in patterns_by_archetype.items():
                if deck_matches_archetype_patterns(slug, patterns):
                    grouped[archetype].update(
                        Counter({card: int(weight * 1000) for card, weight in self.distinctive_card_weights(slug).items()})
                    )
        return grouped


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


_REGISTRY_CACHE: dict[tuple[str, ...], ArchetypeRegistry] = {}


def _resolve_competitive_dirs(root: Path, competitive_dirs: tuple[str, ...] | list[str] | None) -> tuple[Path, ...]:
    dirs = competitive_dirs if competitive_dirs is not None else DEFAULT_COMPETITIVE_DIRS
    resolved: list[Path] = []
    for raw in dirs:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            resolved.append(path)
    return tuple(resolved)


def load_competitive_variants(
    root: Path,
    competitive_dirs: tuple[str, ...] | list[str] | None = None,
) -> list[DeckVariant]:
    """Load tournament decklists grouped by archetype slug from filename."""
    variants: list[DeckVariant] = []
    seen_paths: set[Path] = set()
    for deck_dir in _resolve_competitive_dirs(root, competitive_dirs):
        for path in sorted(deck_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".txt", ".deck"}:
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            slug = competitive_deck_archetype_slug(path.stem)
            try:
                cards = tuple(read_deck_csv(path))
            except ValueError as exc:
                print(f"skipping invalid competitive deck {path.name}: {exc}")
                continue
            variants.append(
                DeckVariant(
                    slug=slug,
                    variant=path.stem,
                    cards=cards,
                    path=path,
                )
            )
    return variants


def load_archetype_registry(
    root: Path,
    *,
    samples_dir: Path | None = None,
    shares_path: Path | None = None,
    competitive_dirs: tuple[str, ...] | list[str] | None = DEFAULT_COMPETITIVE_DIRS,
    include_competitive: bool = True,
) -> ArchetypeRegistry:
    """Build the (read-only) archetype registry, memoized per process.

    The registry globs and parses every deck CSV, so callers that build it in a
    hot loop (e.g. once per generated episode) would otherwise cause heavy,
    contended disk I/O. The result is immutable and safe to share, so we cache
    it keyed by the resolved sample/shares/competitive paths.
    """
    samples_dir = samples_dir or (root / "decks/archetype-samples")
    shares_path = shares_path or (root / "decks/archetype-shares.txt")
    resolved_competitive = _resolve_competitive_dirs(root, competitive_dirs) if include_competitive else ()
    cache_key = (
        str(samples_dir.resolve()),
        str(shares_path.resolve()),
        str(include_competitive),
        *(str(path) for path in resolved_competitive),
    )
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
    if include_competitive:
        for variant in load_competitive_variants(root, competitive_dirs=competitive_dirs):
            variants.append(variant)
            priors.setdefault(variant.slug, 0.0)
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
