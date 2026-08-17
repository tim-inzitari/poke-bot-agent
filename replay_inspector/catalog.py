"""Read-only catalog of the Elmo Kaggle submission replay archive.

The catalog intentionally does *not* infer model identity from a replay file.
It joins numeric archive directories to :mod:`replay_inspector.provenance`
only when there is one checksum-verified mapping.  Missing, malformed, and
ambiguous records remain visible with explicit availability reasons, which
lets a localhost UI render useful diagnostics without ever choosing a model by
guesswork.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from .provenance import (
    PathContainmentError,
    ProvenanceManifest,
    SubmissionLabel,
    SubmissionProvenance,
    resolve_contained_path,
    sha256_file,
)

PathLike: TypeAlias = str | os.PathLike[str]
_SUBMISSION_DIR_RE = re.compile(r"^[0-9]+$")
_REPLAY_FILE_RE = re.compile(r"^episode-([0-9]+)-replay\.json$")


def _unique_reasons(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in result:
            result.append(clean)
    return tuple(result)


def _positive_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _nonnegative_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _explicit_rank(value: Any) -> int | None:
    """Accept a directly supplied non-negative rank, never a score/reward."""

    numeric = _nonnegative_int(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


@dataclass(frozen=True, slots=True)
class PlayerCatalogEntry:
    """Source-supplied identity/rank facts for one fixed replay seat.

    ``name`` and ``rank`` remain independent: a replay can identify a player
    without carrying a leaderboard rank.  The catalog never translates reward,
    score, team id, or submission id into either field.
    """

    seat: int
    name: str | None
    rank: int | None
    name_sources: tuple[str, ...] = ()
    rank_sources: tuple[str, ...] = ()
    name_availability_reasons: tuple[str, ...] = ()
    rank_availability_reasons: tuple[str, ...] = ()

    @property
    def name_available(self) -> bool:
        return self.name is not None and not self.name_availability_reasons

    @property
    def rank_available(self) -> bool:
        return self.rank is not None and not self.rank_availability_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "name": self.name if self.name_available else None,
            "name_available": self.name_available,
            "name_availability_reasons": list(self.name_availability_reasons),
            "name_sources": list(self.name_sources),
            "rank": self.rank if self.rank_available else None,
            "rank_available": self.rank_available,
            "rank_availability_reasons": list(self.rank_availability_reasons),
            "rank_sources": list(self.rank_sources),
        }


@dataclass(frozen=True, slots=True)
class CachedReplayOutcomeSummary:
    """Aggregate only direct archived ``own_agent.reward`` observations.

    This is deliberately a cache summary, not a Kaggle leaderboard result.
    The archive does not declare whether a zero reward represents a draw, a
    loss, or another terminal state, so zero/negative rewards are reported as
    ``nonwins`` and never subdivided into losses or draws.
    """

    games_total: int
    games_with_outcome: int
    wins: int
    nonwins: int
    missing_outcome_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Whether at least one direct own-agent reward is available."""

        return self.games_with_outcome > 0

    @property
    def complete(self) -> bool:
        """Whether every catalogued game supplied a direct own-agent reward."""

        return self.games_total > 0 and self.games_with_outcome == self.games_total

    @property
    def games_without_outcome(self) -> int:
        return self.games_total - self.games_with_outcome

    @property
    def win_rate(self) -> float | None:
        return (
            self.wins / self.games_with_outcome if self.games_with_outcome > 0 else None
        )

    @property
    def availability_reasons(self) -> tuple[str, ...]:
        if self.games_total == 0:
            return ("cached_replay_outcomes_no_games",)
        if self.games_with_outcome == 0:
            return _unique_reasons(
                (
                    "cached_replay_outcomes_no_source_backed_own_agent_rewards",
                    *self.missing_outcome_reasons,
                )
            )
        if not self.complete:
            return _unique_reasons(
                (
                    "cached_replay_outcomes_partial_source_coverage",
                    *self.missing_outcome_reasons,
                )
            )
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "availability_reasons": list(self.availability_reasons),
            "complete": self.complete,
            "games_total": self.games_total,
            "games_with_outcome": self.games_with_outcome,
            "games_without_outcome": self.games_without_outcome,
            "wins": self.wins,
            # Do not call this a loss count: cached own-agent reward semantics
            # alone do not prove whether zero is a draw or another result.
            "nonwins": self.nonwins,
            "nonwins_definition": "source_backed_own_agent_reward_lte_zero",
            "draws": None,
            "draws_available": False,
            "draws_availability_reasons": [
                "draws_not_provable_from_own_agent_reward_semantics"
            ],
            "win_rate": self.win_rate,
            "source": "archived_episode_metadata.own_agent.reward",
        }


@dataclass(frozen=True, slots=True)
class ReplayCatalogEntry:
    """One episode expected by metadata and/or present as a replay file.

    ``available`` is retained as a compatibility alias for
    ``replay_available``.  It deliberately says nothing about whether a model
    may be evaluated: a cached replay remains useful for timeline browsing
    when the matching package/checkpoint provenance is absent.
    """

    submission_id: int
    episode_id: int
    declared_path: Path | None
    path: Path | None
    size_bytes: int | None
    metadata: Mapping[str, Any] | None
    players: tuple[PlayerCatalogEntry, ...] = ()
    expected_replay_sha256: str | None = None
    actual_replay_sha256: str | None = None
    availability_reasons: tuple[str, ...] = ()
    model_analysis_availability_reasons: tuple[str, ...] = ()

    @property
    def replay_available(self) -> bool:
        """Whether the contained replay JSON can be browsed safely."""

        return self.path is not None and not self.availability_reasons

    @property
    def replay_availability_reasons(self) -> tuple[str, ...]:
        """Explicit source-file/metadata reasons a replay is not browseable."""

        return self.availability_reasons

    @property
    def model_analysis_available(self) -> bool:
        """Whether this replay is eligible for exact-model analysis."""

        return self.replay_available and not self.model_analysis_availability_reasons

    @property
    def available(self) -> bool:
        return self.replay_available

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "episode_id": self.episode_id,
            "declared_path": str(self.declared_path) if self.declared_path else None,
            "path": str(self.path) if self.path else None,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata) if self.metadata is not None else None,
            "players": [player.to_dict() for player in self.players],
            "expected_replay_sha256": self.expected_replay_sha256,
            "actual_replay_sha256": self.actual_replay_sha256,
            # ``available`` remains for older callers; the explicit fields
            # prevent a browser endpoint from conflating source readability
            # with model provenance.
            "available": self.replay_available,
            "availability_reasons": list(self.availability_reasons),
            "replay_available": self.replay_available,
            "replay_availability_reasons": list(self.availability_reasons),
            "model_analysis_available": self.model_analysis_available,
            "model_analysis_availability_reasons": list(
                self.model_analysis_availability_reasons
            ),
        }


@dataclass(frozen=True, slots=True)
class SubmissionCatalogEntry:
    """All read-only archive and provenance facts for one submission id."""

    submission_id: int
    declared_archive_directory: Path
    archive_directory: Path | None
    episodes_manifest: Path | None
    replays: tuple[ReplayCatalogEntry, ...]
    provenance: SubmissionProvenance | None
    provenance_candidates: tuple[SubmissionProvenance, ...]
    submission_label: SubmissionLabel
    cached_replay_outcomes: CachedReplayOutcomeSummary
    availability_reasons: tuple[str, ...] = ()
    model_analysis_availability_reasons: tuple[str, ...] = ()

    @property
    def archive_available(self) -> bool:
        """Whether the numeric archive directory stayed inside its root."""

        return self.archive_directory is not None

    @property
    def replay_available(self) -> bool:
        """Whether at least one archived replay can be browsed."""

        return any(row.replay_available for row in self.replays)

    @property
    def replay_availability_reasons(self) -> tuple[str, ...]:
        """Archive/metadata diagnostics, independent of model provenance."""

        return self.availability_reasons

    @property
    def model_analysis_available(self) -> bool:
        """Whether this submission has a sole verified mapping and replay."""

        return (
            any(row.model_analysis_available for row in self.replays)
            and self.provenance is not None
            and self.provenance.available
            and not self.model_analysis_availability_reasons
        )

    @property
    def model_analysis_provenance(self) -> SubmissionProvenance | None:
        """Return a mapping only when at least one replay is analysis-eligible.

        A request handler must still check the selected
        :attr:`ReplayCatalogEntry.model_analysis_available`; this accessor
        merely prevents using a diagnostic-only provenance row as a model
        source.
        """

        return self.provenance if self.model_analysis_available else None

    @property
    def available(self) -> bool:
        return self.replay_available

    def replay(self, episode_id: int) -> ReplayCatalogEntry | None:
        target = int(episode_id)
        return next((row for row in self.replays if row.episode_id == target), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "declared_archive_directory": str(self.declared_archive_directory),
            "archive_directory": (
                str(self.archive_directory) if self.archive_directory else None
            ),
            "episodes_manifest": (
                str(self.episodes_manifest) if self.episodes_manifest else None
            ),
            "label": (
                self.submission_label.text if self.submission_label.available else None
            ),
            "submission_label": self.submission_label.to_dict(),
            "cached_replay_outcomes": self.cached_replay_outcomes.to_dict(),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "provenance_candidates": [
                row.to_dict() for row in self.provenance_candidates
            ],
            "available": self.replay_available,
            "availability_reasons": list(self.availability_reasons),
            "archive_available": self.archive_available,
            "replay_available": self.replay_available,
            "replay_availability_reasons": list(self.availability_reasons),
            "model_analysis_available": self.model_analysis_available,
            "model_analysis_availability_reasons": list(
                self.model_analysis_availability_reasons
            ),
            "replays": [row.to_dict() for row in self.replays],
        }


@dataclass(frozen=True, slots=True)
class ReplayArchiveCatalog:
    """A server-friendly snapshot of one configured replay archive root."""

    declared_archive_root: Path
    archive_root: Path | None
    submissions: tuple[SubmissionCatalogEntry, ...]
    availability_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.archive_root is not None and not self.availability_reasons

    def submission(self, submission_id: int) -> SubmissionCatalogEntry | None:
        target = int(submission_id)
        return next(
            (row for row in self.submissions if row.submission_id == target), None
        )

    def replay(self, submission_id: int, episode_id: int) -> ReplayCatalogEntry | None:
        submission = self.submission(submission_id)
        return submission.replay(episode_id) if submission else None

    def available_submissions(self) -> tuple[SubmissionCatalogEntry, ...]:
        """Return submissions with at least one browseable replay."""

        return tuple(row for row in self.submissions if row.replay_available)

    def model_analysis_submissions(self) -> tuple[SubmissionCatalogEntry, ...]:
        """Return only submissions safe to hand to model inference."""

        return tuple(row for row in self.submissions if row.model_analysis_available)

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_archive_root": str(self.declared_archive_root),
            "archive_root": str(self.archive_root) if self.archive_root else None,
            "available": self.available,
            "availability_reasons": list(self.availability_reasons),
            "submissions": [row.to_dict() for row in self.submissions],
        }

    @classmethod
    def scan(
        cls,
        archive_root: PathLike,
        *,
        provenance: ProvenanceManifest | None = None,
    ) -> ReplayArchiveCatalog:
        return scan_archive(archive_root, provenance=provenance)


def _resolve_archive_file(
    raw_path: Path,
    *,
    submission_root: Path,
    role: str,
) -> tuple[Path | None, str | None]:
    try:
        resolved = resolve_contained_path(
            raw_path,
            roots=(submission_root,),
            require_exists=True,
        )
    except FileNotFoundError:
        return None, f"{role}_missing"
    except PathContainmentError:
        return None, f"{role}_path_outside_submission_directory"
    except OSError:
        return None, f"{role}_unreadable"
    if not resolved.is_file():
        return None, f"{role}_not_a_regular_file"
    return resolved, None


def _read_replay_info(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    """Read the minimum replay shape plus small authoritative ``info`` data.

    The catalog deliberately discards replay steps after validation.  The HTTP
    timeline endpoint opens the selected source again, while this helper keeps
    only player-name/rank metadata that Kaggle placed in the replay header.
    """

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError:
        return None, "replay_file_invalid_json"
    except (OSError, UnicodeDecodeError):
        return None, "replay_file_unreadable"
    if not isinstance(payload, Mapping):
        return None, "replay_file_not_an_object"
    if not isinstance(payload.get("steps"), list):
        return None, "replay_file_steps_missing_or_invalid"
    info = payload.get("info")
    return (dict(info) if isinstance(info, Mapping) else None), None


def _validate_replay_file(path: Path) -> str | None:
    """Compatibility wrapper for the replay readability check."""

    _info, reason = _read_replay_info(path)
    return reason


def _nonempty_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _append_agent_claims(
    agent: Mapping[str, Any],
    *,
    source: str,
    fallback_seat: int | None = None,
    names: dict[int, list[tuple[str, str]]],
    ranks: dict[int, list[tuple[int, str]]],
    invalid_names: set[int],
    invalid_ranks: set[int],
) -> None:
    """Collect direct per-seat claims from an archived agent/player object."""

    seat = _nonnegative_int(agent.get("index"))
    if seat is None and fallback_seat in {0, 1}:
        # Kaggle's authoritative ``info.Agents`` is an ordered seat array.
        # This is an explicit structural seat, not an identity/rank inference.
        seat = fallback_seat
    if seat not in {0, 1}:
        # A seat is never guessed from list position or a submission id.
        return
    for key in ("team_name", "teamName", "name", "Name"):
        if key not in agent:
            continue
        value = _nonempty_text(agent.get(key))
        if value is None:
            invalid_names.add(seat)
        else:
            names[seat].append((value, f"{source}.{key}"))
    for key in (
        "rank",
        "Rank",
        "ranking",
        "Ranking",
        "leaderboard_rank",
        "leaderboardRank",
    ):
        if key not in agent:
            continue
        value = _explicit_rank(agent.get(key))
        if value is None:
            invalid_ranks.add(seat)
        else:
            ranks[seat].append((value, f"{source}.{key}"))


def _append_indexed_claims(
    container: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    source: str,
    values: dict[int, list[tuple[Any, str]]],
    invalid: set[int],
    normalise: Callable[[Any], Any],
) -> None:
    """Collect direct two-seat lists such as ``info.TeamNames`` safely."""

    for key in keys:
        if key not in container:
            continue
        raw_values = container.get(key)
        if not isinstance(raw_values, list):
            invalid.update((0, 1))
            continue
        for seat, raw_value in enumerate(raw_values[:2]):
            value = normalise(raw_value)
            if value is None:
                invalid.add(seat)
            else:
                values[seat].append((value, f"{source}.{key}"))


def _resolve_player_claim(
    values: list[tuple[Any, str]],
    *,
    invalid: bool,
    unavailable_reason: str,
    invalid_reason: str,
    conflict_reason: str,
) -> tuple[Any | None, tuple[str, ...], tuple[str, ...]]:
    """Accept one exact direct value; conflicts stay visible but unavailable."""

    source_values: list[Any] = []
    sources: list[str] = []
    for value, source in values:
        if value not in source_values:
            source_values.append(value)
        sources.append(source)
    if len(source_values) == 1:
        return source_values[0], _unique_reasons(sources), ()
    if len(source_values) > 1:
        return None, _unique_reasons(sources), (conflict_reason,)
    if invalid:
        return None, (), (invalid_reason,)
    return None, (), (unavailable_reason,)


def _catalog_players(
    episode_metadata: Mapping[str, Any] | None,
    replay_info: Mapping[str, Any] | None,
) -> tuple[PlayerCatalogEntry, ...]:
    """Build fixed-seat player context without deriving identities or ranks."""

    names: dict[int, list[tuple[str, str]]] = {0: [], 1: []}
    ranks: dict[int, list[tuple[int, str]]] = {0: [], 1: []}
    invalid_names: set[int] = set()
    invalid_ranks: set[int] = set()

    if isinstance(episode_metadata, Mapping):
        agents = episode_metadata.get("agents")
        if isinstance(agents, list):
            for agent in agents:
                if isinstance(agent, Mapping):
                    _append_agent_claims(
                        agent,
                        source="archived_episode_metadata.agents",
                        names=names,
                        ranks=ranks,
                        invalid_names=invalid_names,
                        invalid_ranks=invalid_ranks,
                    )
        own_agent = episode_metadata.get("own_agent")
        if isinstance(own_agent, Mapping):
            _append_agent_claims(
                own_agent,
                source="archived_episode_metadata.own_agent",
                names=names,
                ranks=ranks,
                invalid_names=invalid_names,
                invalid_ranks=invalid_ranks,
            )

    if isinstance(replay_info, Mapping):
        _append_indexed_claims(
            replay_info,
            keys=("TeamNames", "PlayerNames"),
            source="authoritative_replay_metadata.info",
            values=names,
            invalid=invalid_names,
            normalise=_nonempty_text,
        )
        _append_indexed_claims(
            replay_info,
            keys=("TeamRanks", "PlayerRanks", "Ranks"),
            source="authoritative_replay_metadata.info",
            values=ranks,
            invalid=invalid_ranks,
            normalise=_explicit_rank,
        )
        players = replay_info.get("Players")
        if isinstance(players, list):
            for seat, player in enumerate(players[:2]):
                if isinstance(player, Mapping):
                    _append_agent_claims(
                        player,
                        source="authoritative_replay_metadata.info.Players",
                        fallback_seat=seat,
                        names=names,
                        ranks=ranks,
                        invalid_names=invalid_names,
                        invalid_ranks=invalid_ranks,
                    )
        agents = replay_info.get("Agents")
        if isinstance(agents, list):
            for seat, agent in enumerate(agents[:2]):
                if isinstance(agent, Mapping):
                    _append_agent_claims(
                        agent,
                        source="authoritative_replay_metadata.info.Agents",
                        fallback_seat=seat,
                        names=names,
                        ranks=ranks,
                        invalid_names=invalid_names,
                        invalid_ranks=invalid_ranks,
                    )

    entries: list[PlayerCatalogEntry] = []
    for seat in (0, 1):
        name, name_sources, name_reasons = _resolve_player_claim(
            names[seat],
            invalid=seat in invalid_names,
            unavailable_reason=(
                "player_name_unavailable_not_supplied_by_archived_metadata_or_replay"
            ),
            invalid_reason="player_name_invalid_in_archived_metadata_or_replay",
            conflict_reason="player_name_source_conflict",
        )
        rank, rank_sources, rank_reasons = _resolve_player_claim(
            ranks[seat],
            invalid=seat in invalid_ranks,
            unavailable_reason=(
                "player_rank_unavailable_not_supplied_by_archived_metadata_or_replay"
            ),
            invalid_reason="player_rank_invalid_in_archived_metadata_or_replay",
            conflict_reason="player_rank_source_conflict",
        )
        entries.append(
            PlayerCatalogEntry(
                seat=seat,
                name=name,
                rank=rank,
                name_sources=name_sources,
                rank_sources=rank_sources,
                name_availability_reasons=name_reasons,
                rank_availability_reasons=rank_reasons,
            )
        )
    return tuple(entries)


def _load_episode_metadata(
    manifest: Path,
    *,
    submission_id: int,
) -> tuple[dict[int, tuple[Mapping[str, Any], ...]], tuple[str, ...]]:
    """Read only `episodes.json`; retain duplicate metadata for diagnostics."""

    reasons: list[str] = []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ("episodes_manifest_invalid_json",)
    except OSError:
        return {}, ("episodes_manifest_unreadable",)
    if not isinstance(payload, Mapping):
        return {}, ("episodes_manifest_not_an_object",)

    listed_submission = _positive_int(payload.get("submission_id"))
    if listed_submission is None:
        reasons.append("episodes_manifest_submission_id_missing_or_invalid")
    elif listed_submission != submission_id:
        reasons.append("episodes_manifest_submission_id_mismatch")

    rows = payload.get("episodes")
    if not isinstance(rows, list):
        return {}, _unique_reasons((*reasons, "episodes_manifest_episodes_not_a_list"))

    indexed: dict[int, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            reasons.append(f"episodes_manifest_row_{index}_not_an_object")
            continue
        episode_id = _nonnegative_int(row.get("episode_id"))
        if episode_id is None:
            reasons.append(f"episodes_manifest_row_{index}_invalid_episode_id")
            continue
        indexed.setdefault(episode_id, []).append(row)
    expected_count = payload.get("episode_count")
    if expected_count is not None:
        count = _nonnegative_int(expected_count)
        if count is None or count != len(rows):
            reasons.append("episodes_manifest_episode_count_mismatch")
    return {key: tuple(value) for key, value in indexed.items()}, _unique_reasons(
        reasons
    )


def _source_backed_own_agent_reward(
    replay: ReplayCatalogEntry,
) -> tuple[int | float | None, str | None]:
    """Return only the explicit archived ``own_agent.reward`` observation."""

    metadata = replay.metadata
    if not isinstance(metadata, Mapping):
        return None, "cached_replay_outcome_episode_metadata_missing_or_ambiguous"
    own_agent = metadata.get("own_agent")
    if not isinstance(own_agent, Mapping):
        return None, "cached_replay_outcome_own_agent_missing_or_invalid"
    if "reward" not in own_agent or own_agent.get("reward") is None:
        return None, "cached_replay_outcome_own_agent_reward_missing"
    reward = own_agent.get("reward")
    if isinstance(reward, bool):
        return None, "cached_replay_outcome_own_agent_reward_invalid"
    if isinstance(reward, int):
        return reward, None
    if isinstance(reward, float) and math.isfinite(reward):
        return reward, None
    return None, "cached_replay_outcome_own_agent_reward_invalid"


def _summarize_cached_replay_outcomes(
    replays: Iterable[ReplayCatalogEntry],
) -> CachedReplayOutcomeSummary:
    """Summarize direct own-agent rewards without interpreting zero as a draw."""

    rows = tuple(replays)
    wins = 0
    nonwins = 0
    games_with_outcome = 0
    missing_reasons: list[str] = []
    for replay in rows:
        reward, reason = _source_backed_own_agent_reward(replay)
        if reward is None:
            if reason is not None:
                missing_reasons.append(reason)
            continue
        games_with_outcome += 1
        if reward > 0:
            wins += 1
        else:
            nonwins += 1
    return CachedReplayOutcomeSummary(
        games_total=len(rows),
        games_with_outcome=games_with_outcome,
        wins=wins,
        nonwins=nonwins,
        missing_outcome_reasons=_unique_reasons(missing_reasons),
    )


def _catalog_provenance(
    submission_id: int,
    provenance: ProvenanceManifest | None,
) -> tuple[
    SubmissionProvenance | None,
    tuple[SubmissionProvenance, ...],
    tuple[str, ...],
]:
    if provenance is None:
        return None, (), ("provenance_not_configured",)
    candidates = provenance.candidates_for_submission(submission_id)
    if not candidates:
        return None, (), ("provenance_unresolved",)
    if len(candidates) != 1:
        return None, candidates, ("provenance_ambiguous",)
    selected = candidates[0]
    if not selected.available:
        return (
            selected,
            candidates,
            _unique_reasons(
                (
                    "provenance_unavailable",
                    *(
                        f"provenance:{reason}"
                        for reason in selected.availability_reasons
                    ),
                )
            ),
        )
    return selected, candidates, ()


def _catalog_submission_label(
    *,
    provenance: ProvenanceManifest | None,
    selected: SubmissionProvenance | None,
    candidates: tuple[SubmissionProvenance, ...],
) -> SubmissionLabel:
    """Return only a sole, checksum-backed label; never synthesize one."""

    if provenance is None:
        return SubmissionLabel.unavailable("submission_label_provenance_not_configured")
    if not candidates:
        return SubmissionLabel.unavailable("submission_label_provenance_unresolved")
    if len(candidates) != 1 or selected is None:
        return SubmissionLabel.unavailable("submission_label_provenance_ambiguous")
    return selected.submission_label


def _scan_submission(
    raw_directory: Path,
    *,
    archive_root: Path,
    provenance: ProvenanceManifest | None,
) -> SubmissionCatalogEntry:
    submission_id = int(raw_directory.name)
    reasons: list[str] = []
    try:
        directory = resolve_contained_path(
            raw_directory,
            roots=(archive_root,),
            require_exists=True,
        )
    except FileNotFoundError:
        directory = None
        reasons.append("archive_submission_directory_missing")
    except PathContainmentError:
        directory = None
        reasons.append("archive_submission_directory_outside_configured_root")
    except OSError:
        directory = None
        reasons.append("archive_submission_directory_unreadable")
    if directory is not None and not directory.is_dir():
        directory = None
        reasons.append("archive_submission_path_not_a_directory")

    selected, candidates, provenance_reasons = _catalog_provenance(
        submission_id, provenance
    )
    submission_label = _catalog_submission_label(
        provenance=provenance,
        selected=selected,
        candidates=candidates,
    )
    if directory is None:
        archive_reasons = _unique_reasons(reasons)
        return SubmissionCatalogEntry(
            submission_id=submission_id,
            declared_archive_directory=raw_directory,
            archive_directory=None,
            episodes_manifest=None,
            replays=(),
            provenance=selected,
            provenance_candidates=candidates,
            submission_label=submission_label,
            cached_replay_outcomes=CachedReplayOutcomeSummary(
                games_total=0,
                games_with_outcome=0,
                wins=0,
                nonwins=0,
            ),
            availability_reasons=archive_reasons,
            model_analysis_availability_reasons=_unique_reasons(
                (
                    *archive_reasons,
                    "archive_submission_unavailable",
                    *provenance_reasons,
                )
            ),
        )

    manifest, manifest_reason = _resolve_archive_file(
        directory / "episodes.json",
        submission_root=directory,
        role="episodes_manifest",
    )
    metadata: dict[int, tuple[Mapping[str, Any], ...]] = {}
    metadata_reasons: tuple[str, ...] = ()
    if manifest is None:
        reasons.append(manifest_reason or "episodes_manifest_missing")
    else:
        metadata, metadata_reasons = _load_episode_metadata(
            manifest, submission_id=submission_id
        )
        reasons.extend(metadata_reasons)

    files: dict[
        int,
        tuple[
            Path,
            Path | None,
            int | None,
            str | None,
            Mapping[str, Any] | None,
            tuple[str, ...],
        ],
    ] = {}
    try:
        children = tuple(directory.iterdir())
    except OSError:
        children = ()
        reasons.append("archive_submission_directory_unreadable")
    for child in children:
        match = _REPLAY_FILE_RE.fullmatch(child.name)
        if match is None:
            continue
        episode_id = int(match.group(1))
        path, file_reason = _resolve_archive_file(
            child,
            submission_root=directory,
            role="replay_file",
        )
        file_reasons: list[str] = []
        if file_reason is not None:
            file_reasons.append(file_reason)
        size_bytes: int | None = None
        actual_replay_sha256: str | None = None
        replay_info: Mapping[str, Any] | None = None
        if path is not None:
            try:
                size_bytes = int(path.stat().st_size)
            except OSError:
                file_reasons.append("replay_file_unreadable")
                path = None
        if path is not None:
            replay_info, shape_reason = _read_replay_info(path)
            if shape_reason is not None:
                file_reasons.append(shape_reason)
        if path is not None:
            try:
                actual_replay_sha256 = sha256_file(path)
            except OSError:
                file_reasons.append("replay_sha256_unreadable")
        files[episode_id] = (
            child,
            path,
            size_bytes,
            actual_replay_sha256,
            replay_info,
            _unique_reasons(file_reasons),
        )

    replay_rows: list[ReplayCatalogEntry] = []
    for episode_id in sorted(set(metadata) | set(files)):
        source = files.get(episode_id)
        raw_metadata = metadata.get(episode_id, ())
        replay_reasons: list[str] = []
        replay_model_reasons: list[str] = []
        if source is None:
            declared_path = None
            path = None
            size_bytes = None
            actual_replay_sha256 = None
            replay_info = None
            replay_reasons.append("replay_file_missing")
        else:
            (
                declared_path,
                path,
                size_bytes,
                actual_replay_sha256,
                replay_info,
                file_reasons,
            ) = source
            replay_reasons.extend(file_reasons)
        expected_replay_sha256: str | None = None
        if selected is not None:
            bindings = selected.replay_bindings_for_episode(episode_id)
            if not bindings:
                replay_model_reasons.append("replay_provenance_missing")
            elif len(bindings) != 1:
                replay_model_reasons.append("replay_provenance_ambiguous")
            else:
                binding = bindings[0]
                expected_replay_sha256 = binding.expected_sha256
                replay_model_reasons.extend(binding.availability_reasons)
                if binding.available:
                    if actual_replay_sha256 is None:
                        replay_model_reasons.append("replay_sha256_unreadable")
                    elif actual_replay_sha256 != expected_replay_sha256:
                        replay_model_reasons.append("replay_sha256_mismatch")
        if not raw_metadata:
            # A raw replay can still be rendered without an episode metadata
            # row.  It cannot be used for model analysis because its archive
            # identity has not been independently corroborated.
            replay_model_reasons.append("episode_metadata_missing")
            row_metadata = None
        elif len(raw_metadata) > 1:
            replay_model_reasons.append("episode_metadata_ambiguous")
            row_metadata = None
        else:
            row_metadata = raw_metadata[0]
        players = _catalog_players(row_metadata, replay_info)
        replay_rows.append(
            ReplayCatalogEntry(
                submission_id=submission_id,
                episode_id=episode_id,
                declared_path=declared_path,
                path=path,
                size_bytes=size_bytes,
                metadata=row_metadata,
                players=players,
                expected_replay_sha256=expected_replay_sha256,
                actual_replay_sha256=actual_replay_sha256,
                availability_reasons=_unique_reasons(replay_reasons),
                model_analysis_availability_reasons=_unique_reasons(
                    replay_model_reasons
                ),
            )
        )

    archive_reasons = _unique_reasons(reasons)
    model_submission_reasons = _unique_reasons((*archive_reasons, *provenance_reasons))
    # Keep source browseability and model eligibility separate.  This is both
    # useful to operators (a missing artifact manifest does not erase a
    # timeline) and defensive (an inference endpoint has a dedicated,
    # provenance-inclusive failure signal to check).
    replay_rows = [
        ReplayCatalogEntry(
            submission_id=row.submission_id,
            episode_id=row.episode_id,
            declared_path=row.declared_path,
            path=row.path,
            size_bytes=row.size_bytes,
            metadata=row.metadata,
            players=row.players,
            expected_replay_sha256=row.expected_replay_sha256,
            actual_replay_sha256=row.actual_replay_sha256,
            availability_reasons=row.availability_reasons,
            model_analysis_availability_reasons=_unique_reasons(
                (
                    *row.model_analysis_availability_reasons,
                    *row.availability_reasons,
                    *model_submission_reasons,
                )
            ),
        )
        for row in replay_rows
    ]
    cached_replay_outcomes = _summarize_cached_replay_outcomes(replay_rows)
    return SubmissionCatalogEntry(
        submission_id=submission_id,
        declared_archive_directory=raw_directory,
        archive_directory=directory,
        episodes_manifest=manifest,
        replays=tuple(replay_rows),
        provenance=selected,
        provenance_candidates=candidates,
        submission_label=submission_label,
        cached_replay_outcomes=cached_replay_outcomes,
        availability_reasons=archive_reasons,
        model_analysis_availability_reasons=model_submission_reasons,
    )


def scan_archive(
    archive_root: PathLike,
    *,
    provenance: ProvenanceManifest | None = None,
) -> ReplayArchiveCatalog:
    """Scan `<root>/<numeric submission>/episodes.json` without source writes.

    Only direct numeric children are treated as submissions.  Paths are
    resolved beneath the configured archive root before they are read, so a
    symlinked submission directory or replay file cannot escape into an
    unrelated local filesystem location.
    """

    declared_root = Path(archive_root).expanduser()
    try:
        resolved_root = declared_root.resolve(strict=True)
    except FileNotFoundError:
        return ReplayArchiveCatalog(
            declared_archive_root=declared_root,
            archive_root=None,
            submissions=(),
            availability_reasons=("archive_root_missing",),
        )
    except (OSError, RuntimeError):
        return ReplayArchiveCatalog(
            declared_archive_root=declared_root,
            archive_root=None,
            submissions=(),
            availability_reasons=("archive_root_unreadable",),
        )
    if not resolved_root.is_dir():
        return ReplayArchiveCatalog(
            declared_archive_root=declared_root,
            archive_root=None,
            submissions=(),
            availability_reasons=("archive_root_not_a_directory",),
        )

    try:
        children = tuple(resolved_root.iterdir())
    except OSError:
        return ReplayArchiveCatalog(
            declared_archive_root=declared_root,
            archive_root=resolved_root,
            submissions=(),
            availability_reasons=("archive_root_unreadable",),
        )
    numeric_children = [
        child for child in children if _SUBMISSION_DIR_RE.fullmatch(child.name)
    ]
    submissions = tuple(
        _scan_submission(
            child,
            archive_root=resolved_root,
            provenance=provenance,
        )
        for child in sorted(numeric_children, key=lambda value: int(value.name))
    )
    return ReplayArchiveCatalog(
        declared_archive_root=declared_root,
        archive_root=resolved_root,
        submissions=submissions,
    )


# A descriptive alias is convenient in callers that construct a catalog once
# at server start and rescan it on an explicit refresh action.
build_catalog = scan_archive


__all__ = [
    "CachedReplayOutcomeSummary",
    "PlayerCatalogEntry",
    "ReplayArchiveCatalog",
    "ReplayCatalogEntry",
    "SubmissionCatalogEntry",
    "build_catalog",
    "scan_archive",
]
