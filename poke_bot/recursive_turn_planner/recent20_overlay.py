"""Streaming adapter for the checkpoint-independent recent-20 RTP overlay.

The immutable semantic tensor pack remains the feature authority.  Overlay
rows contain only action-program structure and stable offsets into that pack;
they never copy feature tensors or privileged opponent information.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, BinaryIO


OVERLAY_SCHEMA = "poke_bot.alakazam_recent20_rtp_complete_action_overlay/v1"
MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_recent20_rtp_overlay_completion_receipt/v1"
BASE_SCHEMA = "poke_bot.alakazam_recent20_semantic_tensor_pack/v1"
BASE_COMPLETION_SCHEMA = (
    "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1"
)
DATASET_SAMPLE_SCHEMA = "poke_bot.alakazam_recent20_rtp_joined_sample/v1"


class Recent20OverlayError(RuntimeError):
    """The immutable base/overlay contract was violated."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Recent20OverlayError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise Recent20OverlayError(f"expected JSON object: {path}")
    return value


def _overlay_member_path(manifest_path: Path, declared: str) -> Path:
    """Resolve overlay members declared relative to the artifact root.

    Manifests live in ``<artifact>/manifests`` while their content-addressed
    shards live in ``<artifact>/objects``. Absolute declarations remain valid
    for compatibility with sealed identity documents.
    """
    path = Path(str(declared))
    if path.is_absolute():
        return path
    root = (
        manifest_path.parent.parent
        if manifest_path.parent.name == "manifests"
        else manifest_path.parent
    )
    return root / path


def base_schema_descriptor(completion: Mapping[str, Any]) -> dict[str, Any]:
    packs = list(completion.get("packs") or ())
    if not packs:
        raise Recent20OverlayError("base completion has no packs")
    schemas = {str(row.get("schema") or "") for row in packs}
    widths = {int(row.get("feature_width", -1)) for row in packs}
    dtypes = {str(row.get("feature_dtype") or "") for row in packs}
    roles = {
        tuple(sorted(str(key) for key in dict(row.get("files") or {})))
        for row in packs
    }
    if schemas != {BASE_SCHEMA} or widths != {40} or dtypes != {"float32_le"}:
        raise Recent20OverlayError("base pack schema drifted")
    expected_roles = (
        "decision_key_sha256",
        "decision_offsets_u64",
        "features_f32",
        "selected_option_u32",
    )
    if roles != {expected_roles}:
        raise Recent20OverlayError("base pack file-role inventory drifted")
    return {
        "schema": BASE_SCHEMA,
        "feature_width": 40,
        "feature_dtype": "float32_le",
        "decision_offset_dtype": "uint64_le",
        "selected_option_dtype": "uint32_le",
        "decision_key_dtype": "sha256_raw_32_bytes",
        "file_roles": list(expected_roles),
        "checkpoint_independent": True,
    }


def overlay_schema_document() -> dict[str, Any]:
    return {
        "schema": OVERLAY_SCHEMA,
        "version": 1,
        "row_unit": "recorded_complete_action_program",
        "join_identity": [
            "utc_day",
            "source_archive_sha256",
            "episode_id",
            "acting_seat",
            "env_step",
            "factorized_stage",
            "base_decision_key_sha256",
        ],
        "base_reference": {
            "copied_feature_tensors": False,
            "fields": [
                "base_pack_receipt_sha256",
                "base_source_shard_sha256",
                "base_decision_index",
                "base_option_start",
                "base_option_count",
            ],
        },
        "runtime_public_fields": [
            "canonical_public_observation_hash",
            "ordered_legal_action_programs",
            "selected_action_program",
            "recorded_successor_public_hash",
            "turn",
            "valid_option_mask",
        ],
        "target_only_fields": [
            "recorded_outcome",
            "episode_terminal_state",
            "recorded_successor_program_identity",
        ],
        "forbidden_fields": [
            "opponent_hand_identities",
            "opponent_deck_order",
            "opponent_deck_multiset_sha256",
            "unrevealed_prize_identities",
            "future_transition_as_policy_input",
        ],
        "counterfactual_targets": False,
        "unchosen_action_targets": "absent_masked",
    }


def _resolve_pack_root(completion: Mapping[str, Any], root: Path) -> dict[str, Path]:
    observed: dict[str, Path] = {}
    for row in completion.get("packs") or ():
        # The completion source order mixes hosts.  The pack directory name is
        # always the UTC day and is the only location component consumed here.
        day = ""
        for field in ("receipt_path", "source_path"):
            candidates = [
                part
                for part in Path(str(row.get(field) or "")).parts
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part)
            ]
            if candidates:
                day = candidates[-1]
                break
        if not day or day in observed:
            raise Recent20OverlayError("base completion day inventory is malformed")
        observed[day] = Path(root) / day
    return observed


class _BaseDayReader:
    def __init__(self, day_root: Path, pack: Mapping[str, Any]) -> None:
        self.day_root = Path(day_root)
        self.pack = dict(pack)
        files = dict(pack.get("files") or {})
        self.features: BinaryIO = (self.day_root / "features.f32").open("rb")
        self.width = int(pack.get("feature_width", -1))
        if self.width != 40:
            raise Recent20OverlayError("base feature width is not 40")
        for role, filename in (
            ("features_f32", "features.f32"),
            ("decision_offsets_u64", "decision_offsets.u64"),
            ("selected_option_u32", "selected_option.u32"),
            ("decision_key_sha256", "decision_keys.sha256"),
        ):
            declaration = dict(files.get(role) or {})
            path = self.day_root / filename
            if not path.is_file() or path.stat().st_size != int(
                declaration.get("size_bytes", -1)
            ):
                raise Recent20OverlayError(f"base file size mismatch: {path}")

    def close(self) -> None:
        self.features.close()

    def read_options(self, start: int, count: int) -> list[list[float]]:
        if start < 0 or count < 1:
            raise Recent20OverlayError("invalid base option range")
        byte_start = int(start) * self.width * 4
        byte_count = int(count) * self.width * 4
        self.features.seek(byte_start)
        raw = self.features.read(byte_count)
        if len(raw) != byte_count:
            raise Recent20OverlayError("short read from base feature tensor")
        values = struct.unpack(f"<{count * self.width}f", raw)
        return [
            list(values[index : index + self.width])
            for index in range(0, len(values), self.width)
        ]


class Recent20RTPDataset:
    """Open and stream the exact base-pack-plus-overlay view.

    Only one overlay row and its referenced option vectors are resident at a
    time.  The caller may iterate train, validation, or evaluation days.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        base_pack_root: Path | str,
        base_completion_path: Path | str | None = None,
        expected_manifest_sha256: str = "",
        expected_base_completion_sha256: str = "",
        verify_overlay_shards: bool = True,
        verify_base_shards: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise Recent20OverlayError("overlay manifest must be a regular file")
        actual_manifest = sha256_file(self.manifest_path)
        if expected_manifest_sha256 and actual_manifest != expected_manifest_sha256:
            raise Recent20OverlayError("overlay manifest digest mismatch")
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("schema") != MANIFEST_SCHEMA:
            raise Recent20OverlayError("foreign overlay manifest")
        declared_completion_path = Path(
            str(self.manifest["base_pack"]["completion_path"])
        )
        completion_path = (
            Path(base_completion_path).expanduser().resolve()
            if base_completion_path is not None
            else declared_completion_path.expanduser().resolve()
        )
        if completion_path.is_symlink() or not completion_path.is_file():
            raise Recent20OverlayError("base completion must be a regular file")
        # A transferred view retains the canonical overlay manifest byte for
        # byte, so its embedded completion path can remain an Elmo path.  The
        # optional override only relocates that immutable object; the declared
        # SHA below still binds the exact original identity.
        self.base_completion_path = completion_path
        self.base_completion = read_json(completion_path)
        completion_sha = sha256_file(completion_path)
        declared = str(self.manifest["base_pack"]["completion_sha256"])
        if completion_sha != declared or (
            expected_base_completion_sha256
            and completion_sha != expected_base_completion_sha256
        ):
            raise Recent20OverlayError("base completion digest mismatch")
        if self.base_completion.get("schema") != BASE_COMPLETION_SCHEMA:
            raise Recent20OverlayError("foreign base completion")
        schema_descriptor = base_schema_descriptor(self.base_completion)
        if canonical_sha256(schema_descriptor) != str(
            self.manifest["base_pack"]["schema_sha256"]
        ):
            raise Recent20OverlayError("base schema identity mismatch")
        self.base_roots = _resolve_pack_root(
            self.base_completion, Path(base_pack_root).expanduser().resolve()
        )
        self.pack_by_source = {
            str(row["source_sha256"]): dict(row)
            for row in self.base_completion.get("packs") or ()
        }
        self.shards = list(self.manifest.get("overlay_shards") or ())
        if verify_overlay_shards:
            for row in self.shards:
                path = _overlay_member_path(self.manifest_path, str(row["path"]))
                if path.is_symlink() or not path.is_file():
                    raise Recent20OverlayError(f"overlay shard is not a regular file: {path}")
                if (
                    path.stat().st_size != int(row["size_bytes"])
                    or sha256_file(path) != str(row["sha256"])
                ):
                    raise Recent20OverlayError(f"overlay shard mismatch: {path}")
        if verify_base_shards:
            self._verify_base_objects()

    def _verify_base_objects(self) -> None:
        """Hash each local base object exactly once before a training pass.

        The semantic-pack completion records source-host paths.  A portable
        consumer intentionally resolves by its immutable UTC-day layout under
        ``base_pack_root`` and validates the copied bytes against the recorded
        full digests rather than attempting to dereference those source paths.
        """

        expected_roles = {
            "features_f32": "features.f32",
            "decision_offsets_u64": "decision_offsets.u64",
            "selected_option_u32": "selected_option.u32",
            "decision_key_sha256": "decision_keys.sha256",
        }
        packs = list(self.base_completion.get("packs") or ())
        if not packs or len(self.base_roots) != len(packs):
            raise Recent20OverlayError("base completion pack inventory is malformed")
        for pack in packs:
            # The intentionally explicit loop below avoids trusting a source
            # path from a different host; derive the local day from the pack
            # exactly as _resolve_pack_root did.
            day = ""
            for field in ("receipt_path", "source_path"):
                candidates = [
                    part
                    for part in Path(str(pack.get(field) or "")).parts
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part)
                ]
                if candidates:
                    day = candidates[-1]
                    break
            day_root = self.base_roots.get(day)
            if not day or day_root is None or day_root.is_symlink() or not day_root.is_dir():
                raise Recent20OverlayError("base pack day root is not a real directory")
            files = dict(pack.get("files") or {})
            if set(files) != set(expected_roles):
                raise Recent20OverlayError("base pack file-role inventory drifted")
            for role, filename in expected_roles.items():
                declaration = dict(files.get(role) or {})
                path = day_root / filename
                if path.is_symlink() or not path.is_file():
                    raise Recent20OverlayError(f"base object is not a regular file: {path}")
                if path.stat().st_size != int(declaration.get("size_bytes", -1)):
                    raise Recent20OverlayError(f"base object size mismatch: {path}")
                if sha256_file(path) != str(declaration.get("sha256") or ""):
                    raise Recent20OverlayError(f"base object digest mismatch: {path}")
            receipt = day_root / "receipt.json"
            if receipt.is_symlink() or not receipt.is_file():
                raise Recent20OverlayError("base day receipt is not a regular file")
            if sha256_file(receipt) != str(pack.get("receipt_sha256") or ""):
                raise Recent20OverlayError("base day receipt digest mismatch")

    def iter_samples(self, split: str) -> Iterator[dict[str, Any]]:
        wanted = str(split).strip().lower()
        if wanted not in {"train", "validation", "evaluation"}:
            raise ValueError("split must be train, validation, or evaluation")
        for shard in self.shards:
            if str(shard.get("split")) != wanted:
                continue
            day = str(shard["utc_day"])
            source_sha = str(shard["base_source_shard_sha256"])
            pack = self.pack_by_source.get(source_sha)
            if pack is None:
                raise Recent20OverlayError("overlay source shard is absent from base pack")
            reader = _BaseDayReader(self.base_roots[day], pack)
            try:
                path = _overlay_member_path(self.manifest_path, str(shard["path"]))
                with path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as stream:
                    for line_number, line in enumerate(stream, start=1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise Recent20OverlayError(
                                f"invalid overlay row {path}:{line_number}"
                            ) from exc
                        if row.get("schema") != OVERLAY_SCHEMA:
                            raise Recent20OverlayError("overlay row schema drifted")
                        stage_features = []
                        for stage in row.get("stages") or ():
                            ref = stage["base_ref"]
                            stage_features.append(
                                reader.read_options(
                                    int(ref["option_start"]),
                                    int(ref["option_count"]),
                                )
                            )
                        yield {
                            "schema": DATASET_SAMPLE_SCHEMA,
                            "split": wanted,
                            "program": row,
                            "base_option_features_by_stage": stage_features,
                            "base_feature_width": 40,
                            "public_information_only": True,
                        }
            finally:
                reader.close()


def iter_recent20_overlay_samples_for_job(
    job: Any,
    *,
    split: str,
    verify_overlay_shards: bool = True,
) -> Iterator[dict[str, Any]]:
    """Pipeline-facing adapter for :class:`ArchetypeRTPJob`."""
    dataset = Recent20RTPDataset(
        job.recent20_rtp_overlay_manifest,
        base_pack_root=job.recent20_rtp_base_pack_root,
        expected_manifest_sha256=job.recent20_rtp_overlay_manifest_digest,
        expected_base_completion_sha256=job.recent20_rtp_base_pack_completion_digest,
        verify_overlay_shards=verify_overlay_shards,
    )
    yield from dataset.iter_samples(split)


__all__ = [
    "BASE_COMPLETION_SCHEMA",
    "BASE_SCHEMA",
    "DATASET_SAMPLE_SCHEMA",
    "MANIFEST_SCHEMA",
    "OVERLAY_SCHEMA",
    "RECEIPT_SCHEMA",
    "Recent20OverlayError",
    "Recent20RTPDataset",
    "base_schema_descriptor",
    "canonical_bytes",
    "canonical_sha256",
    "iter_recent20_overlay_samples_for_job",
    "overlay_schema_document",
    "read_json",
    "sha256_file",
]
