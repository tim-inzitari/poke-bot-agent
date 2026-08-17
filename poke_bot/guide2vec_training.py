"""Guide-agnostic training for a tiny, frozen-base Guide2Vec sidecar.

This module deliberately has no dependency on a deck guide, replay parser,
service, owner manifest, or runtime attachment path.  It consumes two already
materialized, content-addressed Torch chunks per shard:

* a guide-independent frozen latent chunk, and
* a guide-versioned label overlay with the same causal stage rows.

The row-level ``stage_key_digest`` and ``legal_options_digest`` tensors are
the join contract.  They prevent a label overlay from being silently applied
to a different observation, factorized stage, or legal-option ordering.  The
only trainable object is :class:`poke_bot.guide2vec.Guide2VecHead`.

There is intentionally no CLI or managed-service integration here.  Training
is deliberately dormant until a separate sealed activation grant has verified
the immutable guide, split, host, and output receipts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import re
import stat
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from poke_bot.guide2vec import (
    FrozenBaseIdentity,
    Guide2VecConfig,
    Guide2VecDecision,
    Guide2VecHead,
)
from poke_bot.guide2vec_activation import (
    Guide2VecTrainingGrant as _Guide2VecTrainingGrant,
)
from poke_bot.guide2vec_activation import (
    validate_activation_receipt as _validate_activation_receipt,
)

LATENT_CHUNK_SCHEMA = "poke_bot.guide2vec_frozen_latent_chunk/v1"
LABEL_CHUNK_SCHEMA = "poke_bot.guide2vec_label_chunk/v1"
CALIBRATION_SCHEMA = "poke_bot.guide2vec_training_calibration/v1"
METRICS_SCHEMA = "poke_bot.guide2vec_training_metrics/v1"
CANDIDATE_SCHEMA = "poke_bot.guide2vec_training_candidate/v1"
RECEIPT_SCHEMA = "poke_bot.guide2vec_training_receipt/v1"
TRAINING_CONFIG_SCHEMA = "poke_bot.guide2vec_training_config/v1"
FROZEN_RUNTIME_SCHEMA = "poke_bot.guide2vec_frozen_runtime/v1"
GRANT_CONSUMPTION_SCHEMA = "poke_bot.guide2vec_training_grant_consumption/v1"

STAGE_DIGEST_BYTES = 32
LATENT_D_MODEL = 96
MAX_EPOCHS = 5
DEFAULT_MINIMUM_PRECISION = 0.70
CHUNK_DTYPE_POLICY_SCHEMA = "poke_bot.guide2vec_chunk_dtype_policy/v1"
DETERMINISM_POLICY = {
    "schema": "poke_bot.guide2vec_training_determinism/v1",
    "torch_deterministic_algorithms": True,
    "warn_only": False,
    "cpu_chunk_order": "seeded_chunk_permutation_only",
    "cuda_cudnn_benchmark": False,
    "cuda_cudnn_deterministic": True,
    "cuda_tf32": False,
    "cuda_cublas_workspace_config": ":4096:8",
    "unsupported_nondeterministic_operation": "fail_closed",
}

_CALIBRATED_STATUS = "validation_calibrated"
_ABSTAIN_ALL_STATUSES = frozenset(
    {
        "abstain_all_no_validation_labels",
        "abstain_all_precision_floor_not_met",
    }
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class Guide2VecTrainingError(ValueError):
    """Raised when paired chunks or a generic Guide2Vec run are invalid."""


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Guide2VecTrainingError("value is not canonical JSON") from exc


def _canonical_json_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _normalize_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise Guide2VecTrainingError(f"{label} must be a sha256 digest string")
    digest = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest):
        digest = "sha256:" + digest
    if _SHA256_RE.fullmatch(digest) is None:
        raise Guide2VecTrainingError(f"{label} must be a canonical sha256 digest")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _head_state_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash dense CPU head weights without requiring NumPy at training time.

    The byte format intentionally matches ``guide2vec.state_dict_sha256``:
    sorted tensor headers followed by raw contiguous bytes.  ``tolist`` on a
    uint8 view is less fast than NumPy, but this happens only at the candidate
    freeze boundary and keeps the generic trainer usable in a minimal Torch
    environment.
    """

    if not isinstance(state_dict, Mapping):
        raise Guide2VecTrainingError("head state must be a mapping")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise Guide2VecTrainingError("head state must contain named tensors")
        if tensor.layout != torch.strided:
            raise Guide2VecTrainingError("head state tensors must be dense")
        value = tensor.detach().cpu().contiguous()
        digest.update(
            json.dumps(
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ContentChunkRef:
    """One immutable Torch chunk referenced by its complete content digest."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "sha256", _normalize_sha256(self.sha256, label="chunk sha256"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ContentChunkRef:
        if not isinstance(value, Mapping):
            raise Guide2VecTrainingError("chunk reference must be a mapping")
        expected = {"path", "sha256"}
        missing = expected.difference(value)
        unknown = set(value).difference(expected)
        if missing or unknown:
            raise Guide2VecTrainingError(
                "chunk reference fields changed: "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        path = value["path"]
        if not isinstance(path, (str, Path)):
            raise Guide2VecTrainingError("chunk reference path must be a path string")
        return cls(path=Path(path), sha256=value["sha256"])  # type: ignore[arg-type]

    def digest_record(self) -> dict[str, str]:
        """Return the portable part of a reference used in immutable receipts."""

        return {"sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class JoinedChunkRef:
    """The explicit frozen-latent and guide-label pair for one chunk."""

    latent: ContentChunkRef
    labels: ContentChunkRef

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> JoinedChunkRef:
        if not isinstance(value, Mapping):
            raise Guide2VecTrainingError("joined chunk reference must be a mapping")
        nested_expected = {"latent", "labels"}
        if set(value) == nested_expected:
            latent = value["latent"]
            labels = value["labels"]
            if not isinstance(latent, Mapping) or not isinstance(labels, Mapping):
                raise Guide2VecTrainingError("joined chunk references must be mappings")
            return cls(
                latent=ContentChunkRef.from_mapping(latent),
                labels=ContentChunkRef.from_mapping(labels),
            )
        flat_expected = {
            "latent_path",
            "latent_sha256",
            "label_path",
            "label_sha256",
        }
        missing = flat_expected.difference(value)
        unknown = set(value).difference(flat_expected)
        if missing or unknown:
            raise Guide2VecTrainingError(
                "joined chunk reference fields changed: "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        return cls(
            latent=ContentChunkRef(
                path=Path(value["latent_path"]),  # type: ignore[arg-type]
                sha256=value["latent_sha256"],  # type: ignore[arg-type]
            ),
            labels=ContentChunkRef(
                path=Path(value["label_path"]),  # type: ignore[arg-type]
                sha256=value["label_sha256"],  # type: ignore[arg-type]
            ),
        )

    @property
    def pair_sha256(self) -> str:
        return _canonical_json_sha256(
            {
                "schema": "poke_bot.guide2vec_joined_chunk_ref/v1",
                "latent_sha256": self.latent.sha256,
                "label_sha256": self.labels.sha256,
            }
        )

    def digest_record(self) -> dict[str, str]:
        return {
            "latent_sha256": self.latent.sha256,
            "label_sha256": self.labels.sha256,
            "pair_sha256": self.pair_sha256,
        }


# Both aliases make the feature/label pairing terminology explicit to callers.
ChunkRef = ContentChunkRef
PairedChunkRef = JoinedChunkRef


@dataclass(frozen=True, slots=True)
class LatentChunk:
    """Validated guide-independent causal latents for a sequence of stages."""

    ref: ContentChunkRef
    stage_key_digest: Tensor
    legal_options_digest: Tensor
    state_vec: Tensor
    option_hidden: Tensor
    base_logits: Tensor
    option_offsets: Tensor

    @property
    def rows(self) -> int:
        return int(self.state_vec.shape[0])

    @property
    def option_rows(self) -> int:
        return int(self.option_hidden.shape[0])

    @property
    def counts(self) -> Tensor:
        return self.option_offsets[1:] - self.option_offsets[:-1]


@dataclass(frozen=True, slots=True)
class LabelChunk:
    """Validated guide-specific target overlay for the same exact stages."""

    ref: ContentChunkRef
    stage_key_digest: Tensor
    legal_options_digest: Tensor
    guide_target_index: Tensor
    guide_confidence: Tensor

    @property
    def rows(self) -> int:
        return int(self.guide_target_index.shape[0])


@dataclass(frozen=True, slots=True)
class JoinedChunk:
    """A verified latent/label pair whose row identities are exactly equal."""

    ref: JoinedChunkRef
    latent: LatentChunk
    labels: LabelChunk

    @property
    def rows(self) -> int:
        return self.latent.rows


@dataclass(frozen=True, slots=True)
class Guide2VecTrainingConfig:
    """Deterministic tiny-head fitting settings, capped at five epochs."""

    head_config: Guide2VecConfig = field(default_factory=Guide2VecConfig)
    epochs: int = MAX_EPOCHS
    batch_rows: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    coverage_weight: float = 0.25
    max_gradient_norm: float = 1.0
    minimum_precision: float = DEFAULT_MINIMUM_PRECISION
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.head_config, Guide2VecConfig):
            raise Guide2VecTrainingError("head_config must be a Guide2VecConfig")
        if self.head_config.d_model != LATENT_D_MODEL:
            raise Guide2VecTrainingError("Guide2Vec latent ABI requires d_model=96")
        if type(self.epochs) is not int or not 1 <= self.epochs <= MAX_EPOCHS:
            raise Guide2VecTrainingError(f"epochs must be an integer in [1, {MAX_EPOCHS}]")
        if type(self.batch_rows) is not int or self.batch_rows <= 0:
            raise Guide2VecTrainingError("batch_rows must be a positive exact integer")
        if type(self.seed) is not int:
            raise Guide2VecTrainingError("seed must be an exact integer")
        for field_name, lower, upper, lower_inclusive in (
            ("learning_rate", 0.0, math.inf, False),
            ("weight_decay", 0.0, math.inf, True),
            ("coverage_weight", 0.0, math.inf, True),
            ("max_gradient_norm", 0.0, math.inf, False),
            ("minimum_precision", 0.0, 1.0, True),
        ):
            value = getattr(self, field_name)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise Guide2VecTrainingError(f"{field_name} must be finite")
            numeric = float(value)
            if (numeric < lower if lower_inclusive else numeric <= lower) or numeric > upper:
                bracket = "[" if lower_inclusive else "("
                raise Guide2VecTrainingError(
                    f"{field_name} must be in {bracket}{lower}, {upper}]"
                )
        try:
            normalized_device = str(torch.device(self.device))
        except (TypeError, RuntimeError) as exc:
            raise Guide2VecTrainingError("device is not a valid torch device") from exc
        object.__setattr__(self, "device", normalized_device)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": TRAINING_CONFIG_SCHEMA,
            "head_config": self.head_config.as_dict(),
            "epochs": self.epochs,
            "batch_rows": self.batch_rows,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "coverage_weight": float(self.coverage_weight),
            "max_gradient_norm": float(self.max_gradient_norm),
            "minimum_precision": float(self.minimum_precision),
            "seed": self.seed,
            "device": self.device,
            "determinism_policy": dict(DETERMINISM_POLICY),
        }


@contextmanager
def _deterministic_torch_scope(device: torch.device) -> Iterator[dict[str, object]]:
    """Temporarily force deterministic Torch/CUDA execution or fail closed.

    The environment is process-local and restored after the isolated fit.  For
    CUDA, cuBLAS workspace selection has to occur before CUDA initialization;
    an already initialized process without the required setting is rejected
    rather than pretending a later flag can make earlier kernels deterministic.
    """

    if not isinstance(device, torch.device):
        raise Guide2VecTrainingError("determinism scope requires a torch device")
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    previous_cudnn_benchmark = (
        bool(cudnn_backend.benchmark) if cudnn_backend is not None else None
    )
    previous_cudnn_deterministic = (
        bool(cudnn_backend.deterministic) if cudnn_backend is not None else None
    )
    previous_cudnn_tf32 = (
        bool(cudnn_backend.allow_tf32)
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32")
        else None
    )
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    previous_matmul_tf32 = (
        bool(matmul_backend.allow_tf32)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32")
        else None
    )
    cublas_before = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    cublas_changed = False
    try:
        if device.type == "cuda":
            if cublas_before not in {None, ":4096:8"}:
                raise Guide2VecTrainingError(
                    "CUDA deterministic training requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
                )
            if cublas_before is None:
                if torch.cuda.is_initialized():
                    raise Guide2VecTrainingError(
                        "CUDA was initialized before deterministic cuBLAS workspace setup"
                    )
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                cublas_changed = True
            if cudnn_backend is not None:
                cudnn_backend.benchmark = False
                cudnn_backend.deterministic = True
                if hasattr(cudnn_backend, "allow_tf32"):
                    cudnn_backend.allow_tf32 = False
            if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
                matmul_backend.allow_tf32 = False
        # ``warn_only=False`` turns a currently unsupported nondeterministic
        # op into a hard error rather than silently accepting an unstable fit.
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield {
            **DETERMINISM_POLICY,
            "device": str(device),
            "cublas_workspace_preinitialized": cublas_before == ":4096:8",
        }
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        if cudnn_backend is not None:
            if previous_cudnn_benchmark is not None:
                cudnn_backend.benchmark = previous_cudnn_benchmark
            if previous_cudnn_deterministic is not None:
                cudnn_backend.deterministic = previous_cudnn_deterministic
            if previous_cudnn_tf32 is not None and hasattr(cudnn_backend, "allow_tf32"):
                cudnn_backend.allow_tf32 = previous_cudnn_tf32
        if previous_matmul_tf32 is not None and matmul_backend is not None and hasattr(
            matmul_backend, "allow_tf32"
        ):
            matmul_backend.allow_tf32 = previous_matmul_tf32
        if cublas_changed:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


TrainingConfig = Guide2VecTrainingConfig


@dataclass(frozen=True, slots=True)
class FrozenGuide2VecRuntime:
    """The sole generic runtime gate for a frozen Guide2Vec candidate.

    ``Guide2VecConfig.min_eligibility == 1.0`` is not an abstain-all switch:
    finite float32 logits can sigmoid-round to exactly one.  The explicit
    ``always_abstain`` bit therefore precedes *every* head rerank call.  It is
    true only for a validation calibration that truthfully selected no row.
    """

    guide2vec_config: Guide2VecConfig
    calibration_status: str
    calibrated_threshold: float
    always_abstain: bool

    def __post_init__(self) -> None:
        if not isinstance(self.guide2vec_config, Guide2VecConfig):
            raise Guide2VecTrainingError("runtime requires a Guide2VecConfig")
        if not isinstance(self.calibration_status, str):
            raise Guide2VecTrainingError("runtime calibration_status must be a string")
        if type(self.calibrated_threshold) not in {int, float} or not math.isfinite(
            float(self.calibrated_threshold)
        ):
            raise Guide2VecTrainingError("runtime calibrated_threshold must be finite")
        threshold = float(self.calibrated_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise Guide2VecTrainingError("runtime calibrated_threshold leaves [0, 1]")
        if type(self.always_abstain) is not bool:
            raise Guide2VecTrainingError("runtime always_abstain must be an exact boolean")
        if float(self.guide2vec_config.min_eligibility) != threshold:
            raise Guide2VecTrainingError(
                "runtime threshold must exactly match Guide2VecConfig.min_eligibility"
            )
        if self.calibration_status == _CALIBRATED_STATUS:
            if self.always_abstain:
                raise Guide2VecTrainingError(
                    "validation-calibrated runtime cannot be always-abstain"
                )
        elif self.calibration_status in _ABSTAIN_ALL_STATUSES:
            if not self.always_abstain or threshold != 1.0:
                raise Guide2VecTrainingError(
                    "abstain-all calibration requires exact always_abstain and threshold=1.0"
                )
        else:
            raise Guide2VecTrainingError("runtime calibration_status is unknown")

    @property
    def runtime_sha256(self) -> str:
        return _canonical_json_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": FROZEN_RUNTIME_SCHEMA,
            "guide2vec_config": self.guide2vec_config.as_dict(),
            "calibration_status": self.calibration_status,
            "calibrated_threshold": float(self.calibrated_threshold),
            "always_abstain": self.always_abstain,
            "application_gate": "always_abstain_precedes_guide2vec_head_rerank/v1",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FrozenGuide2VecRuntime:
        if not isinstance(value, Mapping):
            raise Guide2VecTrainingError("frozen runtime must be a mapping")
        expected = {
            "schema",
            "guide2vec_config",
            "calibration_status",
            "calibrated_threshold",
            "always_abstain",
            "application_gate",
        }
        _require_exact_keys(value, expected=expected, label="frozen runtime")
        if value["schema"] != FROZEN_RUNTIME_SCHEMA:
            raise Guide2VecTrainingError("frozen runtime schema changed")
        if value["application_gate"] != "always_abstain_precedes_guide2vec_head_rerank/v1":
            raise Guide2VecTrainingError("frozen runtime lacks the always-abstain gate")
        config_raw = value["guide2vec_config"]
        if not isinstance(config_raw, Mapping):
            raise Guide2VecTrainingError("frozen runtime Guide2Vec config is invalid")
        return cls(
            guide2vec_config=Guide2VecConfig.from_mapping(config_raw),
            calibration_status=value["calibration_status"],  # type: ignore[arg-type]
            calibrated_threshold=value["calibrated_threshold"],  # type: ignore[arg-type]
            always_abstain=value["always_abstain"],  # type: ignore[arg-type]
        )


def rerank_with_frozen_runtime(
    head: Guide2VecHead,
    runtime: FrozenGuide2VecRuntime,
    state_vec: Tensor,
    option_hidden: Tensor,
    base_logits: Tensor,
    n_options: int | Sequence[int] | Tensor | None = None,
    *,
    expected_base_identity: FrozenBaseIdentity,
    observed_base_identity: FrozenBaseIdentity | None,
) -> Guide2VecDecision:
    """Use the generic frozen runtime gate before delegating to the head.

    The abstain branch intentionally invokes ``Guide2VecHead.rerank`` with no
    valid identity, which follows its existing exact direct-policy fallback
    without executing ``forward``.  Therefore even a finite, saturated
    eligibility logit cannot turn an abstain-all candidate into an applied
    bonus.
    """

    if not isinstance(head, Guide2VecHead):
        raise Guide2VecTrainingError("runtime head must be a Guide2VecHead")
    if not isinstance(runtime, FrozenGuide2VecRuntime):
        raise Guide2VecTrainingError("runtime must be a FrozenGuide2VecRuntime")
    if not isinstance(expected_base_identity, FrozenBaseIdentity):
        raise Guide2VecTrainingError("expected_base_identity must be frozen")
    if head.config != runtime.guide2vec_config:
        raise Guide2VecTrainingError("head config does not match frozen runtime config")
    if runtime.always_abstain:
        fallback = head.rerank(
            state_vec,
            option_hidden,
            base_logits,
            n_options=n_options,
            expected_base_identity=None,
            observed_base_identity=None,
        )
        if bool(fallback.applied.any().item()) or not torch.equal(
            fallback.adjusted_logits, base_logits
        ):
            raise Guide2VecTrainingError("always-abstain runtime did not preserve base logits")
        return replace(
            fallback,
            reasons=("configured_always_abstain",) * len(fallback.reasons),
        )
    return head.rerank(
        state_vec,
        option_hidden,
        base_logits,
        n_options=n_options,
        expected_base_identity=expected_base_identity,
        observed_base_identity=observed_base_identity,
    )


@dataclass(frozen=True, slots=True, init=False)
class FrozenGuide2VecCandidate:
    """Generic candidate loader result that exposes only the gated rerank path."""

    __head: Guide2VecHead = field(repr=False, compare=False)
    runtime: FrozenGuide2VecRuntime
    base_identity: FrozenBaseIdentity
    head_state_sha256: str

    def __init__(
        self,
        *,
        head: Guide2VecHead,
        runtime: FrozenGuide2VecRuntime,
        base_identity: FrozenBaseIdentity,
        head_state_sha256: str,
    ) -> None:
        """Construct an immutable wrapper; raw head access stays private."""

        if not isinstance(head, Guide2VecHead):
            raise Guide2VecTrainingError("frozen candidate head must be a Guide2VecHead")
        if not isinstance(runtime, FrozenGuide2VecRuntime):
            raise Guide2VecTrainingError("frozen candidate runtime is invalid")
        if not isinstance(base_identity, FrozenBaseIdentity):
            raise Guide2VecTrainingError("frozen candidate base identity is invalid")
        if head.config != runtime.guide2vec_config:
            raise Guide2VecTrainingError("frozen candidate head/runtime config mismatch")
        object.__setattr__(self, "_FrozenGuide2VecCandidate__head", head)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "base_identity", base_identity)
        object.__setattr__(
            self,
            "head_state_sha256",
            _normalize_sha256(head_state_sha256, label="frozen candidate head_state_sha256"),
        )

    def rerank(
        self,
        state_vec: Tensor,
        option_hidden: Tensor,
        base_logits: Tensor,
        n_options: int | Sequence[int] | Tensor | None = None,
        *,
        observed_base_identity: FrozenBaseIdentity | None,
    ) -> Guide2VecDecision:
        return rerank_with_frozen_runtime(
            self.__head,
            self.runtime,
            state_vec,
            option_hidden,
            base_logits,
            n_options=n_options,
            expected_base_identity=self.base_identity,
            observed_base_identity=observed_base_identity,
        )


@dataclass
class _MetricAccumulator:
    rank_nll_sum: float = 0.0
    rank_weight_sum: float = 0.0
    rank_correct: int = 0
    rank_rows: int = 0
    coverage_bce_sum: float = 0.0
    coverage_rows: int = 0
    coverage_correct: int = 0
    total_loss_sum: float = 0.0
    batches: int = 0

    def add(self, other: _MetricAccumulator) -> None:
        self.rank_nll_sum += other.rank_nll_sum
        self.rank_weight_sum += other.rank_weight_sum
        self.rank_correct += other.rank_correct
        self.rank_rows += other.rank_rows
        self.coverage_bce_sum += other.coverage_bce_sum
        self.coverage_rows += other.coverage_rows
        self.coverage_correct += other.coverage_correct
        self.total_loss_sum += other.total_loss_sum
        self.batches += other.batches

    def as_dict(self) -> dict[str, object]:
        rank_nll: float | None
        if self.rank_weight_sum > 0.0:
            rank_nll = self.rank_nll_sum / self.rank_weight_sum
        else:
            rank_nll = None
        return {
            "schema": METRICS_SCHEMA,
            "rank_nll": rank_nll,
            "rank_accuracy": self.rank_correct / max(self.rank_rows, 1),
            "rank_rows": self.rank_rows,
            "rank_weight_sum": self.rank_weight_sum,
            "coverage_bce": self.coverage_bce_sum / max(self.coverage_rows, 1),
            "coverage_accuracy": self.coverage_correct / max(self.coverage_rows, 1),
            "coverage_rows": self.coverage_rows,
            "total_loss": self.total_loss_sum / max(self.batches, 1),
            "batches": self.batches,
        }


def _safe_torch_load(ref: ContentChunkRef) -> Mapping[str, object]:
    """Hash a chunk first, then use Torch's tensor-only unpickler.

    A permissive pickle fallback would make a content-addressed training input
    executable.  Old Torch installations that lack ``weights_only`` therefore
    fail closed instead.
    """

    path = ref.path.expanduser()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise Guide2VecTrainingError(f"chunk is not a regular file: {path}")
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
            observed = "sha256:" + digest.hexdigest()
            if not hmac.compare_digest(observed, ref.sha256):
                raise Guide2VecTrainingError(
                    f"chunk digest mismatch for {path}: expected={ref.sha256} observed={observed}"
                )
            # Hash and deserialize the same open object.  A path replacement
            # after the hash cannot change these bytes.  The fstat comparison
            # below also fails closed on an in-place modification during load.
            handle.seek(0)
            try:
                value = torch.load(handle, map_location="cpu", weights_only=True)
            except TypeError as exc:
                raise Guide2VecTrainingError(
                    "Torch weights_only safe loading is required for Guide2Vec chunks"
                ) from exc
            except Exception as exc:  # pragma: no cover - Torch error spelling varies by version.
                raise Guide2VecTrainingError(f"safe Torch load failed for {path}") from exc
            after = os.fstat(handle.fileno())
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if identity_after != identity_before:
                raise Guide2VecTrainingError("chunk changed while being safely loaded")
    except Guide2VecTrainingError:
        raise
    except OSError as exc:
        raise Guide2VecTrainingError(f"cannot open chunk: {path}") from exc
    if not isinstance(value, Mapping):
        raise Guide2VecTrainingError("chunk payload must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, object], *, expected: set[str], label: str
) -> None:
    missing = expected.difference(value)
    unknown = set(value).difference(expected)
    if missing or unknown:
        raise Guide2VecTrainingError(
            f"{label} fields changed: missing={sorted(missing)} unknown={sorted(unknown)}"
        )


def _cpu_dense_tensor(value: object, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise Guide2VecTrainingError(f"{label} must be a torch tensor")
    if value.layout != torch.strided:
        raise Guide2VecTrainingError(f"{label} must be a dense strided tensor")
    if value.device.type != "cpu":
        raise Guide2VecTrainingError(f"{label} must load on CPU")
    return value.detach().contiguous()


def _digest_matrix(value: object, *, label: str, rows: int | None = None) -> Tensor:
    tensor = _cpu_dense_tensor(value, label=label)
    if tensor.dtype != torch.uint8:
        raise Guide2VecTrainingError(f"{label} must use torch.uint8")
    if tensor.ndim != 2 or tensor.shape[1] != STAGE_DIGEST_BYTES:
        raise Guide2VecTrainingError(
            f"{label} must have shape [rows, {STAGE_DIGEST_BYTES}]")
    if rows is not None and tensor.shape[0] != rows:
        raise Guide2VecTrainingError(f"{label} row count does not match latent rows")
    return tensor


def _floating_tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> Tensor:
    tensor = _cpu_dense_tensor(value, label=label)
    if not tensor.is_floating_point() or tensor.is_complex():
        raise Guide2VecTrainingError(f"{label} must be a real floating point tensor")
    if shape is not None and tuple(tensor.shape) != shape:
        raise Guide2VecTrainingError(f"{label} has an invalid shape")
    if ndim is not None and tensor.ndim != ndim:
        raise Guide2VecTrainingError(f"{label} has an invalid rank")
    if not bool(torch.isfinite(tensor).all().item()):
        raise Guide2VecTrainingError(f"{label} contains nonfinite values")
    return tensor


def _integer_vector(value: object, *, label: str, rows: int | None = None) -> Tensor:
    tensor = _cpu_dense_tensor(value, label=label)
    if tensor.dtype not in _INTEGER_DTYPES or tensor.dtype == torch.bool:
        raise Guide2VecTrainingError(f"{label} must be an integer tensor")
    if tensor.ndim != 1:
        raise Guide2VecTrainingError(f"{label} must be one-dimensional")
    if rows is not None and tensor.numel() != rows:
        raise Guide2VecTrainingError(f"{label} row count does not match latent rows")
    # Preserve the content dtype for the partition/manifest ABI check.  Batch
    # construction explicitly widens indexing/target tensors to int64 only at
    # use time, after the immutable input policy has been verified.
    return tensor


def load_latent_chunk(ref: ContentChunkRef | Mapping[str, object]) -> LatentChunk:
    """Safely load and strictly validate one frozen, guide-independent chunk."""

    normalized = _normalize_content_ref(ref)
    payload = _safe_torch_load(normalized)
    _require_exact_keys(
        payload,
        expected={
            "schema",
            "stage_key_digest",
            "legal_options_digest",
            "state_vec",
            "option_hidden",
            "base_logits",
            "option_offsets",
        },
        label="latent chunk",
    )
    if payload["schema"] != LATENT_CHUNK_SCHEMA:
        raise Guide2VecTrainingError("latent chunk schema changed")
    state_vec = _floating_tensor(payload["state_vec"], label="state_vec", ndim=2)
    rows, width = state_vec.shape
    if rows <= 0 or width != LATENT_D_MODEL:
        raise Guide2VecTrainingError("state_vec must have shape [rows, 96] with rows > 0")
    stage_key_digest = _digest_matrix(payload["stage_key_digest"], label="stage_key_digest", rows=rows)
    legal_options_digest = _digest_matrix(
        payload["legal_options_digest"], label="legal_options_digest", rows=rows
    )
    option_hidden = _floating_tensor(payload["option_hidden"], label="option_hidden", ndim=2)
    option_rows, option_width = option_hidden.shape
    if option_rows <= 0 or option_width != LATENT_D_MODEL:
        raise Guide2VecTrainingError(
            "option_hidden must have shape [option_rows, 96] with option_rows > 0"
        )
    base_logits = _floating_tensor(payload["base_logits"], label="base_logits", ndim=1)
    if base_logits.numel() != option_rows:
        raise Guide2VecTrainingError("base_logits must have one entry per option_hidden row")
    if not (state_vec.dtype == option_hidden.dtype == base_logits.dtype):
        raise Guide2VecTrainingError(
            "state_vec, option_hidden, and base_logits must share one declared dtype policy"
        )
    offsets = _integer_vector(payload["option_offsets"], label="option_offsets")
    if offsets.numel() != rows + 1:
        raise Guide2VecTrainingError("option_offsets must have shape [rows + 1]")
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != option_rows:
        raise Guide2VecTrainingError("option_offsets must start at zero and end at option_rows")
    counts = offsets[1:] - offsets[:-1]
    if bool((counts <= 0).any().item()):
        raise Guide2VecTrainingError("every latent row must have at least one legal option")
    return LatentChunk(
        ref=normalized,
        stage_key_digest=stage_key_digest,
        legal_options_digest=legal_options_digest,
        state_vec=state_vec,
        option_hidden=option_hidden,
        base_logits=base_logits,
        option_offsets=offsets,
    )


def load_label_chunk(ref: ContentChunkRef | Mapping[str, object]) -> LabelChunk:
    """Safely load and strictly validate one guide-versioned label overlay."""

    normalized = _normalize_content_ref(ref)
    payload = _safe_torch_load(normalized)
    _require_exact_keys(
        payload,
        expected={
            "schema",
            "stage_key_digest",
            "legal_options_digest",
            "guide_target_index",
            "guide_confidence",
        },
        label="label chunk",
    )
    if payload["schema"] != LABEL_CHUNK_SCHEMA:
        raise Guide2VecTrainingError("label chunk schema changed")
    target = _integer_vector(payload["guide_target_index"], label="guide_target_index")
    rows = int(target.numel())
    if rows <= 0:
        raise Guide2VecTrainingError("label chunk must contain at least one row")
    confidence = _floating_tensor(
        payload["guide_confidence"], label="guide_confidence", shape=(rows,)
    )
    if bool(((confidence < 0.0) | (confidence > 1.0)).any().item()):
        raise Guide2VecTrainingError("guide_confidence must be within [0, 1]")
    return LabelChunk(
        ref=normalized,
        stage_key_digest=_digest_matrix(payload["stage_key_digest"], label="stage_key_digest", rows=rows),
        legal_options_digest=_digest_matrix(
            payload["legal_options_digest"], label="legal_options_digest", rows=rows
        ),
        guide_target_index=target,
        guide_confidence=confidence,
    )


def load_joined_chunk(ref: JoinedChunkRef | Mapping[str, object]) -> JoinedChunk:
    """Load a pair and fail closed unless every row identity joins exactly."""

    normalized = _normalize_joined_ref(ref)
    latent = load_latent_chunk(normalized.latent)
    labels = load_label_chunk(normalized.labels)
    if latent.rows != labels.rows:
        raise Guide2VecTrainingError("latent and label chunks have different row counts")
    if not torch.equal(latent.stage_key_digest, labels.stage_key_digest):
        raise Guide2VecTrainingError("stage_key_digest does not exactly join latent and labels")
    if not torch.equal(latent.legal_options_digest, labels.legal_options_digest):
        raise Guide2VecTrainingError("legal_options_digest does not exactly join latent and labels")
    counts = latent.counts
    target = labels.guide_target_index
    confidence = labels.guide_confidence
    if bool((target < -1).any().item()):
        raise Guide2VecTrainingError("guide_target_index must be -1 or a legal option index")
    labeled = target >= 0
    if bool((target[labeled] >= counts[labeled]).any().item()):
        raise Guide2VecTrainingError("guide_target_index escaped its bound legal option order")
    singleton = counts == 1
    if bool((labeled & singleton).any().item()):
        raise Guide2VecTrainingError("singleton stages must be guide-masked")
    if bool((~labeled & (confidence != 0.0)).any().item()):
        raise Guide2VecTrainingError("masked guide rows must have zero confidence")
    if bool((labeled & (confidence <= 0.0)).any().item()):
        raise Guide2VecTrainingError("labeled guide rows require positive confidence")
    return JoinedChunk(ref=normalized, latent=latent, labels=labels)


def _normalize_content_ref(
    value: ContentChunkRef | Mapping[str, object],
) -> ContentChunkRef:
    if isinstance(value, ContentChunkRef):
        return value
    if isinstance(value, Mapping):
        return ContentChunkRef.from_mapping(value)
    raise Guide2VecTrainingError("chunk reference must be a ContentChunkRef or mapping")


def _normalize_joined_ref(
    value: JoinedChunkRef | Mapping[str, object],
) -> JoinedChunkRef:
    if isinstance(value, JoinedChunkRef):
        return value
    if isinstance(value, Mapping):
        return JoinedChunkRef.from_mapping(value)
    raise Guide2VecTrainingError("joined chunk reference must be a JoinedChunkRef or mapping")


def _normalize_joined_refs(
    values: Sequence[JoinedChunkRef | Mapping[str, object]], *, label: str
) -> tuple[JoinedChunkRef, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise Guide2VecTrainingError(f"{label} must be a sequence of joined chunk references")
    refs = tuple(_normalize_joined_ref(value) for value in values)
    if not refs:
        raise Guide2VecTrainingError(f"{label} must not be empty")
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        key = (ref.latent.sha256, ref.labels.sha256)
        if key in seen:
            raise Guide2VecTrainingError(f"{label} repeats one immutable chunk pair")
        seen.add(key)
    return refs


def _assert_partition_chunk_digests_disjoint(
    partitions: Mapping[str, Sequence[JoinedChunkRef]],
) -> None:
    """Reject any latent, label, or exact pair reuse across partitions pre-I/O."""

    seen_latent: dict[str, str] = {}
    seen_labels: dict[str, str] = {}
    seen_pairs: dict[str, str] = {}
    for partition, refs in partitions.items():
        for ref in refs:
            for kind, digest, seen in (
                ("latent", ref.latent.sha256, seen_latent),
                ("label", ref.labels.sha256, seen_labels),
                ("paired", ref.pair_sha256, seen_pairs),
            ):
                previous = seen.setdefault(digest, partition)
                if previous != partition:
                    raise Guide2VecTrainingError(
                        f"{kind} chunk digest is reused across partitions: "
                        f"{previous} and {partition}"
                    )


def _stage_key_set_sha256(keys: set[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"poke_bot.guide2vec_stage_key_set/v1\0")
    for key in sorted(keys):
        if len(key) != STAGE_DIGEST_BYTES:
            raise Guide2VecTrainingError("stage-key set contains an invalid digest width")
        digest.update(key)
    return "sha256:" + digest.hexdigest()


def _collect_partition_identity(
    refs: Sequence[JoinedChunkRef], *, partition: str
) -> tuple[set[bytes], dict[str, object]]:
    """Verify unique stage identities and one latent dtype policy for a partition."""

    keys: set[bytes] = set()
    dtype_policy: dict[str, str] | None = None
    rows = 0
    for ref in refs:
        chunk = load_joined_chunk(ref)
        current_policy = {
            "state_vec": str(chunk.latent.state_vec.dtype),
            "option_hidden": str(chunk.latent.option_hidden.dtype),
            "base_logits": str(chunk.latent.base_logits.dtype),
            "option_offsets": str(chunk.latent.option_offsets.dtype),
            "guide_target_index": str(chunk.labels.guide_target_index.dtype),
            "guide_confidence": str(chunk.labels.guide_confidence.dtype),
        }
        if dtype_policy is None:
            dtype_policy = current_policy
        elif dtype_policy != current_policy:
            raise Guide2VecTrainingError(
                f"{partition} partition violates one declared latent/label dtype policy"
            )
        for row in chunk.latent.stage_key_digest:
            key = bytes(row.tolist())
            if key in keys:
                raise Guide2VecTrainingError(
                    f"{partition} partition repeats a causal stage_key_digest"
                )
            keys.add(key)
        rows += chunk.rows
    if not keys or dtype_policy is None:
        raise Guide2VecTrainingError(f"{partition} partition has no stage identities")
    policy_sha256 = _canonical_json_sha256(
        {"schema": CHUNK_DTYPE_POLICY_SCHEMA, **dtype_policy}
    )
    return keys, {
        "rows": rows,
        "stage_key_set_sha256": _stage_key_set_sha256(keys),
        "dtype_policy": dtype_policy,
        "dtype_policy_sha256": policy_sha256,
    }


def _assert_stage_sets_disjoint(
    first: set[bytes],
    second: set[bytes],
    *,
    first_label: str,
    second_label: str,
) -> None:
    if first.intersection(second):
        raise Guide2VecTrainingError(
            f"causal stage_key_digest overlaps {first_label} and {second_label} partitions"
        )


def _assert_partition_dtype_policy_matches_grant(
    partition_identity: Mapping[str, object],
    *,
    partition: str,
    activation_grant: _Guide2VecTrainingGrant,
) -> None:
    """Require observed chunk tensor dtypes to match the sealed manifest ABI."""

    observed = partition_identity.get("dtype_policy_sha256")
    expected = activation_grant.compatibility.get("dtype_policy_sha256")
    if not isinstance(observed, str) or not isinstance(expected, str):
        raise Guide2VecTrainingError(
            "activation grant or observed partition lacks a canonical dtype policy identity"
        )
    if observed != expected:
        raise Guide2VecTrainingError(
            f"{partition} partition dtype policy does not match the sealed activation grant"
        )


def _batch_from_joined(
    chunk: JoinedChunk,
    *,
    start: int,
    end: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    if not 0 <= start < end <= chunk.rows:
        raise Guide2VecTrainingError("invalid joined chunk batch range")
    offsets_cpu = chunk.latent.option_offsets[start : end + 1]
    option_start = int(offsets_cpu[0].item())
    option_end = int(offsets_cpu[-1].item())
    local_offsets = offsets_cpu - option_start
    counts_cpu = local_offsets[1:] - local_offsets[:-1]
    rows = end - start
    if int(counts_cpu.numel()) != rows or bool((counts_cpu <= 0).any().item()):
        raise Guide2VecTrainingError("ragged option offsets are not batch-aligned")
    total_options = option_end - option_start
    if total_options != int(counts_cpu.sum().item()):
        raise Guide2VecTrainingError("ragged option offsets have an invalid option span")
    maximum = int(counts_cpu.max().item())
    state = chunk.latent.state_vec[start:end].to(device=device, dtype=dtype)
    hidden = chunk.latent.option_hidden[option_start:option_end].to(device=device, dtype=dtype)
    logits = chunk.latent.base_logits[option_start:option_end].to(device=device, dtype=dtype)
    counts = counts_cpu.to(device=device, dtype=torch.long)
    padded_hidden = torch.zeros((rows, maximum, LATENT_D_MODEL), device=device, dtype=dtype)
    padded_logits = torch.full((rows, maximum), float("-inf"), device=device, dtype=dtype)
    row_ids = torch.repeat_interleave(torch.arange(rows, device=device), counts)
    starts = torch.repeat_interleave(local_offsets[:-1].to(device=device), counts)
    columns = torch.arange(total_options, device=device) - starts
    padded_hidden[row_ids, columns] = hidden
    padded_logits[row_ids, columns] = logits
    return {
        "state_vec": state,
        "option_hidden": padded_hidden,
        "base_logits": padded_logits,
        "n_options": counts,
        "guide_target_index": chunk.labels.guide_target_index[start:end].to(
            device=device, dtype=torch.long
        ),
        "guide_confidence": chunk.labels.guide_confidence[start:end].to(
            device=device, dtype=dtype
        ),
    }


def iter_joined_batches(
    chunk_refs: Sequence[JoinedChunkRef | Mapping[str, object]],
    *,
    batch_rows: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    shuffle_chunks: bool = False,
    seed: int = 0,
) -> Iterator[dict[str, Tensor]]:
    """Yield deterministic padded batches while preserving ragged legal order."""

    if type(batch_rows) is not int or batch_rows <= 0:
        raise Guide2VecTrainingError("batch_rows must be a positive exact integer")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise Guide2VecTrainingError("batch dtype must be a real floating torch dtype")
    refs = list(_normalize_joined_refs(chunk_refs, label="chunk_refs"))
    if shuffle_chunks:
        random.Random(int(seed)).shuffle(refs)
    target_device = torch.device(device)
    for ref in refs:
        # A content digest is verified before *each* safe load.  Apart from
        # keeping long runs honest, this makes post-selection heldout loading
        # an observable hard boundary for callers and tests.
        chunk = load_joined_chunk(ref)
        for start in range(0, chunk.rows, batch_rows):
            yield _batch_from_joined(
                chunk,
                start=start,
                end=min(chunk.rows, start + batch_rows),
                device=target_device,
                dtype=dtype,
            )


def _loss_and_metrics(
    guide_scores: Tensor,
    eligibility_logits: Tensor,
    target: Tensor,
    confidence: Tensor,
    counts: Tensor,
    *,
    coverage_weight: float,
) -> tuple[Tensor, _MetricAccumulator, Tensor, Tensor]:
    if guide_scores.ndim != 2 or eligibility_logits.ndim not in {1, 2}:
        raise Guide2VecTrainingError("Guide2Vec head returned an invalid output shape")
    if eligibility_logits.ndim == 2:
        if eligibility_logits.shape[1] != 1:
            raise Guide2VecTrainingError("eligibility output must be scalar per stage")
        eligibility_logits = eligibility_logits.squeeze(1)
    rows, maximum = guide_scores.shape
    if (
        eligibility_logits.shape != (rows,)
        or target.shape != (rows,)
        or confidence.shape != (rows,)
        or counts.shape != (rows,)
    ):
        raise Guide2VecTrainingError("Guide2Vec batch tensors are not row-aligned")
    legal = torch.arange(maximum, device=guide_scores.device).unsqueeze(0) < counts.unsqueeze(1)
    if bool((counts <= 0).any().item()):
        raise Guide2VecTrainingError("Guide2Vec batch contains an empty legal stage")
    masked_scores = guide_scores.masked_fill(~legal, float("-inf"))
    labeled = target >= 0
    if bool((target[labeled] >= counts[labeled]).any().item()):
        raise Guide2VecTrainingError("guide target escaped the current legal option order")
    if bool((labeled & (counts <= 1)).any().item()):
        raise Guide2VecTrainingError("singleton guide target was not masked")
    coverage_target = labeled.to(dtype=eligibility_logits.dtype)
    positives = int(labeled.sum().item())
    negatives = int((~labeled).sum().item())
    pos_weight: Tensor | None = None
    if positives and negatives:
        pos_weight = torch.tensor(
            negatives / positives,
            device=eligibility_logits.device,
            dtype=eligibility_logits.dtype,
        )
    coverage_bce = F.binary_cross_entropy_with_logits(
        eligibility_logits,
        coverage_target,
        pos_weight=pos_weight,
        reduction="mean",
    )
    if positives:
        labeled_rows = torch.nonzero(labeled, as_tuple=False).flatten()
        per_row = F.cross_entropy(
            masked_scores.index_select(0, labeled_rows),
            target.index_select(0, labeled_rows),
            reduction="none",
        )
        weights = confidence.index_select(0, labeled_rows)
        if bool((weights <= 0.0).any().item()):
            raise Guide2VecTrainingError("labeled guide row has nonpositive confidence")
        rank_loss = (per_row * weights).sum() / weights.sum()
        predicted = masked_scores.index_select(0, labeled_rows).argmax(dim=1)
        rank_correct = int(
            (predicted == target.index_select(0, labeled_rows)).sum().item()
        )
        rank_nll_sum = float((per_row.detach() * weights.detach()).sum().item())
        rank_weight_sum = float(weights.detach().sum().item())
    else:
        # ``guide_scores.sum() * 0`` is unsafe because padding is -inf.  The
        # finite eligibility path supplies an exact zero graph anchor instead.
        rank_loss = eligibility_logits.sum() * 0.0
        rank_correct = 0
        rank_nll_sum = 0.0
        rank_weight_sum = 0.0
    total = rank_loss + float(coverage_weight) * coverage_bce
    if not bool(torch.isfinite(total).item()):
        raise Guide2VecTrainingError("Guide2Vec training loss became nonfinite")
    coverage_prediction = torch.sigmoid(eligibility_logits) >= 0.5
    metrics = _MetricAccumulator(
        rank_nll_sum=rank_nll_sum,
        rank_weight_sum=rank_weight_sum,
        rank_correct=rank_correct,
        rank_rows=positives,
        coverage_bce_sum=float(coverage_bce.detach().item()) * rows,
        coverage_rows=rows,
        coverage_correct=int((coverage_prediction == labeled).sum().item()),
        total_loss_sum=float(total.detach().item()),
        batches=1,
    )
    return total, metrics, masked_scores, labeled


def _runtime_applicable_rows(
    masked_scores: Tensor,
    counts: Tensor,
    *,
    min_score_margin: float,
) -> Tensor:
    """Return the non-threshold half of the actual rerank predicate.

    Calibration must not claim precision over rows that the runtime would
    reject for a tied or too-small guide-score margin.  Eligibility threshold
    selection is the remaining half of the predicate and is swept separately.
    """

    if masked_scores.ndim != 2 or counts.ndim != 1:
        raise Guide2VecTrainingError("runtime applicability tensors are malformed")
    if masked_scores.shape[0] != counts.numel():
        raise Guide2VecTrainingError("runtime applicability rows are misaligned")
    if masked_scores.shape[1] < 2:
        return torch.zeros_like(counts, dtype=torch.bool)
    ranked = masked_scores.topk(k=2, dim=-1).values
    margin = ranked[:, 0] - ranked[:, 1]
    return (counts >= 2) & (margin >= float(min_score_margin))


def _assert_optimizer_is_head_only(
    optimizer: torch.optim.Optimizer, head: Guide2VecHead
) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise Guide2VecTrainingError("optimizer must be a torch optimizer")
    expected = {id(parameter) for parameter in head.parameters()}
    actual: list[int] = []
    for group in optimizer.param_groups:
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise Guide2VecTrainingError("optimizer parameter group is malformed")
        actual.extend(id(parameter) for parameter in parameters)
    if set(actual) != expected or len(actual) != len(expected):
        raise Guide2VecTrainingError("only Guide2VecHead parameters may receive gradients")
    if not all(parameter.requires_grad for parameter in head.parameters()):
        raise Guide2VecTrainingError("training head unexpectedly has frozen parameters")


def _assert_finite_head_gradients(head: Guide2VecHead) -> None:
    for parameter in head.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            raise Guide2VecTrainingError("Guide2Vec head received a nonfinite gradient")


def _train_epoch(
    head: Guide2VecHead,
    chunk_refs: Sequence[JoinedChunkRef | Mapping[str, object]],
    *,
    optimizer: torch.optim.Optimizer,
    training_config: Guide2VecTrainingConfig,
    epoch_seed: int,
) -> dict[str, object]:
    """Run one deterministic, head-only optimization epoch over paired chunks."""

    if not isinstance(head, Guide2VecHead):
        raise Guide2VecTrainingError("head must be a Guide2VecHead")
    if not isinstance(training_config, Guide2VecTrainingConfig):
        raise Guide2VecTrainingError("training_config must be a Guide2VecTrainingConfig")
    _assert_optimizer_is_head_only(optimizer, head)
    refs = _normalize_joined_refs(chunk_refs, label="chunk_refs")
    device = torch.device(training_config.device)
    head.train(True)
    aggregate = _MetricAccumulator()
    dtype = next(head.parameters()).dtype
    for batch in iter_joined_batches(
        refs,
        batch_rows=training_config.batch_rows,
        device=device,
        dtype=dtype,
        shuffle_chunks=True,
        seed=int(epoch_seed),
    ):
        scores, eligibility = head(
            batch["state_vec"],
            batch["option_hidden"],
            batch["base_logits"],
            n_options=batch["n_options"],
        )
        loss, metrics, _masked_scores, _labeled = _loss_and_metrics(
            scores,
            eligibility,
            batch["guide_target_index"],
            batch["guide_confidence"],
            batch["n_options"],
            coverage_weight=training_config.coverage_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _assert_finite_head_gradients(head)
        torch.nn.utils.clip_grad_norm_(head.parameters(), training_config.max_gradient_norm)
        _assert_finite_head_gradients(head)
        optimizer.step()
        aggregate.add(metrics)
    if aggregate.batches == 0:
        raise Guide2VecTrainingError("training epoch received no batches")
    return aggregate.as_dict()


def evaluate_joined_chunks(
    head: Guide2VecHead,
    chunk_refs: Sequence[JoinedChunkRef | Mapping[str, object]],
    *,
    training_config: Guide2VecTrainingConfig,
    collect_calibration: bool = False,
) -> tuple[dict[str, object], dict[str, Tensor] | None]:
    """Deterministically evaluate paired chunks without creating gradients."""

    if not isinstance(head, Guide2VecHead):
        raise Guide2VecTrainingError("head must be a Guide2VecHead")
    if not isinstance(training_config, Guide2VecTrainingConfig):
        raise Guide2VecTrainingError("training_config must be a Guide2VecTrainingConfig")
    refs = _normalize_joined_refs(chunk_refs, label="chunk_refs")
    device = torch.device(training_config.device)
    was_training = head.training
    head.train(False)
    aggregate = _MetricAccumulator()
    calibration: dict[str, list[Tensor]] | None = (
        {"probability": [], "correct": [], "eligible": [], "applicable": []}
        if collect_calibration
        else None
    )
    dtype = next(head.parameters()).dtype
    try:
        with torch.inference_mode():
            for batch in iter_joined_batches(
                refs,
                batch_rows=training_config.batch_rows,
                device=device,
                dtype=dtype,
                shuffle_chunks=False,
                seed=0,
            ):
                scores, eligibility = head(
                    batch["state_vec"],
                    batch["option_hidden"],
                    batch["base_logits"],
                    n_options=batch["n_options"],
                )
                _loss, metrics, masked_scores, labeled = _loss_and_metrics(
                    scores,
                    eligibility,
                    batch["guide_target_index"],
                    batch["guide_confidence"],
                    batch["n_options"],
                    coverage_weight=training_config.coverage_weight,
                )
                aggregate.add(metrics)
                if calibration is not None:
                    if eligibility.ndim == 2:
                        eligibility = eligibility.squeeze(1)
                    predicted = masked_scores.argmax(dim=1)
                    calibration["probability"].append(
                        torch.sigmoid(eligibility).detach().cpu()
                    )
                    calibration["correct"].append(
                        (labeled & (predicted == batch["guide_target_index"])).detach().cpu()
                    )
                    calibration["eligible"].append(labeled.detach().cpu())
                    calibration["applicable"].append(
                        _runtime_applicable_rows(
                            masked_scores,
                            batch["n_options"],
                            min_score_margin=float(head.config.min_score_margin),
                        ).detach().cpu()
                    )
    finally:
        # Evaluation has no authority to silently change a caller's training
        # mode.  train_from_joined_chunks explicitly freezes before heldout.
        head.train(was_training)
    if aggregate.batches == 0:
        raise Guide2VecTrainingError("evaluation received no batches")
    tensors: dict[str, Tensor] | None = None
    if calibration is not None:
        tensors = {
            name: torch.cat(values) if values else torch.empty(0, dtype=torch.float32)
            for name, values in calibration.items()
        }
    return aggregate.as_dict(), tensors


def calibrate_threshold(
    validation: Mapping[str, Tensor], *, minimum_precision: float = DEFAULT_MINIMUM_PRECISION
) -> dict[str, object]:
    """Choose a validation-only eligibility threshold or a truthful abstain-all.

    Precision is measured over every applied non-singleton stage, including
    guide-masked rows, so unlabelled rows cannot disappear from the runtime
    abstention denominator.
    """

    if type(minimum_precision) not in {int, float} or not math.isfinite(float(minimum_precision)):
        raise Guide2VecTrainingError("minimum_precision must be finite")
    if not 0.0 <= float(minimum_precision) <= 1.0:
        raise Guide2VecTrainingError("minimum_precision must be in [0, 1]")
    required = {"probability", "correct", "eligible", "applicable"}
    _require_exact_keys(validation, expected=required, label="validation calibration")
    probability = validation["probability"]
    correct = validation["correct"]
    eligible = validation["eligible"]
    applicable = validation["applicable"]
    if not all(isinstance(value, Tensor) for value in (probability, correct, eligible, applicable)):
        raise Guide2VecTrainingError("validation calibration values must be tensors")
    probability = probability.detach().cpu().to(dtype=torch.float64)
    correct = correct.detach().cpu().to(dtype=torch.bool)
    eligible = eligible.detach().cpu().to(dtype=torch.bool)
    applicable = applicable.detach().cpu().to(dtype=torch.bool)
    if not (
        probability.ndim == correct.ndim == eligible.ndim == applicable.ndim == 1
        and probability.shape == correct.shape == eligible.shape == applicable.shape
    ):
        raise Guide2VecTrainingError("validation calibration rows are misaligned")
    if not bool(torch.isfinite(probability).all().item()):
        raise Guide2VecTrainingError("validation eligibility probability is nonfinite")
    if bool(((probability < 0.0) | (probability > 1.0)).any().item()):
        raise Guide2VecTrainingError("validation eligibility probability leaves [0, 1]")
    base = {
        "schema": CALIBRATION_SCHEMA,
        "minimum_precision": float(minimum_precision),
        "validation_rows": int(probability.numel()),
        "validation_eligible_rows": int(eligible.sum().item()),
        "validation_applicable_rows": int(applicable.sum().item()),
    }
    if probability.numel() == 0 or int(eligible.sum().item()) == 0:
        return {
            **base,
            "status": "abstain_all_no_validation_labels",
            "threshold": 1.0,
            "applied_rows": 0,
            "precision": None,
            "eligible_coverage": 0.0,
            "applied_labeled_rows": 0,
        }
    candidates = torch.unique(
        torch.cat(
            [
                torch.linspace(0.0, 1.0, 101, dtype=torch.float64),
                probability.clamp(0.0, 1.0),
            ]
        )
    ).sort().values
    best: tuple[float, float, int, float] | None = None
    for threshold in candidates.tolist():
        applied = (probability >= threshold) & applicable
        applied_rows = int(applied.sum().item())
        if applied_rows == 0:
            continue
        precision = float(correct[applied].to(dtype=torch.float64).mean().item())
        if precision < float(minimum_precision):
            continue
        labeled_rows = int((applied & eligible).sum().item())
        coverage = labeled_rows / max(int(eligible.sum().item()), 1)
        candidate = (coverage, precision, applied_rows, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return {
            **base,
            "status": "abstain_all_precision_floor_not_met",
            "threshold": 1.0,
            "applied_rows": 0,
            "precision": None,
            "eligible_coverage": 0.0,
            "applied_labeled_rows": 0,
        }
    coverage, precision, applied_rows, threshold = best
    applied = (probability >= threshold) & applicable
    return {
        **base,
        "status": "validation_calibrated",
        "threshold": threshold,
        "applied_rows": applied_rows,
        "precision": precision,
        "eligible_coverage": coverage,
        "applied_labeled_rows": int((applied & eligible).sum().item()),
    }


# Short aliases make the independent train/eval/calibrate surface easy to use
# while preserving the fully descriptive names above.
evaluate = evaluate_joined_chunks
calibrate = calibrate_threshold


def _normalized_training_config(
    value: Guide2VecTrainingConfig | Guide2VecConfig | None,
) -> Guide2VecTrainingConfig:
    if value is None:
        return Guide2VecTrainingConfig()
    if isinstance(value, Guide2VecTrainingConfig):
        return value
    if isinstance(value, Guide2VecConfig):
        return Guide2VecTrainingConfig(head_config=value)
    raise Guide2VecTrainingError("config must be Guide2VecTrainingConfig or Guide2VecConfig")


def _freeze_selected_head(
    head: Guide2VecHead,
    *,
    state_dict: Mapping[str, Tensor],
    threshold: float,
) -> tuple[Guide2VecHead, Guide2VecConfig, str]:
    if not 0.0 <= threshold <= 1.0 or not math.isfinite(threshold):
        raise Guide2VecTrainingError("calibrated threshold leaves [0, 1]")
    try:
        head.load_state_dict(dict(state_dict), strict=True)
    except (RuntimeError, TypeError) as exc:
        raise Guide2VecTrainingError("selected validation state does not fit Guide2VecHead") from exc
    runtime_config = replace(head.config, min_eligibility=float(threshold))
    head.config = runtime_config
    head.train(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if head.training or any(parameter.requires_grad for parameter in head.parameters()):
        raise Guide2VecTrainingError("selected Guide2Vec head did not freeze")
    frozen_state = {
        name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()
    }
    return head, runtime_config, _head_state_sha256(frozen_state)


def _content_addressed_torch_write(
    payload: Mapping[str, object], *, directory: Path, prefix: str
) -> tuple[Path, str]:
    """Publish a Torch payload under its digest without replacing any file."""

    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{prefix}.partial.{os.getpid()}.{time.time_ns()}.pt"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(temporary)
        final = directory / f"{prefix}-{digest.split(':', 1)[1]}.pt"
        try:
            os.link(temporary, final)
        except FileExistsError:
            if not final.is_file() or _sha256_file(final) != digest:
                raise Guide2VecTrainingError(f"content-addressed artifact collision: {final}")
        return final, digest
    finally:
        temporary.unlink(missing_ok=True)


def _content_addressed_json_write(
    payload: Mapping[str, object], *, directory: Path, prefix: str
) -> tuple[Path, str]:
    """Publish canonical JSON under its digest without replacing any file."""

    body = _canonical_json(dict(payload))
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{prefix}-{digest.split(':', 1)[1]}.json"
    if final.exists():
        if not final.is_file() or final.read_bytes() != body:
            raise Guide2VecTrainingError(f"content-addressed artifact collision: {final}")
        return final, digest
    temporary = directory / f".{prefix}.partial.{os.getpid()}.{time.time_ns()}.json"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, final)
        except FileExistsError:
            if not final.is_file() or final.read_bytes() != body:
                raise Guide2VecTrainingError(f"content-addressed artifact collision: {final}")
        return final, digest
    finally:
        temporary.unlink(missing_ok=True)


def load_frozen_candidate_payload(
    payload: Mapping[str, object],
) -> FrozenGuide2VecCandidate:
    """Strictly reconstruct a generic candidate through its gated runtime.

    This loader deliberately returns :class:`FrozenGuide2VecCandidate`, not a
    raw head.  Its only public inference operation is ``candidate.rerank()``,
    which checks the explicit always-abstain gate before the base head can see
    the input tensors.
    """

    if not isinstance(payload, Mapping):
        raise Guide2VecTrainingError("frozen candidate payload must be a mapping")
    required = {
        "schema",
        "base_identity",
        "base_identity_sha256",
        "runtime_head_config",
        "runtime",
        "runtime_sha256",
        "calibrated_threshold",
        "always_abstain",
        "parameter_count",
        "head_state_dict",
        "head_state_sha256",
    }
    missing = required.difference(payload)
    if missing:
        raise Guide2VecTrainingError(
            f"frozen candidate payload is missing fields: {sorted(missing)}"
        )
    if payload["schema"] != CANDIDATE_SCHEMA:
        raise Guide2VecTrainingError("frozen candidate schema changed")
    identity_raw = payload["base_identity"]
    runtime_raw = payload["runtime"]
    config_raw = payload["runtime_head_config"]
    state_raw = payload["head_state_dict"]
    if not isinstance(identity_raw, Mapping) or not isinstance(runtime_raw, Mapping):
        raise Guide2VecTrainingError("frozen candidate identity/runtime is malformed")
    if not isinstance(config_raw, Mapping) or not isinstance(state_raw, Mapping):
        raise Guide2VecTrainingError("frozen candidate config/state is malformed")
    base_identity = FrozenBaseIdentity.from_mapping(identity_raw)
    if _normalize_sha256(
        payload["base_identity_sha256"], label="candidate base_identity_sha256"
    ) != base_identity.identity_sha256:
        raise Guide2VecTrainingError("candidate base identity digest mismatch")
    runtime = FrozenGuide2VecRuntime.from_mapping(runtime_raw)
    if _normalize_sha256(payload["runtime_sha256"], label="candidate runtime_sha256") != runtime.runtime_sha256:
        raise Guide2VecTrainingError("candidate runtime digest mismatch")
    if dict(config_raw) != runtime.guide2vec_config.as_dict():
        raise Guide2VecTrainingError("candidate runtime config is split-brain")
    if payload["always_abstain"] is not runtime.always_abstain:
        raise Guide2VecTrainingError("candidate always-abstain field mismatches runtime")
    threshold = payload["calibrated_threshold"]
    if type(threshold) not in {int, float} or float(threshold) != float(
        runtime.calibrated_threshold
    ):
        raise Guide2VecTrainingError("candidate calibrated threshold mismatches runtime")
    typed_state: dict[str, Tensor] = {}
    for name, tensor in state_raw.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise Guide2VecTrainingError("candidate head state must contain named tensors")
        typed_state[name] = tensor.detach().cpu()
    state_digest = _normalize_sha256(payload["head_state_sha256"], label="candidate head_state_sha256")
    if _head_state_sha256(typed_state) != state_digest:
        raise Guide2VecTrainingError("candidate head state digest mismatch")
    count = payload["parameter_count"]
    if type(count) is not int:
        raise Guide2VecTrainingError("candidate parameter_count must be an exact integer")
    head = Guide2VecHead(runtime.guide2vec_config)
    if count != head.parameter_count:
        raise Guide2VecTrainingError("candidate parameter_count does not fit runtime config")
    try:
        head.load_state_dict(typed_state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise Guide2VecTrainingError("candidate head state does not fit runtime config") from exc
    head.train(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return FrozenGuide2VecCandidate(
        head=head,
        runtime=runtime,
        base_identity=base_identity,
        head_state_sha256=state_digest,
    )


def load_frozen_candidate(
    ref: ContentChunkRef | Mapping[str, object],
) -> FrozenGuide2VecCandidate:
    """Hash-verify and safely load a generic candidate through its runtime gate."""

    payload = _safe_torch_load(_normalize_content_ref(ref))
    return load_frozen_candidate_payload(payload)


def _chunk_digest_records(refs: Sequence[JoinedChunkRef]) -> list[dict[str, str]]:
    return [ref.digest_record() for ref in refs]


def _training_input_binding(
    *,
    base_identity: FrozenBaseIdentity,
    guide_manifest_sha256: str,
    training_manifest_sha256: str,
    train_refs: Sequence[JoinedChunkRef],
    validation_refs: Sequence[JoinedChunkRef],
    heldout_refs: Sequence[JoinedChunkRef],
) -> dict[str, object]:
    return {
        "base_identity": base_identity.as_dict(),
        "base_identity_sha256": base_identity.identity_sha256,
        "guide_manifest_sha256": guide_manifest_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "chunk_digests": {
            "train": _chunk_digest_records(train_refs),
            "validation": _chunk_digest_records(validation_refs),
            "heldout": _chunk_digest_records(heldout_refs),
        },
    }


def _issue_activation_grant(
    *,
    training_manifest_path: str | Path,
    activation_receipt_path: str | Path,
) -> _Guide2VecTrainingGrant:
    """Internally revalidate the manifest and future owner receipt.

    A public caller never supplies a grant object.  In r226 the activation
    validator fails at semantic split readiness before it opens the receipt;
    this is deliberately the first observable operation of the public trainer
    and therefore precedes every chunk/output operation.
    """

    try:
        return _validate_activation_receipt(
            training_manifest_path=training_manifest_path,
            receipt_path=activation_receipt_path,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        detail = str(exc) or type(exc).__name__
        raise Guide2VecTrainingError(
            "Guide2Vec training cannot obtain a semantically-ready activation grant: "
            f"{detail}"
        ) from exc


def _validate_activation_grant(
    activation_grant: object,
    *,
    training_manifest_sha256: str,
    guide_manifest_sha256: str,
    base_identity_sha256: str,
    partition_chunks: Mapping[str, Sequence[Mapping[str, object]]],
    output_dir: str | Path,
) -> _Guide2VecTrainingGrant:
    """Require the opaque grant before opening any feature/label chunk.

    The exact concrete type check prevents a lookalike object from replacing
    the future receipt validator.  The grant itself binds the full resolved
    compatibility declaration, causal partition proof, owner/source/host
    receipts, and each ordered chunk-pair identity at issuance time.
    """

    if type(activation_grant) is not _Guide2VecTrainingGrant:
        raise Guide2VecTrainingError(
            "Guide2Vec training requires an opaque verified activation_grant"
        )
    try:
        activation_grant.validate_for(
            training_manifest_sha256=training_manifest_sha256,
            guide_manifest_sha256=guide_manifest_sha256,
            base_identity_sha256=base_identity_sha256,
            partition_chunks=partition_chunks,
            output_dir=output_dir,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        detail = str(exc) or type(exc).__name__
        raise Guide2VecTrainingError(
            f"Guide2Vec activation grant does not authorize these inputs: {detail}"
        ) from exc
    return activation_grant


def _consume_activation_grant(activation_grant: _Guide2VecTrainingGrant) -> None:
    """Claim the verified one-action grant immediately before optimization."""

    try:
        activation_grant.consume()
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        detail = str(exc) or type(exc).__name__
        raise Guide2VecTrainingError(
            f"Guide2Vec activation grant cannot be consumed: {detail}"
        ) from exc


def _write_durable_grant_consumption_receipt(
    *,
    activation_grant: _Guide2VecTrainingGrant,
    output_dir: Path,
    training_manifest_sha256: str,
    guide_manifest_sha256: str,
    base_identity_sha256: str,
    partition_chunks: Mapping[str, Sequence[Mapping[str, object]]],
    training_config: Guide2VecTrainingConfig,
) -> tuple[Path, str]:
    """Claim a future valid grant durably with ``O_EXCL`` before optimization.

    The current r226 validator cannot issue a grant, so this function is
    unreachable today.  It is intentionally kept adjacent to the optimizer
    boundary for a future semantic-ready contract: a second process that sees
    the same one-action receipt must fail before creating an optimizer.
    """

    payload = {
        "schema": GRANT_CONSUMPTION_SCHEMA,
        "status": "claimed_before_optimizer",
        "activation_receipt_sha256": activation_grant.receipt_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "guide_manifest_sha256": guide_manifest_sha256,
        "base_identity_sha256": base_identity_sha256,
        "partition_chunks": {
            partition: [dict(record) for record in records]
            for partition, records in partition_chunks.items()
        },
        "training_config_sha256": _canonical_json_sha256(training_config.as_dict()),
        "output_root_identity_sha256": activation_grant.output_root_identity_sha256,
    }
    body = _canonical_json(payload)
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    directory = output_dir / "grant-consumptions"
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        "guide2vec-grant-consumption-"
        f"{activation_grant.receipt_sha256.removeprefix('sha256:')}.json"
    )
    target = directory / filename
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise Guide2VecTrainingError(
            "activation receipt already has a durable optimizer-consumption receipt"
        ) from exc
    # Any write/fsync failure leaves the O_EXCL claim in place as forensic
    # evidence; a retry cannot assume an interrupted one-action claim is safe.
    with os.fdopen(fd, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        raise Guide2VecTrainingError(
            "could not open durable grant-consumption directory for fsync"
        ) from exc
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target, digest


def _activation_grant_binding(activation_grant: _Guide2VecTrainingGrant) -> dict[str, object]:
    """Return the immutable receipt identities carried into each artifact."""

    return {
        "activation_receipt_sha256": activation_grant.receipt_sha256,
        "activation_receipt_filename": activation_grant.receipt_path.name,
        "owner_contract_revision": activation_grant.owner_contract_revision,
        "owner_contract_sha256": activation_grant.owner_contract.sha256,
        "guide_contract_sha256": activation_grant.guide_contract.sha256,
        "causal_split_manifest_sha256": activation_grant.causal_split_manifest_sha256,
        "causal_split_receipt_sha256": activation_grant.causal_split_receipt.sha256,
        "source_snapshot_sha256": activation_grant.source_snapshot.sha256,
        "host_noninterference_capability_receipt_sha256": (
            activation_grant.host_noninterference_capability_receipt.sha256
        ),
        "output_root_identity_sha256": activation_grant.output_root_identity_sha256,
    }


def train_from_joined_chunks(
    *,
    train_chunks: Sequence[JoinedChunkRef | Mapping[str, object]],
    validation_chunks: Sequence[JoinedChunkRef | Mapping[str, object]],
    heldout_chunks: Sequence[JoinedChunkRef | Mapping[str, object]],
    base_identity: FrozenBaseIdentity,
    guide_manifest_sha256: str,
    training_manifest_path: str | Path,
    activation_receipt_path: str | Path,
    config: Guide2VecTrainingConfig | Guide2VecConfig | None,
    output_dir: str | Path,
) -> dict[str, object]:
    """Fit, select, freeze, then heldout-evaluate a generic Guide2Vec head.

    ``heldout_chunks`` are deliberately normalized but never opened, hashed, or
    safely loaded until after the best validation state is restored, calibrated,
    and frozen.  The public boundary accepts only a training-manifest path and
    activation-receipt path; it revalidates both itself before a chunk or
    output path is touched.  A future issued grant must then exactly bind the
    guide/base identity, every ordered partition chunk pair, and the dedicated
    output directory.  Its receipt is claimed durably and consumed in-process
    immediately before the first optimizer is constructed.
    """

    # This intentionally comes before config/ref/output validation.  Current
    # r226 stops here at semantic split readiness, before receipt/chunk/output
    # I/O, and a caller cannot smuggle in a fabricated grant object.
    issued_grant = _issue_activation_grant(
        training_manifest_path=training_manifest_path,
        activation_receipt_path=activation_receipt_path,
    )
    if not isinstance(base_identity, FrozenBaseIdentity):
        raise Guide2VecTrainingError("base_identity must be a FrozenBaseIdentity")
    guide_digest = _normalize_sha256(guide_manifest_sha256, label="guide_manifest_sha256")
    training_digest = _normalize_sha256(
        issued_grant.training_manifest_sha256,
        label="issued activation grant training_manifest_sha256",
    )
    training_config = _normalized_training_config(config)
    target_output = Path(output_dir)

    # Normalization is intentionally metadata-only.  In particular, it must
    # not perform a heldout hash or Torch load before validation selection.
    train_refs = _normalize_joined_refs(train_chunks, label="train_chunks")
    validation_refs = _normalize_joined_refs(validation_chunks, label="validation_chunks")
    heldout_refs = _normalize_joined_refs(heldout_chunks, label="heldout_chunks")
    partition_chunk_records: dict[str, list[dict[str, str]]] = {
        "train": _chunk_digest_records(train_refs),
        "validation": _chunk_digest_records(validation_refs),
        "heldout": _chunk_digest_records(heldout_refs),
    }
    verified_grant = _validate_activation_grant(
        issued_grant,
        training_manifest_sha256=training_digest,
        guide_manifest_sha256=guide_digest,
        base_identity_sha256=base_identity.identity_sha256,
        partition_chunks=partition_chunk_records,
        output_dir=target_output,
    )
    _assert_partition_chunk_digests_disjoint(
        {
            "train": train_refs,
            "validation": validation_refs,
            "heldout": heldout_refs,
        }
    )
    if target_output.exists() and not target_output.is_dir():
        raise Guide2VecTrainingError("output_dir must be a directory path")

    # Train/validation split identity is proven before an optimizer is made.
    # Heldout remains unopened until the selected state is frozen below.
    train_stage_keys, train_partition_identity = _collect_partition_identity(
        train_refs, partition="train"
    )
    validation_stage_keys, validation_partition_identity = _collect_partition_identity(
        validation_refs, partition="validation"
    )
    _assert_stage_sets_disjoint(
        train_stage_keys,
        validation_stage_keys,
        first_label="train",
        second_label="validation",
    )
    if (
        train_partition_identity["dtype_policy_sha256"]
        != validation_partition_identity["dtype_policy_sha256"]
    ):
        raise Guide2VecTrainingError(
            "train and validation partitions disagree on latent/label dtype policy"
        )
    _assert_partition_dtype_policy_matches_grant(
        train_partition_identity,
        partition="train",
        activation_grant=verified_grant,
    )
    _assert_partition_dtype_policy_matches_grant(
        validation_partition_identity,
        partition="validation",
        activation_grant=verified_grant,
    )

    device = torch.device(training_config.device)

    # The isolated RNG fork gives reproducible head initialization without
    # changing an unrelated caller's CPU/CUDA stochastic stream.  Enter the
    # deterministic scope first: CUDA's cuBLAS workspace setting must precede
    # any CUDA availability/device initialization.
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with _deterministic_torch_scope(device) as determinism_receipt, torch.random.fork_rng(
        devices=cuda_devices,
        enabled=True,
    ):
        if device.type == "cuda" and not torch.cuda.is_available():
            raise Guide2VecTrainingError("configured CUDA device is unavailable")
        torch.manual_seed(training_config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(training_config.seed)
        head = Guide2VecHead(training_config.head_config).to(device)
        if head.parameter_count != training_config.head_config.expected_parameter_count:
            raise Guide2VecTrainingError("Guide2Vec parameter inventory drifted")
        grant_consumption_path, grant_consumption_digest = (
            _write_durable_grant_consumption_receipt(
                activation_grant=verified_grant,
                output_dir=target_output,
                training_manifest_sha256=training_digest,
                guide_manifest_sha256=guide_digest,
                base_identity_sha256=base_identity.identity_sha256,
                partition_chunks=partition_chunk_records,
                training_config=training_config,
            )
        )
        _consume_activation_grant(verified_grant)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        _assert_optimizer_is_head_only(optimizer, head)

        best_metric = math.inf
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None
        best_calibration: dict[str, object] | None = None
        epoch_records: list[dict[str, object]] = []
        for epoch in range(1, training_config.epochs + 1):
            train_metrics = _train_epoch(
                head,
                train_refs,
                optimizer=optimizer,
                training_config=training_config,
                epoch_seed=training_config.seed + epoch * 10_007,
            )
            validation_metrics, calibration_rows = evaluate_joined_chunks(
                head,
                validation_refs,
                training_config=training_config,
                collect_calibration=True,
            )
            if calibration_rows is None:  # Defensive; collect_calibration is explicit above.
                raise Guide2VecTrainingError("validation calibration rows were not collected")
            calibration = calibrate_threshold(
                calibration_rows, minimum_precision=training_config.minimum_precision
            )
            raw_metric = validation_metrics.get("rank_nll")
            if raw_metric is None:
                metric = math.inf
            elif type(raw_metric) in {int, float} and math.isfinite(float(raw_metric)):
                metric = float(raw_metric)
            else:
                raise Guide2VecTrainingError("validation rank metric is invalid")
            epoch_records.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "validation_calibration": calibration,
                    "selection_metric_name": "validation_confidence_weighted_listwise_cross_entropy",
                    "selection_metric": metric if math.isfinite(metric) else None,
                }
            )
            if metric < best_metric - 1e-12:
                best_metric = metric
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in head.state_dict().items()
                }
                best_calibration = calibration

        if best_state is None or best_calibration is None or best_epoch <= 0:
            raise Guide2VecTrainingError(
                "no validation checkpoint with a finite labeled ranking metric was selected"
            )
        threshold_raw = best_calibration.get("threshold")
        if type(threshold_raw) not in {int, float}:
            raise Guide2VecTrainingError("calibrated validation threshold is invalid")
        calibration_status = best_calibration.get("status")
        if calibration_status not in {_CALIBRATED_STATUS, *_ABSTAIN_ALL_STATUSES}:
            raise Guide2VecTrainingError("validation calibration status is invalid")
        # This call is the mandatory heldout barrier: all parameters are frozen
        # and the calibrated runtime config is bound before any heldout load.
        head, runtime_config, head_state_digest = _freeze_selected_head(
            head,
            state_dict=best_state,
            threshold=float(threshold_raw),
        )
        frozen_runtime = FrozenGuide2VecRuntime(
            guide2vec_config=runtime_config,
            calibration_status=calibration_status,
            calibrated_threshold=float(threshold_raw),
            always_abstain=calibration_status in _ABSTAIN_ALL_STATUSES,
        )

        # Do not move this block above _freeze_selected_head.  It is the first
        # operation that opens, hashes, or safely loads heldout chunks.
        heldout_stage_keys, heldout_partition_identity = _collect_partition_identity(
            heldout_refs, partition="heldout"
        )
        _assert_stage_sets_disjoint(
            train_stage_keys | validation_stage_keys,
            heldout_stage_keys,
            first_label="train/validation",
            second_label="heldout",
        )
        if (
            heldout_partition_identity["dtype_policy_sha256"]
            != train_partition_identity["dtype_policy_sha256"]
        ):
            raise Guide2VecTrainingError(
                "heldout partition disagrees on latent/label dtype policy"
            )
        _assert_partition_dtype_policy_matches_grant(
            heldout_partition_identity,
            partition="heldout",
            activation_grant=verified_grant,
        )
        heldout_metrics, _ = evaluate_joined_chunks(
            head,
            heldout_refs,
            training_config=training_config,
            collect_calibration=False,
        )

    input_binding = _training_input_binding(
        base_identity=base_identity,
        guide_manifest_sha256=guide_digest,
        training_manifest_sha256=training_digest,
        train_refs=train_refs,
        validation_refs=validation_refs,
        heldout_refs=heldout_refs,
    )
    activation_binding = _activation_grant_binding(verified_grant)
    grant_consumption_binding = {
        "schema": GRANT_CONSUMPTION_SCHEMA,
        "sha256": grant_consumption_digest,
        "filename": grant_consumption_path.name,
    }
    partition_identity = {
        "train": train_partition_identity,
        "validation": validation_partition_identity,
        "heldout": heldout_partition_identity,
        "cross_partition_stage_overlap": False,
    }
    frozen_state = {
        name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()
    }
    if _head_state_sha256(frozen_state) != head_state_digest:
        raise Guide2VecTrainingError("frozen head state changed after heldout evaluation")
    candidate_payload: dict[str, object] = {
        "schema": CANDIDATE_SCHEMA,
        "kind": "frozen_generic_guide2vec_candidate",
        "base_identity": base_identity.as_dict(),
        "base_identity_sha256": base_identity.identity_sha256,
        "guide_manifest_sha256": guide_digest,
        "training_manifest_sha256": training_digest,
        "chunk_digests": input_binding["chunk_digests"],
        "partition_identity": partition_identity,
        "activation_grant": activation_binding,
        "activation_grant_consumption": grant_consumption_binding,
        "training_config": training_config.as_dict(),
        "determinism": dict(determinism_receipt),
        "runtime_head_config": runtime_config.as_dict(),
        "runtime": frozen_runtime.as_dict(),
        "runtime_sha256": frozen_runtime.runtime_sha256,
        "calibrated_threshold": float(runtime_config.min_eligibility),
        "always_abstain": frozen_runtime.always_abstain,
        "parameter_count": head.parameter_count,
        "head_state_dict": frozen_state,
        "head_state_sha256": head_state_digest,
        "selected_epoch": best_epoch,
        "selection_metric_name": "validation_confidence_weighted_listwise_cross_entropy",
        "selection_metric": best_metric,
        "metrics": {
            "epochs": epoch_records,
            "validation_calibration": best_calibration,
            "heldout": heldout_metrics,
        },
        "authority": {
            "training_authorized_by_verified_activation_grant": True,
            "activation_receipt_sha256": activation_binding["activation_receipt_sha256"],
            "activation_grant_consumption_sha256": grant_consumption_digest,
            "candidate_runtime_attachment_authorized": False,
            "selector_change_authorized": False,
            "serving_authorized": False,
            "promotion_authorized": False,
            "bo1000_authorized": False,
            "mcts_change_authorized": False,
            "rtp_authorized": False,
            "kaggle_authorized": False,
        },
    }
    candidate_path, candidate_digest = _content_addressed_torch_write(
        candidate_payload,
        directory=target_output / "candidates",
        prefix="guide2vec-candidate",
    )
    receipt_payload: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete_offline_candidate_only",
        "candidate_sha256": candidate_digest,
        "candidate_filename": candidate_path.name,
        "base_identity": base_identity.as_dict(),
        "base_identity_sha256": base_identity.identity_sha256,
        "guide_manifest_sha256": guide_digest,
        "training_manifest_sha256": training_digest,
        "chunk_digests": input_binding["chunk_digests"],
        "partition_identity": partition_identity,
        "activation_grant": activation_binding,
        "activation_grant_consumption": grant_consumption_binding,
        "determinism": dict(determinism_receipt),
        "head_state_sha256": head_state_digest,
        "runtime": frozen_runtime.as_dict(),
        "runtime_sha256": frozen_runtime.runtime_sha256,
        "parameter_count": head.parameter_count,
        "selected_epoch": best_epoch,
        "selection_metric_name": "validation_confidence_weighted_listwise_cross_entropy",
        "selection_metric": best_metric,
        "calibrated_threshold": float(runtime_config.min_eligibility),
        "always_abstain": frozen_runtime.always_abstain,
        "validation_calibration": best_calibration,
        "metrics": {
            "epochs": epoch_records,
            "heldout": heldout_metrics,
        },
        "authority": dict(candidate_payload["authority"]),
    }
    receipt_path, receipt_digest = _content_addressed_json_write(
        receipt_payload,
        directory=target_output / "receipts",
        prefix="guide2vec-training-receipt",
    )
    return {
        "candidate_path": candidate_path,
        "candidate_sha256": candidate_digest,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_digest,
        "selected_epoch": best_epoch,
        "selection_metric": best_metric,
        "calibrated_threshold": float(runtime_config.min_eligibility),
        "always_abstain": frozen_runtime.always_abstain,
        "determinism": dict(determinism_receipt),
        "heldout_metrics": heldout_metrics,
        "head_state_sha256": head_state_digest,
        "activation_receipt_sha256": activation_binding["activation_receipt_sha256"],
        "activation_grant_consumption_sha256": grant_consumption_digest,
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "CANDIDATE_SCHEMA",
    "CHUNK_DTYPE_POLICY_SCHEMA",
    "DETERMINISM_POLICY",
    "FROZEN_RUNTIME_SCHEMA",
    "GRANT_CONSUMPTION_SCHEMA",
    "LABEL_CHUNK_SCHEMA",
    "LATENT_CHUNK_SCHEMA",
    "MAX_EPOCHS",
    "METRICS_SCHEMA",
    "RECEIPT_SCHEMA",
    "STAGE_DIGEST_BYTES",
    "ChunkRef",
    "ContentChunkRef",
    "FrozenGuide2VecCandidate",
    "FrozenGuide2VecRuntime",
    "Guide2VecTrainingConfig",
    "Guide2VecTrainingError",
    "JoinedChunk",
    "JoinedChunkRef",
    "LabelChunk",
    "LatentChunk",
    "PairedChunkRef",
    "TrainingConfig",
    "calibrate",
    "calibrate_threshold",
    "evaluate",
    "evaluate_joined_chunks",
    "iter_joined_batches",
    "load_frozen_candidate",
    "load_frozen_candidate_payload",
    "load_joined_chunk",
    "load_label_chunk",
    "load_latent_chunk",
    "rerank_with_frozen_runtime",
    "train_from_joined_chunks",
]
