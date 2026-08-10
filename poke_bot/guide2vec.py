"""Tiny, frozen-policy Guide2Vec option reranker for Alakazam r212.

This module deliberately owns only a small *sidecar* head.  It never encodes a
board, generates an action, consults hidden information, or runs MCTS/RTP.  A
caller supplies the r195 model's already-causal ``state_vec`` and
``option_hidden`` tensors for the *currently legal* factorized options.  The
head may add a bounded non-negative bonus to those same options, or it returns
the base logits exactly when identity, eligibility, margin, finite-input, or
legality checks do not pass.

The r212 contract permits gradients for ``Guide2VecHead`` only.  The helpers
below make the frozen r195 submission identity and the separate sidecar
checkpoint auditable without coupling this module to a model loader, trainer,
or serving route.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


GUIDE2VEC_CHECKPOINT_SCHEMA = "poke_bot.alakazam_guide2vec/v1"
FROZEN_BASE_IDENTITY_SCHEMA = "poke_bot.alakazam_guide2vec_frozen_base/v1"

ALAKAZAM_SUBMISSION_ID = 55_378_392
ALAKAZAM_R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
ALAKAZAM_R195_CHECKPOINT_BYTES = 127_914_385
ALAKAZAM_R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)

MIN_PARAMETER_COUNT = 100_000
MAX_PARAMETER_COUNT = 500_000
MAX_LOGIT_BONUS = 0.05

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class Guide2VecError(ValueError):
    """Raised when a Guide2Vec identity, artifact, or tensor is invalid."""


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Guide2VecError("value is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalize_digest(value: object, *, label: str, required: bool) -> str | None:
    if value is None:
        if required:
            raise Guide2VecError(f"{label} is required")
        return None
    digest = str(value).strip().lower()
    if not digest:
        if required:
            raise Guide2VecError(f"{label} is required")
        return None
    if not digest.startswith("sha256:") and re.fullmatch(r"[0-9a-f]{64}", digest):
        digest = "sha256:" + digest
    if _SHA256_RE.fullmatch(digest) is None:
        raise Guide2VecError(f"{label} must be a canonical sha256 digest")
    return digest


def _normalize_exact_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or int(value) <= 0:
        raise Guide2VecError(f"{label} must be a positive exact integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class FrozenBaseIdentity:
    """Immutable byte identity for the direct r195 base policy.

    ``bundle_sha256`` binds the submission package separately from the model
    checkpoint.  That keeps a Guide2Vec artifact from being reused with a
    package whose deck/runtime bytes are not the evaluated direct-policy arm.
    """

    submission_id: int
    checkpoint_sha256: str
    checkpoint_bytes: int | None = None
    bundle_sha256: str | None = None
    model_config_sha256: str | None = None
    feature_schema_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.submission_id) is not int or self.submission_id <= 0:
            raise Guide2VecError("submission_id must be a positive exact integer")
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _normalize_digest(
                self.checkpoint_sha256,
                label="checkpoint_sha256",
                required=True,
            ),
        )
        if self.checkpoint_bytes is not None:
            object.__setattr__(
                self,
                "checkpoint_bytes",
                _normalize_exact_positive_int(
                    self.checkpoint_bytes,
                    label="checkpoint_bytes",
                ),
            )
        for field_name in (
            "bundle_sha256",
            "model_config_sha256",
            "feature_schema_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_digest(
                    getattr(self, field_name),
                    label=field_name,
                    required=False,
                ),
            )

    @classmethod
    def alakazam_submission_55378392(cls) -> "FrozenBaseIdentity":
        """The exact r195 NO-RTP submission selected for r212."""

        return cls(
            submission_id=ALAKAZAM_SUBMISSION_ID,
            checkpoint_sha256=ALAKAZAM_R195_CHECKPOINT_SHA256,
            checkpoint_bytes=ALAKAZAM_R195_CHECKPOINT_BYTES,
            bundle_sha256=ALAKAZAM_R195_BUNDLE_SHA256,
        )

    @property
    def identity_sha256(self) -> str:
        return _canonical_json_sha256(
            {
                "schema": FROZEN_BASE_IDENTITY_SCHEMA,
                **self.as_dict(),
            }
        )

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical identity payload, including nulls."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FrozenBaseIdentity":
        if not isinstance(value, Mapping):
            raise Guide2VecError("frozen base identity must be a mapping")
        expected = {
            "submission_id",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "bundle_sha256",
            "model_config_sha256",
            "feature_schema_sha256",
        }
        missing = expected.difference(value)
        unknown = set(value).difference(expected)
        if missing or unknown:
            raise Guide2VecError(
                "frozen base identity fields changed: "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        return cls(
            submission_id=value["submission_id"],  # type: ignore[arg-type]
            checkpoint_sha256=value["checkpoint_sha256"],  # type: ignore[arg-type]
            checkpoint_bytes=value["checkpoint_bytes"],  # type: ignore[arg-type]
            bundle_sha256=value["bundle_sha256"],  # type: ignore[arg-type]
            model_config_sha256=value["model_config_sha256"],  # type: ignore[arg-type]
            feature_schema_sha256=value["feature_schema_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class Guide2VecConfig:
    """Fixed tiny-head layout for r195's 96-dimensional hidden state.

    The default architecture has exactly 155,468 trainable parameters.  The
    eligibility logit is trained separately on stage availability
    (``guide_target_index >= 0``); a high eligibility probability is necessary
    but never sufficient to modify a direct-policy decision.
    """

    d_model: int = 96
    score_hidden_dim: int = 256
    score_bottleneck_dim: int = 64
    eligibility_hidden_dim: int = 128
    max_logit_bonus: float = MAX_LOGIT_BONUS
    min_eligibility: float = 0.60
    min_score_margin: float = 1e-4

    def __post_init__(self) -> None:
        # r212 is intentionally Alakazam/r195-only, not a generic new policy
        # architecture that might silently attach to a different base model.
        if self.d_model != 96:
            raise Guide2VecError("Guide2Vec is bound to the r195 d_model=96")
        for field_name in (
            "score_hidden_dim",
            "score_bottleneck_dim",
            "eligibility_hidden_dim",
        ):
            _normalize_exact_positive_int(getattr(self, field_name), label=field_name)
        for field_name in ("max_logit_bonus", "min_eligibility", "min_score_margin"):
            value = getattr(self, field_name)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise Guide2VecError(f"{field_name} must be finite")
        if float(self.max_logit_bonus) != MAX_LOGIT_BONUS:
            raise Guide2VecError(
                f"max_logit_bonus is fixed by r212 at {MAX_LOGIT_BONUS:.2f}"
            )
        if not 0.0 <= float(self.min_eligibility) <= 1.0:
            raise Guide2VecError("min_eligibility must be in [0, 1]")
        if float(self.min_score_margin) < 0.0:
            raise Guide2VecError("min_score_margin must be non-negative")
        if not (MIN_PARAMETER_COUNT <= self.expected_parameter_count <= MAX_PARAMETER_COUNT):
            raise Guide2VecError(
                "Guide2Vec parameter count is outside the r212 100k--500k envelope"
            )

    @property
    def context_dim(self) -> int:
        # State + option-set extrema + number of legal candidates + best base logit.
        return 3 * self.d_model + 2

    @property
    def score_input_dim(self) -> int:
        # Current option + context + its base logit normalized against the best.
        return self.d_model + self.context_dim + 1

    @property
    def expected_parameter_count(self) -> int:
        score_input = self.score_input_dim
        score_hidden = self.score_hidden_dim
        score_bottleneck = self.score_bottleneck_dim
        eligibility_hidden = self.eligibility_hidden_dim
        # LayerNorms carry one scale and one bias per normalized feature.
        ranking = (
            2 * score_input
            + (score_input * score_hidden + score_hidden)
            + 2 * score_hidden
            + (score_hidden * score_bottleneck + score_bottleneck)
            + 2 * score_bottleneck
            + (score_bottleneck + 1)
        )
        eligibility = (
            2 * self.context_dim
            + (self.context_dim * eligibility_hidden + eligibility_hidden)
            + 2 * eligibility_hidden
            + (eligibility_hidden + 1)
        )
        return ranking + eligibility

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Guide2VecConfig":
        if not isinstance(value, Mapping):
            raise Guide2VecError("Guide2Vec config must be a mapping")
        expected = set(cls.__dataclass_fields__)
        missing = expected.difference(value)
        unknown = set(value).difference(expected)
        if missing or unknown:
            raise Guide2VecError(
                "Guide2Vec config fields changed: "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Guide2VecDecision:
    """Per-row audit result of a bounded overlay or exact direct fallback."""

    base_logits: Tensor
    adjusted_logits: Tensor
    guide_scores: Tensor
    bonus: Tensor
    eligibility_probability: Tensor
    abstain_probability: Tensor
    confidence: Tensor
    applied: Tensor
    fallback: Tensor
    base_indices: Tensor
    selected_indices: Tensor
    reasons: tuple[str, ...]


def freeze_base_model(model: nn.Module) -> nn.Module:
    """Put a parent model in eval mode and remove every gradient route."""

    if not isinstance(model, nn.Module):
        raise Guide2VecError("base model must be a torch.nn.Module")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def assert_base_frozen(model: nn.Module) -> None:
    """Fail closed unless a parent is eval-only and has no trainable tensors."""

    if not isinstance(model, nn.Module):
        raise Guide2VecError("base model must be a torch.nn.Module")
    if model.training:
        raise Guide2VecError("frozen base model must be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise Guide2VecError("frozen base model has trainable parameters")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def verify_base_checkpoint(path: str | Path, identity: FrozenBaseIdentity) -> Path:
    """Verify an on-disk parent checkpoint before a sidecar is attached."""

    if not isinstance(identity, FrozenBaseIdentity):
        raise Guide2VecError("base checkpoint identity is invalid")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise Guide2VecError("base checkpoint is not a regular file")
    if (
        identity.checkpoint_bytes is not None
        and resolved.stat().st_size != identity.checkpoint_bytes
    ):
        raise Guide2VecError("base checkpoint bytes do not match frozen identity")
    observed = _sha256_file(resolved)
    if not hmac.compare_digest(observed, identity.checkpoint_sha256):
        raise Guide2VecError("base checkpoint digest does not match frozen identity")
    return resolved


def state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash dense sidecar weights canonically rather than pickled bytes.

    Tensor names, dtypes, shapes, and raw CPU-contiguous bytes are all bound.
    This keeps the digest stable across a CPU/GPU checkpoint round-trip while
    still detecting an otherwise invisible architecture or weight change.
    """

    if not isinstance(state_dict, Mapping):
        raise Guide2VecError("state_dict must be a mapping")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise Guide2VecError("state_dict must contain string tensor entries")
        if tensor.layout != torch.strided:
            raise Guide2VecError("Guide2Vec state_dict must contain dense tensors")
        value = tensor.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        # Viewing as uint8 supports every dense Torch dtype, including bfloat16.
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def make_checkpoint_payload(
    head: "Guide2VecHead",
    base_identity: FrozenBaseIdentity,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a self-verifying, separately-owned sidecar checkpoint payload."""

    if not isinstance(head, Guide2VecHead):
        raise Guide2VecError("Guide2Vec checkpoint requires a Guide2VecHead")
    if not isinstance(base_identity, FrozenBaseIdentity):
        raise Guide2VecError("Guide2Vec checkpoint requires a frozen base identity")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise Guide2VecError("Guide2Vec checkpoint metadata must be a mapping")
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in head.state_dict().items()
    }
    # Require metadata to be JSON-safe now; it becomes part of the artifact
    # record even though model weights are intentionally not JSON serialised.
    if metadata is not None:
        _canonical_json_sha256(dict(metadata))
    return {
        "schema": GUIDE2VEC_CHECKPOINT_SCHEMA,
        "config": head.config.as_dict(),
        "base_identity": base_identity.as_dict(),
        "base_identity_sha256": base_identity.identity_sha256,
        "parameter_count": head.parameter_count,
        "state_dict": state,
        "state_dict_sha256": state_dict_sha256(state),
        "metadata": dict(metadata or {}),
    }


def load_checkpoint_payload(
    payload: Mapping[str, object],
    expected_base_identity: FrozenBaseIdentity | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple["Guide2VecHead", FrozenBaseIdentity, dict[str, object]]:
    """Strictly reconstruct a sidecar and fail closed on identity mismatch."""

    if not isinstance(payload, Mapping):
        raise Guide2VecError("Guide2Vec checkpoint payload must be a mapping")
    required = {
        "schema",
        "config",
        "base_identity",
        "base_identity_sha256",
        "parameter_count",
        "state_dict",
        "state_dict_sha256",
        "metadata",
    }
    missing = required.difference(payload)
    unknown = set(payload).difference(required)
    if missing or unknown:
        raise Guide2VecError(
            "Guide2Vec checkpoint fields changed: "
            f"missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if payload["schema"] != GUIDE2VEC_CHECKPOINT_SCHEMA:
        raise Guide2VecError("Guide2Vec checkpoint schema changed")
    config_raw = payload["config"]
    identity_raw = payload["base_identity"]
    state_raw = payload["state_dict"]
    metadata_raw = payload["metadata"]
    if not isinstance(config_raw, Mapping):
        raise Guide2VecError("Guide2Vec checkpoint config is invalid")
    if not isinstance(identity_raw, Mapping):
        raise Guide2VecError("Guide2Vec checkpoint base identity is invalid")
    if not isinstance(state_raw, Mapping):
        raise Guide2VecError("Guide2Vec checkpoint state_dict is invalid")
    if not isinstance(metadata_raw, Mapping):
        raise Guide2VecError("Guide2Vec checkpoint metadata is invalid")
    config = Guide2VecConfig.from_mapping(config_raw)
    base_identity = FrozenBaseIdentity.from_mapping(identity_raw)
    claimed_identity_digest = _normalize_digest(
        payload["base_identity_sha256"],
        label="base_identity_sha256",
        required=True,
    )
    if claimed_identity_digest != base_identity.identity_sha256:
        raise Guide2VecError("Guide2Vec checkpoint base identity digest mismatch")
    if (
        expected_base_identity is not None
        and base_identity != expected_base_identity
    ):
        raise Guide2VecError("Guide2Vec checkpoint base identity mismatch")
    try:
        claimed_count = int(payload["parameter_count"])
    except (TypeError, ValueError) as exc:
        raise Guide2VecError("Guide2Vec checkpoint parameter count is invalid") from exc
    if type(payload["parameter_count"]) is bool:
        raise Guide2VecError("Guide2Vec checkpoint parameter count is invalid")
    expected_digest = _normalize_digest(
        payload["state_dict_sha256"],
        label="state_dict_sha256",
        required=True,
    )
    typed_state: dict[str, Tensor] = {}
    for name, value in state_raw.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise Guide2VecError("Guide2Vec checkpoint state_dict is invalid")
        typed_state[name] = value.detach().to(map_location)
    if state_dict_sha256(typed_state) != expected_digest:
        raise Guide2VecError("Guide2Vec checkpoint state_dict digest mismatch")
    head = Guide2VecHead(config)
    if claimed_count != head.parameter_count:
        raise Guide2VecError("Guide2Vec checkpoint parameter count mismatch")
    try:
        head.load_state_dict(typed_state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise Guide2VecError("Guide2Vec checkpoint state_dict does not fit head") from exc
    # Preserve the normal PyTorch load contract but place all tensor parameters
    # on the requested device for a caller that explicitly supplies one.
    head.to(map_location)
    return head, base_identity, dict(metadata_raw)


class Guide2VecHead(nn.Module):
    """155,468-parameter legal-option ranker plus stage eligibility head.

    ``state_vec`` is the frozen base's causal public-board/history state.
    ``option_hidden`` is the base decoder's representation of the exact current
    legal factorized candidates.  The module has no positional option input;
    reordering candidates reorders scores exactly.
    """

    def __init__(self, config: Guide2VecConfig | None = None) -> None:
        super().__init__()
        self.config = config or Guide2VecConfig()
        if not isinstance(self.config, Guide2VecConfig):
            raise Guide2VecError("config must be a Guide2VecConfig")

        self.score_net = nn.Sequential(
            nn.LayerNorm(self.config.score_input_dim),
            nn.Linear(self.config.score_input_dim, self.config.score_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.score_hidden_dim),
            nn.Linear(self.config.score_hidden_dim, self.config.score_bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.score_bottleneck_dim),
            nn.Linear(self.config.score_bottleneck_dim, 1),
        )
        self.eligibility_net = nn.Sequential(
            nn.LayerNorm(self.config.context_dim),
            nn.Linear(self.config.context_dim, self.config.eligibility_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.eligibility_hidden_dim),
            nn.Linear(self.config.eligibility_hidden_dim, 1),
        )
        if self.parameter_count != self.config.expected_parameter_count:
            raise AssertionError("Guide2Vec layer inventory disagrees with config")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _normalize_shapes(
        state_vec: Tensor,
        option_hidden: Tensor,
        base_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not all(isinstance(value, Tensor) for value in (state_vec, option_hidden, base_logits)):
            raise Guide2VecError("Guide2Vec inputs must be torch tensors")
        if state_vec.ndim != 2:
            raise Guide2VecError("state_vec must have shape [batch, d_model]")
        if option_hidden.ndim != 3:
            raise Guide2VecError(
                "option_hidden must have shape [batch, max_options, d_model]"
            )
        if base_logits.ndim != 2:
            raise Guide2VecError("base_logits must have shape [batch, max_options]")
        if not (
            state_vec.is_floating_point()
            and option_hidden.is_floating_point()
            and base_logits.is_floating_point()
        ):
            raise Guide2VecError("Guide2Vec inputs must use floating point tensors")
        batch, width, d_model = option_hidden.shape
        if (
            state_vec.shape != (batch, d_model)
            or base_logits.shape != (batch, width)
        ):
            raise Guide2VecError("Guide2Vec input shapes do not align")
        return state_vec, option_hidden, base_logits

    def _counts_and_mask(
        self,
        base_logits: Tensor,
        n_options: int | Sequence[int] | Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch, width = base_logits.shape
        if width <= 0:
            raise Guide2VecError("Guide2Vec requires at least one option column")
        if n_options is None:
            counts = torch.full(
                (batch,), width, dtype=torch.long, device=base_logits.device
            )
        elif isinstance(n_options, int) and not isinstance(n_options, bool):
            if batch != 1:
                raise Guide2VecError("scalar n_options is valid only for batch size one")
            counts = torch.tensor([n_options], dtype=torch.long, device=base_logits.device)
        else:
            try:
                raw_counts = torch.as_tensor(n_options, device=base_logits.device).reshape(-1)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise Guide2VecError("n_options must be an integer vector") from exc
            if raw_counts.dtype == torch.bool or raw_counts.is_complex():
                raise Guide2VecError("n_options must be an integer vector")
            if raw_counts.is_floating_point():
                if not bool(torch.isfinite(raw_counts).all().item()) or not bool(
                    (raw_counts == torch.floor(raw_counts)).all().item()
                ):
                    raise Guide2VecError("n_options must be an integer vector")
            elif raw_counts.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise Guide2VecError("n_options must be an integer vector")
            counts = raw_counts.to(dtype=torch.long)
        if counts.numel() != batch:
            raise Guide2VecError("n_options does not match Guide2Vec batch size")
        if torch.any(counts < 1) or torch.any(counts > width):
            raise Guide2VecError("n_options must be within [1, max_options]")
        mask = torch.arange(width, device=base_logits.device).unsqueeze(0) < counts.unsqueeze(1)
        return counts, mask

    def _prepare(
        self,
        state_vec: Tensor,
        option_hidden: Tensor,
        base_logits: Tensor,
        n_options: int | Sequence[int] | Tensor | None,
        *,
        require_finite: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        state_vec, option_hidden, base_logits = self._normalize_shapes(
            state_vec, option_hidden, base_logits
        )
        if state_vec.size(-1) != self.config.d_model:
            raise Guide2VecError("state_vec d_model does not match Guide2Vec config")
        if option_hidden.size(-1) != self.config.d_model:
            raise Guide2VecError("option_hidden d_model does not match Guide2Vec config")
        if not (
            state_vec.device == option_hidden.device == base_logits.device
        ):
            raise Guide2VecError("Guide2Vec inputs must be on one device")
        parameter_dtype = next(self.parameters()).dtype
        if not (
            state_vec.dtype == option_hidden.dtype == base_logits.dtype == parameter_dtype
        ):
            raise Guide2VecError(
                "Guide2Vec inputs must use the sidecar parameter dtype"
            )
        counts, mask = self._counts_and_mask(base_logits, n_options)
        finite = torch.isfinite(state_vec).all(dim=1)
        # Padded representations/logits have no action authority.  In
        # particular, r195 policy logits intentionally use ``-inf`` padding.
        valid_hidden_finite = torch.where(
            mask.unsqueeze(-1), torch.isfinite(option_hidden), True
        ).all(dim=(1, 2))
        valid_base_finite = torch.where(mask, torch.isfinite(base_logits), True).all(dim=1)
        finite = finite & valid_hidden_finite & valid_base_finite
        if require_finite and not bool(finite.all().item()):
            raise Guide2VecError("Guide2Vec received nonfinite causal inputs")
        return state_vec, option_hidden, base_logits, counts, mask

    def _context(
        self,
        state_vec: Tensor,
        option_hidden: Tensor,
        base_logits: Tensor,
        counts: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build candidate-order-invariant, legal-only context features."""

        # amax/amin avoid order-dependent floating point sums, which makes
        # score permutation equivariance exact rather than approximate.
        option_mask = mask.unsqueeze(-1)
        floor = torch.finfo(option_hidden.dtype).min
        ceiling = torch.finfo(option_hidden.dtype).max
        option_max = option_hidden.masked_fill(~option_mask, floor).amax(dim=1)
        option_min = option_hidden.masked_fill(~option_mask, ceiling).amin(dim=1)
        base_floor = torch.finfo(base_logits.dtype).min
        base_max = base_logits.masked_fill(~mask, base_floor).amax(dim=1, keepdim=True)
        log_count = counts.to(dtype=state_vec.dtype).log().unsqueeze(1)
        context = torch.cat([state_vec, option_max, option_min, log_count, base_max], dim=-1)
        # Normalizing base logits by the legal maximum is shift invariant and
        # cannot introduce information about padded or unavailable options.
        normalized_base = torch.where(mask, base_logits - base_max, torch.zeros_like(base_logits))
        return context, normalized_base

    def forward(
        self,
        state_vec: Tensor,
        option_hidden: Tensor,
        base_logits: Tensor,
        n_options: int | Sequence[int] | Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return legal option guide scores and one stage eligibility logit.

        Training callers use the raw eligibility logits against the causal
        stage-availability target.  Padding is set to ``-inf`` exactly like the
        parent policy head, so it cannot enter a ranking loss or a runtime
        action decision.
        """

        state_vec, option_hidden, base_logits, counts, mask = self._prepare(
            state_vec,
            option_hidden,
            base_logits,
            n_options,
            require_finite=True,
        )
        context, normalized_base = self._context(
            state_vec, option_hidden, base_logits, counts, mask
        )
        width = option_hidden.size(1)
        repeated_context = context.unsqueeze(1).expand(-1, width, -1)
        score_input = torch.cat(
            [option_hidden, repeated_context, normalized_base.unsqueeze(-1)],
            dim=-1,
        )
        guide_scores = self.score_net(score_input).squeeze(-1)
        guide_scores = guide_scores.masked_fill(~mask, float("-inf"))
        eligibility_logits = self.eligibility_net(context).squeeze(-1)
        return guide_scores, eligibility_logits

    @staticmethod
    def _base_indices(base_logits: Tensor, mask: Tensor) -> Tensor:
        # Legal direct-policy scores are finite by construction before this is
        # used.  Re-mask nevertheless so an accidental finite padding logit
        # never becomes the returned fallback choice.
        return base_logits.masked_fill(~mask, float("-inf")).argmax(dim=-1)

    def _fallback_decision(
        self,
        base_logits: Tensor,
        mask: Tensor,
        *,
        reasons: Sequence[str],
        guide_scores: Tensor | None = None,
        eligibility_probability: Tensor | None = None,
    ) -> Guide2VecDecision:
        batch = base_logits.size(0)
        if guide_scores is None:
            guide_scores = torch.full_like(base_logits, float("-inf"))
        else:
            guide_scores = guide_scores.masked_fill(~mask, float("-inf"))
        if eligibility_probability is None:
            eligibility_probability = torch.zeros(
                batch, dtype=base_logits.dtype, device=base_logits.device
            )
        eligibility_probability = eligibility_probability.to(dtype=base_logits.dtype)
        applied = torch.zeros(batch, dtype=torch.bool, device=base_logits.device)
        bonus = torch.zeros_like(base_logits)
        base_indices = self._base_indices(base_logits, mask)
        return Guide2VecDecision(
            base_logits=base_logits,
            adjusted_logits=base_logits.clone(),
            guide_scores=guide_scores,
            bonus=bonus,
            eligibility_probability=eligibility_probability,
            abstain_probability=1.0 - eligibility_probability,
            confidence=eligibility_probability,
            applied=applied,
            fallback=~applied,
            base_indices=base_indices,
            selected_indices=base_indices,
            reasons=tuple(reasons),
        )

    @torch.no_grad()
    def rerank(
        self,
        state_vec: Tensor,
        option_hidden: Tensor,
        base_logits: Tensor,
        n_options: int | Sequence[int] | Tensor | None = None,
        *,
        expected_base_identity: FrozenBaseIdentity | None = None,
        observed_base_identity: FrozenBaseIdentity | None = None,
    ) -> Guide2VecDecision:
        """Apply the bounded overlay only to eligible, uniquely ranked rows.

        An identity mismatch, malformed tensor, singleton legal stage,
        nonfinite value, low eligibility, or tied/low-margin score all produces
        an *exact* direct-policy-logit fallback for that row.  ``bonus`` is
        always in ``[0, 0.05]`` and is zero on padding/fallback rows.
        """

        # Shape/count validation is intentionally outside the generic failure
        # handler: without a rectangular base tensor there is no safe direct
        # fallback artifact to return.
        state_vec, option_hidden, base_logits = self._normalize_shapes(
            state_vec, option_hidden, base_logits
        )
        _, mask = self._counts_and_mask(base_logits, n_options)
        batch = base_logits.size(0)
        if (
            not isinstance(expected_base_identity, FrozenBaseIdentity)
            or not isinstance(observed_base_identity, FrozenBaseIdentity)
            or observed_base_identity != expected_base_identity
        ):
            return self._fallback_decision(
                base_logits,
                mask,
                reasons=("base_identity_mismatch",) * batch,
            )
        try:
            guide_scores, eligibility_logits = self.forward(
                state_vec, option_hidden, base_logits, n_options
            )
        except Guide2VecError:
            return self._fallback_decision(
                base_logits,
                mask,
                reasons=("nonfinite_or_invalid_input",) * batch,
            )

        # A malformed/corrupt sidecar must never turn into a NaN bonus or an
        # unverifiable audit field.  Padding scores are intentionally -inf;
        # only legal option scores must be finite.
        legal_scores_finite = torch.where(
            mask, torch.isfinite(guide_scores), True
        ).all()
        if not bool(legal_scores_finite.item()) or not bool(
            torch.isfinite(eligibility_logits).all().item()
        ):
            return self._fallback_decision(
                base_logits,
                mask,
                reasons=("nonfinite_sidecar_output",) * batch,
            )

        _, _, _, counts, mask = self._prepare(
            state_vec,
            option_hidden,
            base_logits,
            n_options,
            require_finite=True,
        )
        eligibility_probability = torch.sigmoid(eligibility_logits)
        # Scores at padding are -inf.  Use a legal-only top-two margin and
        # explicitly reject singleton rows rather than treating their margin as
        # infinite eligibility.
        if guide_scores.size(1) < 2:
            margin = torch.full_like(eligibility_probability, float("-inf"))
        else:
            ranked = guide_scores.masked_fill(~mask, float("-inf")).topk(
                k=2, dim=-1
            ).values
            margin = ranked[:, 0] - ranked[:, 1]
        enough_options = counts >= 2
        eligible = eligibility_probability >= float(self.config.min_eligibility)
        unique = margin >= float(self.config.min_score_margin)
        apply = enough_options & eligible & unique

        minimum = guide_scores.masked_fill(~mask, float("inf")).amin(dim=-1, keepdim=True)
        maximum = guide_scores.masked_fill(~mask, float("-inf")).amax(dim=-1, keepdim=True)
        spread = maximum - minimum
        # Rows failing uniqueness must retain a zero bonus; clamp protects a
        # future calibration from producing a negative or >0.05 adjustment.
        normalized = (guide_scores - minimum) / spread.clamp_min(torch.finfo(guide_scores.dtype).eps)
        normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
        bonus = normalized * (float(self.config.max_logit_bonus) * eligibility_probability.unsqueeze(1))
        bonus = bonus.clamp(min=0.0, max=float(self.config.max_logit_bonus))
        bonus = torch.where(apply.unsqueeze(1) & mask, bonus, torch.zeros_like(bonus))
        # An applied row must keep padded columns impossible.  A fallback row
        # is different: it is an exact record of the frozen direct-policy
        # logits, including whatever padding representation the caller used.
        # Use a separately masked candidate tensor only for applied rows.
        candidate_adjusted = (base_logits + bonus).masked_fill(~mask, float("-inf"))
        adjusted = torch.where(apply.unsqueeze(1), candidate_adjusted, base_logits)
        base_indices = self._base_indices(base_logits, mask)
        candidate_indices = candidate_adjusted.argmax(dim=-1)
        selected_indices = torch.where(apply, candidate_indices, base_indices)
        reasons: list[str] = []
        for row in range(batch):
            if bool(apply[row].item()):
                reasons.append("bounded_eligible_guide_bonus")
            elif not bool(enough_options[row].item()):
                reasons.append("singleton_legal_stage")
            elif not bool(eligible[row].item()):
                reasons.append("low_eligibility")
            else:
                reasons.append("guide_margin_not_unique")
        return Guide2VecDecision(
            base_logits=base_logits,
            adjusted_logits=adjusted,
            guide_scores=guide_scores,
            bonus=bonus,
            eligibility_probability=eligibility_probability,
            abstain_probability=1.0 - eligibility_probability,
            confidence=eligibility_probability,
            applied=apply,
            fallback=~apply,
            base_indices=base_indices,
            selected_indices=selected_indices,
            reasons=tuple(reasons),
        )


__all__ = [
    "ALAKAZAM_R195_BUNDLE_SHA256",
    "ALAKAZAM_R195_CHECKPOINT_BYTES",
    "ALAKAZAM_R195_CHECKPOINT_SHA256",
    "ALAKAZAM_SUBMISSION_ID",
    "FROZEN_BASE_IDENTITY_SCHEMA",
    "FrozenBaseIdentity",
    "GUIDE2VEC_CHECKPOINT_SCHEMA",
    "Guide2VecConfig",
    "Guide2VecDecision",
    "Guide2VecError",
    "Guide2VecHead",
    "MAX_LOGIT_BONUS",
    "MAX_PARAMETER_COUNT",
    "MIN_PARAMETER_COUNT",
    "assert_base_frozen",
    "freeze_base_model",
    "load_checkpoint_payload",
    "make_checkpoint_payload",
    "state_dict_sha256",
    "verify_base_checkpoint",
]
