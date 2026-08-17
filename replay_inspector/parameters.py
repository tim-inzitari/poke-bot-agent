"""Read-only, bounded model-parameter inspection helpers.

The replay inspector deliberately works from a model which has already been
loaded through the checksum-bound inference cache.  This module never opens a
checkpoint path itself, and it never returns an unbounded tensor payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

PARAMETER_INSPECTION_SCHEMA = "poke_bot.replay_model_inspector.parameters/v1"
DEFAULT_HISTOGRAM_BINS = 64
MAX_HISTOGRAM_BINS = 256
DEFAULT_SLICE_LIMIT = 256
MAX_SLICE_LIMIT = 4096


class ParameterInspectionError(ValueError):
    """A requested tensor name or bounded inspection request is invalid."""


def _availability(*, available: bool, reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"available": bool(available)}
    if reason is not None:
        result["reason"] = str(reason)
    return result


def _json_number(value: float) -> float | None:
    """Produce strict-JSON-friendly numbers without silently changing infinities."""

    value = float(value)
    return value if math.isfinite(value) else None


def _tensor_values(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a tensor into a CPU floating view suitable for read-only stats."""

    # Statistics for integer buffers are still meaningful, while float64 keeps
    # the summary stable for half/bfloat16 checkpoint tensors.
    return tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64)


def _tensor_metadata(
    name: str,
    tensor: torch.Tensor,
    *,
    kind: str,
) -> dict[str, Any]:
    return {
        "name": str(name),
        "kind": str(kind),
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "element_count": int(tensor.numel()),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }


def tensor_summary(
    name: str,
    tensor: torch.Tensor,
    *,
    kind: str = "parameter",
) -> dict[str, Any]:
    """Return finite-aware descriptive statistics for one tensor.

    Norms and descriptive statistics are calculated over finite entries only;
    their basis and the non-finite count are returned explicitly rather than
    converting NaN/Inf to zero.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_summary requires a torch.Tensor")
    values = _tensor_values(tensor)
    finite = torch.isfinite(values)
    finite_values = values[finite]
    count = int(values.numel())
    finite_count = int(finite_values.numel())
    zero_count = int((values == 0).sum().item())
    result = {
        "schema": PARAMETER_INSPECTION_SCHEMA,
        "availability": _availability(available=True),
        **_tensor_metadata(name, tensor, kind=kind),
        "finite_count": finite_count,
        "non_finite_count": count - finite_count,
        "zero_count": zero_count,
        "zero_fraction": (float(zero_count) / count) if count else None,
        "statistics_basis": "finite_values_only",
    }
    if not finite_count:
        result.update(
            {
                "minimum": None,
                "maximum": None,
                "mean": None,
                "standard_deviation": None,
                "l1_norm": None,
                "l2_norm": None,
            }
        )
        return result
    result.update(
        {
            "minimum": _json_number(float(finite_values.min().item())),
            "maximum": _json_number(float(finite_values.max().item())),
            "mean": _json_number(float(finite_values.mean().item())),
            # ``unbiased=False`` gives a meaningful zero for a single value.
            "standard_deviation": _json_number(
                float(finite_values.std(unbiased=False).item())
            ),
            "l1_norm": _json_number(float(finite_values.abs().sum().item())),
            "l2_norm": _json_number(
                float(torch.linalg.vector_norm(finite_values, ord=2).item())
            ),
        }
    )
    return result


def tensor_histogram(
    name: str,
    tensor: torch.Tensor,
    *,
    kind: str = "parameter",
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> dict[str, Any]:
    """Return a bounded equal-width finite-value histogram for a tensor."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_histogram requires a torch.Tensor")
    bins = int(bins)
    if not 1 <= bins <= MAX_HISTOGRAM_BINS:
        raise ParameterInspectionError(
            f"histogram bins must be in 1..{MAX_HISTOGRAM_BINS}"
        )
    values = _tensor_values(tensor)
    finite_values = values[torch.isfinite(values)]
    metadata = _tensor_metadata(name, tensor, kind=kind)
    if not int(finite_values.numel()):
        return {
            "schema": PARAMETER_INSPECTION_SCHEMA,
            "availability": _availability(
                available=False,
                reason="tensor has no finite values for a histogram",
            ),
            **metadata,
            "bins": bins,
            "finite_count": 0,
            "edges": [],
            "counts": [],
        }

    lower = float(finite_values.min().item())
    upper = float(finite_values.max().item())
    if lower == upper:
        return {
            "schema": PARAMETER_INSPECTION_SCHEMA,
            "availability": _availability(available=True),
            **metadata,
            "bins": 1,
            "finite_count": int(finite_values.numel()),
            "edges": [_json_number(lower), _json_number(upper)],
            "counts": [int(finite_values.numel())],
            "constant_value": _json_number(lower),
        }

    # ``histc`` is a CPU-only reduction here; it neither alters nor aliases the
    # resident model tensor.  Values were widened to float64 above, but histc
    # accepts float32 consistently on the supported PyTorch versions.
    counts_tensor = torch.histc(
        finite_values.to(dtype=torch.float32), bins=bins, min=lower, max=upper
    )
    width = (upper - lower) / bins
    edges = [lower + width * index for index in range(bins + 1)]
    # Use the exact extrema for stable presentation despite float multiplication.
    edges[0] = lower
    edges[-1] = upper
    return {
        "schema": PARAMETER_INSPECTION_SCHEMA,
        "availability": _availability(available=True),
        **metadata,
        "bins": bins,
        "finite_count": int(finite_values.numel()),
        "edges": [_json_number(value) for value in edges],
        "counts": [int(value) for value in counts_tensor.tolist()],
        "constant_value": None,
    }


def tensor_slice(
    name: str,
    tensor: torch.Tensor,
    *,
    kind: str = "parameter",
    offset: int = 0,
    limit: int = DEFAULT_SLICE_LIMIT,
) -> dict[str, Any]:
    """Return one bounded flat slice without exposing a whole checkpoint tensor."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_slice requires a torch.Tensor")
    offset = int(offset)
    limit = int(limit)
    if offset < 0:
        raise ParameterInspectionError("slice offset must be nonnegative")
    if not 1 <= limit <= MAX_SLICE_LIMIT:
        raise ParameterInspectionError(f"slice limit must be in 1..{MAX_SLICE_LIMIT}")
    values = _tensor_values(tensor)
    total = int(values.numel())
    if offset > total:
        raise ParameterInspectionError(
            f"slice offset {offset} exceeds tensor element count {total}"
        )
    end = min(total, offset + limit)
    selected = values[offset:end]
    return {
        "schema": PARAMETER_INSPECTION_SCHEMA,
        "availability": _availability(available=True),
        **_tensor_metadata(name, tensor, kind=kind),
        "offset": offset,
        "end_offset_exclusive": end,
        "limit": limit,
        "returned_count": end - offset,
        "next_offset": end if end < total else None,
        "values": [_json_number(value) for value in selected.tolist()],
    }


@dataclass(frozen=True)
class TensorReference:
    """A registered tensor name and whether it is a parameter or a buffer."""

    name: str
    kind: str
    tensor: torch.Tensor


class ParameterInspector:
    """Expose a loaded model's state through strict names and bounded views."""

    def __init__(self, model: torch.nn.Module, *, include_buffers: bool = True) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("ParameterInspector requires a torch.nn.Module")
        entries: dict[str, TensorReference] = {}
        for name, tensor in model.named_parameters(recurse=True):
            entries[name] = TensorReference(name, "parameter", tensor)
        if include_buffers:
            for name, tensor in model.named_buffers(recurse=True):
                # PyTorch state keys cannot normally collide, but preserve the
                # learned parameter if a custom module makes one ambiguous.
                entries.setdefault(name, TensorReference(name, "buffer", tensor))
        self._entries = entries

    @property
    def tensor_count(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def _entry(self, name: str) -> TensorReference:
        try:
            return self._entries[str(name)]
        except KeyError as exc:
            raise ParameterInspectionError(f"unknown model tensor: {name}") from exc

    def inventory(self) -> dict[str, Any]:
        """Return metadata only; summaries remain individual lazy operations."""

        tensors = [
            _tensor_metadata(reference.name, reference.tensor, kind=reference.kind)
            for reference in (self._entries[name] for name in self.names())
        ]
        return {
            "schema": PARAMETER_INSPECTION_SCHEMA,
            "availability": _availability(available=True),
            "tensor_count": len(tensors),
            "parameter_count": sum(
                1
                for reference in self._entries.values()
                if reference.kind == "parameter"
            ),
            "buffer_count": sum(
                1 for reference in self._entries.values() if reference.kind == "buffer"
            ),
            "element_count": sum(
                int(reference.tensor.numel()) for reference in self._entries.values()
            ),
            "tensors": tensors,
        }

    def summary(self, name: str) -> dict[str, Any]:
        reference = self._entry(name)
        return tensor_summary(reference.name, reference.tensor, kind=reference.kind)

    def histogram(
        self, name: str, *, bins: int = DEFAULT_HISTOGRAM_BINS
    ) -> dict[str, Any]:
        reference = self._entry(name)
        return tensor_histogram(
            reference.name, reference.tensor, kind=reference.kind, bins=bins
        )

    def slice(
        self,
        name: str,
        *,
        offset: int = 0,
        limit: int = DEFAULT_SLICE_LIMIT,
    ) -> dict[str, Any]:
        reference = self._entry(name)
        return tensor_slice(
            reference.name,
            reference.tensor,
            kind=reference.kind,
            offset=offset,
            limit=limit,
        )


__all__ = [
    "DEFAULT_HISTOGRAM_BINS",
    "DEFAULT_SLICE_LIMIT",
    "MAX_HISTOGRAM_BINS",
    "MAX_SLICE_LIMIT",
    "PARAMETER_INSPECTION_SCHEMA",
    "ParameterInspectionError",
    "ParameterInspector",
    "TensorReference",
    "tensor_histogram",
    "tensor_slice",
    "tensor_summary",
]
