"""Complexity router: direct / root-only / recursive."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
from torch import Tensor

from .config import PokeRLMConfig
from .reasons import ReasonCode


class Route(str, Enum):
    DIRECT = "direct"
    ROOT = "root"
    RECURSIVE = "recursive"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: ReasonCode
    diagnostics: dict[str, Any]


def policy_entropy(logits: Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    finite = torch.isfinite(logits)
    if not bool(finite.any()):
        return 0.0
    x = logits[finite]
    probs = torch.softmax(x.float(), dim=-1)
    return float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())


def choose_route(
    config: PokeRLMConfig,
    *,
    n_legal: int,
    policy_logits: Optional[Tensor] = None,
    force: Optional[Route] = None,
) -> RouteDecision:
    if force is not None:
        return RouteDecision(force, ReasonCode.OK, {"forced": force.value})
    if config.skip_trivial_decisions and n_legal <= 1:
        return RouteDecision(
            Route.DIRECT,
            ReasonCode.TRIVIAL_DIRECT,
            {"n_legal": n_legal},
        )
    entropy = policy_entropy(policy_logits) if policy_logits is not None else 0.0
    by_options = n_legal >= config.complexity_option_threshold
    by_entropy = entropy >= config.complexity_entropy_threshold
    if by_options and by_entropy:
        route = Route.RECURSIVE
    elif by_options or by_entropy:
        route = Route.ROOT
    else:
        route = Route.DIRECT
    return RouteDecision(
        route,
        ReasonCode.OK,
        {
            "n_legal": n_legal,
            "entropy": entropy,
            "by_options": by_options,
            "by_entropy": by_entropy,
        },
    )
