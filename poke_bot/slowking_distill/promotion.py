"""Promotion gate for Slowking distill artifacts — always fail-closed.

This module never grants selector / serving / training / freeze / registration
authority. Even with every research flag satisfied, ``promoted`` remains false
unless an *external* human operator sets ``allow_external_research_tag`` for a
non-authoritative research tag only (still not a production selector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class PromotionRequest:
    eval_gate_passed: bool = False
    paired_win_delta: float = 0.0
    action_agreement: float = 0.0
    stage_d_win_rate: float = 0.0
    actor_checkpoint: str = ""
    selector_receipt_present: bool = False
    training_authority_requested: bool = False
    serving_authority_requested: bool = False
    #: Research-only tag (never a production selector key).
    allow_external_research_tag: bool = False
    research_tag: str = ""


@dataclass
class PromotionDecision:
    promoted: bool
    research_tag_allowed: bool
    reasons: list[str] = field(default_factory=list)
    authority: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "research_tag_allowed": self.research_tag_allowed,
            "reasons": list(self.reasons),
            "authority": dict(self.authority),
        }


def evaluate_promotion(request: PromotionRequest | Mapping[str, Any]) -> PromotionDecision:
    """Hard fail-closed promotion evaluator.

    - ``promoted`` is **always** False (no self-promotion into production).
    - ``research_tag_allowed`` may be True only when eval passes *and*
      ``allow_external_research_tag`` is explicitly set — still grants no
      selector/serving/training authority.
    """
    if isinstance(request, Mapping):
        req = PromotionRequest(**{
            k: request[k]
            for k in PromotionRequest.__dataclass_fields__
            if k in request
        })
    else:
        req = request

    reasons: list[str] = []
    if req.training_authority_requested:
        reasons.append("training_authority_forbidden")
    if req.serving_authority_requested:
        reasons.append("serving_authority_forbidden")
    if req.selector_receipt_present:
        reasons.append("selector_receipt_ignored_research_lane")
    if not req.eval_gate_passed:
        reasons.append("eval_gate_not_passed")
    if not req.actor_checkpoint:
        reasons.append("missing_actor_checkpoint")

    research_tag_allowed = bool(
        req.allow_external_research_tag
        and req.eval_gate_passed
        and req.actor_checkpoint
        and not req.training_authority_requested
        and not req.serving_authority_requested
        and bool(req.research_tag)
    )
    if req.allow_external_research_tag and not research_tag_allowed:
        reasons.append("external_research_tag_rejected")
    elif research_tag_allowed:
        reasons.append("research_tag_only_no_production_authority")

    # Hard: never promote into production selectors / serving.
    reasons.append("pipeline_never_self_promotes")

    return PromotionDecision(
        promoted=False,
        research_tag_allowed=research_tag_allowed,
        reasons=reasons,
        authority={
            "research_only": True,
            "training_authority": False,
            "serving_authority": False,
            "selector_authority": False,
            "runtime_authority": "none",
            "research_tag": req.research_tag if research_tag_allowed else "",
        },
    )


__all__ = [
    "PromotionDecision",
    "PromotionRequest",
    "evaluate_promotion",
]
