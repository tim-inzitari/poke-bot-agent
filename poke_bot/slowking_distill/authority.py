"""Hard authority stamps for the Slowking distill research lane."""

from __future__ import annotations

RESEARCH_ONLY = True
RUNTIME_AUTHORITY = "none"
TRAINING_AUTHORITY = False
SERVING_AUTHORITY = False
SELECTOR_AUTHORITY = False

PIPELINE_SCHEMA = "poke_bot.slowking_distill.pipeline/v1"
DECISION_SCHEMA = "poke_bot.slowking_distill.decision/v1"
SEARCH_RECEIPT_SCHEMA = "poke_bot.slowking_distill.search_receipt/v1"
SPLIT_SCHEMA = "poke_bot.slowking_distill.day_split/v1"
EVAL_GATE_SCHEMA = "poke_bot.slowking_distill.eval_gate/v1"
