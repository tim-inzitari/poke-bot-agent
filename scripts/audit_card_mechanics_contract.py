#!/usr/bin/env python3
"""Audit simulator -> observation -> feature -> model card-mechanics coverage.

The report keeps three claims separate:

* ``simulator``: represented/enforced by the competition engine,
* ``feature``: explicitly visible to the current policy/value input, and
* ``learnable``: direct, ID-memorized, history-inferred, or not identifiable.

It is intentionally fail-closed.  Known high-severity representation aliases
produce a non-zero exit unless ``--allow-known-gaps`` is explicitly supplied.
No production model, service, checkpoint, or engine is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import card2vec, cg_env, features  # noqa: E402
from poke_bot.card_metadata import (  # noqa: E402
    CARD_METADATA_SCHEMA,
    MetadataCatalog,
    MetadataContractError,
    build_metadata_catalog,
)


REPORT_SCHEMA = "pokebot.card-mechanics-contract-audit.v1"
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, Any]:
    suffixes = {".h", ".hpp", ".c", ".cc", ".cpp", ".py", ".toml", ".txt"}
    files: dict[str, str] = {}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and (
                path.suffix.lower() in suffixes or path.name == "LICENSE"
            ):
                files[str(path.relative_to(root))] = _sha256_file(path)
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "root": str(root),
        "file_count": len(files),
        "tree_sha256": "sha256:" + digest,
        "files": files,
    }


def compare_source_trees(
    official: Optional[Path], rebuilt: Optional[Path]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if official is not None:
        result["official"] = _tree_manifest(official)
    if rebuilt is not None:
        result["rebuilt"] = _tree_manifest(rebuilt)
        licenses = list(rebuilt.glob("LICENSE*")) + list(rebuilt.glob("LICENSES/*"))
        result["rebuilt"]["competition_license_present"] = bool(licenses)
        guard_file = rebuilt / "BattleData.h"
        guard_text = (
            guard_file.read_text(encoding="utf-8", errors="replace")
            if guard_file.is_file()
            else ""
        )
        result["rebuilt"]["action_count_3000_guard_present"] = (
            "actionCount >= 3000" in guard_text or "actionCount>=3000" in guard_text
        )
    if official is not None and rebuilt is not None:
        left = result["official"]["files"]
        right = result["rebuilt"]["files"]
        names = sorted(set(left).union(right))
        result["comparison"] = {
            "changed": [
                name
                for name in names
                if name in left and name in right and left[name] != right[name]
            ],
            "only_official": [name for name in names if name not in right],
            "only_rebuilt": [name for name in names if name not in left],
            "parity_proven": False,
            "note": "source equality is useful evidence; seeded transition parity is still required",
        }
    return result


def _card_type_profile(catalog: MetadataCatalog) -> dict[str, Any]:
    type_names = {
        0: "pokemon",
        1: "item",
        2: "tool",
        3: "supporter",
        4: "stadium",
        5: "basic_energy",
        6: "special_energy",
    }
    types = Counter(type_names[int(card["cardType"])] for card in catalog.cards)
    fossil_hybrids = [
        int(card["cardId"])
        for card in catalog.cards
        if int(card["cardType"]) != 0 and bool(card["basic"])
    ]
    return {
        "types": dict(sorted(types.items())),
        "stages": {
            "basic_flag": sum(bool(card["basic"]) for card in catalog.cards),
            "stage1_flag": sum(bool(card["stage1"]) for card in catalog.cards),
            "stage2_flag": sum(bool(card["stage2"]) for card in catalog.cards),
            "evolves_from": len(catalog.evolution_parents),
        },
        "special_rules": {
            "ex": sum(bool(card["ex"]) for card in catalog.cards),
            "mega_ex": sum(bool(card["megaEx"]) for card in catalog.cards),
            "tera": sum(bool(card["tera"]) for card in catalog.cards),
            "ace_spec": sum(bool(card["aceSpec"]) for card in catalog.cards),
            "cards_with_skills": sum(bool(card["skills"]) for card in catalog.cards),
            "skill_count": sum(len(card["skills"]) for card in catalog.cards),
        },
        "mechanics": {
            "cards_with_attacks": sum(bool(card["attacks"]) for card in catalog.cards),
            "cards_with_weakness": sum(
                card["weakness"] is not None for card in catalog.cards
            ),
            "cards_with_resistance": sum(
                card["resistance"] is not None for card in catalog.cards
            ),
        },
        "non_pokemon_basic_flag_hybrids": fossil_hybrids,
    }


def factor_table_audit(catalog: MetadataCatalog) -> dict[str, Any]:
    card_vocab = catalog.card_vocab
    attack_vocab = catalog.attack_vocab
    encoder_vocab = features.encoder_vocab_size()
    decoder_vocab = features.decoder_vocab_size()
    if (
        features.card_vocab_size() != card_vocab
        or features.attack_vocab_size() != attack_vocab
    ):
        raise MetadataContractError(
            "feature vocab differs from metadata catalog: "
            f"features=({features.card_vocab_size()},{features.attack_vocab_size()}), "
            f"catalog=({card_vocab},{attack_vocab})"
        )
    board_kind, board_eid, _ = card2vec.build_encoder_factor_tables(
        card_vocab, encoder_vocab
    )
    option_kind, option_eid, _ = card2vec.build_decoder_factor_tables(
        card_vocab, attack_vocab, decoder_vocab
    )
    board_cards = set(int(v) for v in board_eid[board_kind == 1].tolist())
    option_cards = set(int(v) for v in option_eid[option_kind == 1].tolist())
    option_attacks = set(int(v) for v in option_eid[option_kind == 2].tolist())
    expected_cards = set(range(1, card_vocab))
    expected_attacks = set(range(1, attack_vocab))
    return {
        "encoder_vocab": encoder_vocab,
        "decoder_vocab": decoder_vocab,
        "all_engine_cards_reachable_on_board": expected_cards.issubset(board_cards),
        "all_engine_cards_reachable_in_options": expected_cards.issubset(option_cards),
        "all_engine_attacks_reachable_in_options": expected_attacks.issubset(
            option_attacks
        ),
        "missing_board_card_ids": sorted(expected_cards - board_cards),
        "missing_option_card_ids": sorted(expected_cards - option_cards),
        "missing_option_attack_ids": sorted(expected_attacks - option_attacks),
        "metadata_zero_card_rows": int(
            (catalog.card_features[1:].abs().sum(dim=1) == 0).sum().item()
        ),
        "metadata_zero_attack_rows": int(
            (catalog.attack_features[1:].abs().sum(dim=1) == 0).sum().item()
        ),
        "distinct_card_mechanics_vectors": len(
            {bytes(row.numpy()) for row in catalog.card_features[1:]}
        ),
        "distinct_attack_mechanics_vectors": len(
            {bytes(row.numpy()) for row in catalog.attack_features[1:]}
        ),
        "note": (
            "fixed mechanics vectors may match for mechanically identical IDs; "
            "the existing learned identity table remains the unique identity source"
        ),
    }


def _read_deck_loose(path: Path) -> Optional[list[int]]:
    cards: list[int] = []
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            try:
                if len(parts) == 1:
                    cards.append(int(parts[0]))
                elif len(parts) >= 2:
                    cards.extend([int(parts[0])] * int(parts[1]))
            except ValueError:
                return None
    except OSError:
        return None
    return cards if len(cards) == 60 else None


def deck_census(
    catalog: MetadataCatalog,
    roots: Sequence[Path],
    json_paths: Sequence[Path],
) -> dict[str, Any]:
    decks: list[tuple[str, list[int]]] = []
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".deck"}:
                deck = _read_deck_loose(path)
                if deck is not None:
                    decks.append((str(path), deck))
    for path in json_paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name, value in (payload.get("decks") or {}).items():
            cards = value.get("card_ids") if isinstance(value, Mapping) else None
            if isinstance(cards, list) and len(cards) == 60:
                decks.append((f"{path}:{name}", [int(v) for v in cards]))
    emitted = {card for _, deck in decks for card in deck}
    valid = set(range(1, catalog.card_vocab))
    return {
        "valid_decks": len(decks),
        "distinct_emitted_card_ids": len(emitted),
        "oov_card_ids": sorted(emitted - valid),
        "engine_cards_absent_from_decks": len(valid - emitted),
        "deck_sources": [name for name, _ in decks[:50]],
        "deck_sources_truncated": len(decks) > 50,
    }


def _walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _observations(value: Any) -> Iterator[dict[str, Any]]:
    seen: set[int] = set()
    for item in _walk(value):
        if not isinstance(item, dict):
            continue
        if id(item) in seen:
            continue
        if isinstance(item.get("current"), dict) and isinstance(
            item.get("select"), dict
        ):
            seen.add(id(item))
            yield item


def _load_json_units(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        yield json.loads(path.read_text(encoding="utf-8"))


def _word_signature(
    vector: features.SparseVector, word: int
) -> tuple[tuple[int, float], ...]:
    start = vector.offset[word]
    end = vector.offset[word + 1] if word + 1 < vector.num_words else len(vector.index)
    coalesced: dict[int, float] = defaultdict(float)
    for index, value in zip(vector.index[start:end], vector.value[start:end]):
        coalesced[int(index)] += float(value)
    return tuple(sorted(coalesced.items()))


def _raw_option(option: Mapping[str, Any]) -> str:
    return json.dumps(option, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def trace_census(
    catalog: MetadataCatalog,
    paths: Sequence[Path],
    *,
    max_context: int,
) -> dict[str, Any]:
    decision_count = 0
    game_count = 0
    game_lengths: list[int] = []
    card_ids: Counter[int] = Counter()
    attack_ids: Counter[int] = Counter()
    option_types: Counter[int] = Counter()
    raw_duplicate_states = 0
    encoded_collision_states = 0
    collision_examples: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for path in paths:
        try:
            for unit in _load_json_units(path):
                observations = list(_observations(unit))
                if observations:
                    game_count += 1
                    game_lengths.append(len(observations))
                for obs in observations:
                    decision_count += 1
                    for item in _walk(obs):
                        if not isinstance(item, Mapping):
                            continue
                        if type(item.get("id")) is int:
                            card_ids[int(item["id"])] += 1
                        if type(item.get("cardId")) is int:
                            card_ids[int(item["cardId"])] += 1
                        if type(item.get("attackId")) is int:
                            attack_ids[int(item["attackId"])] += 1
                    options = (obs.get("select") or {}).get("option") or []
                    for option in options:
                        if (
                            isinstance(option, Mapping)
                            and type(option.get("type")) is int
                        ):
                            option_types[int(option["type"])] += 1
                    raw = [_raw_option(option) for option in options]
                    if len(raw) != len(set(raw)):
                        raw_duplicate_states += 1
                    try:
                        encoded = features.build_option_tokens(
                            obs, [[index] for index in range(len(options))]
                        )
                        groups: dict[tuple[tuple[int, float], ...], list[int]] = (
                            defaultdict(list)
                        )
                        for index in range(len(options)):
                            groups[_word_signature(encoded, index)].append(index)
                        distinct_collisions = []
                        for indices in groups.values():
                            if len(indices) > 1 and len({raw[i] for i in indices}) > 1:
                                distinct_collisions.append(indices)
                        if distinct_collisions:
                            encoded_collision_states += 1
                            if len(collision_examples) < 20:
                                collision_examples.append(
                                    {
                                        "path": str(path),
                                        "context": (obs.get("select") or {}).get(
                                            "context"
                                        ),
                                        "groups": [
                                            [options[i] for i in indices]
                                            for indices in distinct_collisions
                                        ],
                                    }
                                )
                    except Exception as exc:
                        if len(parse_errors) < 20:
                            parse_errors.append(f"{path}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            parse_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    valid_cards = set(range(1, catalog.card_vocab))
    valid_attacks = set(range(1, catalog.attack_vocab))
    over_context = sum(length > max_context for length in game_lengths)
    return {
        "files": len(paths),
        "games": game_count,
        "decisions": decision_count,
        "distinct_card_ids": len(card_ids),
        "distinct_attack_ids": len(attack_ids),
        "oov_card_ids": sorted(set(card_ids) - valid_cards),
        "oov_attack_ids": sorted(set(attack_ids) - valid_attacks),
        "option_type_counts": dict(sorted(option_types.items())),
        "raw_duplicate_option_states": raw_duplicate_states,
        "distinct_raw_options_with_same_feature_states": encoded_collision_states,
        "collision_examples": collision_examples,
        "temporal": {
            "max_context": max_context,
            "games_over_context": over_context,
            "games_measured": len(game_lengths),
            "over_context_rate": (
                over_context / len(game_lengths) if game_lengths else None
            ),
            "policy": "retain newest 320 decisions; track oldest-history truncation; do not expand",
        },
        "parse_errors": parse_errors,
    }


def representation_matrix() -> list[dict[str, str]]:
    return [
        {
            "mechanic": "card identity",
            "simulator": "exact",
            "feature": "exact validated cardId",
            "learnable": "direct learned identity",
        },
        {
            "mechanic": "attack identity",
            "simulator": "exact",
            "feature": "exact validated attackId",
            "learnable": "direct learned identity",
        },
        {
            "mechanic": "card type / energy type",
            "simulator": "exact",
            "feature": "not explicit",
            "learnable": "seen-ID memorization only",
        },
        {
            "mechanic": "stage / evolvesFrom / stack",
            "simulator": "exact",
            "feature": "top card only; stack omitted",
            "learnable": "legal-set proxy + seen-ID memorization",
        },
        {
            "mechanic": "evolution timing / Rare Candy",
            "simulator": "enforced",
            "feature": "appearThisTurn and prior stack omitted",
            "learnable": "current legality direct; future value partially hidden",
        },
        {
            "mechanic": "ex / Mega ex prize liability",
            "simulator": "1/2/3 plus effects",
            "feature": "not explicit",
            "learnable": "seen-ID memorization + prize deltas",
        },
        {
            "mechanic": "attacks: cost/damage/text",
            "simulator": "enforced",
            "feature": "attackId only",
            "learnable": "seen-ID memorization",
        },
        {
            "mechanic": "abilities / skills",
            "simulator": "enforced",
            "feature": "cardId; SKILL serial omitted",
            "learnable": "ambiguous when same card emits multiple skills",
        },
        {
            "mechanic": "weakness / resistance / Tera",
            "simulator": "enforced",
            "feature": "not explicit",
            "learnable": "seen-ID memorization",
        },
        {
            "mechanic": "HP / damage",
            "simulator": "exact",
            "feature": "current HP only; maxHp omitted",
            "learnable": "damage inferable only with known card",
        },
        {
            "mechanic": "special conditions",
            "simulator": "five flags",
            "feature": "five explicit flags",
            "learnable": "direct",
        },
        {
            "mechanic": "attached energy",
            "simulator": "typed units",
            "feature": "energy-card IDs only; raw energy units omitted",
            "learnable": "partial / ID memorization",
        },
        {
            "mechanic": "energy/damage selection residuals",
            "simulator": "ENERGY.count + remaining cost/counters are exact",
            "feature": "count, remainEnergyCost and remainDamageCounter omitted",
            "learnable": "legal-set/ID proxy; residual amount not direct",
        },
        {
            "mechanic": "retreat cost / readiness",
            "simulator": "cost, conditions and once-per-turn legality enforced",
            "feature": "RETREAT legality + active/energy-card identities only",
            "learnable": "seen-ID and legal-set proxy",
        },
        {
            "mechanic": "board zones / owners / slots",
            "simulator": "exact",
            "feature": "board slots + normalized exact option bindings",
            "learnable": "direct for exposed zones",
        },
        {
            "mechanic": "turn once-per-turn flags",
            "simulator": "enforced",
            "feature": "supporter/stadium/energy/retreat flags omitted",
            "learnable": "legal-set/history proxy",
        },
        {
            "mechanic": "prizes / deck-out / win",
            "simulator": "enforced",
            "feature": "counts visible; result/reason omitted",
            "learnable": "value target external; terminal reason hidden",
        },
        {
            "mechanic": "opponent hidden hand / remainder",
            "simulator": "exact only in privileged training fork/visual trace",
            "feature": "never an input; target-only binary multi-hot heads",
            "learnable": "masked BCE presence, not card multiplicity",
        },
        {
            "mechanic": "lethal / prize-take threat",
            "simulator": "exact prize deltas along recorded trajectory",
            "feature": "target-only played-line horizon label",
            "learnable": "aux BCE; terminal prize-takes can be right-censored",
        },
        {
            "mechanic": "prize race",
            "simulator": "public remaining-prize counts",
            "feature": "public counts plus redundant two-value aux target",
            "learnable": "direct board input + smooth-L1 reconstruction",
        },
        {
            "mechanic": "NUMBER options",
            "simulator": "exact integer",
            "feature": "all values >=4 share one bucket",
            "learnable": "not identifiable",
        },
        {
            "mechanic": "candidate ordinal",
            "simulator": "ordered option list",
            "feature": "not encoded for single-option tokens",
            "learnable": "duplicate payloads forced equal",
        },
    ]


def known_findings(
    trace: Mapping[str, Any], sources: Mapping[str, Any]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = [
        {
            "id": "FEATURE-NUMBER-ALIAS",
            "severity": "high",
            "confidence": "confirmed",
            "failed": "NUMBER values 4, 5, and every larger value encode identically",
            "impact": "legal choices can be impossible for the policy to distinguish",
            "remediation": "append exact bounded number/candidate-ordinal features and fail on overflow",
        },
        {
            "id": "FEATURE-SKILL-SERIAL-ALIAS",
            "severity": "high",
            "confidence": "confirmed",
            "failed": "SKILL serial is discarded; same-card skills can alias",
            "impact": "different effects can receive exactly the same policy logit",
            "remediation": "expose/encode serial plus a candidate ordinal residual",
        },
        {
            "id": "FEATURE-DUPLICATE-ATTACK-ORDINAL",
            "severity": "high",
            "confidence": "confirmed",
            "failed": "official API can emit identical ATTACK payloads at distinct option indices",
            "impact": "copy/source/bench-origin choices are not identifiable",
            "remediation": "encode exact candidate ordinal; extend ABI with source identity where possible",
        },
        {
            "id": "FEATURE-MECHANICS-IMPLICIT",
            "severity": "medium",
            "confidence": "code proof",
            "failed": "type/stage/evolution/prize liability/cost/text/weakness/resistance are identity-only",
            "impact": "unseen and rare cards cannot share rule knowledge systematically",
            "remediation": (
                "keep learned FactorizedCard2Vec identity and add a zero-gated, "
                "provenance-checked fixed metadata residual"
            ),
        },
        {
            "id": "FEATURE-STATE-OMISSIONS",
            "severity": "medium",
            "confidence": "code proof",
            "failed": "maxHp, evolution stack, appearThisTurn and turn-action flags are omitted",
            "impact": "value/search state can be partially aliased even when current legal actions differ less",
            "remediation": "add explicit stable scalars/flags in a schema migration with metamorphic tests",
        },
        {
            "id": "FEATURE-ENERGY-RESIDUAL-OMISSIONS",
            "severity": "medium",
            "confidence": "code proof",
            "failed": (
                "typed Pokemon.energies, ENERGY.count, remainEnergyCost and "
                "remainDamageCounter are not encoded"
            ),
            "impact": (
                "special-energy unit value and partial cost/counter selections "
                "must be inferred from card identity and legal-set shape"
            ),
            "remediation": (
                "append bounded typed-energy and remaining-budget features in "
                "a checkpointed schema migration"
            ),
        },
        {
            "id": "AUX-LETHAL-RIGHT-CENSORING",
            "severity": "medium",
            "confidence": "code proof",
            "failed": (
                "lethal labels inspect later own decision frames, so a prize "
                "taken on a terminal action has no later frame to expose it"
            ),
            "impact": "winning terminal prize lines can receive a false-zero target",
            "remediation": (
                "derive the label from exact post-transition prize deltas, "
                "version the target contract, rebuild the corpus, and retrain"
            ),
        },
        {
            "id": "AUX-HIDDEN-MULTIPLICITY-COLLAPSE",
            "severity": "medium",
            "confidence": "code proof",
            "failed": "opponent hidden-card targets are binary multi-hots",
            "impact": "one versus multiple hidden copies share the same target",
            "remediation": (
                "benchmark count-distribution targets before changing the "
                "belief-head output/loss contract"
            ),
        },
    ]
    if int(trace.get("distinct_raw_options_with_same_feature_states") or 0):
        findings[0]["trace_collision_states_total"] = trace[
            "distinct_raw_options_with_same_feature_states"
        ]
    rebuilt = sources.get("rebuilt") if isinstance(sources, Mapping) else None
    if rebuilt and rebuilt.get("action_count_3000_guard_present") is False:
        findings.append(
            {
                "id": "ENGINE-LONG-GAME-GUARD-DIVERGENCE",
                "severity": "high",
                "confidence": "source diff",
                "failed": "rebuilt source lacks the official actionCount>=3000 draw guard",
                "impact": "extreme games may diverge or loop compared with the reference engine",
                "remediation": "restore the guard or prove seeded parity including long/effect-loop games",
            }
        )
    return findings


def pretraining_contract() -> dict[str, Any]:
    return {
        "immutable_rule_tasks": [
            "card type, stage, energy type, weakness, resistance, ex/mega/tera/ace classification",
            "attack cost multiset, base damage, retreat cost and prize-liability prediction",
            "evolvesFrom contrastive link and Basic->Stage1->Stage2 path prediction",
            "weakness/resistance damage-transform and retreat-sufficiency calculations",
            "legal timing flags for evolve/supporter/stadium/energy/retreat",
            "action target/serial/candidate-ordinal reconstruction",
            "simulator board-delta and event prediction from exact transitions",
        ],
        "strategy_separation": (
            "deck heuristics, ladder policy, matchup plans and top-player behavior are mutable "
            "strategy labels and must not be mixed into immutable engine metadata"
        ),
        "integration": "learned identity + zero-gated fixed metadata residual + existing role embedding",
        "legacy_checkpoint_migration": "load legacy weights; initialize both metadata gates to exactly 0; verify bit-exact logits before opening gates",
    }


def competition_risks() -> list[str]:
    return [
        "Treat competition simulator behavior as authoritative even when it differs from paper TCG rules.",
        "Keep the competition-only engine license and do not redistribute engine/card payloads outside allowed use.",
        "Do not train against known serialization/API bugs as exploits; pin and hash the updated engine bundle.",
        "A source diff is not parity proof: require seeded transition, legality, terminal-result and long-game parity.",
        "Do not package hidden or privileged simulator state into a submission-time observation path.",
    ]


def _markdown(report: Mapping[str, Any]) -> str:
    profile = report["catalog_profile"]
    factor = report["factor_table"]
    lines = [
        "# Card mechanics contract audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Dataset and grain",
        "",
        f"- Engine catalog: {report['metadata_provenance']['card_count']} cards and {report['metadata_provenance']['attack_count']} attacks, one row per engine ID.",
        "- CSV: provenance-checked identity join; attack/ability/Tera rows are intentionally not treated as a single grain.",
        f"- Card feature reachability: board={factor['all_engine_cards_reachable_on_board']}, options={factor['all_engine_cards_reachable_in_options']}; attacks={factor['all_engine_attacks_reachable_in_options']}.",
        f"- Non-Pokémon cards with `basic=True`: {len(profile['non_pokemon_basic_flag_hybrids'])} (Fossil hybrids; preserved).",
        "",
        "## Simulator / feature / learnability matrix",
        "",
        "| Mechanic | Simulator | Current feature | Learnability |",
        "|---|---|---|---|",
    ]
    for row in report["representation_matrix"]:
        lines.append(
            f"| {row['mechanic']} | {row['simulator']} | {row['feature']} | {row['learnable']} |"
        )
    lines.extend(["", "## Findings", ""])
    for finding in report["findings"]:
        lines.append(
            f"- **{finding['severity'].upper()} — {finding['id']}**: "
            f"{finding['failed']}. Impact: {finding['impact']} Remediation: {finding['remediation']}"
        )
    lines.extend(
        [
            "",
            "## Temporal contract",
            "",
            "The context remains fixed at 320 decisions. Track the measured share of games that truncate only the oldest decisions; do not expand the window based on this audit.",
            "",
            "## Safe metadata extension",
            "",
            report["pretraining_contract"]["integration"] + ".",
            "",
            "Immutable rule-derived tasks stay separate from deck strategy or ladder-policy labels.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.runtime is not None:
        os.environ["CG_LIB_PATH"] = str(args.runtime)
    cards = cg_env.all_card_data()
    attacks = cg_env.all_attack()
    catalog = build_metadata_catalog(cards, attacks, csv_path=args.csv)
    sources = compare_source_trees(args.official_source, args.rebuilt_source)
    traces = trace_census(catalog, args.trace, max_context=args.max_context)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "intent": "training representation safety; not a production mutation",
        "metadata_schema": CARD_METADATA_SCHEMA,
        "metadata_provenance": dict(catalog.provenance),
        "catalog_profile": _card_type_profile(catalog),
        "factor_table": factor_table_audit(catalog),
        "deck_census": deck_census(catalog, args.deck_root, args.deck_json),
        "trace_census": traces,
        "source_audit": sources,
        "representation_matrix": representation_matrix(),
        "pretraining_contract": pretraining_contract(),
        "competition_risks": competition_risks(),
    }
    report["findings"] = known_findings(traces, sources)
    report["quality_gate"] = {
        "highest_severity": max(
            (finding["severity"] for finding in report["findings"]),
            key=lambda value: SEVERITY_ORDER[value],
            default="low",
        ),
        "passes_without_known_gap_override": not any(
            SEVERITY_ORDER[finding["severity"]] >= SEVERITY_ORDER["high"]
            for finding in report["findings"]
        ),
    }
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime", type=Path, help="directory containing the cg package"
    )
    parser.add_argument("--csv", type=Path, required=True, help="EN_Card_Data.csv")
    parser.add_argument("--deck-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--deck-json",
        type=Path,
        action="append",
        default=[ROOT / "data/training_mixes/top_ladder_representatives.v1.json"],
    )
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--official-source", type=Path)
    parser.add_argument(
        "--rebuilt-source", type=Path, default=ROOT / ".private/ptcg_engine"
    )
    parser.add_argument("--max-context", type=int, default=320)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-known-gaps", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
    except (MetadataContractError, OSError, ValueError) as exc:
        print(f"FAIL CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
        args.report.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
        print(args.report)
    else:
        print(payload, end="")
    if (
        not args.allow_known_gaps
        and not report["quality_gate"]["passes_without_known_gap_override"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
