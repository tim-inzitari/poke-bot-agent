#!/usr/bin/env python3
"""Build the canonical portable-report artifact for Alakazam replay analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TITLE = "Why the submitted Alakazam policy is underperforming"


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analysis = load(args.analysis_dir / "analysis.json")
    attribution = load(args.analysis_dir / "head-attribution-iter9.json")
    guide = load(args.analysis_dir / "guide-alignment-iter9.json")
    training = load(args.analysis_dir / "training-evidence.json")
    summary = analysis["summary"]
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    submitted_scores = {0: 888.1, 4: 899.6, 9: 892.0}
    iteration_rows = []
    for iteration in (0, 4, 9):
        row = summary["by_iteration"][str(iteration)]
        iteration_rows.append(
            {
                "iteration": f"Iteration {iteration}",
                "iteration_number": iteration,
                "games": row["games"],
                "win_rate": row["win_rate"],
                "went_first_rate": row["went_first_rate"],
                "setup_by_turn_2_rate": row["alakazam_by_own_turn_2_rate"],
                "kaggle_score": submitted_scores[iteration],
            }
        )

    key_matchup_rows = []
    labels = {
        "marnie_grimmsnarl": "Marnie / Grimmsnarl",
        "alakazam_mirror": "Alakazam mirror",
        "crustle": "Crustle",
    }
    for iteration in (0, 4, 9):
        for archetype, label in labels.items():
            row = summary["by_iteration_archetype"].get(
                f"iter_{iteration:05d}:{archetype}"
            )
            if row:
                key_matchup_rows.append(
                    {
                        "iteration": f"Iteration {iteration}",
                        "iteration_number": iteration,
                        "matchup": label,
                        "games": row["games"],
                        "wins": row["wins"],
                        "win_rate": row["win_rate"],
                        "went_first_rate": row["went_first_rate"],
                        "setup_by_turn_2_rate": row[
                            "alakazam_by_own_turn_2_rate"
                        ],
                    }
                )

    matchup_rows = []
    for archetype, row in summary["by_archetype"].items():
        if row["games"] < 3:
            continue
        matchup_rows.append(
            {
                "matchup": labels.get(archetype, archetype.replace("_", " ").title()),
                "games": row["games"],
                "wins": row["wins"],
                "win_rate": row["win_rate"],
                "went_first_win_rate": row["win_rate_went_first"],
                "went_second_win_rate": row["win_rate_went_second"],
                "setup_by_turn_2_rate": row["alakazam_by_own_turn_2_rate"],
                "deckout_losses": row["probable_deckout_losses"],
            }
        )
    matchup_rows.sort(key=lambda row: (-row["games"], row["matchup"]))

    gate_gap_rows = []
    gate_ids = {
        "Marnie / Grimmsnarl": "specialist-marnie-s-grimmsnarl-ex-gate-iter5-52a5207e4c98",
        "Alakazam mirror": "specialist-alakazam-owner-accepted-iter39-roster18-v5",
        "Lucifer baseline": "lucifer19-battlecore",
    }
    public_key = {
        "Marnie / Grimmsnarl": "marnie_grimmsnarl",
        "Alakazam mirror": "alakazam_mirror",
    }
    for iteration in (0, 4, 9):
        gate = training["gates"][str(iteration)]
        for matchup, opponent_id in gate_ids.items():
            gate_row = gate["opponents"].get(opponent_id)
            if gate_row is None:
                continue
            public_row = None
            if matchup in public_key:
                public_row = summary["by_iteration_archetype"].get(
                    f"iter_{iteration:05d}:{public_key[matchup]}"
                )
            gate_gap_rows.append(
                {
                    "iteration": f"Iteration {iteration}",
                    "matchup": matchup,
                    "gate_games": gate_row["games"],
                    "gate_win_rate": gate_row["wr"],
                    "public_games": None if public_row is None else public_row["games"],
                    "public_win_rate": None
                    if public_row is None
                    else public_row["win_rate"],
                    "gap_pp": None
                    if public_row is None
                    else 100.0 * (gate_row["wr"] - public_row["win_rate"]),
                }
            )

    head_rows = []
    all_heads = attribution["summary"]["all"]["heads"]
    win_heads = attribution["summary"]["wins"]["heads"]
    loss_heads = attribution["summary"]["losses"]["heads"]
    training_heads = training["checkpoints"]["12"]["expanded_heads"]
    for name, row in all_heads.items():
        train_name = "tactical_outcome" if name == "tactical_outcomes" else name
        train_row = training_heads.get(train_name, {})
        head_rows.append(
            {
                "head": name,
                "all_effect": row["mean_abs_logit_effect"],
                "win_effect": win_heads[name]["mean_abs_logit_effect"],
                "loss_effect": loss_heads[name]["mean_abs_logit_effect"],
                "loss_to_win_ratio": loss_heads[name]["mean_abs_logit_effect"]
                / max(win_heads[name]["mean_abs_logit_effect"], 1e-12),
                "ablation_flip_rate": row["choice_flip_rate_when_ablated"],
                "supervision_coverage": train_row.get("coverage"),
                "validation_loss": train_row.get("validation_loss"),
                "loss_weight": train_row.get("loss_weight"),
            }
        )
    head_rows.sort(key=lambda row: -row["loss_effect"])

    guide_rows = []
    for group, label in (
        ("all", "All iteration-9 stages"),
        ("wins", "Winning games"),
        ("losses", "Losing games"),
        ("matchup:marnie_grimmsnarl:losses", "Marnie losses"),
        ("matchup:crustle:losses", "Crustle losses"),
    ):
        row = guide["summary"][group]
        guide_rows.append(
            {
                "cohort": label,
                "guide_rows": row["guide_rows"],
                "agreement": row["agreement"],
                "mean_confidence": row["mean_confidence"],
            }
        )

    all_summary = summary["all"]
    marnie = summary["by_archetype"]["marnie_grimmsnarl"]
    crustle = summary["by_archetype"]["crustle"]
    mirror = summary["by_archetype"]["alakazam_mirror"]
    setup_effect = all_heads["setup_board_outcome"]["mean_abs_logit_effect"]
    opp_hand_effect = all_heads["opponent_hand"]["mean_abs_logit_effect"]
    action_type = training_heads["action_type"]
    setup_ratio = opp_hand_effect / max(setup_effect, 1e-12)
    recommendations = [
        {
            "priority": 1,
            "change": "Make every dedicated route typed-output-centered",
            "why": "Subtract the zero-typed-output baseline (or use a pure typed interaction), then add positive bounded reliability gates. The current setup route can mostly ignore its setup output.",
            "activation": "Future architecture migration at a receipt-backed boundary; paired replay and gate test first.",
        },
        {
            "priority": 2,
            "change": "Version action_type supervision; cap its current reliability at 0.25×",
            "why": "Factorized stages do not present cross-type alternatives, so the current scalar option head is structurally unidentifiable—not merely missing labels. Keep it nonzero while a state-to-next-action-type compatibility target is validated.",
            "activation": "Prospective fusion-v3 checkpoints after label-contract tests and a paired no-regression gate.",
        },
        {
            "priority": 3,
            "change": "Train and gate on current public Marnie and Crustle variants",
            "why": "Routing recognizes both matchups, but local gates overstate strength. Public Crustle losses are often deckouts; public Marnie is strongly seat/setup dependent.",
            "activation": "Corpus and frozen-opponent update; no core-shape change required.",
        },
        {
            "priority": 4,
            "change": "Use the 0.05 guide as a pairwise direction term on causal heads",
            "why": "Keep the guide training-only and low-weight, but let its preferred-vs-alternative ordering steer action_utility, action_resource, setup, and Q heads. Do not add guide logits to serving.",
            "activation": "Future specialists first; isolated guide-on/off evidence required.",
        },
        {
            "priority": 5,
            "change": "Prototype a deck-survival head",
            "why": "Existing resource_forecast predicts only the next state. A new head would predict safe draw budget, deckout probability, turns to deckout, and recovery-loop value; admit it only if Crustle counterfactual tests beat expanded resource targets.",
            "activation": "Final-model experiment, not the live Alakazam run.",
        },
    ]

    def source(source_id: str, label: str, filename: str, filters: list[str]) -> dict[str, Any]:
        path = f"outputs/analysis/kaggle-alakazam-iter0-4-9-20260801/{filename}"
        return {
            "id": source_id,
            "label": label,
            "path": path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": f"SELECT * FROM read_json_auto('{path}')",
                "description": f"Read the deterministic {label.lower()} source artifact.",
                "tables_used": [path],
                "filters": filters,
                "executed_at": generated,
            },
        }

    sources = [
        source(
            "replay_analysis",
            "Decoded Kaggle episode analysis",
            "analysis.json",
            ["submission iterations in (0, 4, 9)", "all 207 listed completed episodes"],
        ),
        source(
            "head_attribution",
            "Iteration 9 leave-one-head-out attribution",
            "head-attribution-iter9.json",
            ["iteration = 9", "all 61 downloaded episodes", "4,546 factorized stages"],
        ),
        source(
            "guide_alignment",
            "Iteration 9 guide-alignment analysis",
            "guide-alignment-iter9.json",
            ["iteration = 9", "high-confidence unique guide preferences"],
        ),
        source(
            "training_evidence",
            "Checkpoint head coverage and local-gate evidence",
            "training-evidence.json",
            ["checkpoints in (0, 9, 12)"],
        ),
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Technical diagnosis of three submitted Alakazam checkpoints using 207 Kaggle episodes and instrumented fusion replay.",
        "generatedAt": generated,
        "cards": [],
        "charts": [
            {
                "id": "key_matchup_chart",
                "title": "Public win rate for the three decision-critical matchups",
                "subtitle": "Iteration 9 recovered Marnie and mirror performance, while Crustle remained unstable on a small sample.",
                "type": "bar",
                "dataset": "key_matchups",
                "sourceId": "replay_analysis",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "matchup", "type": "nominal", "label": "Matchup"},
                    "y": {"field": "win_rate", "type": "quantitative", "label": "Win rate", "format": "percent"},
                    "color": {"field": "iteration", "type": "nominal", "label": "Checkpoint"},
                    "tooltip": [
                        {"field": "games", "type": "quantitative", "label": "Games"},
                        {"field": "setup_by_turn_2_rate", "type": "quantitative", "label": "Alakazam by own turn 2", "format": "percent"},
                    ],
                },
                "layout": "full",
            },
            {
                "id": "head_effect_chart",
                "title": "Learned-head action influence in iteration-9 losses",
                "subtitle": "Mean absolute leave-one-head-out logit effect over replayed factorized stages; setup_board_outcome is effectively invisible at this scale.",
                "type": "horizontalBar",
                "dataset": "head_effects",
                "sourceId": "head_attribution",
                "valueFormat": "number",
                "encodings": {
                    "x": {"field": "head", "type": "nominal", "label": "Head"},
                    "y": {"field": "loss_effect", "type": "quantitative", "label": "Mean absolute logit effect"},
                    "tooltip": [
                        {"field": "loss_to_win_ratio", "type": "quantitative", "label": "Loss / win effect ratio"},
                        {"field": "ablation_flip_rate", "type": "quantitative", "label": "Choice flip rate", "format": "percent"},
                        {"field": "supervision_coverage", "type": "quantitative", "label": "Supervision coverage", "format": "percent"},
                    ],
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "matchup_table",
                "title": "Public matchup outcomes",
                "subtitle": "All archetypes with at least three observed games across iterations 0, 4, and 9.",
                "dataset": "matchups",
                "sourceId": "replay_analysis",
                "defaultSort": {"field": "games", "direction": "desc"},
                "columns": [
                    {"field": "matchup", "label": "Matchup", "type": "text"},
                    {"field": "games", "label": "Games", "type": "number"},
                    {"field": "win_rate", "label": "Win rate", "format": "percent"},
                    {"field": "went_first_win_rate", "label": "Went first", "format": "percent"},
                    {"field": "went_second_win_rate", "label": "Went second", "format": "percent"},
                    {"field": "setup_by_turn_2_rate", "label": "Setup by own turn 2", "format": "percent"},
                    {"field": "deckout_losses", "label": "Deckout losses", "type": "number"},
                ],
            },
            {
                "id": "gate_gap_table",
                "title": "Local gate versus public Kaggle evidence",
                "subtitle": "Gate opponents use 250 balanced games; public rows are the observed Kaggle episode samples. Lucifer is a baseline, not a clean Crustle-equivalent comparison.",
                "dataset": "gate_gaps",
                "sourceId": "training_evidence",
                "defaultSort": {"field": "iteration", "direction": "asc"},
                "columns": [
                    {"field": "iteration", "label": "Checkpoint", "type": "text"},
                    {"field": "matchup", "label": "Opponent", "type": "text"},
                    {"field": "gate_games", "label": "Gate games", "type": "number"},
                    {"field": "gate_win_rate", "label": "Gate win rate", "format": "percent"},
                    {"field": "public_games", "label": "Public games", "type": "number"},
                    {"field": "public_win_rate", "label": "Public win rate", "format": "percent"},
                    {"field": "gap_pp", "label": "Gate gap", "type": "number", "unit": "pp"},
                ],
            },
            {
                "id": "guide_table",
                "title": "Guide alignment on high-confidence iteration-9 stages",
                "subtitle": "Agreement compares the realized submitted action with the training-only guide's unique preferred action.",
                "dataset": "guide_alignment",
                "sourceId": "guide_alignment",
                "defaultSort": {"field": "guide_rows", "direction": "desc"},
                "columns": [
                    {"field": "cohort", "label": "Cohort", "type": "text"},
                    {"field": "guide_rows", "label": "Guide rows", "type": "number"},
                    {"field": "agreement", "label": "Agreement", "format": "percent"},
                    {"field": "mean_confidence", "label": "Mean confidence", "format": "percent"},
                ],
            },
            {
                "id": "recommendation_table",
                "title": "Proposed changes and activation boundaries",
                "subtitle": "Ordered by expected leverage and evidentiary support; none are applied to the live iteration by this report.",
                "dataset": "recommendations",
                "sourceId": "head_attribution",
                "defaultSort": {"field": "priority", "direction": "asc"},
                "columns": [
                    {"field": "priority", "label": "Priority", "type": "number"},
                    {"field": "change", "label": "Change", "type": "text"},
                    {"field": "why", "label": "Evidence and rationale", "type": "text"},
                    {"field": "activation", "label": "Activation", "type": "text"},
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## Technical summary\n\n"
                    f"The policy is **not broadly broken**: it won {pct(all_summary['win_rate'])} of 207 public episodes and iteration 9 reached {pct(summary['by_iteration']['9']['win_rate'])}. The rating ceiling is dominated by two matchup-specific failures: **Marnie / Grimmsnarl ({pct(marnie['win_rate'])}, 65 games)** and **Crustle ({pct(crustle['win_rate'])}, 18 games)**.\n\n"
                    f"The setup system is present but not causally strong. On 4,546 instrumented iteration-9 stages, `setup_board_outcome` changed logits by only **{setup_effect:.6f} on average**, versus **{opp_hand_effect:.6f}** for `opponent_hand`—a **{setup_ratio:.0f}× gap**—and setup-head ablation changed no selected action.\n\n"
                    "The immediate design priority is therefore **typed-route calibration**, not adding another setup head or blindly increasing every auxiliary loss. The strongest new-head candidate is a separate deck-survival head for long Crustle games, but it should be admitted only after better public Crustle training/gate coverage and a counterfactual comparison against expanded resource-forecast targets."
                ),
            },
            {
                "id": "public_pattern",
                "type": "markdown",
                "body": (
                    "## Public losses are concentrated in Marnie and Crustle\n\n"
                    f"The model is already strong in the Alakazam mirror ({pct(mirror['win_rate'])}) and against Archaludon ({pct(summary['by_archetype']['archaludon']['win_rate'])}). Marnie accounts for 36 of the 88 losses; Crustle contributes 11 more. Together those two matchups explain **53.4% of all observed losses**.\n\n"
                    "Marnie is a seat-plus-setup interaction: going first and establishing Alakazam by own turn 2 won 10/12 games, while the other three seat/setup cells won only 31.8–40.0%. Crustle is different: 6/11 losses ended with an empty deck and a surviving board, pointing to draw-loop/resource-horizon errors rather than failure to establish the evolution line."
                ),
                "sourceId": "replay_analysis",
            },
            {"id": "key_matchup_chart_block", "type": "chart", "chartId": "key_matchup_chart"},
            {"id": "matchup_table_block", "type": "table", "tableId": "matchup_table"},
            {
                "id": "gate_domain_gap",
                "type": "markdown",
                "body": (
                    "## The local gate overstates public robustness\n\n"
                    "Iteration 9 scored 65.8% against the frozen Marnie specialist but only 56.2% against public Marnie variants. For iteration 0 the gap was 33.4 percentage points. The router is not missing these decks: iteration-9 replay attribution used Marnie's route on 1,010 of 1,075 Marnie stages and Crustle's route on 422 of 443 Crustle stages. The remaining gap is adapter/opponent-policy coverage, not route recognition.\n\n"
                    "The current Lucifer baseline is also not a valid substitute for the public Crustle family: Alakazam holds roughly 85% against that gate opponent while winning only 38.9% against the observed public Crustle decks."
                ),
            },
            {"id": "gate_gap_table_block", "type": "table", "tableId": "gate_gap_table"},
            {
                "id": "head_result",
                "type": "markdown",
                "body": (
                    "## Equal route averaging is not equal head influence\n\n"
                    f"`opponent_hand` is the largest observed head route ({opp_hand_effect:.6f} mean absolute effect), followed by `remaining_turns`, `archetype`, and `opponent_response`. Their effects are 16–31% larger in losses than wins, but that is descriptive: harder states can activate them more strongly. A direct downweight is not justified without paired counterfactual games.\n\n"
                    f"Two findings *are* actionable. First, `setup_board_outcome` is effectively silent ({setup_effect:.6f}; zero choice flips). Second, `action_type` has **{action_type['labeled_rows']} labeled rows and {pct(action_type['coverage'])} coverage**, yet still has a measurable route effect. Inspection of the target contract shows this is structural: factorized stages almost never compare different action types. The current route network also concatenates option hidden state with typed output, so a route can produce action deltas while largely ignoring the head it is supposed to represent."
                ),
            },
            {"id": "head_effect_chart_block", "type": "chart", "chartId": "head_effect_chart"},
            {
                "id": "guide_result",
                "type": "markdown",
                "body": (
                    "## The 0.05 guide identifies useful states but supplies little directional force\n\n"
                    f"The guide issued a unique high-confidence preference on 2,308 of 4,546 iteration-9 stages. The submitted action agreed on {pct(guide['summary']['all']['agreement'])}; agreement was {pct(guide['summary']['wins']['agreement'])} in wins and {pct(guide['summary']['losses']['agreement'])} in losses. Main-action agreement was only 38.0%.\n\n"
                    "This does not mean the guide should become a serving policy. The checkpoint contract confirms `direct_policy_cross_entropy=false` and `guide_preferred_action_consumed=false`: the guide currently selects/emphasizes observed head-target rows but does not communicate its preferred-vs-alternative direction. A small pairwise ranking term on the causal learned heads would match the owner's intended “lead the heads uphill” behavior while leaving the guide absent from runtime."
                ),
            },
            {"id": "guide_table_block", "type": "table", "tableId": "guide_table"},
            {
                "id": "scope_methods",
                "type": "markdown",
                "body": (
                    "## Scope, definitions, and methodology\n\n"
                    "The public cohort is every currently listed episode for submissions 55146726 (iteration 0), 55154133 (iteration 4), and 55165133 (iteration 9): 207 completed games total. Win rate is reward-positive episodes divided by episodes. Archetypes come from exact core-card signatures in the decoded initial deck. `Alakazam by own turn 2` means the card appeared active or benched no later than the second distinct own turn.\n\n"
                    "Head attribution replays all 61 iteration-9 games through checkpoint `iter_00009.pt`, restores temporal action history and the public matchup tree, and records the full-logit change when each typed source is zeroed. It covers 4,546 factorized decision stages and reproduces the realized submitted action on 99.4% of them. Association between head magnitude and losses is not treated as causal; only zero-label coverage, route silence, exact router activation, and replay-state outcomes are used as direct design evidence."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and robustness checks\n\n"
                    "Kaggle ratings are opponent-strength weighted and time-varying, so the report diagnoses episode outcomes rather than treating 888.1/899.6/892.0 as directly comparable win rates. Per-iteration matchup samples are small; the 18-game Crustle finding is strong enough to motivate a test, not to prove a universal counterfactual.\n\n"
                    "Leave-one-head-out attribution establishes action sensitivity, not whether the alternative action would win. The two strongest proposed weight changes—capping unsupervised `action_type` reliability and restoring meaningful setup sensitivity—must therefore pass paired replay, fixed-seed gate, public-variant holdout, and rating-simulation checks before activation. No live checkpoint or training service was changed during this analysis."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Implement a **typed-output-centered route** (`route(option, head) - route(option, 0)` or an equivalent pure interaction) plus positive bounded per-head reliability gates. Add an influence-band gate so every learned head has measurable, non-dominant action authority.\n"
                    "2. Replace the unidentifiable scalar `action_type` objective with a versioned state-to-next-action-type compatibility target. Until that contract validates, use a **0.25× nonzero reliability cap**. Do not set it to zero.\n"
                    "3. Add current public Marnie and frozen public Crustle agents to the training mix and formal gate. Preserve the existing matchup routes; recognition is already working.\n"
                    "4. For future specialists, keep guide weight at **0.05** but add a guide-qualified pairwise ranking term on the other causal heads. The guide remains training-only and never enters fusion.\n"
                    "5. Prototype a four-output **deck_survival** head (`safe_draw_budget`, `deckout_probability`, `turns_to_deckout`, `recovery_loop_value`) only after comparing it with expanded multi-horizon resource-forecast targets."
                ),
            },
            {"id": "recommendation_table_block", "type": "table", "tableId": "recommendation_table"},
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- Does a typed-output-centered route recover setup-head sensitivity without reducing the iteration-9 mirror and Marnie gains?\n"
                    "- Which public Crustle actions cause the draw budget to cross zero, and can multi-horizon resource targets solve them without a new head?\n"
                    "- Does the frozen public Marnie family require one adapter or variant-specific subroutes once enough public replays exist?"
                ),
            },
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "overall_win_rate": all_summary["win_rate"],
                        "marnie_win_rate": marnie["win_rate"],
                        "crustle_win_rate": crustle["win_rate"],
                        "setup_effect": setup_effect,
                    }
                ],
                "iterations": iteration_rows,
                "key_matchups": key_matchup_rows,
                "matchups": matchup_rows,
                "gate_gaps": gate_gap_rows,
                "head_effects": head_rows,
                "guide_alignment": guide_rows,
                "recommendations": recommendations,
            },
        },
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
