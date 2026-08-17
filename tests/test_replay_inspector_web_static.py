"""Dependency-free smoke checks for the standalone replay inspector UI."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "replay_inspector" / "web"


class ReplayInspectorWebStaticTests(unittest.TestCase):
    def test_required_static_assets_and_local_api_contract_are_present(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn("Replay Model Inspector", index)
        self.assertIn('href="./styles.css"', index)
        self.assertIn('src="./app.js"', index)
        self.assertIn('const API = "./api";', app)
        for selector_id in (
            "submission-select",
            "game-select",
            "step-select",
            "stage-select",
            "parameter-search",
            "reproduction-status",
        ):
            self.assertIn(f'id="{selector_id}"', index)

        for endpoint_fragment in (
            "`${API}/health`",
            "`${API}/submissions`",
            "`${API}/submissions/${encodedSubmission}/games`",
            "/steps/${encodeURIComponent(state.stepIndex)}?stage=${encodeURIComponent(state.stage)}`",
            "/parameters`",
            "/parameters/${encodeURIComponent(name)}?offset=${encodeURIComponent(state.parameterOffset)}&limit=${SLICE_LIMIT}`",
        ):
            self.assertIn(endpoint_fragment, app)
        self.assertIn('credentials: "same-origin"', app)
        combined = index + app + styles
        self.assertEqual(combined.count("https://"), 1)
        self.assertIn("https://ptcgvis.heroz.jp/Visualizer/Replay/", combined)
        self.assertNotIn("http://", index + app + styles)

    def test_ui_explicitly_distinguishes_replay_record_from_re_evaluation(self) -> None:
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

        for status in (
            "exact_reproduced",
            "exact_runtime_short_circuit",
            "hypothetical_model_forward_not_submitted_runtime",
            "recomputed_not_historical",
            "diverged_or_fallback_unknown",
            'status === "unavailable"',
        ):
            self.assertIn(status, app)
        self.assertIn("not values stored in the raw replay", app)
        self.assertIn("The displayed 100% / 0% policy is the exact runtime choice", app)
        self.assertIn("Hypothetical neural rerun", app)
        self.assertIn("actual_runtime_probability", app)
        self.assertIn("Causal replay reconstruction", index)
        self.assertIn("RE-EVALUATED POLICY SURFACE", index)

    def test_plain_english_actions_and_head_removal_impact_are_rendered(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn("Action in plain English", index)
        self.assertIn('id="head-impact-summary"', index)
        self.assertIn('id="head-impact-grid"', index)
        self.assertIn("Policy impact if removed", index)
        self.assertIn('colspan="9"', index)
        self.assertIn("setEmptyRow(elements.headsBody, 9", app)
        for fragment in (
            "function friendlyActionTranscript",
            "function firstActionTranscript",
            "function optionForRecordedAction",
            "function renderOverviewAction",
            "Technical action data",
            "Technical recorded action",
            "Plain-English recorded action transcript not supplied",
            "selected_action_transcript",
            "Plain-English action transcript not supplied",
            "function headImpactFor",
            "policy_influence",
            "effect_logits",
            "selected_option_probability_delta",
            "signedPercentagePoints",
            "Current raw route signal only",
            "What changes if this head were removed?",
            "View policy-impact calculation and source tensors",
        ):
            self.assertIn(fragment, app)
        for selector in (
            ".action-transcript",
            ".head-impact-grid",
            ".head-impact-card",
            ".impact-metric-grid",
        ):
            self.assertIn(selector, styles)

    def test_head_faq_explains_all_policy_inputs_and_live_effects_in_plain_english(
        self,
    ) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        for fragment in (
            'id="head-faq-title"',
            'id="head-faq-list"',
            "Looking for “are we going to lose?”",
            "offensive danger to the opponent",
            "Head effects are nonlinear",
        ):
            self.assertIn(fragment, index)
        for head in (
            "value",
            "archetype",
            "opponent_hand",
            "opponent_remainder",
            "lethal_threat",
            "prize_race",
            "action_q",
            "action_type",
            "action_target",
            "action_resource",
            "action_utility",
            "tactical_outcomes",
            "opponent_response",
            "resource_forecast",
            "game_phase",
            "outcome_distribution",
            "remaining_turns",
            "setup_board_outcome",
            "combo_state",
        ):
            self.assertIn(f'id: "{head}"', app)
        for fragment in (
            "function renderHeadFaq",
            "function headFaqCurrentEffect",
            "Current policy setting: 1× baseline.",
            "Total policy-distribution shift",
            "a technical fusion multiplier, not a percent importance or training-loss weight",
            "our offensive threat to the opponent, not the chance that we lose",
            "Its loss side is the closest direct learned signal",
            "Its knockout output is the clearest near-term",
        ):
            self.assertIn(fragment, app)
        for selector in (
            ".head-faq-card",
            ".head-faq-list",
            ".head-faq-item",
            ".head-faq-body",
            ".head-faq-callout",
        ):
            self.assertIn(selector, styles)

    def test_training_guide_shadow_is_explicitly_non_authoritative(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        for fragment in (
            'id="guide-shadow-title"',
            'id="guide-shadow-availability"',
            'id="guide-shadow-content"',
            'id="guide-shadow-toggle"',
            'id="guide-shadow-toggle-state"',
            "Compare Guide ON / OFF",
            "Ordinary training guides appear here as zero-authority second opinions",
        ):
            self.assertIn(fragment, index)
        for fragment in (
            "function renderGuideShadow",
            "trace?.guide_shadow",
            "guideShadowEnabled: false",
            'elements.guideShadowToggle.addEventListener("change"',
            "Guide comparison on · model policy unchanged",
            "Guide comparison off · exact model policy shown",
            "If the guide were allowed to choose",
            "it would replace the model’s",
            "Model would do",
            "Guide would do",
            "shadow.policy_authority !== false",
            "Effect on production policy",
            "Exactly zero",
            "training-teacher ranking, not a probability",
            "submittedPolicy.policy_authority === true",
            "Production guide active",
            "Guide ON · exact submitted-runtime policy",
            "Guide OFF · neural-only comparison",
            "Neural-only would do",
            "Submitted runtime did",
        ):
            self.assertIn(fragment, app)
        for selector in (
            ".guide-shadow-card",
            ".guide-shadow-toggle",
            ".guide-shadow-comparison.is-same",
            ".guide-shadow-comparison.is-different",
        ):
            self.assertIn(selector, styles)

    def test_selection_identity_player_ranks_and_matchup_adapter_status_are_explicit(
        self,
    ) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for selector_id in (
            "selection-identity",
            "matchup-adapter-availability",
            "matchup-adapter-status",
            "matchup-adapter-content",
        ):
            self.assertIn(f'id="{selector_id}"', index)
        for fragment in (
            "submission_text",
            "function gamePlayers",
            "Rank unavailable",
            "Exact submission ID:",
            "function renderMatchupAdapter",
            "function renderMatchupAdapterComparison",
            "function matchupAdapterPolicyShiftSummary",
            "function matchupAdapterReliabilitySummary",
            "matchup_adapter_status",
            "model.adapter_status",
            "Active for this decision",
            "Bypassed —",
            "Unavailable —",
            "Installed adapter weights alone do not mean the adapter was active.",
            "What changed with the Matchup Adapter ON?",
            "Adapter OFF is an exact rerun",
            "Change = ON − OFF",
            "on_off_comparison",
        ):
            self.assertIn(fragment, app)
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")
        for selector in (
            ".adapter-comparison",
            ".adapter-comparison-table",
            ".adapter-delta-positive",
            ".adapter-delta-negative",
        ):
            self.assertIn(selector, styles)

    def test_submission_selector_starts_with_id_and_reports_cached_replay_outcomes(
        self,
    ) -> None:
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for fragment in (
            "function cachedReplayOutcome",
            "cached_replay_outcomes",
            "games_with_outcome",
            "win_rate",
            "win rate unavailable",
            "function cachedReplayOutcomeText",
            "function submissionModelStatus",
            "`${identity.idText} · ${identity.primary}`",
        ):
            self.assertIn(fragment, app)
        self.assertIn(
            "return [\n    identity.idText,\n    `label: ${label}`,\n    `weights: ${readiness.weightsText}`,\n    `trace: ${readiness.traceText}`,\n    cachedReplayOutcomeText(submission),",
            app,
        )

    def test_weights_only_readiness_keeps_weight_inspection_and_explains_trace_gap(
        self,
    ) -> None:
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for fragment in (
            "function submissionReadiness",
            '"trace_ready", "weights_only", "replay_only"',
            "submission?.weights",
            "submission?.dynamic_trace",
            "runtime_parity_receipt_missing",
            "missing exact-runtime parity receipt",
            "runtime_package_artifact_missing",
            "runtime_package_sha256_invalid_or_missing",
            "runtime_package_module_path_unavailable",
            "exact submitted runtime is not yet checksum-attested",
            "exact submitted runtime cannot be loaded from the checksum-attested source",
            "Exact model weights available",
            "Dynamic decision trace unavailable",
            "Model weights: ${readiness.weightsText}",
            "Decision trace: ${readiness.traceText}",
            "Exact model weights available · parameter inventory returned by source",
            "Weights and decision-trace status",
        ):
            self.assertIn(fragment, app)
        self.assertNotIn('"model_ready"', app)
        self.assertNotIn(
            "isUnavailable(modelAnalysis) || isUnavailable(parameterPayload)", app
        )

    def test_game_search_is_client_side_and_preserves_the_native_chooser(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        for fragment in (
            'id="game-search"',
            'type="search"',
            'placeholder="Type game ID or player name"',
            'inputmode="search"',
            'aria-describedby="game-filter-status"',
            'id="game-filter-status"',
            'id="game-select"',
        ):
            self.assertIn(fragment, index)
        for fragment in (
            'gameFilter: ""',
            "function normalisedGameFilter",
            "function gameSearchTerms",
            "function gameMatchesFilter",
            "function filteredGames",
            "function renderFilteredGameSelect",
            "function applyGameFilter",
            "function clearGameForNoMatch",
            "function clearPendingGameFilterLoad",
            "function scheduleFirstMatchingGameLoad",
            "gameFilterLoadTimer: null",
            "window.setTimeout",
            "}, 250);",
            "gameIdOf(game)",  # exact/partial decimal game ID starts the search terms
            "toLocaleLowerCase().includes(filter)",
            "No games match",
            "state.games.filter",
            'elements.gameSearch.addEventListener("input"',
            "resetGameFilter",
        ):
            self.assertIn(fragment, app)
        for selector in (
            ".game-selector-field",
            ".game-filter-status",
        ):
            self.assertIn(selector, styles)
        apply_start = app.index("function applyGameFilter")
        apply_end = app.index("\nfunction ", apply_start + 1)
        apply_source = app[apply_start:apply_end]
        self.assertIn("selectedStillMatches", apply_source)
        self.assertIn("scheduleFirstMatchingGameLoad()", apply_source)
        self.assertNotIn("fetchJson", apply_source)
        reset_start = app.index("function resetGameFilter")
        reset_end = app.index("\nfunction ", reset_start + 1)
        self.assertIn("clearPendingGameFilterLoad()", app[reset_start:reset_end])

    def test_raw_replay_link_quick_find_uses_submission_and_episode_query_params(
        self,
    ) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        for fragment in (
            'id="replay-link-form"',
            'id="replay-link-input"',
            'placeholder="Paste a link with submissionId and episodeId"',
            'id="replay-link-button"',
            "Open replay",
            'id="replay-link-status"',
        ):
            self.assertIn(fragment, index)
        for fragment in (
            "function replayAddressFromLink",
            "new URL(text)",
            'parsed.searchParams.get("submissionId")',
            'parsed.searchParams.get("episodeId")',
            "function quickFindReplay",
            "await selectSubmission(address.submissionId, { targetGameId: address.episodeId })",
            'targetGameId = ""',
            "requestedGameMissing",
            'outcome === "game_not_found"',
            "sameId(state.gameId, address.episodeId)",
            "Use Check Kaggle now, then retry.",
            'elements.replayLinkForm.addEventListener("submit"',
        ):
            self.assertIn(fragment, app)
        parser_start = app.index("function replayAddressFromLink")
        parser_end = app.index("\nfunction ", parser_start + 1)
        parser_source = app[parser_start:parser_end]
        self.assertNotIn("hostname", parser_source)
        self.assertNotIn("fetch", parser_source)
        for selector in (
            ".replay-link-quick-find",
            ".replay-link-status",
            ".replay-link-status.is-success",
            ".replay-link-status.is-error",
        ):
            self.assertIn(selector, styles)

        contract_path = ROOT / "state/replay-model-inspector-replay-link-quick-find-r220.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        goal = (ROOT / "GOAL.md").read_text(encoding="utf-8")
        self.assertEqual(contract["owner_decision_revision"], 220)
        self.assertEqual(contract["quick_find"]["query_parameters"], {
            "submission": "submissionId",
            "episode": "episodeId",
        })
        self.assertFalse(contract["network_and_safety"]["pasted_url_fetched"])
        self.assertFalse(contract["network_and_safety"]["pasted_url_navigated"])
        self.assertIn("Under revision 220", goal)
        self.assertIn(str(contract_path.relative_to(ROOT)), goal)

    def test_on_demand_kaggle_sync_is_explicit_and_keeps_hourly_refresh(self) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for fragment in (
            'id="sync-now-button"',
            "Check Kaggle now",
        ):
            self.assertIn(fragment, index)
        for fragment in (
            "async function requestReplaySync",
            "async function waitForReplaySync",
            'method: "POST"',
            '"X-Replay-Sync-Intent": "manual"',
            "`${API}/sync-status`",
            "Kaggle check accepted",
            "await refreshIndex()",
            'elements.syncNowButton.addEventListener("click"',
        ):
            self.assertIn(fragment, app)

    def test_submission_search_and_direct_step_entry_preserve_native_choosers(
        self,
    ) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for fragment in (
            'id="submission-search"',
            'placeholder="Type submission ID or label"',
            'id="submission-filter-status"',
            'id="submission-select"',
            'id="step-search"',
            'placeholder="Type exact step number"',
            'inputmode="numeric"',
            'id="step-filter-status"',
            'id="step-select"',
        ):
            self.assertIn(fragment, index)
        for fragment in (
            'submissionFilter: ""',
            "function filteredSubmissions",
            "function renderFilteredSubmissionSelect",
            "function applySubmissionFilter",
            "optionLabelForSubmission(submission)",
            'stepFilter: ""',
            "function filteredSteps",
            "function renderFilteredStepSelect",
            "function applyStepFilter",
            'replace(/\\D/g, "")',
            "sameId(stepIdOf(step), state.stepFilter)",
            "void selectStep(stepIdOf(currentExact))",
            'elements.submissionSearch.addEventListener("input"',
            'elements.stepSearch.addEventListener("input"',
            "traceCache: new Map()",
            "traceFetches: new Map()",
            "traceAbortControllers: new Map()",
            "function fetchBaseTraceCached",
            "const MAX_BROWSER_BASE_TRACES = 8;",
            "while (state.traceCache.size > MAX_BROWSER_BASE_TRACES)",
            "new AbortController()",
            "controller.abort()",
            "a cold exact runtime may take longer than 20 seconds",
            "await fetchBaseTraceCached(state.stepIndex, state.stage)",
        ):
            self.assertIn(fragment, app)
        self.assertNotIn("TRACE_REQUEST_TIMEOUT_MS", app)
        self.assertNotIn("Selected trace reconstruction exceeded 20 seconds", app)
        self.assertNotIn("abortTraceFetchesExcept", app)
        for fragment in (
            "gameMaterialization",
            "gameTraceAddresses",
            "ensureGameMaterialization",
            "runGameMaterialization",
            "fetchMaterializedBaseTrace",
        ):
            self.assertNotIn(fragment, app)

        contract_path = ROOT / "state/replay-model-inspector-on-demand-gpu-trace-r222.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertFalse(contract["trace_loading"]["whole_game_background_prefetch"])
        self.assertEqual(contract["trace_loading"]["selected_trace_timeout_seconds"], 20)
        self.assertEqual(contract["runtime"]["resident_submission_id"], "55410353")

        forward_contract_path = (
            ROOT / "state/replay-model-inspector-forward-pass-reconstruction-r237.json"
        )
        forward_contract = json.loads(
            forward_contract_path.read_text(encoding="utf-8")
        )
        self.assertEqual(forward_contract["owner_decision_revision"], 237)
        self.assertIsNone(
            forward_contract["trace_loading"]["browser_visible_timeout_seconds"]
        )
        self.assertFalse(
            forward_contract["trace_loading"][
                "elapsed_browser_wait_marks_trace_unavailable"
            ]
        )
        self.assertTrue(
            forward_contract["trace_loading"]["stale_browser_requests_aborted"]
        )
        self.assertEqual(forward_contract["execution"]["host"], "elmo")
        self.assertFalse(forward_contract["execution"]["bert_inference_authority"])

        game_contract_path = (
            ROOT
            / "state/replay-model-inspector-physical-game-materialization-r243.json"
        )
        game_contract = json.loads(game_contract_path.read_text(encoding="utf-8"))
        self.assertEqual(game_contract["owner_decision_revision"], 243)
        self.assertEqual(
            game_contract["physical_game_materialization"]["step_or_stage_navigation"],
            "cache_read_or_join_existing_materialization_never_independent_forward",
        )
        self.assertTrue(
            game_contract["physical_game_materialization"][
                "factorized_stages_reuse_decision_state"
            ]
        )
        self.assertEqual(
            game_contract["physical_game_materialization"]["stale_browser_abort"],
            "detach_only; never_corrupt_verified_game_materialization",
        )
        self.assertEqual(game_contract["execution"]["host"], "elmo")
        self.assertFalse(game_contract["execution"]["bert_inference_authority"])

    def test_submission_and_episode_indexes_are_newest_first_but_steps_stay_chronological(
        self,
    ) -> None:
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for fragment in (
            "function newestSubmissionFirst",
            "function newestGameFirst",
            ".sort(newestSubmissionFirst)",
            ".sort(newestGameFirst)",
            'state.steps = collectionFrom(payload, "steps");',
        ):
            self.assertIn(fragment, app)
        self.assertNotIn(".sort(newestStepFirst)", app)

    def test_submission_identifiers_stay_exact_decimal_text(self) -> None:
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for fragment in (
            "function strictDecimalIdentifier",
            '"submission_id_text"',
            "Number.isSafeInteger(value)",
            "function decimalIdentifierFrom",
            "function traceAddressForDisplay",
            "Exact submission ID: ${identity.idText}",
            "Submission ID: ${identity.idText}",
            'state.submissionId = submissionIdOf(submission) || ""',
        ):
            self.assertIn(fragment, app)
        self.assertNotIn("formatValue(identity.id)", app)

    def test_parameter_and_provenance_panels_use_exact_submission_id_text(self) -> None:
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        for fragment in (
            "function isSubmissionIdField",
            "function hasExactSubmissionIdText",
            "function formatRecordFieldValue",
            '"submission_id_text", "submissionIdText"',
            'return exact === undefined ? "Exact submission ID unavailable" : exact;',
            "detail.textContent = formatRecordFieldValue(source, key, value);",
            'appendSourceObject(panes, "Submission provenance", provenance)',
            '"Parameter index response"',
        ):
            self.assertIn(fragment, app)

        grid_start = app.index("function keyValueGrid")
        grid_end = app.index("\nfunction ", grid_start + 1)
        grid_source = app[grid_start:grid_end]
        self.assertIn("formatRecordFieldValue(source, key, value)", grid_source)
        self.assertNotIn("detail.textContent = formatValue(value)", grid_source)

    def test_selected_game_links_to_ptcg_visualizer_with_only_exact_replay_id(self) -> None:
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        for fragment in (
            "function ptcgVisualizerReplayUrl(game)",
            "https://ptcgvis.heroz.jp/Visualizer/Replay/${encodeURIComponent(replayId)}/0",
            'visualizerLink.target = "_blank"',
            'visualizerLink.rel = "noopener noreferrer"',
            'visualizerLink.referrerPolicy = "no-referrer"',
            "Open replay in PTCG Visualizer ↗",
        ):
            self.assertIn(fragment, app)
        self.assertIn(".external-replay-link", styles)

    def test_matchup_adapter_legal_action_table_is_sortable_by_every_header(self) -> None:
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        for fragment in (
            "function adapterComparisonDisplayLabel",
            "function adapterComparisonChoiceRank",
            "function adapterComparisonSortValue",
            "function compareAdapterComparisonOptions",
            "ADAPTER_COMPARISON_COLLATOR = new Intl.Collator",
            '{ key: "action", label: "Legal action", initialDirection: "ascending" }',
            '{ key: "adapter_on", label: "Adapter ON chance", initialDirection: "descending" }',
            '{ key: "adapter_off", label: "Adapter OFF chance", initialDirection: "descending" }',
            '{ key: "change", label: "Change caused by adapter", initialDirection: "descending" }',
            '{ key: "chosen", label: "Chosen", initialDirection: "descending" }',
            'cell.setAttribute("aria-sort", "none")',
            'button.type = "button"',
            'button.className = "adapter-comparison-sort-button"',
            "button.dataset.sortKey = column.key",
            "finiteNumber(entry.option.adapter_on_probability)",
            "finiteNumber(entry.option.adapter_off_probability)",
            "finiteNumber(entry.option.probability_delta)",
            "option.adapter_on_choice === true ? 2 : 0",
            "option.adapter_off_choice === true ? 1 : 0",
            'recordValue(left.option, "position", "index")',
            "return left.sourceIndex - right.sourceIndex",
            'button.addEventListener("click"',
        ):
            self.assertIn(fragment, app)
        for selector in (
            ".adapter-comparison-sort-button",
            ".adapter-comparison-sort-button:focus-visible",
            ".adapter-comparison-sort-indicator",
        ):
            self.assertIn(selector, styles)

        contract_path = ROOT / "state/replay-model-inspector-adapter-comparison-sorting-r223.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(contract["owner_decision_revision"], 223)
        self.assertEqual(
            contract["sorting"]["columns"],
            ["legal_action", "adapter_on_chance", "adapter_off_chance", "adapter_change", "chosen"],
        )
        self.assertTrue(contract["safety"]["client_side_only"])

    def test_r187_distinguishes_decision_counterfactuals_from_training_weights(
        self,
    ) -> None:
        index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        app = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")

        for selector_id in (
            "decision-influence-availability",
            "decision-influence-reset",
            "decision-influence-apply",
            "decision-influence-controls",
            "decision-influence-choice",
            "decision-influence-body",
            "decision-influence-detail",
            "training-recipe-availability",
            "training-recipe-content",
        ):
            self.assertIn(f'id="{selector_id}"', index)
        for fragment in (
            "not training, is not saved, and is not a historical replay/model trace",
            "Training-loss weights require a real, isolated fine-tune",
            "do not change this submitted model’s current forward pass",
            "cached replays remain evaluation-only",
            "function renderDecisionInfluenceAwaiting",
            "function loadDecisionInfluence",
            "function scheduleDecisionInfluenceRecompute",
            "function decisionInfluencePath",
            "exact_nonlinear_fusion_source_recomputation",
            "decision_only_counterfactual",
            "counterfactual_minus_baseline",
            "?stage=${encodeURIComponent(state.stage)}",
            'query.set("scales"',
            "Reset to 1× baseline",
            "function renderTrainingRecipeAwaiting",
            "function loadTrainingRecipe",
            "/training-recipe",
            "source_backed_training_loss_multipliers",
            "guide",
            "range.min = String(minimum)",
            "range.max = String(maximum)",
            'range.step = "0.05"',
            "never sums leave-one-out effects",
            "Missing source-backed weights are unavailable, not zero.",
            "Baseline head coefficient:",
            "nominal_policy_coefficient",
        ):
            self.assertIn(fragment, index + app)
        for selector in (
            ".decision-influence-playground",
            ".decision-influence-controls",
            ".decision-influence-baseline-weight",
            ".training-recipe-card",
        ):
            self.assertIn(selector, styles)


if __name__ == "__main__":
    unittest.main()
