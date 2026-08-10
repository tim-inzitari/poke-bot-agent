const API = "./api";
const SLICE_LIMIT = 256;
const TRACE_REQUEST_TIMEOUT_MS = 20000;
const ADAPTER_COMPARISON_COLLATOR = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: "base",
});

const state = {
  health: null,
  healthError: null,
  submissions: [],
  submissionFilter: "",
  submissionFilterLoadTimer: null,
  submissionId: "",
  games: [],
  gameId: "",
  gameFilter: "",
  gameFilterLoadTimer: null,
  steps: [],
  stepFilter: "",
  stepFilterLoadTimer: null,
  stepIndex: "",
  stage: "",
  trace: null,
  traceError: null,
  traceCache: new Map(),
  traceFetches: new Map(),
  traceAbortControllers: new Map(),
  guideShadowEnabled: false,
  decisionInfluence: {
    payload: null,
    error: null,
    scales: {},
    eligibleHeads: [],
    debounceTimer: null,
  },
  trainingRecipe: null,
  trainingRecipeError: null,
  parameters: [],
  parametersPayload: null,
  parameterName: "",
  parameterDetail: null,
  parameterError: null,
  parameterOffset: 0,
  activeView: "decision",
  requests: {
    index: 0,
    submission: 0,
    game: 0,
    trace: 0,
    decisionInfluence: 0,
    trainingRecipe: 0,
    parameter: 0,
  },
};

const HEAD_FAQ = [
  {
    id: "value",
    title: "Overall position value",
    question: "If the game continued from here, how good is this position for us?",
    answer: "A single whole-game estimate learned from final results. Higher means the position looks better for us; lower means it looks worse.",
    horizon: "The rest of the game",
    caution: "This is an expected result signal, not a guaranteed win percentage.",
  },
  {
    id: "archetype",
    title: "Opponent deck style",
    question: "What kind of deck does the visible opponent evidence most resemble?",
    answer: "It recognizes likely opponent archetypes so the policy can interpret the same board differently against different strategies.",
    horizon: "Current matchup",
    caution: "It is a belief from visible information, not access to the opponent’s hidden deck list.",
  },
  {
    id: "opponent_hand",
    title: "Opponent hand belief",
    question: "Which cards might currently be in the opponent’s hidden hand?",
    answer: "It assigns plausibility to card identities that could be in the opponent’s hand based on information seen so far.",
    horizon: "Right now",
    caution: "This is a hidden-card belief, never a claim that the model saw the private hand.",
  },
  {
    id: "opponent_remainder",
    title: "Opponent unseen-card belief",
    question: "Which cards are likely still somewhere in the opponent’s unseen remainder?",
    answer: "It estimates likely card identities among the opponent’s still-hidden cards, helping the model reason about what may remain available.",
    horizon: "Rest of this game",
    caution: "It does not know the exact card order or reveal hidden zones.",
  },
  {
    id: "lethal_threat",
    title: "Our near-term prize-taking chance",
    question: "Are we likely to take a prize soon along this line?",
    answer: "It was trained on whether our own prize count decreases within the next eight later decisions made by us—roughly, whether our attack or knockout path pays off soon.",
    horizon: "Next 8 of our decision frames",
    caution: "Naming trap: this is our offensive threat to the opponent, not the chance that we lose or get knocked out.",
  },
  {
    id: "prize_race",
    title: "Prize-race scoreboard",
    question: "How many prizes do we and the opponent still have left?",
    answer: "It reconstructs the public prize race as our remaining prizes and their remaining prizes, normalized against six.",
    horizon: "Current public scoreboard",
    caution: "It is a race-position scaffold, not a prediction of who will take the next prize.",
  },
  {
    id: "action_q",
    title: "Long-term value of this action",
    question: "If we choose this legal option, how good does the eventual result look?",
    answer: "It gives each legal candidate its own long-term action-value estimate, trained from the game’s terminal result for the action that was actually taken.",
    horizon: "From this action to game end",
    caution: "Unchosen actions need real counterfactual evidence; their scores should not be treated as simulator truth.",
  },
  {
    id: "action_type",
    title: "Right kind of action",
    question: "Does this candidate use the kind of move that fits the situation?",
    answer: "It distinguishes broad action kinds—such as play, attach, retreat, attack, or end—before considering the finer details.",
    horizon: "This choice",
    caution: "It judges the action category, not whether every target and resource inside it is correct.",
  },
  {
    id: "action_target",
    title: "Right target",
    question: "Is this action aimed at the right Pokémon, card, or board slot?",
    answer: "It helps separate candidates that do the same kind of thing but point at different legal targets.",
    horizon: "This choice",
    caution: "It is about target selection, not the action’s whole strategic value.",
  },
  {
    id: "action_resource",
    title: "Right card or resource",
    question: "Is this candidate using the right source card, Tool, or Energy?",
    answer: "It helps distinguish which legal resource should be spent or attached when several sources could perform a similar action.",
    horizon: "This choice",
    caution: "It evaluates resource binding, not long-term resource conservation by itself.",
  },
  {
    id: "action_utility",
    title: "Immediate payoff",
    question: "What changes immediately after taking this action?",
    answer: "For each legal option it estimates damage dealt, cards drawn, Energy change, open Bench change, prize change, and whether a knockout occurs.",
    horizon: "Immediate post-action transition",
    caution: "This is the immediate payoff, not the opponent’s later response or the final game result.",
  },
  {
    id: "tactical_outcomes",
    title: "Short tactical future",
    question: "What is likely to happen over the next few times we get to decide?",
    answer: "At 1-, 2-, and 3-own-decision horizons it tracks our prizes, their prizes, our knockouts suffered, their knockouts suffered, net damage, and net prize movement.",
    horizon: "Next 1, 2, and 3 of our decision frames",
    caution: "For ‘could one of ours be knocked out soon?’, inspect its own-knockout outputs.",
  },
  {
    id: "opponent_response",
    title: "Opponent’s next response",
    question: "What is the opponent likely to do before we get another decision?",
    answer: "It predicts whether they attack, take a prize, score a knockout, switch their Active, reduce their hand, add board Energy, or end without attacking.",
    horizon: "After our action, before our next decision",
    caution: "Its knockout output is the clearest near-term ‘can they knock one of ours out next?’ signal.",
  },
  {
    id: "resource_forecast",
    title: "Resources on our next decision",
    question: "What will our usable position probably look like when we act again?",
    answer: "It forecasts hand size, deck size, attached Energy, open Bench slots, whether an Energy attachment is available, and whether retreat is available.",
    horizon: "Our next decision",
    caution: "It forecasts availability, not whether using each resource will be correct.",
  },
  {
    id: "game_phase",
    title: "Stage of the game",
    question: "Are we setting up, stabilizing, applying pressure, racing prizes, or closing out?",
    answer: "It classifies the current strategic phase so the same action can be valued differently early, midgame, or near the finish.",
    horizon: "Current phase",
    caution: "One label summarizes the phase; real positions can contain features of several phases.",
  },
  {
    id: "outcome_distribution",
    title: "Lose / draw / win outlook",
    question: "How does the model divide the possible final result among loss, draw, and win?",
    answer: "It produces the model’s game-level outcome distribution. Its loss side is the closest direct learned signal for ‘what is our chance of eventually losing?’",
    horizon: "Final game result",
    caution: "Use calibration evidence before reading the displayed loss score as a perfectly calibrated real-world percentage.",
  },
  {
    id: "remaining_turns",
    title: "How long the game may last",
    question: "Roughly how many complete turns remain before the game ends?",
    answer: "It estimates the logarithm of remaining complete game turns, giving the policy a sense of urgency without letting very long games dominate training.",
    horizon: "Until game end",
    caution: "The raw head value is log-scaled; it is not a literal turn count unless decoded.",
  },
  {
    id: "setup_board_outcome",
    title: "Opening-board quality",
    question: "Which setup choice leads to a healthier next position and eventual result?",
    answer: "For setup Active and Bench choices it predicts next-decision resources plus the eventual loss/draw/win outcome for each legal setup option.",
    horizon: "Next own decision and game end",
    caution: "It is specialized for setup choices and may correctly have no effect later in the game.",
  },
  {
    id: "combo_state",
    title: "Deck-combo readiness",
    question: "Does this option preserve or assemble the specialist deck’s important combo pieces?",
    answer: "It represents deck-specific combo state such as top-deck plans, search sources, copied attacks, visible pieces, Energy routes, and Bench continuity when that specialist has a valid combo contract.",
    horizon: "Current and near-future combo line",
    caution: "It is specialist-specific and may be deliberately disabled even though its tensors exist in the checkpoint.",
  },
];

const elements = {
  healthBadge: byId("health-badge"),
  replayLinkForm: byId("replay-link-form"),
  replayLinkInput: byId("replay-link-input"),
  replayLinkButton: byId("replay-link-button"),
  replayLinkStatus: byId("replay-link-status"),
  submissionSelect: byId("submission-select"),
  submissionSearch: byId("submission-search"),
  submissionFilterStatus: byId("submission-filter-status"),
  gameSearch: byId("game-search"),
  gameSelect: byId("game-select"),
  gameFilterStatus: byId("game-filter-status"),
  stepSelect: byId("step-select"),
  stepSearch: byId("step-search"),
  stepFilterStatus: byId("step-filter-status"),
  stageSelect: byId("stage-select"),
  selectionContext: byId("selection-context"),
  selectionIdentity: byId("selection-identity"),
  syncNowButton: byId("sync-now-button"),
  refreshButton: byId("refresh-button"),
  globalNotice: byId("global-notice"),
  tabs: Array.from(document.querySelectorAll(".view-tab")),
  decisionView: byId("decision-view"),
  weightsView: byId("weights-view"),
  traceStatus: byId("trace-status"),
  provenanceAvailability: byId("provenance-availability"),
  provenanceContent: byId("provenance-content"),
  reproductionStatus: byId("reproduction-status"),
  recordedAction: byId("recorded-action"),
  recordedActionNote: byId("recorded-action-note"),
  modelAction: byId("model-action"),
  modelActionNote: byId("model-action-note"),
  actionAgreement: byId("action-agreement"),
  actionAgreementNote: byId("action-agreement-note"),
  modelValue: byId("model-value"),
  modelRouteNote: byId("model-route-note"),
  matchupAdapterAvailability: byId("matchup-adapter-availability"),
  matchupAdapterStatus: byId("matchup-adapter-status"),
  matchupAdapterContent: byId("matchup-adapter-content"),
  guideShadowAvailability: byId("guide-shadow-availability"),
  guideShadowToggle: byId("guide-shadow-toggle"),
  guideShadowToggleState: byId("guide-shadow-toggle-state"),
  guideShadowContent: byId("guide-shadow-content"),
  observationAvailability: byId("observation-availability"),
  observationContent: byId("observation-content"),
  optionsAvailability: byId("options-availability"),
  optionsBody: byId("options-body"),
  optionContributionDetail: byId("option-contribution-detail"),
  fusionAvailability: byId("fusion-availability"),
  fusionContent: byId("fusion-content"),
  headsAvailability: byId("heads-availability"),
  headImpactSummary: byId("head-impact-summary"),
  headImpactGrid: byId("head-impact-grid"),
  headsBody: byId("heads-body"),
  headsNote: byId("heads-note"),
  headFaqList: byId("head-faq-list"),
  decisionInfluenceAvailability: byId("decision-influence-availability"),
  decisionInfluenceReset: byId("decision-influence-reset"),
  decisionInfluenceApply: byId("decision-influence-apply"),
  decisionInfluenceDebounce: byId("decision-influence-debounce"),
  decisionInfluenceControls: byId("decision-influence-controls"),
  decisionInfluenceChoice: byId("decision-influence-choice"),
  decisionInfluenceBody: byId("decision-influence-body"),
  decisionInfluenceDetail: byId("decision-influence-detail"),
  trainingRecipeAvailability: byId("training-recipe-availability"),
  trainingRecipeContent: byId("training-recipe-content"),
  warningsSection: byId("warnings-section"),
  warningsList: byId("warnings-list"),
  weightsStatus: byId("weights-status"),
  weightsAvailability: byId("weights-availability"),
  weightsProvenanceContent: byId("weights-provenance-content"),
  parameterSearch: byId("parameter-search"),
  parameterCount: byId("parameter-count"),
  parametersBody: byId("parameters-body"),
  parameterDetailStatus: byId("parameter-detail-status"),
  parameterDetail: byId("parameter-detail"),
};

function byId(id) {
  return document.getElementById(id);
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function stringValue(value) {
  if (value === undefined || value === null) return "";
  return String(value);
}

function sameId(left, right) {
  return stringValue(left) === stringValue(right);
}

function collectionFrom(payload, key) {
  if (Array.isArray(payload)) return payload;
  if (isObject(payload) && Array.isArray(payload[key])) return payload[key];
  return [];
}

function reasonFrom(source, fallback = "Availability was not stated by the source.") {
  if (!source || typeof source !== "object") return fallback;
  const reason = firstDefined(
    source.reason,
    source.unavailable_reason,
    source.availability_reason,
    source.availability && source.availability.reason,
    source.detail,
    source.message,
  );
  return reason === undefined || reason === null || reason === "" ? fallback : stringValue(reason);
}

function isUnavailable(source) {
  if (!source || typeof source !== "object") return false;
  return (
    source.available === false ||
    source.availability === false ||
    (isObject(source.availability) && source.availability.available === false) ||
    source.status === "unavailable" ||
    source.status === "not_available" ||
    source.status === "error"
  );
}

function reproductionStatusOf(trace) {
  return recordValue(trace, "reproduction_status", "reconstruction_status", "replay_status");
}

function reproductionStatusMessage(trace) {
  const status = reproductionStatusOf(trace);
  if (status === "exact_reproduced") {
    return "Exact reproduced: dynamic model values were checksum-bound causal re-evaluation results, not values stored in the raw replay.";
  }
  if (status === "recomputed_not_historical") {
    return "Recomputed, not historical: the raw replay records the decision, while dynamic model values are a checksum-bound causal re-evaluation.";
  }
  if (status === "exact_runtime_short_circuit") {
    return "Exact runtime decision: the submitted entrypoint made this setup choice deterministically before the neural model ran. The displayed 100% / 0% policy is the exact runtime choice, not a neural softmax.";
  }
  if (status === "hypothetical_model_forward_not_submitted_runtime") {
    return "Hypothetical neural rerun: the submitted runtime skipped the model for this setup prompt, so the displayed percentages were freshly calculated by asking the checksum-bound archived model what it would choose. These are not historical Kaggle policy outputs.";
  }
  if (status === "diverged_or_fallback_unknown") {
    return "Diverged or fallback unknown: this reconstruction must not be treated as the historical model output.";
  }
  if (status === "unavailable") {
    return plainAvailabilityReason(trace, "Reproduction is unavailable; dynamic model values are not displayed as recorded outputs.");
  }
  if (status !== undefined && status !== null) {
    return `Reproduction status: ${formatValue(status)}. Dynamic model values are re-evaluated evidence, not raw replay fields.`;
  }
  return "Reproduction status was not stated. Dynamic model values, when present, are causal re-evaluation evidence rather than raw replay fields.";
}

function renderReproductionStatus(trace) {
  const status = trace ? reproductionStatusOf(trace) : undefined;
  elements.reproductionStatus.className = "reproduction-status";
  if (status === "exact_reproduced" || status === "exact_runtime_short_circuit") {
    elements.reproductionStatus.classList.add("is-exact");
  }
  if (status === "unavailable" || status === "diverged_or_fallback_unknown") {
    elements.reproductionStatus.classList.add("is-unavailable");
  }
  elements.reproductionStatus.textContent = trace
    ? reproductionStatusMessage(trace)
    : "Reproduction status is not available until a trace is selected.";
}

function asBooleanText(value) {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return formatValue(value);
}

function formatNumber(value, digits = 6) {
  if (typeof value !== "number") return formatValue(value);
  if (Number.isNaN(value)) return "NaN";
  if (value === Infinity) return "∞";
  if (value === -Infinity) return "−∞";
  if (!Number.isFinite(value)) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 1e6 || magnitude < 1e-4)) {
    return value.toExponential(Math.min(5, digits));
  }
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    useGrouping: magnitude >= 1000,
  });
}

function formatProbability(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return formatValue(value);
  return `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 4 })}% (${formatNumber(value)})`;
}

function compactJson(value, limit = 180) {
  let serialized;
  try {
    serialized = JSON.stringify(value);
  } catch {
    serialized = String(value);
  }
  if (serialized === undefined) return "—";
  return serialized.length > limit ? `${serialized.slice(0, Math.max(0, limit - 1))}…` : serialized;
}

function fullJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatValue(value) {
  if (value === undefined || value === null || value === "") return "—";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value;
  return compactJson(value);
}

function setText(element, value) {
  element.textContent = value;
}

function clearNode(node) {
  node.replaceChildren();
}

function setBadge(element, label, kind = "pending") {
  element.className = `status-badge status-${kind}`;
  element.textContent = label;
}

function setAvailability(element, source, loadedLabel = "Available", pendingLabel = "Awaiting trace") {
  if (source === null || source === undefined) {
    element.textContent = pendingLabel;
    return;
  }
  if (isUnavailable(source)) {
    element.textContent = reasonFrom(source, "Source marked this data unavailable.");
    return;
  }
  element.textContent = loadedLabel;
}

function setGlobalNotice(message = "", kind = "") {
  elements.globalNotice.textContent = message;
  elements.globalNotice.className = `global-notice${kind ? ` is-${kind}` : ""}`;
}

function setEmptyRow(body, colSpan, text) {
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = colSpan;
  cell.className = "empty-cell";
  cell.textContent = text;
  row.append(cell);
  body.replaceChildren(row);
}

function appendCell(row, value, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  cell.textContent = formatValue(value);
  row.append(cell);
  return cell;
}

function appendStructuredValue(cell, value, empty = "—") {
  if (value === undefined || value === null) {
    cell.textContent = empty;
    return;
  }
  if (typeof value !== "object") {
    cell.textContent = formatValue(value);
    return;
  }
  const details = document.createElement("details");
  details.className = "cell-details";
  const summary = document.createElement("summary");
  summary.textContent = compactJson(value, 92);
  const pre = document.createElement("pre");
  pre.className = "json-block";
  pre.textContent = fullJson(value);
  details.append(summary, pre);
  cell.append(details);
}

function structuredValueCell(row, value, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  appendStructuredValue(cell, value);
  row.append(cell);
  return cell;
}

function recordValue(record, ...keys) {
  if (!record || typeof record !== "object") return undefined;
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) return record[key];
  }
  return undefined;
}

async function fetchJson(path, { signal } = {}) {
  const response = await fetch(path, {
    method: "GET",
    credentials: "same-origin",
    headers: { Accept: "application/json" },
    signal,
  });
  const raw = await response.text();
  let payload = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      if (response.ok) throw new Error("The inspector API returned a non-JSON response.");
    }
  }
  if (!response.ok) {
    const sourceMessage = isObject(payload)
      ? reasonFrom(payload, `Request failed with HTTP ${response.status}.`)
      : `Request failed with HTTP ${response.status}.`;
    throw new Error(sourceMessage);
  }
  return payload || {};
}

function strictDecimalIdentifier(value) {
  if (typeof value === "string") {
    const text = value.trim();
    return /^\d+$/.test(text) ? text : undefined;
  }
  if (typeof value === "bigint") return value >= 0n ? value.toString() : undefined;
  if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) return String(value);
  return undefined;
}

function replayAddressFromLink(value) {
  const text = stringValue(value).trim();
  if (!text) return undefined;
  let parsed;
  try {
    parsed = new URL(text);
  } catch (_error) {
    return undefined;
  }
  const submissionId = strictDecimalIdentifier(parsed.searchParams.get("submissionId"));
  const episodeId = strictDecimalIdentifier(parsed.searchParams.get("episodeId"));
  return submissionId !== undefined && episodeId !== undefined
    ? { submissionId, episodeId }
    : undefined;
}

function setReplayLinkStatus(message, stateName = "") {
  elements.replayLinkStatus.className = `replay-link-status${stateName ? ` is-${stateName}` : ""}`;
  elements.replayLinkStatus.textContent = message;
}

function setReplayLinkControlsEnabled(enabled) {
  elements.replayLinkInput.disabled = !enabled;
  elements.replayLinkButton.disabled = !enabled;
}

function decimalIdentifierFrom(record, ...keys) {
  for (const key of keys) {
    const text = strictDecimalIdentifier(recordValue(record, key));
    if (text !== undefined) return text;
  }
  return undefined;
}

function submissionIdOf(submission) {
  return decimalIdentifierFrom(submission, "submission_id_text", "submission_id", "id", "submission");
}

function newestSubmissionFirst(left, right) {
  const leftId = submissionIdOf(left) || "";
  const rightId = submissionIdOf(right) || "";
  if (leftId.length !== rightId.length) return rightId.length - leftId.length;
  return rightId.localeCompare(leftId);
}

function gameIdOf(game) {
  return decimalIdentifierFrom(game, "episode_id_text", "episode_id", "game_id", "id", "episode");
}

function ptcgVisualizerReplayUrl(game) {
  const replayId = gameIdOf(game);
  return replayId === undefined
    ? undefined
    : `https://ptcgvis.heroz.jp/Visualizer/Replay/${encodeURIComponent(replayId)}/0`;
}

function newestGameFirst(left, right) {
  const leftId = gameIdOf(left) || "";
  const rightId = gameIdOf(right) || "";
  if (leftId.length !== rightId.length) return rightId.length - leftId.length;
  return rightId.localeCompare(leftId);
}

function stepIdOf(step) {
  return decimalIdentifierFrom(step, "step_index_text", "step_index", "step", "index");
}

function currentSubmission() {
  return state.submissions.find((submission) => sameId(submissionIdOf(submission), state.submissionId));
}

function currentGame() {
  return state.games.find((game) => sameId(gameIdOf(game), state.gameId));
}

function currentStep() {
  return state.steps.find((step) => sameId(stepIdOf(step), state.stepIndex));
}

function submissionIdentityOf(submission) {
  const id = submissionIdOf(submission);
  const idText = id === undefined ? "submission ID unavailable" : id;
  const exactText = firstNonEmptyText(
    recordValue(submission, "submission_text", "submission_label", "submitted_text"),
  );
  const label = firstNonEmptyText(recordValue(submission, "label", "name", "title"));
  return {
    id,
    idText,
    exactText,
    label,
    primary: exactText || label || (id === undefined ? "Unidentified submission" : `Submission ${idText}`),
  };
}

function availabilityState(source) {
  if (source === undefined || source === null) return { known: false, available: undefined, reason: undefined };
  if (isUnavailable(source)) {
    return { known: true, available: false, reason: reasonFrom(source, "The source marked this unavailable.") };
  }
  if (isObject(source) && recordValue(source, "available") === true) {
    return { known: true, available: true, reason: undefined };
  }
  if (source === true) return { known: true, available: true, reason: undefined };
  if (source === false) return { known: true, available: false, reason: "The source marked this unavailable." };
  return { known: false, available: undefined, reason: undefined };
}

function plainAvailabilityReason(value, fallback) {
  const raw = isObject(value) ? reasonFrom(value, fallback) : stringValue(value || fallback);
  const normalized = raw.trim();
  const known = {
    runtime_parity_receipt_missing: "missing exact-runtime parity receipt",
    runtime_source_parity_receipt_missing: "missing exact-runtime parity receipt",
    runtime_source_provenance_unavailable: "missing exact-runtime parity evidence",
    runtime_parity_unavailable: "missing exact-runtime parity",
    // A package may exist on disk without being declared and checksum-bound
    // to this submission. Do not present that provenance gap as a missing
    // local file or an available dynamic model trace.
    runtime_package_artifact_missing: "exact submitted runtime is not yet checksum-attested",
    runtime_package_path_missing: "exact submitted runtime is not yet checksum-attested",
    runtime_package_sha256_invalid_or_missing: "exact submitted runtime is not yet checksum-attested",
    runtime_package_source_roots_not_configured: "exact submitted runtime cannot be checksum-attested from approved artifact roots",
    runtime_package_file_missing: "exact submitted runtime artifact is unavailable for checksum verification",
    runtime_package_path_outside_configured_roots: "exact submitted runtime is outside approved checksum roots",
    runtime_package_path_unreadable: "exact submitted runtime artifact cannot be read for checksum verification",
    runtime_package_not_a_regular_file: "exact submitted runtime artifact is not a regular file",
    runtime_package_hash_unreadable: "exact submitted runtime artifact cannot be hashed for checksum verification",
    runtime_package_sha256_mismatch: "exact submitted runtime checksum does not match its attestation",
    runtime_package_module_path_unavailable: "exact submitted runtime cannot be loaded from the checksum-attested source",
    runtime_parity_binding_invalid: "exact submitted runtime is not bound by a valid parity receipt",
    runtime_parity_receipt_unavailable: "missing usable exact-runtime parity receipt",
  };
  if (known[normalized]) return known[normalized];
  return normalized.includes("_") ? normalized.replaceAll("_", " ") : normalized;
}

function submissionReadiness(submission) {
  const declared = firstNonEmptyText(recordValue(submission, "status", "submission_status"))?.toLowerCase();
  const status = ["trace_ready", "weights_only", "replay_only"].includes(declared) ? declared : undefined;
  const weightsSource = firstDefined(
    submission?.weights,
    submission?.checkpoint_weights,
    submission?.model_analysis,
  );
  const dynamicTraceSource = firstDefined(
    submission?.dynamic_trace,
    submission?.dynamic_trace_availability,
    submission?.trace_availability,
    isObject(submission?.model_analysis) ? submission.model_analysis.dynamic_trace : undefined,
  );
  const weightsState = availabilityState(weightsSource);
  const dynamicTraceState = availabilityState(dynamicTraceSource);
  const weightsAvailable = weightsState.known
    ? weightsState.available
    : status === "trace_ready" || status === "weights_only"
      ? true
      : status === "replay_only" ? false : undefined;
  const traceAvailable = dynamicTraceState.known
    ? dynamicTraceState.available
    : status === "trace_ready" ? true
      : status === "weights_only" || status === "replay_only" ? false : undefined;
  const weightsReason = weightsState.reason;
  const traceFallback = status === "weights_only"
    ? "missing exact-runtime parity"
    : status === "replay_only"
      ? "exact submitted-model analysis is unavailable"
      : "The source did not state why dynamic decision tracing is unavailable.";
  const traceReason = traceAvailable === false
    ? plainAvailabilityReason(dynamicTraceState.reason, traceFallback)
    : undefined;
  const weightsText = weightsAvailable === true
    ? "Exact model weights available"
    : weightsAvailable === false
      ? `Exact model weights unavailable${weightsReason ? ` — ${plainAvailabilityReason(weightsReason, "")}` : ""}`
      : "Exact model-weight availability not supplied";
  const traceText = traceAvailable === true
    ? "Dynamic decision trace available"
    : traceAvailable === false
      ? `Dynamic decision trace unavailable — ${traceReason}`
      : "Dynamic decision-trace availability not supplied";
  return {
    status,
    weights: { available: weightsAvailable, reason: weightsReason, source: weightsSource },
    dynamicTrace: { available: traceAvailable, reason: traceReason, source: dynamicTraceSource },
    weightsText,
    traceText,
    evidence: {
      status: status || "status_not_supplied",
      weights: weightsSource,
      dynamic_trace: dynamicTraceSource,
    },
  };
}

function submissionModelStatus(submission) {
  const readiness = submissionReadiness(submission);
  if (readiness.weights.available === true && readiness.dynamicTrace.available === true) {
    return "weights and dynamic trace available";
  }
  if (readiness.weights.available === true && readiness.dynamicTrace.available === false) {
    return "weights available; dynamic trace unavailable";
  }
  if (readiness.weights.available === false) return "replay-only; model weights unavailable";
  return "model availability not supplied";
}

function cachedReplayOutcome(submission) {
  const source = firstDefined(
    submission?.cached_replay_outcomes,
    submission?.cached_replay_outcome,
    submission?.replay_outcomes,
    submission?.replay_outcome,
  );
  if (!isObject(source) || isUnavailable(source)) {
    return {
      available: false,
      reason: reasonFrom(source, "Cached replay outcomes were not supplied."),
    };
  }
  const wins = finiteNumber(recordValue(source, "wins", "win_count", "cached_replay_wins"));
  const denominator = finiteNumber(recordValue(
    source,
    "games_with_outcome",
    "outcome_count",
    "denominator",
    "cached_replay_denominator",
  ));
  const directRate = finiteNumber(recordValue(source, "win_rate", "win_rate_percent", "percent"));
  if (
    wins === undefined
    || denominator === undefined
    || denominator <= 0
    || wins < 0
    || wins > denominator
  ) {
    return {
      available: false,
      reason: reasonFrom(source, "Cached replay outcomes do not include a usable win denominator."),
    };
  }
  const rate = directRate === undefined
    ? wins / denominator
    : directRate > 1 ? directRate / 100 : directRate;
  if (!Number.isFinite(rate) || rate < 0 || rate > 1) {
    return { available: false, reason: "Cached replay win rate is not finite." };
  }
  return { available: true, wins, denominator, rate, source };
}

function cachedReplayOutcomeText(submission) {
  const outcome = cachedReplayOutcome(submission);
  if (!outcome.available) return "win rate unavailable";
  const percentage = (outcome.rate * 100).toLocaleString(undefined, { maximumFractionDigits: 2 });
  return `cached replay ${formatNumber(outcome.wins)}/${formatNumber(outcome.denominator)} (${percentage}%)`;
}

function optionLabelForSubmission(submission) {
  const identity = submissionIdentityOf(submission);
  const label = identity.primary === "Unidentified submission" ? "label unavailable" : identity.primary;
  const readiness = submissionReadiness(submission);
  return [
    identity.idText,
    `label: ${label}`,
    `weights: ${readiness.weightsText}`,
    `trace: ${readiness.traceText}`,
    cachedReplayOutcomeText(submission),
  ].join(" · ");
}

function playerNameOf(player, fallback = "Player") {
  if (!player || typeof player !== "object") return fallback;
  return firstNonEmptyText(
    recordValue(player, "name", "player_name", "submission_text", "label", "display_name", "submission"),
  ) || fallback;
}

function gamePlayers(game, submission = currentSubmission()) {
  const ownSeat = recordValue(game, "own_seat", "seat");
  const indexed = collectionFrom(game, "players").filter(isObject);
  if (indexed.length) {
    return indexed.map((player, index) => ({
      name: playerNameOf(player, `Player ${index + 1}`),
      seat: recordValue(player, "seat", "player_seat", "position", "index"),
      rank: recordValue(player, "rank"),
      isOwn: recordValue(player, "is_own", "is_self", "is_submission") === true
        || sameId(recordValue(player, "seat", "player_seat", "position", "index"), ownSeat),
      source: player,
    }));
  }

  const ownIdentity = submissionIdentityOf(submission || {});
  const ownName = firstNonEmptyText(
    recordValue(game, "own_player", "own_player_name", "player_name"),
    ownIdentity.exactText,
    ownIdentity.label,
  ) || "This submission";
  const opponentValue = recordValue(game, "opponent", "opponent_name", "opponent_submission");
  const opponentName = isObject(opponentValue)
    ? playerNameOf(opponentValue, "Opponent")
    : firstNonEmptyText(opponentValue) || "Opponent";
  const opponentSeat = recordValue(game, "opponent_seat");
  const ownRank = recordValue(game, "own_rank", "player_rank");
  const opponentRank = isObject(opponentValue)
    ? recordValue(opponentValue, "rank")
    : recordValue(game, "opponent_rank");
  return [
    { name: ownName, seat: ownSeat, rank: ownRank, isOwn: true },
    { name: opponentName, seat: opponentSeat, rank: opponentRank, isOwn: false },
  ];
}

function playerSummary(player) {
  const parts = [player.name];
  parts.push(player.seat === undefined || player.seat === null || player.seat === ""
    ? "Seat unavailable"
    : `seat ${formatValue(player.seat)}`);
  parts.push(player.rank === undefined || player.rank === null || player.rank === ""
    ? "Rank unavailable"
    : `Rank ${formatValue(player.rank)}`);
  return parts.join(" · ");
}

function optionLabelForGame(game) {
  const episodeId = gameIdOf(game);
  const parts = [episodeId === undefined ? "Unidentified game" : `Game ${episodeId}`];
  const steps = recordValue(game, "step_count", "decision_count");
  const players = gamePlayers(game);
  if (players.length) parts.push(players.map(playerSummary).join(" vs "));
  if (steps !== undefined) parts.push(`${formatNumber(Number(steps))} steps`);
  return parts.join(" · ");
}

function optionLabelForStep(step) {
  const stepIndex = stepIdOf(step);
  const parts = [stepIndex === undefined ? "Unidentified step" : `Step ${stepIndex}`];
  const turn = recordValue(step, "turn", "turn_index");
  const context = recordValue(step, "context", "phase");
  const candidates = recordValue(step, "candidate_count", "legal_option_count");
  if (turn !== undefined) parts.push(`turn ${turn}`);
  if (context !== undefined) parts.push(stringValue(context));
  if (candidates !== undefined) parts.push(`${formatNumber(Number(candidates))} options`);
  return parts.join(" · ");
}

function resetSelect(select, message, disabled = true) {
  const option = new Option(message, "");
  select.replaceChildren(option);
  select.value = "";
  select.disabled = disabled;
}

function fillSelect(select, records, idOf, labelOf, selectedId, placeholder) {
  const options = [new Option(placeholder, "")];
  for (const record of records) {
    const id = idOf(record);
    if (id === undefined || id === null || id === "") continue;
    options.push(new Option(labelOf(record), String(id)));
  }
  select.replaceChildren(...options);
  select.disabled = records.length === 0;
  const matching = records.some((record) => sameId(idOf(record), selectedId));
  select.value = matching ? String(selectedId) : "";
}

function normalisedSubmissionFilter(value = state.submissionFilter) {
  return stringValue(value).trim().toLocaleLowerCase();
}

function filteredSubmissions() {
  const filter = normalisedSubmissionFilter();
  if (!filter) return state.submissions;
  return state.submissions.filter((submission) => [
    submissionIdOf(submission),
    recordValue(submission, "label", "submission_text", "text"),
    optionLabelForSubmission(submission),
  ].some((value) => stringValue(value).toLocaleLowerCase().includes(filter)));
}

function clearPendingSubmissionFilterLoad() {
  if (state.submissionFilterLoadTimer !== null) clearTimeout(state.submissionFilterLoadTimer);
  state.submissionFilterLoadTimer = null;
}

function renderFilteredSubmissionSelect() {
  const matches = filteredSubmissions();
  const noMatches = state.submissions.length > 0 && matches.length === 0;
  fillSelect(
    elements.submissionSelect,
    matches,
    submissionIdOf,
    optionLabelForSubmission,
    state.submissionId,
    noMatches ? "No matching submissions" : state.submissions.length ? "Choose submission" : "No submissions indexed",
  );
  elements.submissionSearch.disabled = state.submissions.length === 0;
  elements.submissionFilterStatus.className = `selector-filter-status${noMatches ? " is-no-match" : ""}`;
  const filter = state.submissionFilter.trim();
  elements.submissionFilterStatus.textContent = !state.submissions.length
    ? "No submissions are indexed."
    : !filter
      ? `${state.submissions.length} submissions available. Type an ID or label to narrow the list.`
      : !matches.length
        ? `No submissions match “${filter}”.`
        : `${matches.length} of ${state.submissions.length} submissions match “${filter}”.`;
  return matches;
}

function resetSubmissionFilter(message = "Loading submissions…", disabled = true) {
  clearPendingSubmissionFilterLoad();
  state.submissionFilter = "";
  elements.submissionSearch.value = "";
  elements.submissionSearch.disabled = disabled;
  elements.submissionFilterStatus.className = "selector-filter-status";
  elements.submissionFilterStatus.textContent = message;
}

function applySubmissionFilter(value) {
  clearPendingSubmissionFilterLoad();
  state.submissionFilter = stringValue(value).trim();
  elements.submissionSearch.value = state.submissionFilter;
  const matches = renderFilteredSubmissionSelect();
  if (!matches.length || matches.some((item) => sameId(submissionIdOf(item), state.submissionId))) return;
  state.submissionFilterLoadTimer = window.setTimeout(() => {
    state.submissionFilterLoadTimer = null;
    const currentMatches = filteredSubmissions();
    const nextId = currentMatches.length ? submissionIdOf(currentMatches[0]) : undefined;
    if (nextId !== undefined) void selectSubmission(nextId);
  }, 250);
}

function normalisedGameFilter(value = state.gameFilter) {
  return stringValue(value).trim().toLocaleLowerCase();
}

function gameSearchTerms(game) {
  const directNames = [
    recordValue(game, "own_player", "own_player_name", "player_name"),
    recordValue(game, "opponent", "opponent_name", "opponent_submission"),
  ];
  const terms = [gameIdOf(game), ...gamePlayers(game).map((player) => player.name)];
  for (const value of directNames) {
    if (isObject(value)) terms.push(playerNameOf(value, ""));
    else terms.push(stringValue(value));
  }
  return terms.filter((term) => typeof term === "string" && term.trim() !== "");
}

function gameMatchesFilter(game, filter = normalisedGameFilter()) {
  if (!filter) return true;
  return gameSearchTerms(game).some((term) => term.toLocaleLowerCase().includes(filter));
}

function filteredGames() {
  const filter = normalisedGameFilter();
  return filter ? state.games.filter((game) => gameMatchesFilter(game, filter)) : state.games;
}

function gameFilterMessage(matches) {
  const filter = state.gameFilter.trim();
  if (!state.games.length) return "No games are available for this submission.";
  if (!filter) return `${state.games.length} game${state.games.length === 1 ? "" : "s"} available. Type an ID or player name to narrow the list.`;
  if (!matches.length) return `No games match “${filter}”. Clear the search to see every game.`;
  return `${matches.length} of ${state.games.length} game${state.games.length === 1 ? "" : "s"} match “${filter}”.`;
}

function renderFilteredGameSelect() {
  const matches = filteredGames();
  const noSourceGames = state.games.length === 0;
  const noMatches = !noSourceGames && matches.length === 0;
  fillSelect(
    elements.gameSelect,
    matches,
    gameIdOf,
    optionLabelForGame,
    state.gameId,
    noSourceGames ? "No games indexed" : noMatches ? "No matching games" : "Choose game",
  );
  elements.gameSearch.disabled = noSourceGames;
  elements.gameFilterStatus.className = `game-filter-status${noMatches ? " is-no-match" : ""}`;
  elements.gameFilterStatus.textContent = gameFilterMessage(matches);
  return matches;
}

function resetGameFilter(message = "Choose a submission to search its games.", disabled = true) {
  clearPendingGameFilterLoad();
  state.gameFilter = "";
  elements.gameSearch.value = "";
  elements.gameSearch.disabled = disabled;
  elements.gameFilterStatus.className = "game-filter-status";
  elements.gameFilterStatus.textContent = message;
}

function clearPendingGameFilterLoad() {
  if (state.gameFilterLoadTimer !== null) clearTimeout(state.gameFilterLoadTimer);
  state.gameFilterLoadTimer = null;
}

function clearGameForNoMatch() {
  clearPendingGameFilterLoad();
  const reason = gameFilterMessage([]);
  ++state.requests.game;
  ++state.requests.trace;
  state.gameId = "";
  state.steps = [];
  state.stepIndex = "";
  state.stage = "";
  resetSelect(elements.stepSelect, "No matching game selected", true);
  resetStepFilter("No matching game selected.", true);
  resetSelect(elements.stageSelect, "No matching game selected", true);
  clearTrace(reason);
  updateSelectionContext();
}

function scheduleFirstMatchingGameLoad() {
  clearPendingGameFilterLoad();
  state.gameFilterLoadTimer = window.setTimeout(() => {
    state.gameFilterLoadTimer = null;
    const matches = filteredGames();
    const selectedStillMatches = matches.some((game) => sameId(gameIdOf(game), state.gameId));
    if (selectedStillMatches || !matches.length) return;
    const nextId = gameIdOf(matches.find((game) => gameIdOf(game) !== undefined));
    if (nextId !== undefined) void selectGame(nextId);
  }, 250);
}

function applyGameFilter(value) {
  clearPendingGameFilterLoad();
  state.gameFilter = stringValue(value).trim();
  elements.gameSearch.value = state.gameFilter;
  const matches = renderFilteredGameSelect();
  if (!state.games.length) return;
  const selectedStillMatches = matches.some((game) => sameId(gameIdOf(game), state.gameId));
  if (selectedStillMatches) return;
  if (!matches.length) {
    clearGameForNoMatch();
    return;
  }
  scheduleFirstMatchingGameLoad();
}

function clearPendingStepFilterLoad() {
  if (state.stepFilterLoadTimer !== null) clearTimeout(state.stepFilterLoadTimer);
  state.stepFilterLoadTimer = null;
}

function filteredSteps() {
  const filter = state.stepFilter.trim();
  return filter
    ? state.steps.filter((step) => stringValue(stepIdOf(step)).includes(filter))
    : state.steps;
}

function renderFilteredStepSelect() {
  const matches = filteredSteps();
  const noMatches = state.steps.length > 0 && matches.length === 0;
  fillSelect(
    elements.stepSelect,
    matches,
    stepIdOf,
    optionLabelForStep,
    state.stepIndex,
    noMatches ? "No matching steps" : state.steps.length ? "Choose step" : "No decision steps indexed",
  );
  elements.stepSearch.disabled = state.steps.length === 0;
  elements.stepFilterStatus.className = `selector-filter-status${noMatches ? " is-no-match" : ""}`;
  const filter = state.stepFilter.trim();
  elements.stepFilterStatus.textContent = !state.steps.length
    ? "No decision steps are available."
    : !filter
      ? `${state.steps.length} recorded decision steps available. Type an exact step number to open it.`
      : !matches.length
        ? `No recorded decision step matches “${filter}”.`
        : `${matches.length} step${matches.length === 1 ? "" : "s"} match “${filter}”.`;
  return matches;
}

function resetStepFilter(message = "Choose a game to enter a step number.", disabled = true) {
  clearPendingStepFilterLoad();
  state.stepFilter = "";
  elements.stepSearch.value = "";
  elements.stepSearch.disabled = disabled;
  elements.stepFilterStatus.className = "selector-filter-status";
  elements.stepFilterStatus.textContent = message;
}

function applyStepFilter(value) {
  clearPendingStepFilterLoad();
  state.stepFilter = stringValue(value).replace(/\D/g, "");
  elements.stepSearch.value = state.stepFilter;
  renderFilteredStepSelect();
  if (!state.stepFilter) return;
  const exact = state.steps.find((step) => sameId(stepIdOf(step), state.stepFilter));
  if (!exact || sameId(stepIdOf(exact), state.stepIndex)) return;
  state.stepFilterLoadTimer = window.setTimeout(() => {
    state.stepFilterLoadTimer = null;
    const currentExact = state.steps.find((step) => sameId(stepIdOf(step), state.stepFilter));
    if (currentExact) void selectStep(stepIdOf(currentExact));
  }, 250);
}

function baseTraceKey(stepIndex, stage) {
  return [state.submissionId, state.gameId, stepIndex, stage].map(stringValue).join(":");
}

function baseTracePath(stepIndex, stage) {
  // Canonical selected-step suffix remains
  // `/steps/${encodeURIComponent(state.stepIndex)}?stage=${encodeURIComponent(state.stage)}`
  // while this helper also supports background warming of other addresses.
  return `${API}/submissions/${encodeURIComponent(state.submissionId)}/games/${encodeURIComponent(state.gameId)}/steps/${encodeURIComponent(stepIndex)}?stage=${encodeURIComponent(stage)}`;
}

function clearGameTraceCache() {
  for (const controller of state.traceAbortControllers.values()) controller.abort();
  state.traceAbortControllers.clear();
  state.traceCache.clear();
  state.traceFetches.clear();
}

function abortTraceFetchesExcept(selectedKey) {
  for (const [key, controller] of state.traceAbortControllers) {
    if (key !== selectedKey) controller.abort();
  }
}

function fetchBaseTraceCached(stepIndex, stage) {
  const key = baseTraceKey(stepIndex, stage);
  if (state.traceCache.has(key)) return Promise.resolve(state.traceCache.get(key));
  if (state.traceFetches.has(key)) return state.traceFetches.get(key);
  const controller = new AbortController();
  let timedOut = false;
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, TRACE_REQUEST_TIMEOUT_MS);
  state.traceAbortControllers.set(key, controller);
  const request = fetchJson(baseTracePath(stepIndex, stage), { signal: controller.signal })
    .then((payload) => {
      state.traceCache.set(key, payload);
      return payload;
    })
    .catch((error) => {
      if (timedOut) {
        throw new Error("Selected trace reconstruction exceeded 20 seconds on Elmo's GPU. Choose the step again to retry.");
      }
      throw error;
    })
    .finally(() => {
      window.clearTimeout(timeout);
      state.traceFetches.delete(key);
      state.traceAbortControllers.delete(key);
    });
  state.traceFetches.set(key, request);
  return request;
}

function stagesForStep(step) {
  if (!step || typeof step !== "object") return [];
  const raw = firstDefined(step.factorized_stages, step.stages, step.factorized_stage, step.stage);
  const source = Array.isArray(raw) ? raw : raw === undefined || raw === null || raw === "" ? [] : [raw];
  const values = [];
  for (const item of source) {
    const value = isObject(item) ? recordValue(item, "stage", "index", "id", "value") : item;
    if (value === undefined || value === null || value === "") continue;
    if (!values.some((existing) => sameId(existing, value))) values.push(value);
  }
  return values;
}

function populateStages(step, preserve = "") {
  const stages = stagesForStep(step);
  if (!stages.length) {
    resetSelect(elements.stageSelect, "Stage not stated by replay", true);
    state.stage = "";
    return;
  }
  const options = [new Option("Choose stage", "")];
  for (const stage of stages) options.push(new Option(`Stage ${stage}`, String(stage)));
  elements.stageSelect.replaceChildren(...options);
  elements.stageSelect.disabled = false;
  const next = stages.some((stage) => sameId(stage, preserve)) ? preserve : stages[0];
  state.stage = String(next);
  elements.stageSelect.value = state.stage;
}

function updateSelectionContext() {
  const bits = [];
  const submission = currentSubmission();
  const game = currentGame();
  const step = currentStep();
  if (submission) bits.push(optionLabelForSubmission(submission));
  if (game) bits.push(optionLabelForGame(game));
  if (step) bits.push(`step ${stepIdOf(step)}`);
  if (state.stage !== "") bits.push(`stage ${state.stage}`);
  elements.selectionContext.textContent = bits.length
    ? bits.join("  /  ")
    : "The inspector will only show evidence returned by the selected replay and checkpoint.";
  renderSelectionIdentity(submission, game);
}

function identityCard(kicker, primary, metadata = []) {
  const card = document.createElement("article");
  card.className = "selection-identity-card";
  const label = document.createElement("p");
  label.className = "selection-identity-kicker";
  label.textContent = kicker;
  const value = document.createElement("p");
  value.className = "selection-identity-value";
  value.textContent = primary;
  card.append(label, value);
  for (const item of metadata) {
    if (!item) continue;
    const detail = document.createElement("p");
    detail.className = "selection-identity-detail";
    detail.textContent = item;
    card.append(detail);
  }
  return card;
}

function renderSelectionIdentity(submission, game) {
  clearNode(elements.selectionIdentity);
  if (!submission) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Choose a submission to see its exact label and the replay players.";
    elements.selectionIdentity.append(empty);
    return;
  }

  const identity = submissionIdentityOf(submission);
  const readiness = submissionReadiness(submission);
  const outcome = cachedReplayOutcome(submission);
  const submissionDetails = [];
  if (identity.id !== undefined) {
    submissionDetails.push(`Exact submission ID: ${identity.idText}`);
  }
  if (identity.exactText && identity.label && identity.exactText !== identity.label) {
    submissionDetails.push(`Submission label: ${identity.label}`);
  }
  submissionDetails.push(`Model weights: ${readiness.weightsText}`);
  submissionDetails.push(`Decision trace: ${readiness.traceText}`);
  submissionDetails.push(outcome.available
    ? `Cached replay wins: ${formatNumber(outcome.wins)}/${formatNumber(outcome.denominator)} (${(outcome.rate * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%)`
    : "Cached replay win rate unavailable");
  elements.selectionIdentity.append(identityCard(
    "Selected submission",
    `${identity.idText} · ${identity.primary}`,
    submissionDetails,
  ));

  if (!game) return;
  const players = gamePlayers(game, submission);
  const playerDetails = [
    `Submission ID: ${identity.idText}`,
    ...players.map((player) => `${player.isOwn ? "This submission" : "Opponent"}: ${playerSummary(player)}`),
  ];
  const gameId = gameIdOf(game);
  const gameCard = identityCard(
    gameId === undefined ? "Selected game" : `Game ${formatValue(gameId)}`,
    players.length ? players.map((player) => player.name).join(" vs ") : "Player details unavailable",
    playerDetails,
  );
  const visualizerUrl = ptcgVisualizerReplayUrl(game);
  if (visualizerUrl) {
    const visualizerLink = document.createElement("a");
    visualizerLink.className = "external-replay-link";
    visualizerLink.href = visualizerUrl;
    visualizerLink.target = "_blank";
    visualizerLink.rel = "noopener noreferrer";
    visualizerLink.referrerPolicy = "no-referrer";
    visualizerLink.textContent = "Open replay in PTCG Visualizer ↗";
    visualizerLink.setAttribute("aria-label", `Open game ${gameId} in PTCG Visualizer (new tab)`);
    gameCard.append(visualizerLink);
  }
  elements.selectionIdentity.append(gameCard);
}

function renderHealth() {
  if (state.healthError) {
    setBadge(elements.healthBadge, "Service unavailable", "error");
    return;
  }
  if (!state.health) {
    setBadge(elements.healthBadge, "Checking service", "pending");
    return;
  }
  if (isUnavailable(state.health)) {
    setBadge(elements.healthBadge, "Service reports unavailable", "unavailable");
    return;
  }
  const status = recordValue(state.health, "status", "state");
  setBadge(elements.healthBadge, status ? `Service: ${status}` : "Service reachable", "ready");
}

function clearTrace(reason = "Choose a replay decision to load its causal re-evaluation.") {
  resetDecisionInfluenceState();
  state.trace = null;
  state.traceError = null;
  renderTrace(reason);
}

function clearParameterDetail(reason = "Choose a parameter from the exact checkpoint inventory. Full tensors are intentionally not exposed.") {
  state.parameterName = "";
  state.parameterDetail = null;
  state.parameterError = null;
  state.parameterOffset = 0;
  setAvailability(elements.parameterDetailStatus, null, "Available", "No parameter selected");
  elements.parameterDetail.className = "parameter-detail empty-state";
  elements.parameterDetail.textContent = reason;
}

function submitMetaSource() {
  const submission = currentSubmission();
  if (!submission) return null;
  return {
    available: recordValue(submission, "model_analysis")?.available,
    reason: recordValue(submission, "model_analysis")?.reason,
  };
}

function normalizedRecordFieldName(key) {
  return String(key).replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

function isSubmissionIdField(key) {
  const normalized = normalizedRecordFieldName(key);
  return normalized === "submission_id" || normalized === "submission_id_text";
}

function hasExactSubmissionIdText(source) {
  return strictDecimalIdentifier(recordValue(source, "submission_id_text", "submissionIdText")) !== undefined;
}

function primitiveEntries(source) {
  if (!isObject(source)) return [];
  return Object.entries(source).filter(([key, value]) => {
    // Public envelopes retain the numeric compatibility field, but the exact
    // decimal text is the only display-safe submission identity. Keep one
    // canonical row when both are present; the unmodified source remains in
    // the expandable technical payload.
    if (normalizedRecordFieldName(key) === "submission_id" && hasExactSubmissionIdText(source)) {
      return false;
    }
    return value === null || typeof value !== "object";
  });
}

function recordFieldLabel(key) {
  return isSubmissionIdField(key) ? "submission id" : String(key).replaceAll("_", " ");
}

function formatRecordFieldValue(source, key, value) {
  if (!isSubmissionIdField(key)) return formatValue(value);
  const exact = submissionIdOf(source) || strictDecimalIdentifier(value);
  return exact === undefined ? "Exact submission ID unavailable" : exact;
}

function keyValueGrid(source, { includeEmpty = false } = {}) {
  const entries = primitiveEntries(source).filter(([, value]) => includeEmpty || value !== undefined);
  if (!entries.length) return null;
  const list = document.createElement("dl");
  list.className = "key-value-grid";
  for (const [key, value] of entries) {
    const item = document.createElement("div");
    item.className = "key-value";
    const name = document.createElement("dt");
    name.textContent = recordFieldLabel(key);
    const detail = document.createElement("dd");
    detail.textContent = formatRecordFieldValue(source, key, value);
    item.append(name, detail);
    list.append(item);
  }
  return list;
}

function detailsBlock(summaryText, payload, open = false) {
  const details = document.createElement("details");
  details.className = "structured-details";
  details.open = open;
  const summary = document.createElement("summary");
  summary.textContent = summaryText;
  const pre = document.createElement("pre");
  pre.className = "json-block";
  pre.textContent = fullJson(payload);
  details.append(summary, pre);
  return details;
}

function renderStructuredContent(container, payload, emptyText) {
  clearNode(container);
  if (payload === undefined || payload === null) {
    container.className = "structured-content empty-state";
    container.textContent = emptyText;
    return;
  }
  if (isUnavailable(payload)) {
    container.className = "structured-content empty-state";
    container.textContent = reasonFrom(payload, "Source marked this data unavailable.");
    return;
  }
  container.className = "structured-content";
  if (typeof payload !== "object") {
    const text = document.createElement("p");
    text.textContent = formatValue(payload);
    container.append(text);
    return;
  }
  const grid = keyValueGrid(payload);
  if (grid) container.append(grid);
  container.append(detailsBlock("View complete source payload", payload));
}

function appendSourceObject(container, title, source) {
  if (source === undefined || source === null) return false;
  const pane = document.createElement("section");
  pane.className = "fusion-pane";
  const heading = document.createElement("h4");
  heading.textContent = title;
  pane.append(heading);
  if (isUnavailable(source)) {
    const message = document.createElement("p");
    message.className = "empty-state";
    message.textContent = reasonFrom(source, "Source marked this data unavailable.");
    pane.append(message);
  } else if (typeof source === "object") {
    const grid = keyValueGrid(source);
    if (grid) pane.append(grid);
    pane.append(detailsBlock("View source payload", source));
  } else {
    const message = document.createElement("p");
    message.textContent = formatValue(source);
    pane.append(message);
  }
  container.append(pane);
  return true;
}

function actionSummary(action) {
  if (action === undefined || action === null) return "—";
  if (typeof action !== "object") return formatValue(action);
  const meaningful = firstDefined(action.label, action.summary, action.action, action.name, action.index, action.action_index);
  return meaningful === undefined ? compactJson(action) : formatValue(meaningful);
}

function optionForModelChoice(trace) {
  const selectedIndex = recordValue(trace?.model, "selected_index", "selected_action_index");
  const options = collectionFrom(trace, "legal_options");
  return options.find((option) => sameId(recordValue(option, "index", "action_index"), selectedIndex));
}

function sameActionPayload(left, right) {
  if (left === right) return true;
  try {
    return JSON.stringify(left) === JSON.stringify(right);
  } catch {
    return false;
  }
}

function optionForRecordedAction(trace) {
  const options = collectionFrom(trace, "legal_options");
  const marked = options.find((option) => option?.is_recorded === true);
  if (marked) return marked;
  const recorded = trace?.recorded_action;
  return options.find((option) => sameActionPayload(recordValue(option, "action", "raw_action", "candidate"), recorded));
}

function maskDescription(head) {
  if (!head || typeof head !== "object") return "Not supplied";
  if (head.available === false) return reasonFrom(head, "Source marked this head unavailable.");
  const mask = firstDefined(head.mask, head.output_mask, head.valid_mask, head.availability_mask);
  if (mask !== undefined) return formatValue(mask);
  return "Mask not supplied";
}

function routeContributionOf(option) {
  return firstDefined(
    option?.route_contributions,
    option?.per_route_contributions,
    option?.fusion_contribution,
    option?.route_delta,
    option?.contribution,
  );
}

function hasOptionRouteContributions(options) {
  return options.some((option) => routeContributionOf(option) !== undefined);
}

function renderTrace(emptyReason = "Choose a replay decision to load its causal re-evaluation.") {
  const trace = state.trace;
  if (state.traceError) {
    const reason = plainAvailabilityReason(state.traceError.message, "Trace request failed.");
    setBadge(elements.traceStatus, "Trace request failed", "error");
    elements.provenanceAvailability.textContent = reason;
    clearNode(elements.provenanceContent);
    elements.provenanceContent.className = "provenance-content empty-state";
    elements.provenanceContent.textContent = reason;
    renderReproductionStatus(null);
    renderDecisionUnavailable(reason);
    return;
  }
  if (!trace) {
    setBadge(elements.traceStatus, "Awaiting selection", "pending");
    elements.provenanceAvailability.textContent = "No trace selected";
    clearNode(elements.provenanceContent);
    elements.provenanceContent.className = "provenance-content empty-state";
    elements.provenanceContent.textContent = emptyReason;
    renderReproductionStatus(null);
    renderDecisionUnavailable(emptyReason);
    return;
  }
  if (isUnavailable(trace)) {
    const reason = plainAvailabilityReason(trace, "The requested replay trace is unavailable.");
    setBadge(elements.traceStatus, "Trace unavailable", "unavailable");
    elements.provenanceAvailability.textContent = reason;
    clearNode(elements.provenanceContent);
    elements.provenanceContent.className = "provenance-content empty-state";
    elements.provenanceContent.textContent = reason;
    renderReproductionStatus(trace);
    renderDecisionUnavailable(reason);
    return;
  }
  const reproductionStatus = reproductionStatusOf(trace);
  if (reproductionStatus === "diverged_or_fallback_unknown") {
    setBadge(elements.traceStatus, "Re-evaluation diverged", "unavailable");
  } else if (reproductionStatus === "unavailable") {
    setBadge(elements.traceStatus, "Re-evaluation unavailable", "unavailable");
  } else if (
    reproductionStatus === "exact_reproduced" ||
    reproductionStatus === "exact_runtime_short_circuit"
  ) {
    setBadge(elements.traceStatus, "Exact re-evaluation", "ready");
  } else if (reproductionStatus === "hypothetical_model_forward_not_submitted_runtime") {
    setBadge(elements.traceStatus, "Hypothetical model rerun", "ready");
  } else {
    setBadge(elements.traceStatus, "Re-evaluation loaded", "ready");
  }
  renderReproductionStatus(trace);
  renderProvenance(trace);
  renderDecisionOverview(trace);
  renderMatchupAdapter(trace);
  renderGuideShadow(trace);
  renderObservation(trace);
  renderOptions(trace);
  renderFusion(trace);
  renderHeads(trace);
  renderDecisionInfluenceAwaiting(trace);
  renderTrainingRecipeAwaiting(trace);
  renderWarnings(trace);
}

function renderDecisionUnavailable(reason) {
  const items = [
    [elements.recordedAction, elements.recordedActionNote, "—", reason],
    [elements.modelAction, elements.modelActionNote, "—", reason],
    [elements.actionAgreement, elements.actionAgreementNote, "—", reason],
    [elements.modelValue, elements.modelRouteNote, "—", reason],
  ];
  for (const [value, note, label, detail] of items) {
    value.classList.add("muted-value");
    setText(value, label);
    setText(note, detail);
  }
  elements.matchupAdapterAvailability.textContent = reason;
  elements.matchupAdapterStatus.className = "matchup-adapter-status is-unavailable";
  elements.matchupAdapterStatus.textContent = `Unavailable — ${reason}`;
  renderStructuredContent(elements.matchupAdapterContent, null, reason);
  elements.guideShadowAvailability.textContent = `Unavailable — ${reason}`;
  renderStructuredContent(elements.guideShadowContent, null, reason);
  state.guideShadowEnabled = false;
  elements.guideShadowToggle.disabled = true;
  elements.guideShadowToggle.checked = false;
  elements.guideShadowToggleState.textContent = "Guide comparison unavailable";
  elements.observationAvailability.textContent = reason;
  renderStructuredContent(elements.observationContent, null, reason);
  elements.optionsAvailability.textContent = reason;
  setEmptyRow(elements.optionsBody, 7, reason);
  elements.optionContributionDetail.textContent = reason;
  elements.fusionAvailability.textContent = reason;
  renderStructuredContent(elements.fusionContent, null, reason);
  elements.headsAvailability.textContent = reason;
  elements.headImpactSummary.textContent = reason;
  clearNode(elements.headImpactGrid);
  const headImpactEmpty = document.createElement("p");
  headImpactEmpty.className = "empty-state";
  headImpactEmpty.textContent = reason;
  elements.headImpactGrid.append(headImpactEmpty);
  setEmptyRow(elements.headsBody, 9, reason);
  elements.headsNote.textContent = reason;
  renderHeadFaq();
  renderDecisionInfluenceUnavailable(reason);
  renderTrainingRecipeUnavailable(reason);
  elements.warningsSection.hidden = true;
  clearNode(elements.warningsList);
}

function renderProvenance(trace) {
  const address = traceAddressForDisplay(trace.address);
  const provenance = trace.provenance;
  const traceAvailable = !isUnavailable(trace);
  elements.provenanceAvailability.textContent = traceAvailable
    ? "Replay record and reconstruction evidence returned by source"
    : reasonFrom(trace, "Trace unavailable.");
  clearNode(elements.provenanceContent);
  elements.provenanceContent.className = "provenance-content";
  const gridSource = isObject(address) ? address : provenance;
  const grid = keyValueGrid(gridSource);
  if (grid) elements.provenanceContent.append(grid);
  const panes = document.createElement("div");
  panes.className = "fusion-grid";
  const hasAddress = appendSourceObject(panes, "Trace address", address);
  const hasProvenance = appendSourceObject(panes, "Bundle & checkpoint provenance", provenance);
  if (hasAddress || hasProvenance) elements.provenanceContent.append(panes);
  if (!hasAddress && !hasProvenance) {
    const message = document.createElement("p");
    message.className = "empty-state";
    message.textContent = "The source did not include trace address or provenance fields.";
    elements.provenanceContent.append(message);
  }
}

function traceAddressForDisplay(address) {
  if (!isObject(address)) return address;
  const result = { ...address };
  const submissionId = submissionIdOf(address);
  if (submissionId !== undefined) {
    result.submission_id_text = submissionId;
    result.submission_id = submissionId;
  }
  const episodeId = gameIdOf(address);
  if (episodeId !== undefined) {
    result.episode_id_text = episodeId;
    result.episode_id = episodeId;
  }
  const stepId = stepIdOf(address);
  if (stepId !== undefined) {
    result.step_index_text = stepId;
    result.step_index = stepId;
  }
  return result;
}

function renderDecisionOverview(trace) {
  const model = isObject(trace.model) ? trace.model : {};
  const recorded = trace.recorded_action;
  const recordedOption = optionForRecordedAction(trace);
  const modelOption = optionForModelChoice(trace);
  const selectedIndex = recordValue(model, "selected_index", "selected_action_index");
  const recordedTranscript = firstDefined(
    recordedOption ? friendlyActionTranscript(recordedOption) : undefined,
    isObject(recorded)
      ? firstActionTranscript(recordValue(recorded, "selected_action_transcript", "action_transcript", "transcript"))
      : undefined,
    isObject(recorded) ? friendlyActionTranscript(recorded) : undefined,
    typeof recorded === "string" ? recorded : undefined,
  );
  const modelTranscript = firstDefined(
    modelOption ? friendlyActionTranscript(modelOption) : undefined,
    firstActionTranscript(recordValue(model, "selected_action_transcript", "action_transcript", "transcript")),
    isObject(model.selected_action) ? friendlyActionTranscript(model.selected_action) : undefined,
    typeof model.selected_action === "string" ? model.selected_action : undefined,
    isObject(model.action) ? friendlyActionTranscript(model.action) : undefined,
    typeof model.action === "string" ? model.action : undefined,
  );
  const agreement = recordValue(model, "agreement", "recorded_action_agreement", "matches_recorded");
  const value = recordValue(model, "value", "state_value", "value_prediction");
  const route = recordValue(model, "route", "decision_route", "route_name");

  renderOverviewAction(
    elements.recordedAction,
    recordedTranscript || "Plain-English recorded action transcript not supplied",
    "Technical recorded action",
    {
      replay_recorded_action: recorded,
      matching_legal_option: recordedOption ? actionTechnicalPayload(recordedOption) : undefined,
    },
    !recordedTranscript,
  );
  elements.recordedActionNote.textContent = recorded === undefined || recorded === null
    ? "Recorded action was not supplied by this replay trace."
    : recordedTranscript
      ? "Plain-English action transcript supplied; the exact replay action remains in technical details."
      : "The source did not supply a plain-English transcript. The exact replay action remains in technical details.";

  renderOverviewAction(
    elements.modelAction,
    modelTranscript || "Plain-English model action transcript not supplied",
    "Technical model action",
    {
      selected_legal_option_index: selectedIndex,
      model_selected_action: model.selected_action,
      matching_legal_option: modelOption ? actionTechnicalPayload(modelOption) : undefined,
    },
    !modelTranscript,
  );
  elements.modelActionNote.textContent = selectedIndex === undefined
    ? "Re-evaluated model selected index was not supplied by this trace."
    : modelTranscript
      ? "Plain-English selected action supplied; its exact legal-option index remains in technical details."
      : "The source did not supply a plain-English transcript; its exact selected index remains in technical details.";

  let agreementText = "—";
  if (agreement === true) agreementText = "Match";
  else if (agreement === false) agreementText = "Different";
  else if (agreement !== undefined && agreement !== null) agreementText = formatValue(agreement);
  elements.actionAgreement.classList.toggle("muted-value", agreementText === "—");
  elements.actionAgreement.textContent = agreementText;
  elements.actionAgreementNote.textContent = agreement === undefined || agreement === null
    ? "The source did not provide an agreement result."
    : "Recorded and model actions compared by the source trace.";

  elements.modelValue.classList.toggle("muted-value", value === undefined || value === null);
  elements.modelValue.textContent = formatValue(value);
  elements.modelRouteNote.textContent = route === undefined || route === null
    ? "Decision route not supplied by the trace."
    : `Route: ${formatValue(route)}`;
}

function renderOverviewAction(element, headline, detailsSummary, payload, muted = false) {
  clearNode(element);
  element.classList.toggle("muted-value", muted);
  const text = document.createElement("span");
  text.className = "overview-action-headline";
  text.textContent = headline;
  element.append(text);
  const technical = Object.fromEntries(
    Object.entries(payload).filter(([, value]) => value !== undefined && value !== null),
  );
  if (Object.keys(technical).length) element.append(detailsBlock(detailsSummary, technical));
}

function matchupAdapterSource(trace) {
  const model = isObject(trace?.model) ? trace.model : {};
  return firstDefined(
    trace?.matchup_adapter_status,
    model.matchup_adapter_status,
    model.adapter_status,
    model.adapter,
    trace?.adapter,
  );
}

function matchupAdapterState(source) {
  if (source === undefined || source === null) {
    return { kind: "unavailable", reason: "The trace did not include a Matchup Adapter decision status." };
  }
  if (isUnavailable(source)) {
    return { kind: "unavailable", reason: reasonFrom(source, "The source marked Matchup Adapter status unavailable.") };
  }
  if (!isObject(source)) {
    const status = stringValue(source).trim().toLowerCase();
    if (["active", "active_for_decision", "applied"].includes(status)) return { kind: "active", reason: "" };
    if (["bypassed", "bypass", "inactive", "disabled"].includes(status)) return { kind: "bypassed", reason: "The source marked the adapter bypassed." };
    return { kind: "unavailable", reason: "The source did not provide a usable Matchup Adapter decision status." };
  }
  const status = stringValue(recordValue(source, "status", "decision_status", "state")).trim().toLowerCase();
  const active = recordValue(source, "active_for_decision", "active", "decision_active", "runtime_active", "applied");
  const bypassed = recordValue(source, "bypassed", "is_bypassed", "decision_bypassed");
  if (active === true || ["active", "active_for_decision", "applied", "routed"].includes(status)) {
    return { kind: "active", reason: "" };
  }
  if (bypassed === true || recordValue(source, "enabled") === false || ["bypassed", "bypass", "inactive", "disabled", "not_active"].includes(status)) {
    return {
      kind: "bypassed",
      reason: reasonFrom(source, "The adapter was not active for this decision."),
    };
  }
  if (["unavailable", "not_available", "unknown"].includes(status)) {
    return {
      kind: "unavailable",
      reason: reasonFrom(source, "The source marked Matchup Adapter status unavailable."),
    };
  }
  return {
    kind: "unavailable",
    reason: "Adapter metadata may be present, but activation for this decision was not stated.",
  };
}

function renderMatchupAdapter(trace) {
  const source = matchupAdapterSource(trace);
  const status = matchupAdapterState(source);
  const label = status.kind === "active"
    ? "Active for this decision"
    : status.kind === "bypassed"
      ? `Bypassed — ${status.reason}`
      : `Unavailable — ${status.reason}`;
  elements.matchupAdapterStatus.className = `matchup-adapter-status is-${status.kind}`;
  elements.matchupAdapterStatus.textContent = label;
  elements.matchupAdapterAvailability.textContent = label;
  clearNode(elements.matchupAdapterContent);

  if (status.kind !== "active") {
    elements.matchupAdapterContent.className = "structured-content empty-state";
    elements.matchupAdapterContent.textContent = status.kind === "bypassed"
      ? "The adapter was deliberately bypassed for this decision. There is no truthful force-on comparison because the router did not select an applied matchup route. Installed adapter weights alone do not mean the adapter was active."
      : "The trace does not establish that the adapter affected this decision, so an Adapter ON result cannot be invented. Installed adapter weights alone do not mean the adapter was active.";
    if (isObject(source)) elements.matchupAdapterContent.append(detailsBlock("View exact Matchup Adapter source payload", source));
    return;
  }

  elements.matchupAdapterContent.className = "structured-content";
  const policyShift = recordValue(source, "policy_shift", "policy_influence", "policy_delta", "logit_shift");
  const policyShiftSummary = matchupAdapterPolicyShiftSummary(policyShift);
  const reliabilitySummary = matchupAdapterReliabilitySummary(
    recordValue(source, "reliability", "route_reliability", "confidence"),
  );
  const facts = {
    matched_archetype: recordValue(source, "matched_archetype", "archetype", "archetype_name"),
    route: recordValue(source, "route", "matched_route", "decision_route"),
    slot: recordValue(source, "slot", "matched_slot", "adapter_slot"),
    reliability: reliabilitySummary,
    policy_shift: policyShiftSummary,
  };
  const factGrid = document.createElement("dl");
  factGrid.className = "key-value-grid matchup-adapter-facts";
  const labels = {
    matched_archetype: "Matched archetype",
    route: "Route",
    slot: "Slot",
    reliability: "Reliability",
    policy_shift: "Policy shift",
  };
  for (const [key, value] of Object.entries(facts)) {
    const item = document.createElement("div");
    item.className = "key-value";
    const name = document.createElement("dt");
    name.textContent = labels[key];
    const detail = document.createElement("dd");
    if (value === undefined || value === null) {
      detail.textContent = "Not supplied";
    } else {
      detail.textContent = formatValue(value);
    }
    item.append(name, detail);
    factGrid.append(item);
  }
  elements.matchupAdapterContent.append(factGrid);
  const plainEnglish = firstNonEmptyText(recordValue(source, "plain_english", "plain_english_interpretation"));
  if (plainEnglish) {
    const note = document.createElement("p");
    note.className = "matchup-adapter-note";
    note.textContent = plainEnglish;
    elements.matchupAdapterContent.append(note);
  }
  renderMatchupAdapterComparison(
    elements.matchupAdapterContent,
    recordValue(source, "on_off_comparison", "adapter_on_off_comparison"),
  );
  if (isObject(source)) elements.matchupAdapterContent.append(detailsBlock("View exact Matchup Adapter source payload", source));
}

function renderGuideShadow(trace) {
  const shadow = trace?.guide_shadow;
  const submittedPolicy = trace?.model?.policy?.submitted_runtime_policy;
  clearNode(elements.guideShadowContent);
  if (isObject(submittedPolicy)
      && !isUnavailable(submittedPolicy)
      && submittedPolicy.policy_authority === true) {
    const options = collectionFrom(trace, "legal_options");
    const audit = isObject(submittedPolicy.audit) ? submittedPolicy.audit : {};
    const neuralProbabilities = Array.isArray(trace?.model?.policy?.neural_probabilities)
      ? trace.model.policy.neural_probabilities
      : [];
    const guideOnProbabilities = options.map((option) => finiteNumber(option.probability));
    const neuralIndex = finiteNumber(submittedPolicy.neural_selected_index);
    const selectedIndex = finiteNumber(submittedPolicy.selected_index);
    const neuralAction = neuralIndex === undefined
      ? "Neural-only action unavailable"
      : actionNameAtPosition(options, optionPositionForIndex(options, neuralIndex));
    const submittedAction = selectedIndex === undefined
      ? "Submitted action unavailable"
      : actionNameAtPosition(options, optionPositionForIndex(options, selectedIndex));
    const weight = finiteNumber(audit.guide_logit_weight);
    elements.guideShadowToggle.disabled = false;
    elements.guideShadowToggle.checked = state.guideShadowEnabled;
    elements.guideShadowToggleState.textContent = state.guideShadowEnabled
      ? "Guide ON · exact submitted-runtime policy"
      : "Guide OFF · neural-only comparison";
    elements.guideShadowAvailability.textContent = submittedPolicy.applied === true
      ? `Production guide active · ${weight === undefined ? "weight unavailable" : `weight ${formatNumber(weight)}`}`
      : "Production guide checked · exact neural fallback";
    elements.guideShadowContent.className = "structured-content";
    const headline = document.createElement("p");
    headline.className = "guide-shadow-answer";
    headline.textContent = state.guideShadowEnabled
      ? `Guide ON chose: ${submittedAction}`
      : `Guide OFF would choose: ${neuralAction}`;
    const comparison = document.createElement("p");
    comparison.className = `guide-shadow-comparison ${submittedPolicy.changed_neural_choice === true ? "is-different" : "is-same"}`;
    comparison.textContent = submittedPolicy.changed_neural_choice === true
      ? `The packaged guide changed the neural choice from ${neuralAction} to ${submittedAction}.`
      : `The packaged guide kept the neural choice: ${submittedAction}.`;
    const facts = document.createElement("dl");
    facts.className = "key-value-grid guide-shadow-facts";
    for (const [label, value] of [
      ["Production authority", "Yes — exact package-local decision layer"],
      ["Guide logit weight", weight === undefined ? "Unavailable" : formatNumber(weight)],
      ["Neural-only would do", neuralAction],
      ["Submitted runtime did", submittedAction],
      ["Changed the choice?", submittedPolicy.changed_neural_choice === true ? "Yes" : "No"],
    ]) {
      const item = document.createElement("div");
      item.className = "key-value";
      const term = document.createElement("dt");
      term.textContent = label;
      const detail = document.createElement("dd");
      detail.textContent = value;
      item.append(term, detail);
      facts.append(item);
    }
    const scroller = document.createElement("div");
    scroller.className = "table-scroll";
    const table = document.createElement("table");
    table.className = "data-table";
    table.innerHTML = "<thead><tr><th>Legal action</th><th>Guide OFF</th><th>Guide ON</th><th>Change</th></tr></thead>";
    const body = document.createElement("tbody");
    options.forEach((option, position) => {
      const off = finiteNumber(neuralProbabilities[position]);
      const on = guideOnProbabilities[position];
      const row = document.createElement("tr");
      const action = document.createElement("td");
      action.textContent = actionNameAtPosition(options, position);
      const offCell = document.createElement("td");
      offCell.textContent = formatProbability(off);
      const onCell = document.createElement("td");
      onCell.textContent = formatProbability(on);
      const delta = document.createElement("td");
      delta.textContent = off === undefined || on === undefined
        ? "Unavailable"
        : signedPercentagePoints(on - off);
      row.append(action, offCell, onCell, delta);
      body.append(row);
    });
    table.append(body);
    scroller.append(table);
    const note = document.createElement("p");
    note.className = "guide-shadow-note";
    note.textContent = "Guide ON is the exact recomputed submitted policy. Guide OFF is the preserved neural-only policy from the same forward pass. Neither is recorded historical telemetry.";
    elements.guideShadowContent.append(
      headline,
      comparison,
      facts,
      scroller,
      note,
      detailsBlock("View exact package guide evidence", submittedPolicy),
    );
    return;
  }
  if (!isObject(shadow) || isUnavailable(shadow)) {
    const reason = reasonFrom(shadow, "No matching exact-runtime deck guide produced a safe ranking for this decision.");
    elements.guideShadowAvailability.textContent = `Unavailable — ${reason}`;
    elements.guideShadowContent.className = "structured-content empty-state";
    elements.guideShadowContent.textContent = reason;
    state.guideShadowEnabled = false;
    elements.guideShadowToggle.disabled = true;
    elements.guideShadowToggle.checked = false;
    elements.guideShadowToggleState.textContent = "Guide comparison unavailable";
    if (isObject(shadow)) elements.guideShadowContent.append(detailsBlock("View guide-shadow evidence", shadow));
    return;
  }
  if (shadow.policy_authority !== false || finiteNumber(shadow.policy_logit_delta) !== 0) {
    const reason = "Guide payload did not prove zero policy authority and an exact zero logit delta.";
    elements.guideShadowAvailability.textContent = `Unavailable — ${reason}`;
    elements.guideShadowContent.className = "structured-content empty-state";
    elements.guideShadowContent.textContent = reason;
    state.guideShadowEnabled = false;
    elements.guideShadowToggle.disabled = true;
    elements.guideShadowToggle.checked = false;
    elements.guideShadowToggleState.textContent = "Guide comparison unavailable";
    return;
  }
  elements.guideShadowToggle.disabled = false;
  elements.guideShadowToggle.checked = state.guideShadowEnabled;
  elements.guideShadowToggleState.textContent = state.guideShadowEnabled
    ? "Guide comparison on · model policy unchanged"
    : "Guide comparison off · exact model policy shown";
  if (!state.guideShadowEnabled) {
    elements.guideShadowAvailability.textContent = "Calculated · comparison off";
    elements.guideShadowContent.className = "structured-content empty-state";
    elements.guideShadowContent.textContent = "Turn on “Show guide recommendation” to compare the guide’s preferred action with the model. The model probabilities stay unchanged in both positions.";
    return;
  }
  const options = collectionFrom(trace, "legal_options");
  const recommendedIndex = finiteNumber(recordValue(shadow, "recommended_index"));
  const recommendedAction = recommendedIndex === undefined
    ? "Recommendation unavailable"
    : actionNameAtPosition(options, optionPositionForIndex(options, recommendedIndex));
  const modelIndex = finiteNumber(recordValue(trace?.model, "selected_index", "selected_action_index"));
  const modelAction = modelIndex === undefined
    ? "Model action unavailable"
    : actionNameAtPosition(options, optionPositionForIndex(options, modelIndex));
  elements.guideShadowAvailability.textContent = "Calculated · zero policy authority";
  elements.guideShadowContent.className = "structured-content";
  const headline = document.createElement("p");
  headline.className = "guide-shadow-answer";
  headline.textContent = `Guide recommendation: ${recommendedAction}`;
  const comparison = document.createElement("p");
  comparison.className = `guide-shadow-comparison ${shadow.agrees_with_model_action === true ? "is-same" : "is-different"}`;
  comparison.textContent = shadow.agrees_with_model_action === true
    ? `If the guide were allowed to choose, it would also choose ${recommendedAction}.`
    : shadow.agrees_with_model_action === false
      ? `If the guide were allowed to choose, it would replace the model’s ${modelAction} with ${recommendedAction}.`
      : "Choice comparison unavailable.";
  const facts = document.createElement("dl");
  facts.className = "key-value-grid guide-shadow-facts";
  const values = [
    ["Guide", firstNonEmptyText(shadow.guide_id) || "Unavailable"],
    ["Guide version", firstNonEmptyText(shadow.guide_version) || "Unavailable"],
    ["Agrees with model?", shadow.agrees_with_model_action === true ? "Yes" : shadow.agrees_with_model_action === false ? "No" : "Unavailable"],
    ["Agrees with replay action?", shadow.agrees_with_recorded_action === true ? "Yes" : shadow.agrees_with_recorded_action === false ? "No" : "Unavailable"],
    ["Guide score margin", finiteNumber(shadow.score_margin) === undefined ? "Unavailable" : formatNumber(shadow.score_margin)],
    ["Model would do", modelAction],
    ["Guide would do", recommendedAction],
    ["Effect on production policy", "Exactly zero"],
  ];
  for (const [label, value] of values) {
    const item = document.createElement("div");
    item.className = "key-value";
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = value;
    item.append(term, detail);
    facts.append(item);
  }
  const note = document.createElement("p");
  note.className = "guide-shadow-note";
  note.textContent = "The guide score is a training-teacher ranking, not a probability. This shadow calculation never changes the model’s action.";
  elements.guideShadowContent.append(headline, comparison, facts, note, detailsBlock("View exact guide scores and provenance", shadow));
}

function adapterComparisonDisplayLabel(option, sourceIndex = 0) {
  const label = firstNonEmptyText(recordValue(option, "label", "action_transcript"));
  if (label) return label;
  const position = recordValue(option, "index", "position");
  return `Legal option ${position === undefined ? sourceIndex + 1 : formatValue(position)}`;
}

function adapterComparisonChoiceRank(option) {
  return (option.adapter_on_choice === true ? 2 : 0) + (option.adapter_off_choice === true ? 1 : 0);
}

function adapterComparisonSortValue(entry, key) {
  if (key === "action") return adapterComparisonDisplayLabel(entry.option, entry.sourceIndex);
  if (key === "adapter_on") return finiteNumber(entry.option.adapter_on_probability);
  if (key === "adapter_off") return finiteNumber(entry.option.adapter_off_probability);
  if (key === "change") return finiteNumber(entry.option.probability_delta);
  if (key === "chosen") return adapterComparisonChoiceRank(entry.option);
  return undefined;
}

function compareAdapterComparisonOptions(left, right, key, direction) {
  const leftValue = adapterComparisonSortValue(left, key);
  const rightValue = adapterComparisonSortValue(right, key);
  const leftMissing = leftValue === undefined;
  const rightMissing = rightValue === undefined;
  if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
  let comparison = 0;
  if (!leftMissing && typeof leftValue === "string" && typeof rightValue === "string") {
    comparison = ADAPTER_COMPARISON_COLLATOR.compare(leftValue, rightValue);
  } else if (!leftMissing) {
    comparison = leftValue - rightValue;
  }
  if (comparison !== 0) return direction === "ascending" ? comparison : -comparison;
  const leftPosition = finiteNumber(recordValue(left.option, "position", "index"));
  const rightPosition = finiteNumber(recordValue(right.option, "position", "index"));
  if (leftPosition !== undefined && rightPosition !== undefined && leftPosition !== rightPosition) {
    return leftPosition - rightPosition;
  }
  if ((leftPosition === undefined) !== (rightPosition === undefined)) {
    return leftPosition === undefined ? 1 : -1;
  }
  return left.sourceIndex - right.sourceIndex;
}

function renderMatchupAdapterComparison(container, comparison) {
  const section = document.createElement("section");
  section.className = "adapter-comparison";
  const title = document.createElement("h4");
  title.textContent = "What changed with the Matchup Adapter ON?";
  section.append(title);

  if (!isObject(comparison) || isUnavailable(comparison)) {
    const unavailable = document.createElement("p");
    unavailable.className = "matchup-adapter-note adapter-comparison-unavailable";
    unavailable.textContent = `Adapter ON versus OFF comparison unavailable — ${reasonFrom(comparison, "the exact no-adapter rerun was not supplied")}.`;
    section.append(unavailable);
    container.append(section);
    return;
  }

  const adapterOn = isObject(comparison.adapter_on) ? comparison.adapter_on : {};
  const adapterOff = isObject(comparison.adapter_off) ? comparison.adapter_off : {};
  const onChoice = firstNonEmptyText(recordValue(adapterOn.selected_option, "label", "action_transcript")) || "Action unavailable";
  const offChoice = firstNonEmptyText(recordValue(adapterOff.selected_option, "label", "action_transcript")) || "Action unavailable";
  const intro = document.createElement("p");
  intro.className = "matchup-adapter-note";
  intro.textContent = "Adapter ON is the checksum-bound submitted-runtime decision. Adapter OFF is an exact rerun with only its matchup route bypassed. Change = ON − OFF. These are recomputed values, not recorded telemetry.";
  section.append(intro);

  const summary = document.createElement("dl");
  summary.className = "key-value-grid adapter-comparison-summary";
  const summaryValues = [
    ["Adapter ON choice", onChoice],
    ["Adapter OFF choice", offChoice],
    ["Choice changed?", comparison.choice_changed === true ? "Yes" : comparison.choice_changed === false ? "No" : "Unavailable"],
    ["Largest chance shift", finiteNumber(comparison.maximum_absolute_probability_shift) === undefined ? "Unavailable" : signedPercentagePoints(Math.abs(finiteNumber(comparison.maximum_absolute_probability_shift))).replace("+", "")],
  ];
  for (const [nameText, valueText] of summaryValues) {
    const item = document.createElement("div");
    item.className = "key-value";
    const name = document.createElement("dt");
    name.textContent = nameText;
    const value = document.createElement("dd");
    value.textContent = valueText;
    item.append(name, value);
    summary.append(item);
  }
  section.append(summary);

  const options = Array.isArray(comparison.options) ? comparison.options : [];
  const scroller = document.createElement("div");
  scroller.className = "table-scroll adapter-comparison-scroll";
  const table = document.createElement("table");
  table.className = "data-table adapter-comparison-table";
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const columns = [
    { key: "action", label: "Legal action", initialDirection: "ascending" },
    { key: "adapter_on", label: "Adapter ON chance", initialDirection: "descending" },
    { key: "adapter_off", label: "Adapter OFF chance", initialDirection: "descending" },
    { key: "change", label: "Change caused by adapter", initialDirection: "descending" },
    { key: "chosen", label: "Chosen", initialDirection: "descending" },
  ];
  const headerControls = new Map();
  let activeSortKey = "";
  let activeSortDirection = "ascending";
  for (const column of columns) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.setAttribute("aria-sort", "none");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "adapter-comparison-sort-button";
    button.dataset.sortKey = column.key;
    const label = document.createElement("span");
    label.textContent = column.label;
    const indicator = document.createElement("span");
    indicator.className = "adapter-comparison-sort-indicator";
    indicator.setAttribute("aria-hidden", "true");
    indicator.textContent = "↕";
    button.append(label, indicator);
    cell.append(button);
    headerRow.append(cell);
    headerControls.set(column.key, { cell, button, indicator, column });
  }
  head.append(headerRow);
  const body = document.createElement("tbody");

  const entries = options.map((option, sourceIndex) => ({ option, sourceIndex }));
  function updateSortHeaders() {
    for (const [key, control] of headerControls) {
      const active = key === activeSortKey;
      const direction = active ? activeSortDirection : "none";
      control.cell.setAttribute("aria-sort", direction);
      control.indicator.textContent = active
        ? activeSortDirection === "ascending" ? "▲" : "▼"
        : "↕";
      const nextDirection = active
        ? activeSortDirection === "ascending" ? "descending" : "ascending"
        : control.column.initialDirection;
      control.button.setAttribute("aria-label", `${control.column.label}: sort ${nextDirection}`);
    }
  }

  function renderRows() {
    body.replaceChildren();
    const visibleEntries = activeSortKey
      ? [...entries].sort((left, right) => compareAdapterComparisonOptions(
        left,
        right,
        activeSortKey,
        activeSortDirection,
      ))
      : entries;
    for (const entry of visibleEntries) {
      const option = entry.option;
      const row = document.createElement("tr");
      const action = document.createElement("td");
      action.textContent = adapterComparisonDisplayLabel(option, entry.sourceIndex);
      const onChance = document.createElement("td");
      onChance.textContent = formatProbability(option.adapter_on_probability);
      const offChance = document.createElement("td");
      offChance.textContent = formatProbability(option.adapter_off_probability);
      const delta = document.createElement("td");
      const probabilityDelta = finiteNumber(option.probability_delta);
      delta.textContent = probabilityDelta === undefined ? "Unavailable" : signedPercentagePoints(probabilityDelta);
      delta.className = probabilityDelta === undefined || probabilityDelta === 0 ? "" : probabilityDelta > 0 ? "adapter-delta-positive" : "adapter-delta-negative";
      const chosen = document.createElement("td");
      if (option.adapter_on_choice === true) chosen.append(flag("ON choice", "model"));
      if (option.adapter_off_choice === true) chosen.append(flag("OFF choice", "recorded"));
      if (!chosen.childNodes.length) chosen.textContent = "—";
      row.append(action, onChance, offChance, delta, chosen);
      body.append(row);
    }
    if (!visibleEntries.length) {
      const empty = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.textContent = "No exact per-action comparison was supplied.";
      empty.append(cell);
      body.append(empty);
    }
  }

  for (const { button, column } of headerControls.values()) {
    button.addEventListener("click", () => {
      if (activeSortKey === column.key) {
        activeSortDirection = activeSortDirection === "ascending" ? "descending" : "ascending";
      } else {
        activeSortKey = column.key;
        activeSortDirection = column.initialDirection;
      }
      updateSortHeaders();
      renderRows();
    });
  }
  updateSortHeaders();
  renderRows();
  table.append(head, body);
  scroller.append(table);
  section.append(scroller, detailsBlock("View exact Adapter ON/OFF calculation", comparison));
  container.append(section);
}

function matchupAdapterPolicyShiftSummary(value) {
  if (value === undefined || value === null) return undefined;
  if (!isObject(value)) return formatValue(value);
  const selectedProbability = finiteNumber(recordValue(value, "selected_option_probability_delta", "selected_probability_delta", "probability_delta"));
  const selectedLogit = finiteNumber(recordValue(value, "selected_option_logit_delta", "selected_logit_delta", "logit_delta"));
  const pieces = [];
  if (selectedProbability !== undefined) {
    pieces.push(`selected chance ${selectedProbability >= 0 ? "+" : "−"}${(Math.abs(selectedProbability) * 100).toLocaleString(undefined, { maximumFractionDigits: 4 })} pp`);
  }
  if (selectedLogit !== undefined) {
    pieces.push(`selected score ${selectedLogit >= 0 ? "+" : "−"}${formatNumber(Math.abs(selectedLogit))} logit`);
  }
  return pieces.length ? pieces.join(" · ") : "Structured source value below";
}

function matchupAdapterReliabilitySummary(value) {
  if (value === undefined || value === null) return undefined;
  if (!isObject(value)) return formatValue(value);
  if (isUnavailable(value)) return reasonFrom(value, "Reliability was not supplied.");
  const kind = firstNonEmptyText(recordValue(value, "kind", "method"));
  const residualScale = finiteNumber(recordValue(value, "residual_scale", "scale", "effective_multiplier"));
  const pieces = [];
  if (kind) pieces.push(kind.replaceAll("_", " "));
  if (residualScale !== undefined) pieces.push(`scale ${formatNumber(residualScale)}`);
  return pieces.length ? pieces.join(" · ") : "Structured source value below";
}

function renderObservation(trace) {
  const observation = trace.observation;
  setAvailability(
    elements.observationAvailability,
    observation,
    observation === undefined || observation === null ? "Observation not supplied" : "Source observation loaded",
  );
  renderStructuredContent(
    elements.observationContent,
    observation,
    "The source did not provide a causal observation for this trace.",
  );
}

function firstNonEmptyText(...values) {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const clean = value.trim();
    if (clean) return clean;
  }
  return undefined;
}

function firstActionTranscript(...values) {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const clean = value.trim();
    if (!clean) continue;
    if (/^\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]$/.test(clean)) continue;
    return clean;
  }
  return undefined;
}

function actionObjectOf(option) {
  return isObject(option?.action) ? option.action : null;
}

function friendlyActionTranscript(option) {
  const action = actionObjectOf(option);
  return firstActionTranscript(
    option?.action_transcript,
    option?.transcript,
    option?.plain_english_action,
    option?.plain_english,
    option?.human_readable_action,
    option?.human_readable,
    option?.display_text,
    option?.display_name,
    option?.label,
    option?.summary,
    action?.action_transcript,
    action?.transcript,
    action?.plain_english_action,
    action?.plain_english,
    action?.human_readable_action,
    action?.human_readable,
    action?.display_text,
    action?.display_name,
    action?.label,
    action?.summary,
    typeof option?.action === "string" ? option.action : undefined,
  );
}

function friendlyActionMeaning(option, transcript) {
  const action = actionObjectOf(option);
  const meaning = firstNonEmptyText(
    option?.action_meaning,
    option?.meaning,
    option?.explanation,
    option?.intent,
    option?.description,
    action?.action_meaning,
    action?.meaning,
    action?.explanation,
    action?.intent,
    action?.description,
  );
  return meaning && meaning !== transcript ? meaning : undefined;
}

function optionActionLabel(option) {
  return firstDefined(
    friendlyActionTranscript(option),
    option?.name,
    option?.action_id,
    option?.action,
    option?.summary,
  );
}

function actionTechnicalPayload(option) {
  const payload = {
    option_index: recordValue(option, "index", "action_index"),
    encoded_action: recordValue(option, "action", "raw_action", "candidate", "encoded_action"),
    action_id: recordValue(option, "action_id", "id"),
    source_label: recordValue(option, "label"),
    source_transcript: recordValue(option, "action_transcript", "transcript"),
    source_meaning: recordValue(option, "action_meaning", "meaning", "explanation"),
    actual_runtime_probability: recordValue(option, "actual_runtime_probability"),
    probability_semantics: recordValue(option, "probability_semantics"),
  };
  return Object.fromEntries(
    Object.entries(payload).filter(([, value]) => value !== undefined && value !== null),
  );
}

function appendFriendlyAction(cell, option) {
  const transcript = friendlyActionTranscript(option);
  const meaning = friendlyActionMeaning(option, transcript);
  const stack = document.createElement("div");
  stack.className = "action-presentation";

  const primary = document.createElement("span");
  primary.className = "action-transcript";
  if (transcript) {
    primary.textContent = transcript;
  } else {
    primary.classList.add("is-missing");
    primary.textContent = "Plain-English action transcript not supplied";
  }
  stack.append(primary);

  if (meaning) {
    const explanation = document.createElement("span");
    explanation.className = "action-meaning";
    explanation.textContent = meaning;
    stack.append(explanation);
  }

  const technical = actionTechnicalPayload(option);
  if (Object.keys(technical).length) {
    stack.append(detailsBlock("Technical action data", technical));
  }
  cell.append(stack);
}

function renderOptions(trace) {
  const options = collectionFrom(trace, "legal_options");
  if (!options.length) {
    const reason = reasonFrom(trace.legal_options, "No legal options were supplied by this trace.");
    elements.optionsAvailability.textContent = reason;
    setEmptyRow(elements.optionsBody, 7, reason);
    elements.optionContributionDetail.textContent = reason;
    return;
  }
  const hypotheticalSetup = reproductionStatusOf(trace) === "hypothetical_model_forward_not_submitted_runtime";
  elements.optionsAvailability.textContent = hypotheticalSetup
    ? `${options.length} legal options · percentages are a fresh hypothetical neural rerun; actual runtime choice remains in Technical action data`
    : `${options.length} legal option${options.length === 1 ? "" : "s"} returned by source`;
  clearNode(elements.optionsBody);
  for (const option of options) {
    const row = document.createElement("tr");
    appendCell(row, recordValue(option, "index", "action_index"), "number");

    const actionCell = document.createElement("td");
    appendFriendlyAction(actionCell, option);
    row.append(actionCell);

    appendCell(row, recordValue(option, "base_logit"), "number");
    appendCell(row, recordValue(option, "final_logit"), "number");
    appendCell(row, formatProbability(recordValue(option, "probability")), "number");
    structuredValueCell(row, routeContributionOf(option), "number");

    const flagsCell = document.createElement("td");
    const flags = document.createElement("div");
    flags.className = "decision-flags";
    if (option.is_recorded === true) flags.append(flag("Recorded", "recorded"));
    if (option.is_model_choice === true) flags.append(flag("Model", "model"));
    if (option.is_recorded !== true && option.is_model_choice !== true) flags.append(flag("—", ""));
    flagsCell.append(flags);
    row.append(flagsCell);
    elements.optionsBody.append(row);
  }
  const transcriptCount = options.filter((option) => Boolean(friendlyActionTranscript(option))).length;
  const transcriptNote = transcriptCount === options.length
    ? "Plain-English action transcripts were supplied for every legal option."
    : transcriptCount
      ? `Plain-English action transcripts were supplied for ${transcriptCount} of ${options.length} legal options; expand Technical action data for the exact encoded candidate.`
      : "The source did not supply plain-English action transcripts. Exact encoded candidates remain available under Technical action data; this view does not invent an action meaning.";
  const contributionNote = hasOptionRouteContributions(options)
    ? " Per-option route evidence is shown in the table; expand structured values when the source returns a route map."
    : " Per-route, per-option evidence was not supplied for this trace.";
  elements.optionContributionDetail.textContent = `${transcriptNote}${contributionNote}`;
}

function flag(text, kind) {
  const item = document.createElement("span");
  item.className = `mini-badge${kind ? ` ${kind}` : ""}`;
  item.textContent = text;
  return item;
}

function renderFusion(trace) {
  const model = isObject(trace.model) ? trace.model : {};
  const fusion = firstDefined(model.fusion, trace.fusion, trace.fusion_v3);
  const route = firstDefined(model.route, trace.route, trace.route_reliability);
  const perOption = firstDefined(
    fusion && fusion.per_option_contributions,
    fusion && fusion.option_contributions,
    model.per_option_contributions,
    trace.per_option_contributions,
  );
  clearNode(elements.fusionContent);
  elements.fusionContent.className = "structured-content";
  const grid = document.createElement("div");
  grid.className = "fusion-grid";
  const hasRoute = appendSourceObject(grid, "Selected route / reliability", route);
  const hasFusion = appendSourceObject(grid, "Fusion payload", fusion);
  const hasPerOption = appendSourceObject(grid, "Per-option route source data", perOption);
  if (hasRoute || hasFusion || hasPerOption) {
    elements.fusionContent.append(grid);
    elements.fusionAvailability.textContent = "Fusion source payload loaded";
  } else {
    elements.fusionContent.className = "structured-content empty-state";
    elements.fusionContent.textContent = "Fusion route payload was not supplied by this trace.";
    elements.fusionAvailability.textContent = "Fusion route payload unavailable";
  }
}

function finiteNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function numericVector(value) {
  if (!Array.isArray(value) || value.length === 0) return null;
  const values = [];
  for (const item of value) {
    const numeric = finiteNumber(item);
    if (numeric === undefined) return null;
    values.push(numeric);
  }
  return values;
}

function firstAvailableObject(...values) {
  return values.find((value) => isObject(value) && !isUnavailable(value)) || null;
}

function metricFromSources(sources, keys) {
  for (const source of sources) {
    if (!isObject(source)) continue;
    const scopes = [source];
    for (const name of ["metrics", "summary", "policy_influence", "impact"]) {
      if (isObject(source[name])) scopes.push(source[name]);
    }
    for (const scope of scopes) {
      const value = recordValue(scope, ...keys);
      if (value !== undefined && value !== null) return value;
    }
  }
  return undefined;
}

function numericMetric(sources, keys) {
  return finiteNumber(metricFromSources(sources, keys));
}

function vectorMetric(sources, keys) {
  for (const source of sources) {
    if (!isObject(source)) continue;
    const scopes = [source];
    for (const name of ["metrics", "summary", "policy_influence", "impact"]) {
      if (isObject(source[name])) scopes.push(source[name]);
    }
    for (const scope of scopes) {
      const vector = numericVector(recordValue(scope, ...keys));
      if (vector) return vector;
    }
  }
  return null;
}

function booleanMetric(sources, keys) {
  const value = metricFromSources(sources, keys);
  return typeof value === "boolean" ? value : undefined;
}

function vectorDifference(left, right) {
  if (!left || !right || left.length !== right.length) return null;
  return left.map((value, index) => value - right[index]);
}

function negatedVector(vector) {
  return vector ? vector.map((value) => -value) : null;
}

function maxAbsoluteValue(vector) {
  if (!vector || !vector.length) return undefined;
  return Math.max(...vector.map((value) => Math.abs(value)));
}

function indexOfLargest(vector) {
  if (!vector || !vector.length) return undefined;
  return vector.reduce((best, value, index) => value > vector[best] ? index : best, 0);
}

function indexOfSmallest(vector) {
  if (!vector || !vector.length) return undefined;
  return vector.reduce((best, value, index) => value < vector[best] ? index : best, 0);
}

function optionPositionForIndex(options, index) {
  if (index === undefined || index === null || index === "") return undefined;
  const explicit = options.findIndex((option) => sameId(recordValue(option, "index", "action_index"), index));
  if (explicit >= 0) return explicit;
  const numeric = finiteNumber(index);
  return numeric !== undefined && Number.isInteger(numeric) && numeric >= 0 && numeric < options.length
    ? numeric
    : undefined;
}

function optionIndexAtPosition(options, position) {
  if (position === undefined || position === null || !options[position]) return undefined;
  return firstDefined(recordValue(options[position], "index", "action_index"), position);
}

function actionNameAtPosition(options, position) {
  if (position === undefined || position === null) return "the selected legal option";
  const option = options[position];
  const explicitIndex = optionIndexAtPosition(options, position);
  if (!option) return `legal option #${formatValue(explicitIndex === undefined ? position : explicitIndex)}`;
  return friendlyActionTranscript(option) || `legal option #${formatValue(explicitIndex === undefined ? position : explicitIndex)}`;
}

function actionNameFromImpactValue(value, options) {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "number" || (typeof value === "string" && /^\d+$/.test(value.trim()))) {
    const position = optionPositionForIndex(options, value);
    return position === undefined ? `legal option #${formatValue(value)}` : actionNameAtPosition(options, position);
  }
  if (typeof value === "string") return value;
  if (!isObject(value)) return undefined;
  const index = recordValue(value, "index", "option_index", "action_index", "selected_option_index");
  const position = optionPositionForIndex(options, index);
  if (position !== undefined) return actionNameAtPosition(options, position);
  if (index !== undefined && index !== null) return `legal option #${formatValue(index)}`;
  return firstNonEmptyText(
    recordValue(value, "action_transcript", "transcript", "label", "display_name", "name", "action"),
  );
}

function ablationInfoForHead(head) {
  const envelope = firstDefined(head?.ablation_effect, head?.ablation, head?.leave_one_out);
  if (!isObject(envelope)) {
    return { envelope, payload: null, label: "No leave-one-out result", reason: undefined };
  }
  const candidates = [
    ["Runtime leave-one-out", envelope.runtime_path],
    ["Counterfactual all-routes leave-one-out", envelope.counterfactual_all_routes],
    ["Leave-one-out result", envelope.removal],
    ["Leave-one-out result", envelope.policy_impact],
  ];
  for (const [label, payload] of candidates) {
    if (isObject(payload) && !isUnavailable(payload)) return { envelope, payload, label, reason: undefined };
  }
  if (!isUnavailable(envelope)) {
    return { envelope, payload: envelope, label: "Leave-one-out result", reason: undefined };
  }
  return {
    envelope,
    payload: null,
    label: "No leave-one-out result",
    reason: reasonFrom(envelope, "The source did not provide a usable leave-one-out result."),
  };
}

function softmaxVector(values) {
  if (!values || !values.length || !values.every((value) => Number.isFinite(value))) return null;
  const maximum = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return total > 0 && Number.isFinite(total) ? exponents.map((value) => value / total) : null;
}

function signedLogitChange(value) {
  if (value === undefined) return "Not supplied";
  if (Math.abs(value) < 1e-12) return "No score change";
  return `${value > 0 ? "Rises" : "Falls"} ${formatNumber(Math.abs(value))} logit`;
}

function signedProbabilityChange(value) {
  if (value === undefined) return "Not supplied";
  if (Math.abs(value) < 1e-12) return "No chance change";
  return `${value > 0 ? "Rises" : "Falls"} ${(Math.abs(value) * 100).toLocaleString(undefined, { maximumFractionDigits: 4 })} percentage points`;
}

function signedPercentagePoints(value) {
  if (value === undefined) return "Probability delta not supplied";
  return `${value >= 0 ? "+" : "−"}${(Math.abs(value) * 100).toLocaleString(undefined, { maximumFractionDigits: 4 })} percentage points`;
}

function probabilityDeltaFromImpactValue(value) {
  if (!isObject(value)) return undefined;
  return finiteNumber(recordValue(value, "probability_delta", "probability_effect", "effect_probability", "delta"));
}

function positionFromImpactValue(value, options) {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "number" || (typeof value === "string" && /^\d+$/.test(value.trim()))) {
    return optionPositionForIndex(options, value);
  }
  return isObject(value)
    ? optionPositionForIndex(options, recordValue(value, "index", "option_index", "action_index", "selected_option_index"))
    : undefined;
}

function headImpactFor(head, trace) {
  const name = formatValue(recordValue(head, "name", "head", "id"));
  const options = collectionFrom(trace, "legal_options");
  const unavailable = isUnavailable(head);
  const policyInfluence = firstAvailableObject(
    head?.policy_influence,
    head?.policy_impact,
    head?.influence,
    head?.decision_impact,
    head?.removal_impact,
    head?.impact,
  );
  const ablation = ablationInfoForHead(head);
  const metricSources = [policyInfluence, ablation.payload].filter(Boolean);
  const selectedIndex = firstDefined(
    recordValue(trace?.model, "selected_index", "selected_action_index"),
    metricFromSources(metricSources, ["selected_option_index", "selected_index", "model_choice_index"]),
  );
  const selectedPosition = optionPositionForIndex(options, selectedIndex);
  const selectedAction = actionNameAtPosition(options, selectedPosition);

  let effectLogits = vectorMetric(metricSources, [
    "effect_logits",
    "full_minus_without_head_logits",
    "full_minus_removed_logits",
    "head_effect_logits",
    "policy_effect_logits",
  ]);
  let removalLogits = vectorMetric(metricSources, [
    "option_logit_change_if_removed",
    "option_logit_changes_if_removed",
    "removed_minus_full_logits",
    "policy_without_head_minus_full_logits",
    "removal_logit_changes",
  ]);
  const fullLogits = vectorMetric(metricSources, ["full_policy_logits", "full_logits", "policy_logits"]);
  const withoutHeadLogits = vectorMetric(metricSources, ["policy_without_head_logits", "without_head_logits", "removed_policy_logits"]);
  if (!removalLogits && fullLogits && withoutHeadLogits) removalLogits = vectorDifference(withoutHeadLogits, fullLogits);
  if (!effectLogits && removalLogits) effectLogits = negatedVector(removalLogits);
  if (!removalLogits && effectLogits) removalLogits = negatedVector(effectLogits);

  let effectProbabilities = vectorMetric(metricSources, [
    "effect_probabilities",
    "full_minus_without_head_probabilities",
    "policy_effect_probabilities",
  ]);
  let removalProbabilities = vectorMetric(metricSources, [
    "option_probability_change_if_removed",
    "option_probability_changes_if_removed",
    "removed_minus_full_probabilities",
    "policy_without_head_minus_full_probabilities",
    "removal_probability_changes",
  ]);
  const fullProbabilities = vectorMetric(metricSources, ["full_policy_probabilities", "full_probabilities", "policy_probabilities"])
    || numericVector(options.map((option) => recordValue(option, "probability")));
  const withoutHeadProbabilities = vectorMetric(metricSources, ["policy_without_head_probabilities", "without_head_probabilities", "removed_policy_probabilities"]);
  if (!removalProbabilities && fullProbabilities && withoutHeadProbabilities) {
    removalProbabilities = vectorDifference(withoutHeadProbabilities, fullProbabilities);
  }
  if (!effectProbabilities && removalProbabilities) effectProbabilities = negatedVector(removalProbabilities);
  if (!removalProbabilities && effectProbabilities) removalProbabilities = negatedVector(effectProbabilities);
  if (!removalProbabilities && removalLogits) {
    const currentLogits = numericVector(options.map((option) => recordValue(option, "final_logit"))) || fullLogits;
    const removedLogits = currentLogits && currentLogits.length === removalLogits.length
      ? currentLogits.map((value, index) => value + removalLogits[index])
      : null;
    const currentProbabilities = softmaxVector(currentLogits);
    const removedProbabilities = softmaxVector(removedLogits);
    if (currentProbabilities && removedProbabilities) {
      removalProbabilities = vectorDifference(removedProbabilities, currentProbabilities);
      effectProbabilities = negatedVector(removalProbabilities);
    }
  }

  let selectedLogitChange = numericMetric(metricSources, [
    "selected_option_logit_change_if_removed",
    "selected_logit_change_if_removed",
    "removed_selected_option_logit_delta",
    "removal_selected_logit_change",
  ]);
  if (selectedLogitChange === undefined) {
    const effect = numericMetric(metricSources, ["selected_option_logit_delta", "selected_logit_delta", "selected_option_logit_effect"]);
    if (effect !== undefined) selectedLogitChange = -effect;
  }
  if (selectedLogitChange === undefined && removalLogits && selectedPosition !== undefined) {
    selectedLogitChange = removalLogits[selectedPosition];
  }

  let selectedProbabilityChange = numericMetric(metricSources, [
    "selected_option_probability_change_if_removed",
    "selected_probability_change_if_removed",
    "removed_selected_option_probability_delta",
    "removal_selected_probability_change",
  ]);
  if (selectedProbabilityChange === undefined) {
    const effect = numericMetric(metricSources, [
      "selected_option_probability_delta",
      "selected_probability_delta",
      "selected_option_probability_effect",
    ]);
    if (effect !== undefined) selectedProbabilityChange = -effect;
  }
  if (selectedProbabilityChange === undefined && removalProbabilities && selectedPosition !== undefined) {
    selectedProbabilityChange = removalProbabilities[selectedPosition];
  }

  const routeContribution = numericVector(recordValue(head, "route_delta", "logit_delta", "contribution"));
  const hasRemovalMeasurement = Boolean(
    effectLogits
    || removalLogits
    || effectProbabilities
    || removalProbabilities
    || selectedLogitChange !== undefined
    || selectedProbabilityChange !== undefined
    || booleanMetric(metricSources, ["changes_model_choice", "model_choice_changed"]) !== undefined,
  );
  const sourceKind = hasRemovalMeasurement
    ? policyInfluence
      ? "Source policy-influence result"
      : ablation.label
    : routeContribution
      ? "Current raw route signal only"
      : "No policy-impact result";
  const policyEffectForRanking = effectProbabilities || effectLogits;
  const suppliedHelped = metricFromSources(metricSources, ["most_helped_option", "most_helped_action", "highest_helped_option"]);
  const suppliedHurt = metricFromSources(metricSources, ["most_hurt_option", "most_hurt_action", "highest_hurt_option"]);
  const helpedPosition = positionFromImpactValue(suppliedHelped, options) ?? indexOfLargest(policyEffectForRanking);
  const hurtPosition = positionFromImpactValue(suppliedHurt, options) ?? indexOfSmallest(policyEffectForRanking);
  const mostHelped = actionNameFromImpactValue(suppliedHelped, options)
    || (helpedPosition === undefined ? undefined : actionNameAtPosition(options, helpedPosition));
  const mostHurt = actionNameFromImpactValue(suppliedHurt, options)
    || (hurtPosition === undefined ? undefined : actionNameAtPosition(options, hurtPosition));
  const mostHelpedProbabilityDelta = probabilityDeltaFromImpactValue(suppliedHelped)
    ?? (effectProbabilities && helpedPosition !== undefined ? effectProbabilities[helpedPosition] : undefined);
  const mostHurtProbabilityDelta = probabilityDeltaFromImpactValue(suppliedHurt)
    ?? (effectProbabilities && hurtPosition !== undefined ? effectProbabilities[hurtPosition] : undefined);
  const rawRouteHelpedPosition = indexOfLargest(routeContribution);
  const rawRouteHurtPosition = indexOfSmallest(routeContribution);
  const rawRouteHelped = rawRouteHelpedPosition === undefined ? undefined : actionNameAtPosition(options, rawRouteHelpedPosition);
  const rawRouteHurt = rawRouteHurtPosition === undefined ? undefined : actionNameAtPosition(options, rawRouteHurtPosition);
  const rawRouteHelpedValue = routeContribution && rawRouteHelpedPosition !== undefined ? routeContribution[rawRouteHelpedPosition] : undefined;
  const rawRouteHurtValue = routeContribution && rawRouteHurtPosition !== undefined ? routeContribution[rawRouteHurtPosition] : undefined;
  const choiceChanged = booleanMetric(metricSources, ["changes_model_choice", "model_choice_changed"]);
  const ablatedIndex = metricFromSources(metricSources, ["ablated_model_choice_index", "model_choice_without_head_index", "removed_model_choice_index"]);
  const ablatedPosition = optionPositionForIndex(options, ablatedIndex);
  const ablatedAction = ablatedPosition === undefined ? undefined : actionNameAtPosition(options, ablatedPosition);
  const suppliedMaximumLogitShift = numericMetric(metricSources, [
    "maximum_absolute_option_logit_delta",
    "max_absolute_logit_effect",
    "max_absolute_option_shift",
    "maximum_option_shift",
  ]);
  const maximumLogitShift = suppliedMaximumLogitShift === undefined
    ? maxAbsoluteValue(removalLogits || effectLogits || (hasRemovalMeasurement ? null : routeContribution))
    : suppliedMaximumLogitShift;
  const suppliedMaximumProbabilityShift = numericMetric(metricSources, [
    "maximum_absolute_option_probability_delta",
    "max_absolute_probability_effect",
    "max_absolute_probability_shift",
  ]);
  const maximumProbabilityShift = suppliedMaximumProbabilityShift === undefined
    ? maxAbsoluteValue(removalProbabilities || effectProbabilities)
    : suppliedMaximumProbabilityShift;
  const unavailableReason = unavailable
    ? reasonFrom(head, "The source marked this head unavailable.")
    : !hasRemovalMeasurement && !routeContribution
      ? reasonFrom(policyInfluence || ablation.envelope || head, ablation.reason || "The source did not provide a policy-impact measurement for this head.")
      : undefined;
  const impact = {
    name,
    unavailable,
    unavailableReason,
    sourceKind,
    hasRemovalMeasurement,
    selectedAction,
    selectedLogitChange,
    selectedProbabilityChange,
    maximumLogitShift,
    maximumProbabilityShift,
    mostHelped,
    mostHurt,
    mostHelpedProbabilityDelta,
    mostHurtProbabilityDelta,
    rawRouteHelped,
    rawRouteHurt,
    rawRouteHelpedValue,
    rawRouteHurtValue,
    choiceChanged,
    ablatedAction,
    policyInfluence,
    ablation,
    routeContribution,
    effectLogits,
    removalLogits,
    effectProbabilities,
    removalProbabilities,
  };
  impact.explanation = headImpactExplanation(impact);
  return impact;
}

function headImpactExplanation(impact) {
  if (impact.unavailable) return `This head is unavailable: ${impact.unavailableReason}`;
  if (!impact.hasRemovalMeasurement) {
    if (impact.routeContribution) {
      return "The source supplied this head's raw route signal, but not a leave-one-out removal result. A raw route signal is not labelled here as an additive policy contribution or proof of what would happen if the head were removed.";
    }
    return impact.unavailableReason || "The source did not provide a head-removal or policy-influence result for this head.";
  }
  const pieces = [];
  if (impact.selectedLogitChange !== undefined) {
    pieces.push(`If this head were removed, ${impact.selectedAction}'s score would ${impact.selectedLogitChange > 0 ? "rise" : impact.selectedLogitChange < 0 ? "fall" : "stay unchanged"}${Math.abs(impact.selectedLogitChange) < 1e-12 ? "" : ` by ${formatNumber(Math.abs(impact.selectedLogitChange))} logit`}.`);
  }
  if (impact.selectedProbabilityChange !== undefined) {
    pieces.push(`Its policy chance would ${impact.selectedProbabilityChange > 0 ? "rise" : impact.selectedProbabilityChange < 0 ? "fall" : "stay unchanged"}${Math.abs(impact.selectedProbabilityChange) < 1e-12 ? "" : ` by ${(Math.abs(impact.selectedProbabilityChange) * 100).toLocaleString(undefined, { maximumFractionDigits: 4 })} percentage points`}.`);
  }
  if (impact.choiceChanged === true) {
    pieces.push(`The model's top legal option would change${impact.ablatedAction ? ` to ${impact.ablatedAction}` : ""}.`);
  } else if (impact.choiceChanged === false) {
    pieces.push("The source reports that the model's top legal option would stay the same.");
  }
  if (impact.mostHelped && impact.mostHurt) {
    pieces.push(`With this head present, it most helped ${impact.mostHelped} (${signedPercentagePoints(impact.mostHelpedProbabilityDelta)}) and most hurt ${impact.mostHurt} (${signedPercentagePoints(impact.mostHurtProbabilityDelta)}).`);
  }
  return pieces.length
    ? pieces.join(" ")
    : "The source returned a leave-one-out result, but not enough per-option data to describe the change in plain language.";
}

function compactHeadImpact(impact) {
  if (impact.unavailable) return `Unavailable — ${impact.unavailableReason}`;
  if (!impact.hasRemovalMeasurement) {
    return impact.routeContribution
      ? "Current raw route signal only; no head-removal result supplied."
      : impact.unavailableReason || "No policy-impact result supplied.";
  }
  const pieces = [];
  if (impact.selectedLogitChange !== undefined) pieces.push(`selected score: ${signedLogitChange(impact.selectedLogitChange)}`);
  if (impact.selectedProbabilityChange !== undefined) pieces.push(`selected chance: ${signedProbabilityChange(impact.selectedProbabilityChange)}`);
  if (impact.maximumLogitShift !== undefined) pieces.push(`largest score shift: ${formatNumber(impact.maximumLogitShift)} logit`);
  if (impact.mostHelpedProbabilityDelta !== undefined) pieces.push(`most helped: ${signedPercentagePoints(impact.mostHelpedProbabilityDelta)}`);
  if (impact.mostHurtProbabilityDelta !== undefined) pieces.push(`most hurt: ${signedPercentagePoints(impact.mostHurtProbabilityDelta)}`);
  if (impact.choiceChanged === true) pieces.push("top option changes");
  return pieces.length ? pieces.join(" · ") : "Leave-one-out data supplied; see card for source details.";
}

function impactMetric(label, value, className = "") {
  const item = document.createElement("div");
  item.className = `impact-metric${className ? ` ${className}` : ""}`;
  const name = document.createElement("span");
  name.textContent = label;
  const detail = document.createElement("strong");
  detail.textContent = value;
  item.append(name, detail);
  return item;
}

function renderHeadImpactCard(impact) {
  const card = document.createElement("article");
  card.className = "head-impact-card";
  if (impact.unavailable) card.classList.add("is-unavailable");
  else if (!impact.hasRemovalMeasurement) card.classList.add("is-limited");
  else if (impact.selectedLogitChange !== undefined) card.classList.add(impact.selectedLogitChange >= 0 ? "is-positive" : "is-negative");

  const heading = document.createElement("div");
  heading.className = "head-impact-heading";
  const title = document.createElement("h4");
  title.textContent = impact.name;
  const source = document.createElement("span");
  source.className = "mini-badge";
  source.textContent = impact.sourceKind;
  heading.append(title, source);

  const question = document.createElement("p");
  question.className = "head-impact-question";
  question.textContent = "What changes if this head were removed?";
  const answer = document.createElement("p");
  answer.className = "head-impact-answer";
  answer.textContent = impact.explanation;

  const metrics = document.createElement("div");
  metrics.className = "impact-metric-grid";
  const helpedLabel = impact.hasRemovalMeasurement ? "Most helped (with head)" : "Largest positive raw route signal";
  const hurtLabel = impact.hasRemovalMeasurement ? "Most hurt (with head)" : "Largest negative raw route signal";
  const helpedValue = impact.hasRemovalMeasurement
    ? impact.mostHelped
      ? `${impact.mostHelped} (${signedPercentagePoints(impact.mostHelpedProbabilityDelta)})`
      : "Not supplied"
    : impact.rawRouteHelped
      ? `${impact.rawRouteHelped} (${formatNumber(impact.rawRouteHelpedValue)})`
      : "Not supplied";
  const hurtValue = impact.hasRemovalMeasurement
    ? impact.mostHurt
      ? `${impact.mostHurt} (${signedPercentagePoints(impact.mostHurtProbabilityDelta)})`
      : "Not supplied"
    : impact.rawRouteHurt
      ? `${impact.rawRouteHurt} (${formatNumber(impact.rawRouteHurtValue)})`
      : "Not supplied";
  metrics.append(
    impactMetric("Selected score", signedLogitChange(impact.selectedLogitChange)),
    impactMetric("Selected chance", signedProbabilityChange(impact.selectedProbabilityChange)),
    impactMetric("Largest score shift", impact.maximumLogitShift === undefined ? "Not supplied" : `${formatNumber(impact.maximumLogitShift)} logit`),
    impactMetric("Largest chance shift", impact.maximumProbabilityShift === undefined ? "Not supplied" : `${(impact.maximumProbabilityShift * 100).toLocaleString(undefined, { maximumFractionDigits: 4 })} percentage points`),
    impactMetric(helpedLabel, helpedValue),
    impactMetric(hurtLabel, hurtValue),
    impactMetric(
      "Top legal option",
      impact.choiceChanged === true
        ? `Changes${impact.ablatedAction ? ` to ${impact.ablatedAction}` : ""}`
        : impact.choiceChanged === false ? "Stays the same" : "Not supplied",
    ),
  );
  card.append(heading, question, answer, metrics);
  card.append(detailsBlock("View policy-impact calculation and source tensors", {
    policy_influence: impact.policyInfluence,
    leave_one_out: impact.ablation.envelope,
    selected_leave_one_out_path: impact.ablation.payload,
    route_contribution: impact.routeContribution,
    full_minus_removed_logits: impact.effectLogits,
    removed_minus_full_logits: impact.removalLogits,
    full_minus_removed_probabilities: impact.effectProbabilities,
    removed_minus_full_probabilities: impact.removalProbabilities,
    source_kind: impact.sourceKind,
  }));
  return card;
}

function renderHeadImpactGrid(impacts) {
  clearNode(elements.headImpactGrid);
  if (!impacts.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No head policy-impact result was supplied.";
    elements.headImpactGrid.append(empty);
    elements.headImpactSummary.textContent = empty.textContent;
    return;
  }
  const removalCount = impacts.filter((impact) => impact.hasRemovalMeasurement && !impact.unavailable).length;
  const routeOnlyCount = impacts.filter((impact) => !impact.hasRemovalMeasurement && impact.routeContribution && !impact.unavailable).length;
  elements.headImpactSummary.textContent = removalCount
    ? `${removalCount} of ${impacts.length} heads include a source policy-influence or leave-one-out result. “Removed” values compare the current policy with the same reconstruction after one head source is removed; raw tensors remain expandable.`
    : routeOnlyCount
      ? `${routeOnlyCount} heads expose a raw route signal, but the source did not supply a leave-one-out removal result. These cards do not call that raw signal an additive policy contribution or a counterfactual removal effect.`
      : "The source did not supply a head-removal or policy-influence result. Raw head values and availability remain below.";
  for (const impact of impacts) elements.headImpactGrid.append(renderHeadImpactCard(impact));
}

function headFaqCurrentEffect(impact) {
  if (!impact) return "Choose a decision to calculate this head’s current policy effect.";
  if (impact.unavailable) return `Unavailable for this decision — ${impact.unavailableReason}`;
  if (!impact.hasRemovalMeasurement) {
    return impact.routeContribution
      ? "A raw route signal exists, but no exact remove-this-head policy result was supplied."
      : impact.unavailableReason || "No exact current-decision effect was supplied.";
  }
  const pieces = ["Current policy setting: 1× baseline."];
  if (impact.selectedProbabilityChange !== undefined) {
    pieces.push(`If this head were removed, the chosen action’s chance would change by ${signedPercentagePoints(impact.selectedProbabilityChange)}.`);
  }
  const probabilityEffect = impact.effectProbabilities || impact.removalProbabilities;
  if (probabilityEffect) {
    const totalVariation = 0.5 * probabilityEffect.reduce((total, value) => total + Math.abs(value), 0);
    pieces.push(`Total policy-distribution shift: ${(totalVariation * 100).toLocaleString(undefined, { maximumFractionDigits: 4 })}%.`);
  }
  if (impact.choiceChanged === true) pieces.push("Removing it would change the top legal action.");
  else if (impact.choiceChanged === false) pieces.push("Removing it would not change the top legal action.");
  return pieces.join(" ");
}

function faqFact(label, text) {
  const item = document.createElement("div");
  item.className = "head-faq-fact";
  const term = document.createElement("span");
  term.textContent = label;
  const value = document.createElement("p");
  value.textContent = text;
  item.append(term, value);
  return item;
}

function renderHeadFaq(heads = [], impacts = []) {
  clearNode(elements.headFaqList);
  const liveHeads = new Map();
  for (const [index, head] of heads.entries()) {
    const rawName = recordValue(head, "name", "head", "id");
    if (typeof rawName !== "string") continue;
    liveHeads.set(rawName, { head, impact: impacts[index] });
  }
  const definitions = [...HEAD_FAQ];
  for (const name of liveHeads.keys()) {
    if (definitions.some((definition) => definition.id === name)) continue;
    definitions.push({
      id: name,
      title: name.replaceAll("_", " "),
      question: "What does this architecture-specific head represent?",
      answer: "This checkpoint returned the head, but this inspector version has no source-backed plain-English definition for it yet.",
      horizon: "Not documented",
      caution: "Use the raw tensors and provenance; do not guess its meaning from the name alone.",
    });
  }
  for (const definition of definitions) {
    const live = liveHeads.get(definition.id);
    const details = document.createElement("details");
    details.className = "head-faq-item";
    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.className = "head-faq-title";
    title.textContent = definition.title;
    const code = document.createElement("code");
    code.textContent = definition.id;
    const badge = document.createElement("span");
    badge.className = `mini-badge ${live ? "model" : ""}`;
    badge.textContent = live ? "In this trace" : "Architecture reference";
    summary.append(title, code, badge);
    const body = document.createElement("div");
    body.className = "head-faq-body";
    body.append(
      faqFact("What it is looking for", definition.question),
      faqFact("What that means", definition.answer),
      faqFact("Time horizon", definition.horizon),
      faqFact("Important caveat", definition.caution),
      faqFact("Current decision policy effect", headFaqCurrentEffect(live?.impact)),
    );
    if (live) {
      const reliability = finiteNumber(recordValue(live.head, "route_reliability", "reliability"));
      if (reliability !== undefined) {
        body.append(faqFact(
          "Current route reliability",
          `${formatNumber(reliability)} (a technical fusion multiplier, not a percent importance or training-loss weight).`,
        ));
      }
    }
    details.append(summary, body);
    elements.headFaqList.append(details);
  }
}

function appendHeadImpactCell(row, impact) {
  const cell = document.createElement("td");
  cell.className = "head-impact-cell";
  const source = document.createElement("span");
  source.className = "cell-note";
  source.textContent = impact.sourceKind;
  const text = document.createElement("span");
  text.textContent = compactHeadImpact(impact);
  cell.append(source, text);
  row.append(cell);
}

function renderHeads(trace) {
  const heads = collectionFrom(trace, "heads");
  if (!heads.length) {
    const reason = reasonFrom(trace.heads, "No architecture-present head payload was supplied by this trace.");
    elements.headsAvailability.textContent = reason;
    elements.headImpactSummary.textContent = reason;
    clearNode(elements.headImpactGrid);
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = reason;
    elements.headImpactGrid.append(empty);
    setEmptyRow(elements.headsBody, 9, reason);
    elements.headsNote.textContent = reason;
    renderHeadFaq();
    return;
  }
  const impacts = heads.map((head) => headImpactFor(head, trace));
  elements.headsAvailability.textContent = `${heads.length} head${heads.length === 1 ? "" : "s"} returned by source`;
  elements.headsNote.textContent = "The cards translate source policy-influence or leave-one-out data; raw, normalized, route, and ablation payloads remain expandable in the technical table.";
  renderHeadImpactGrid(impacts);
  renderHeadFaq(heads, impacts);
  clearNode(elements.headsBody);
  for (const [index, head] of heads.entries()) {
    const impact = impacts[index];
    const row = document.createElement("tr");
    const nameCell = document.createElement("td");
    const nameStack = document.createElement("div");
    nameStack.className = "label-stack";
    const name = document.createElement("span");
    name.className = "mono-value";
    name.textContent = formatValue(recordValue(head, "name", "head", "id"));
    nameStack.append(name);
    if (isUnavailable(head)) nameStack.append(flag("Unavailable", "unavailable"));
    nameCell.append(nameStack);
    row.append(nameCell);
    appendCell(row, recordValue(head, "scope", "role", "fusion_role"));
    structuredValueCell(row, recordValue(head, "raw", "raw_value"));
    structuredValueCell(row, recordValue(head, "normalized", "normalised", "normalized_value"));
    appendCell(row, maskDescription(head));
    structuredValueCell(row, recordValue(head, "route_reliability", "reliability"));
    structuredValueCell(row, recordValue(head, "route_delta", "logit_delta", "contribution"));
    structuredValueCell(row, recordValue(head, "ablation_effect", "ablation", "leave_one_out"));
    appendHeadImpactCell(row, impact);
    elements.headsBody.append(row);
  }
}

function decisionInfluenceEnvelope(trace) {
  return isObject(trace?.decision_influence) ? trace.decision_influence : null;
}

function decisionInfluenceAvailable(influence) {
  return Boolean(
    influence
    && !isUnavailable(influence)
    && recordValue(influence, "mode") === "decision_only_counterfactual"
    && recordValue(influence, "method") === "exact_nonlinear_fusion_source_recomputation",
  );
}

function decisionInfluenceReason(influence) {
  if (!influence) return "The source did not return an exact decision-influence payload for this trace.";
  if (isUnavailable(influence)) return reasonFrom(influence, "Exact decision influence is unavailable for this trace.");
  if (recordValue(influence, "mode") !== "decision_only_counterfactual") {
    return "The source did not identify this payload as a decision-only counterfactual.";
  }
  if (recordValue(influence, "method") !== "exact_nonlinear_fusion_source_recomputation") {
    return "The source did not confirm exact nonlinear fusion recomputation for this decision.";
  }
  return "Exact decision influence is unavailable for this trace.";
}

function decisionInfluenceEligibleHeads(influence) {
  const seen = new Set();
  const names = [];
  for (const entry of collectionFrom(influence, "eligible_heads")) {
    const name = typeof entry === "string" ? entry.trim() : "";
    if (!/^[a-z][a-z0-9_]*$/.test(name) || seen.has(name)) continue;
    seen.add(name);
    names.push(name);
  }
  return names.slice(0, 32);
}

function decisionInfluenceBounds(influence) {
  const source = Array.isArray(influence?.scale_bounds) ? influence.scale_bounds : [];
  const minimum = finiteNumber(source[0]);
  const maximum = finiteNumber(source[1]);
  if (minimum === undefined || maximum === undefined || minimum > maximum) return [0, 2];
  return [minimum, maximum];
}

function decisionInfluenceDefaultScale(influence) {
  const value = finiteNumber(recordValue(influence, "default_scale"));
  return value === undefined ? 1 : value;
}

function scaleForDecisionHead(name, influence) {
  const requested = finiteNumber(state.decisionInfluence.scales[name]);
  if (requested !== undefined) return requested;
  const effective = isObject(influence?.effective_scales)
    ? finiteNumber(influence.effective_scales[name])
    : undefined;
  return effective === undefined ? decisionInfluenceDefaultScale(influence) : effective;
}

function baselineHeadWeight(name, influence) {
  const weights = isObject(influence?.baseline_head_weights)
    ? influence.baseline_head_weights
    : {};
  return isObject(weights[name]) ? weights[name] : null;
}

function setDecisionInfluenceControlsDisabled(disabled) {
  elements.decisionInfluenceReset.disabled = disabled;
  elements.decisionInfluenceApply.disabled = disabled;
  for (const range of elements.decisionInfluenceControls.querySelectorAll("input[data-decision-head]")) {
    range.disabled = disabled;
  }
}

function decisionInfluenceActionLabel(choice, options, index) {
  const fromSource = firstActionTranscript(
    recordValue(choice, "action_transcript", "transcript", "label", "display_name", "name"),
  );
  if (fromSource) return fromSource;
  const position = optionPositionForIndex(options, index);
  return actionNameAtPosition(options, position === undefined ? index : position);
}

function decisionInfluenceOptionPosition(options, index) {
  return optionPositionForIndex(options, index);
}

function decisionInfluenceProbabilityDelta(effect, baseline, counterfactual, position) {
  const supplied = numericVector(effect?.probability_delta);
  if (supplied && position < supplied.length) return supplied[position];
  if (baseline && counterfactual && position < baseline.length && position < counterfactual.length) {
    return counterfactual[position] - baseline[position];
  }
  return undefined;
}

function renderDecisionInfluenceControls(influence) {
  const heads = decisionInfluenceEligibleHeads(influence);
  state.decisionInfluence.eligibleHeads = heads;
  const [minimum, maximum] = decisionInfluenceBounds(influence);
  const defaultScale = decisionInfluenceDefaultScale(influence);
  clearNode(elements.decisionInfluenceControls);
  for (const name of heads) {
    const scale = Math.min(maximum, Math.max(minimum, scaleForDecisionHead(name, influence)));
    state.decisionInfluence.scales[name] = scale;
    const control = document.createElement("label");
    control.className = "decision-influence-control";
    const title = document.createElement("span");
    title.textContent = name;
    const range = document.createElement("input");
    range.type = "range";
    range.min = String(minimum);
    range.max = String(maximum);
    range.step = "0.05";
    range.value = String(scale);
    range.dataset.decisionHead = name;
    range.setAttribute("aria-label", `${name} decision-only hypothetical scale; ${formatNumber(defaultScale, 2)}× is the exact baseline`);
    const value = document.createElement("output");
    value.dataset.decisionOutput = name;
    value.textContent = `${scale.toFixed(2)}×${Math.abs(scale - defaultScale) < 1e-9 ? " baseline" : " hypothetical"}`;
    const baselineWeight = baselineHeadWeight(name, influence);
    const coefficient = finiteNumber(baselineWeight?.nominal_policy_coefficient);
    const reliability = finiteNumber(baselineWeight?.learned_route_multiplier);
    const routeCount = finiteNumber(baselineWeight?.shared_active_route_count);
    const cap = finiteNumber(baselineWeight?.shared_total_delta_cap);
    const detail = document.createElement("span");
    detail.className = "decision-influence-baseline-weight";
    detail.textContent = coefficient === undefined
      ? "Baseline head coefficient unavailable for this submitted Fusion runtime."
      : `Baseline head coefficient: ${formatNumber(coefficient, 6)} = cap ${formatNumber(cap, 4)} × learned reliability ${formatNumber(reliability, 4)} ÷ ${formatNumber(routeCount, 0)} active heads. Final policy effect is nonlinear and decision-specific.`;
    control.append(title, range, value, detail);
    elements.decisionInfluenceControls.append(control);
  }
}

function renderDecisionInfluenceResult(influence, trace) {
  const baseline = isObject(influence?.baseline) ? influence.baseline : null;
  const counterfactual = isObject(influence?.counterfactual) ? influence.counterfactual : null;
  const effect = isObject(influence?.effect) ? influence.effect : {};
  const options = collectionFrom(trace, "legal_options");
  const baselineProbabilities = numericVector(baseline?.probabilities);
  const changedProbabilities = numericVector(counterfactual?.probabilities);
  const count = Math.max(
    baselineProbabilities?.length || 0,
    changedProbabilities?.length || 0,
    options.length,
  );
  const baselineIndex = recordValue(baseline, "selected_option_index");
  const changedIndex = recordValue(counterfactual, "selected_option_index");
  const baselineLabel = decisionInfluenceActionLabel(baseline?.selected_option, options, baselineIndex);
  const changedLabel = decisionInfluenceActionLabel(counterfactual?.selected_option, options, changedIndex);
  const actionChanged = recordValue(effect, "selected_action_changed");
  const sourceSign = recordValue(effect, "sign_convention");
  const reproductionStatus = recordValue(influence, "reproduction_status");
  const shift = finiteNumber(recordValue(effect, "maximum_absolute_probability_shift"));
  const distance = finiteNumber(recordValue(effect, "total_variation_distance"));
  const helped = isObject(effect.most_helped_option) ? effect.most_helped_option : null;
  const hurt = isObject(effect.most_hurt_option) ? effect.most_hurt_option : null;
  const helpedLabel = helped
    ? decisionInfluenceActionLabel(helped, options, recordValue(helped, "index", "option_index"))
    : undefined;
  const hurtLabel = hurt
    ? decisionInfluenceActionLabel(hurt, options, recordValue(hurt, "index", "option_index"))
    : undefined;

  const choiceParts = [];
  choiceParts.push(`Exact nonlinear 1× causal baseline chooses ${baselineLabel}.`);
  choiceParts.push(`Hypothetical decision chooses ${changedLabel}.`);
  if (reproductionStatus === "recomputed_not_historical") {
    choiceParts.push("This result is recomputed, not historical.");
  }
  if (actionChanged === true) choiceParts.push("The chosen legal action changed.");
  if (actionChanged === false) choiceParts.push("The chosen legal action did not change.");
  if (sourceSign === "counterfactual_minus_baseline") {
    choiceParts.push("Differences are counterfactual minus baseline.");
  }
  if (shift !== undefined) choiceParts.push(`Largest probability shift: ${signedPercentagePoints(shift).replace("+", "").replace("−", "")}.`);
  if (distance !== undefined) choiceParts.push(`Total variation distance: ${formatProbability(distance)}.`);
  if (helpedLabel) choiceParts.push(`Most helped: ${helpedLabel}${finiteNumber(helped.probability_delta) === undefined ? "" : ` (${signedPercentagePoints(finiteNumber(helped.probability_delta))})`}.`);
  if (hurtLabel) choiceParts.push(`Most hurt: ${hurtLabel}${finiteNumber(hurt.probability_delta) === undefined ? "" : ` (${signedPercentagePoints(finiteNumber(hurt.probability_delta))})`}.`);
  elements.decisionInfluenceChoice.textContent = choiceParts.join(" ");

  if (!count || !baselineProbabilities || !changedProbabilities) {
    setEmptyRow(
      elements.decisionInfluenceBody,
      5,
      "The source returned an exact counterfactual contract without complete baseline and changed probability vectors.",
    );
  } else {
    clearNode(elements.decisionInfluenceBody);
    for (let position = 0; position < count; position += 1) {
      const row = document.createElement("tr");
      const option = options[position];
      const optionIndex = optionIndexAtPosition(options, position);
      const action = document.createElement("td");
      action.textContent = friendlyActionTranscript(option) || `legal option #${formatValue(optionIndex === undefined ? position : optionIndex)}`;
      const baselineCell = document.createElement("td");
      baselineCell.textContent = formatProbability(baselineProbabilities[position]);
      const changedCell = document.createElement("td");
      changedCell.textContent = formatProbability(changedProbabilities[position]);
      const deltaCell = document.createElement("td");
      const delta = decisionInfluenceProbabilityDelta(effect, baselineProbabilities, changedProbabilities, position);
      deltaCell.textContent = delta === undefined ? "Probability delta not supplied" : signedPercentagePoints(delta);
      const chosenCell = document.createElement("td");
      const baselinePosition = decisionInfluenceOptionPosition(options, baselineIndex);
      const changedPosition = decisionInfluenceOptionPosition(options, changedIndex);
      if (position === baselinePosition) chosenCell.append(flag("Baseline", "recorded"));
      if (position === changedPosition) chosenCell.append(flag("Hypothetical", "model"));
      if (!chosenCell.childNodes.length) chosenCell.textContent = "—";
      row.append(action, baselineCell, changedCell, deltaCell, chosenCell);
      elements.decisionInfluenceBody.append(row);
    }
  }
  clearNode(elements.decisionInfluenceDetail);
  elements.decisionInfluenceDetail.append(detailsBlock("View exact decision counterfactual source payload", influence));
}

function renderDecisionInfluenceUnavailable(reason) {
  elements.decisionInfluenceAvailability.textContent = reason;
  elements.decisionInfluenceReset.disabled = true;
  elements.decisionInfluenceApply.disabled = true;
  elements.decisionInfluenceDebounce.textContent = "No counterfactual request was sent.";
  clearNode(elements.decisionInfluenceControls);
  const empty = document.createElement("p");
  empty.className = "empty-state";
  empty.textContent = reason;
  elements.decisionInfluenceControls.append(empty);
  elements.decisionInfluenceChoice.textContent = "No baseline or changed decision is shown.";
  setEmptyRow(elements.decisionInfluenceBody, 5, reason);
  clearNode(elements.decisionInfluenceDetail);
}

function renderDecisionInfluenceAwaiting(trace) {
  const sourceInfluence = decisionInfluenceEnvelope(trace);
  if (!decisionInfluenceAvailable(sourceInfluence)) {
    renderDecisionInfluenceUnavailable(decisionInfluenceReason(sourceInfluence));
    return;
  }
  const activeInfluence = decisionInfluenceAvailable(state.decisionInfluence.payload)
    ? state.decisionInfluence.payload
    : sourceInfluence;
  const heads = decisionInfluenceEligibleHeads(sourceInfluence);
  if (!heads.length) {
    renderDecisionInfluenceUnavailable("The exact source returned no eligible fused heads for this decision.");
    return;
  }
  renderDecisionInfluenceControls(sourceInfluence);
  elements.decisionInfluenceAvailability.textContent = "Exact nonlinear causal re-evaluation · recomputed, not historical";
  elements.decisionInfluenceReset.disabled = false;
  elements.decisionInfluenceApply.disabled = false;
  if (state.decisionInfluence.error) {
    elements.decisionInfluenceDebounce.textContent = `Could not recompute: ${state.decisionInfluence.error.message}`;
  } else if (state.decisionInfluence.debounceTimer) {
    elements.decisionInfluenceDebounce.textContent = "Changes ready; recomputing this decision in 250 ms.";
  } else {
    const parity = activeInfluence?.parity;
    elements.decisionInfluenceDebounce.textContent = parity?.all_scales_one === true
      ? "Exact 1× causal-re-evaluation baseline loaded. Adjust a scale for a decision-only hypothetical."
      : "Exact source-backed hypothetical loaded. Adjust a scale or reset to the 1× baseline.";
  }
  renderDecisionInfluenceResult(activeInfluence, trace);
}

function updateDecisionInfluenceOutput(head, value) {
  const output = elements.decisionInfluenceControls.querySelector(`output[data-decision-output="${head}"]`);
  if (!output) return;
  const defaultScale = decisionInfluenceDefaultScale(decisionInfluenceEnvelope(state.trace));
  output.textContent = `${value.toFixed(2)}×${Math.abs(value - defaultScale) < 1e-9 ? " baseline" : " hypothetical"}`;
}

function currentDecisionInfluenceScales() {
  const influence = decisionInfluenceEnvelope(state.trace);
  const [minimum, maximum] = decisionInfluenceBounds(influence);
  const scales = {};
  for (const head of state.decisionInfluence.eligibleHeads) {
    const value = finiteNumber(state.decisionInfluence.scales[head]);
    if (value === undefined) continue;
    scales[head] = Math.min(maximum, Math.max(minimum, value));
  }
  return scales;
}

function requestedDecisionInfluenceScales() {
  const influence = decisionInfluenceEnvelope(state.trace);
  const defaultScale = decisionInfluenceDefaultScale(influence);
  return Object.fromEntries(
    Object.entries(currentDecisionInfluenceScales())
      .filter(([, value]) => Math.abs(value - defaultScale) > 1e-9)
      .map(([head, value]) => [head, Number(value.toFixed(4))]),
  );
}

function decisionInfluencePath(scales) {
  const query = new URLSearchParams({ stage: String(state.stage) });
  const entries = Object.entries(scales);
  if (entries.length) {
    query.set("scales", entries.map(([head, scale]) => `${head}:${scale}`).join(","));
  }
  return `${API}/submissions/${encodeURIComponent(state.submissionId)}/games/${encodeURIComponent(state.gameId)}/steps/${encodeURIComponent(state.stepIndex)}?${query.toString()}`;
}

function resetDecisionInfluenceState() {
  if (state.decisionInfluence.debounceTimer) clearTimeout(state.decisionInfluence.debounceTimer);
  state.decisionInfluence.payload = null;
  state.decisionInfluence.error = null;
  state.decisionInfluence.scales = {};
  state.decisionInfluence.eligibleHeads = [];
  state.decisionInfluence.debounceTimer = null;
  ++state.requests.decisionInfluence;
}

function scheduleDecisionInfluenceRecompute() {
  if (!decisionInfluenceAvailable(decisionInfluenceEnvelope(state.trace))) return;
  if (state.decisionInfluence.debounceTimer) clearTimeout(state.decisionInfluence.debounceTimer);
  state.decisionInfluence.debounceTimer = window.setTimeout(() => {
    state.decisionInfluence.debounceTimer = null;
    loadDecisionInfluence();
  }, 250);
  renderDecisionInfluenceAwaiting(state.trace);
}

async function loadDecisionInfluence() {
  if (!state.trace || !decisionInfluenceAvailable(decisionInfluenceEnvelope(state.trace))) return;
  const token = ++state.requests.decisionInfluence;
  const traceAtRequest = state.trace;
  const scales = requestedDecisionInfluenceScales();
  state.decisionInfluence.error = null;
  elements.decisionInfluenceAvailability.textContent = "Recomputing exact nonlinear decision…";
  elements.decisionInfluenceDebounce.textContent = "Request is scoped to this decision only; no checkpoint or training state is changed.";
  elements.decisionInfluenceApply.disabled = true;
  try {
    const response = await fetchJson(decisionInfluencePath(scales));
    if (token !== state.requests.decisionInfluence || traceAtRequest !== state.trace) return;
    const influence = decisionInfluenceEnvelope(response);
    if (!decisionInfluenceAvailable(influence)) {
      state.decisionInfluence.payload = null;
      state.decisionInfluence.error = new Error(decisionInfluenceReason(influence));
    } else {
      state.decisionInfluence.payload = influence;
      state.decisionInfluence.error = null;
      if (isObject(influence.effective_scales)) {
        for (const head of state.decisionInfluence.eligibleHeads) {
          const effective = finiteNumber(influence.effective_scales[head]);
          if (effective !== undefined) state.decisionInfluence.scales[head] = effective;
        }
      }
    }
  } catch (error) {
    if (token !== state.requests.decisionInfluence || traceAtRequest !== state.trace) return;
    state.decisionInfluence.payload = null;
    state.decisionInfluence.error = error instanceof Error ? error : new Error("Decision counterfactual request failed.");
  }
  if (token === state.requests.decisionInfluence && traceAtRequest === state.trace) {
    renderDecisionInfluenceAwaiting(state.trace);
  }
}

function resetDecisionInfluence() {
  const influence = decisionInfluenceEnvelope(state.trace);
  if (!decisionInfluenceAvailable(influence)) return;
  const defaultScale = decisionInfluenceDefaultScale(influence);
  for (const head of decisionInfluenceEligibleHeads(influence)) {
    state.decisionInfluence.scales[head] = defaultScale;
  }
  if (state.decisionInfluence.debounceTimer) clearTimeout(state.decisionInfluence.debounceTimer);
  state.decisionInfluence.debounceTimer = null;
  state.decisionInfluence.payload = null;
  state.decisionInfluence.error = null;
  renderDecisionInfluenceAwaiting(state.trace);
  loadDecisionInfluence();
}

function trainingRecipeEnvelope(trace) {
  const candidates = [
    trace?.training_recipe,
    currentGame()?.training_recipe,
    currentSubmission()?.training_recipe,
    state.trainingRecipe?.training_recipe,
    state.trainingRecipe,
  ];
  return candidates.find((candidate) => candidate !== undefined && candidate !== null) || null;
}

function trainingRecipeWeightEntries(recipe) {
  const collections = [
    recipe?.loss_weights,
    recipe?.strategic_head_loss_weights,
    recipe?.head_loss_weights,
    recipe?.loss_multipliers,
    recipe?.weights,
    recipe?.multipliers,
  ];
  const entries = [];
  for (const source of collections) {
    if (Array.isArray(source)) {
      for (const item of source) {
        if (!isObject(item)) continue;
        const name = firstNonEmptyText(recordValue(item, "name", "head", "loss", "key", "id"));
        const weight = finiteNumber(recordValue(item, "weight", "multiplier", "value", "loss_weight"));
        if (name && weight !== undefined) entries.push([name, weight]);
      }
    } else if (isObject(source)) {
      for (const [name, value] of Object.entries(source)) {
        const direct = finiteNumber(value);
        const nested = isObject(value)
          ? finiteNumber(recordValue(value, "weight", "multiplier", "value", "loss_weight"))
          : undefined;
        if (direct !== undefined || nested !== undefined) entries.push([name, direct ?? nested]);
      }
    }
  }
  const guideWeight = isObject(recipe?.guide)
    ? finiteNumber(recordValue(recipe.guide, "loss_weight", "weight", "multiplier", "value"))
    : undefined;
  if (guideWeight !== undefined) entries.unshift(["guide", guideWeight]);
  return entries.filter(([name], index) => entries.findIndex(([other]) => other === name) === index);
}

function trainingRecipeWeightTable(entries) {
  const table = document.createElement("table");
  table.className = "data-table training-recipe-table";
  const heading = document.createElement("thead");
  heading.innerHTML = "<tr><th scope=\"col\">Training loss / head</th><th scope=\"col\">Multiplier</th></tr>";
  const body = document.createElement("tbody");
  for (const [name, value] of entries) {
    const row = document.createElement("tr");
    const nameCell = document.createElement("td");
    nameCell.textContent = name.replaceAll("_", " ");
    const valueCell = document.createElement("td");
    valueCell.textContent = `${formatNumber(value)}×`;
    row.append(nameCell, valueCell);
    body.append(row);
  }
  table.append(heading, body);
  return table;
}

function renderTrainingRecipeUnavailable(reason) {
  elements.trainingRecipeAvailability.textContent = reason;
  renderStructuredContent(elements.trainingRecipeContent, null, reason);
}

function renderTrainingRecipeAwaiting(trace) {
  const envelope = trainingRecipeEnvelope(trace);
  if (!envelope) {
    if (state.trainingRecipeError) {
      renderTrainingRecipeUnavailable(state.trainingRecipeError.message);
      return;
    }
    elements.trainingRecipeAvailability.textContent = "Exact checkpoint recipe not supplied";
    renderStructuredContent(
      elements.trainingRecipeContent,
      null,
      "An exact checkpoint-bound baseline recipe is required before displaying guide or head loss weights. Missing source-backed weights are unavailable, not zero.",
    );
    return;
  }
  if (isUnavailable(envelope)) {
    renderTrainingRecipeUnavailable(reasonFrom(envelope, "The selected checkpoint has no source-backed training recipe."));
    return;
  }
  const recipe = recordValue(envelope, "recipe") ?? envelope;
  if (
    !isObject(recipe)
    || recipe.scope !== "source_backed_training_loss_multipliers"
    || recipe.training_only !== true
  ) {
    renderTrainingRecipeUnavailable(
      "The source did not confirm a checkpoint-bound, training-only loss-multiplier recipe for this submission.",
    );
    return;
  }
  const checkpoint = firstNonEmptyText(
    recordValue(envelope, "checkpoint_sha256", "checkpoint_digest", "checkpoint_id"),
    recordValue(recipe, "checkpoint_sha256", "checkpoint_digest", "checkpoint_id"),
  );
  const entries = trainingRecipeWeightEntries(recipe);
  elements.trainingRecipeAvailability.textContent = checkpoint
    ? `Source-backed checkpoint recipe · ${checkpoint.slice(0, 16)}…`
    : "Source-backed training recipe returned";
  clearNode(elements.trainingRecipeContent);
  elements.trainingRecipeContent.className = "structured-content";
  const intro = document.createElement("p");
  intro.className = "recipe-reference-note";
  intro.textContent = "These are training-loss multipliers from the source. They are read-only reference values, not decision scales; using them requires a real fine-tune and a new checksum-distinct checkpoint.";
  elements.trainingRecipeContent.append(intro);
  const sourceSummary = firstNonEmptyText(recordValue(recipe, "source_summary", "summary", "description"));
  if (sourceSummary) {
    const summary = document.createElement("p");
    summary.className = "recipe-reference-note";
    summary.textContent = sourceSummary;
    elements.trainingRecipeContent.append(summary);
  }
  if (entries.length) {
    const heading = document.createElement("h4");
    heading.className = "subsection-title";
    heading.textContent = "Source-provided training loss multipliers";
    elements.trainingRecipeContent.append(heading, trainingRecipeWeightTable(entries));
  } else {
    const missing = document.createElement("p");
    missing.className = "inline-availability";
    missing.textContent = "The source returned recipe provenance without a readable guide or head-loss multiplier map.";
    elements.trainingRecipeContent.append(missing);
  }
  const evidence = recordValue(envelope, "evidence", "provenance", "source_evidence");
  if (evidence !== undefined) elements.trainingRecipeContent.append(detailsBlock("View recipe evidence", evidence));
  elements.trainingRecipeContent.append(detailsBlock("View complete training recipe payload", envelope));
}

async function loadTrainingRecipe(submissionId) {
  if (!submissionId) return;
  const token = ++state.requests.trainingRecipe;
  state.trainingRecipe = null;
  state.trainingRecipeError = null;
  renderTrainingRecipeAwaiting(state.trace);
  try {
    const payload = await fetchJson(`${API}/submissions/${encodeURIComponent(submissionId)}/training-recipe`);
    if (token !== state.requests.trainingRecipe || !sameId(submissionId, state.submissionId)) return;
    state.trainingRecipe = payload;
    state.trainingRecipeError = null;
  } catch (error) {
    if (token !== state.requests.trainingRecipe || !sameId(submissionId, state.submissionId)) return;
    state.trainingRecipe = null;
    state.trainingRecipeError = error instanceof Error ? error : new Error("Training recipe request failed.");
  }
  if (token === state.requests.trainingRecipe && sameId(submissionId, state.submissionId)) {
    renderTrainingRecipeAwaiting(state.trace);
  }
}

function renderWarnings(trace) {
  const warnings = collectionFrom(trace, "warnings");
  elements.warningsSection.hidden = warnings.length === 0;
  clearNode(elements.warningsList);
  for (const warning of warnings) {
    const item = document.createElement("li");
    item.textContent = typeof warning === "string" ? warning : compactJson(warning, 400);
    elements.warningsList.append(item);
  }
}

function renderWeights() {
  const submission = currentSubmission();
  if (!submission) {
    setBadge(elements.weightsStatus, "Awaiting submission", "pending");
    elements.weightsAvailability.textContent = "Awaiting submission";
    clearNode(elements.weightsProvenanceContent);
    elements.weightsProvenanceContent.className = "provenance-content empty-state";
    elements.weightsProvenanceContent.textContent = "Select a submission to inspect only its checksum-bound parameter metadata and bounded slices.";
    elements.parameterSearch.disabled = true;
    elements.parameterCount.textContent = "0";
    setEmptyRow(elements.parametersBody, 5, "Choose a submission to load its parameter index.");
    clearParameterDetail();
    return;
  }
  renderWeightsProvenance(submission);
  renderParameterInventory();
}

function renderWeightsProvenance(submission) {
  const modelAnalysis = isObject(submission.model_analysis) ? submission.model_analysis : null;
  const readiness = submissionReadiness(submission);
  const parameterPayload = state.parametersPayload;
  if (isUnavailable(parameterPayload)) {
    setBadge(elements.weightsStatus, "Weights unavailable", "unavailable");
    elements.weightsAvailability.textContent = reasonFrom(parameterPayload, "Parameter inspection is unavailable for this submission.");
  } else if (parameterPayload) {
    setBadge(elements.weightsStatus, "Parameter index loaded", "ready");
    elements.weightsAvailability.textContent = readiness.weights.available === true
      ? "Exact model weights available · parameter inventory returned by source"
      : "Parameter inventory returned by source";
  } else if (readiness.weights.available === false) {
    setBadge(elements.weightsStatus, "Weights unavailable", "unavailable");
    elements.weightsAvailability.textContent = readiness.weightsText;
  } else {
    setBadge(elements.weightsStatus, "Loading parameter index", "pending");
    elements.weightsAvailability.textContent = readiness.weights.available === true
      ? "Exact model weights available · loading parameter inventory"
      : "Loading checkpoint parameter inventory";
  }
  clearNode(elements.weightsProvenanceContent);
  elements.weightsProvenanceContent.className = "provenance-content";
  const provenance = recordValue(submission, "provenance");
  const gridSource = isObject(provenance) ? provenance : submission;
  const grid = keyValueGrid(gridSource);
  if (grid) elements.weightsProvenanceContent.append(grid);
  const panes = document.createElement("div");
  panes.className = "fusion-grid";
  const hasReadiness = appendSourceObject(panes, "Weights and decision-trace status", readiness.evidence);
  const hasAnalysis = appendSourceObject(panes, "Model analysis availability", modelAnalysis);
  const hasProvenance = appendSourceObject(panes, "Submission provenance", provenance);
  const hasParameterPayload = appendSourceObject(
    panes,
    "Parameter index response",
    parameterPayload && isObject(parameterPayload) ? omitCollection(parameterPayload, "parameters") : parameterPayload,
  );
  if (hasReadiness || hasAnalysis || hasProvenance || hasParameterPayload) elements.weightsProvenanceContent.append(panes);
  if (!hasReadiness && !hasAnalysis && !hasProvenance && !hasParameterPayload) {
    const paragraph = document.createElement("p");
    paragraph.className = "empty-state";
    paragraph.textContent = "The submission response did not include checkpoint provenance or parameter availability fields.";
    elements.weightsProvenanceContent.append(paragraph);
  }
}

function omitCollection(payload, key) {
  if (!isObject(payload)) return payload;
  const copy = { ...payload };
  if (Array.isArray(copy[key])) copy[key] = `[${copy[key].length} indexed entries]`;
  return copy;
}

function renderParameterInventory() {
  const payload = state.parametersPayload;
  if (!payload) {
    elements.parameterSearch.disabled = true;
    elements.parameterCount.textContent = "0";
    setEmptyRow(elements.parametersBody, 5, "Loading the selected submission's parameter index…");
    return;
  }
  if (isUnavailable(payload)) {
    const reason = reasonFrom(payload, "Parameter inventory is unavailable for this submission.");
    elements.parameterSearch.disabled = true;
    elements.parameterCount.textContent = "0";
    setEmptyRow(elements.parametersBody, 5, reason);
    return;
  }
  const query = elements.parameterSearch.value.trim().toLocaleLowerCase();
  const parameters = state.parameters.filter((parameter) => {
    const searchable = [parameter.name, parameter.module, parameter.dtype, parameter.shape]
      .filter((value) => value !== undefined && value !== null)
      .map((value) => (Array.isArray(value) ? value.join("x") : String(value)).toLocaleLowerCase())
      .join(" ");
    return !query || searchable.includes(query);
  });
  elements.parameterSearch.disabled = false;
  elements.parameterCount.textContent = `${parameters.length}/${state.parameters.length}`;
  if (!parameters.length) {
    setEmptyRow(
      elements.parametersBody,
      5,
      query ? "No indexed parameters match this search." : "The source returned an empty parameter inventory.",
    );
    return;
  }
  clearNode(elements.parametersBody);
  for (const parameter of parameters) {
    const row = document.createElement("tr");
    const nameCell = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "parameter-name-button";
    button.dataset.parameterName = String(parameter.name ?? "");
    button.textContent = formatValue(parameter.name);
    if (sameId(parameter.name, state.parameterName)) button.classList.add("is-selected");
    nameCell.append(button);
    row.append(nameCell);
    appendCell(row, Array.isArray(parameter.shape) ? `[${parameter.shape.join(", ")}]` : parameter.shape, "mono-value");
    appendCell(row, parameter.dtype, "mono-value");
    appendCell(row, recordValue(parameter, "numel", "element_count", "elements"), "number");
    appendCell(row, asBooleanText(parameter.trainable));
    elements.parametersBody.append(row);
  }
}

function statsValue(stats, ...keys) {
  return recordValue(stats, ...keys);
}

function renderParameterDetail() {
  const payload = state.parameterDetail;
  if (state.parameterError) {
    elements.parameterDetailStatus.textContent = "Parameter request failed";
    elements.parameterDetail.className = "parameter-detail empty-state";
    elements.parameterDetail.textContent = state.parameterError.message;
    return;
  }
  if (!payload) return;
  if (isUnavailable(payload) || isUnavailable(payload.parameter)) {
    const reason = reasonFrom(isUnavailable(payload.parameter) ? payload.parameter : payload, "Parameter detail is unavailable.");
    elements.parameterDetailStatus.textContent = reason;
    elements.parameterDetail.className = "parameter-detail empty-state";
    elements.parameterDetail.textContent = reason;
    return;
  }
  const parameter = payload.parameter;
  if (!parameter || typeof parameter !== "object") {
    elements.parameterDetailStatus.textContent = "Parameter response incomplete";
    elements.parameterDetail.className = "parameter-detail empty-state";
    elements.parameterDetail.textContent = "The source did not provide a parameter payload.";
    return;
  }
  elements.parameterDetailStatus.textContent = "Bounded detail loaded";
  elements.parameterDetail.className = "parameter-detail";
  clearNode(elements.parameterDetail);
  const title = document.createElement("p");
  title.className = "detail-title";
  title.textContent = formatValue(parameter.name);
  elements.parameterDetail.append(title);

  const metadata = {
    shape: Array.isArray(parameter.shape) ? `[${parameter.shape.join(", ")}]` : parameter.shape,
    dtype: parameter.dtype,
    elements: recordValue(parameter, "numel", "element_count", "elements"),
    trainable: parameter.trainable,
    module: parameter.module,
  };
  const metadataGrid = keyValueGrid(metadata);
  if (metadataGrid) elements.parameterDetail.append(metadataGrid);

  const stats = isObject(parameter.stats) ? parameter.stats : {};
  const statEntries = [
    ["Finite count", statsValue(stats, "finite_count", "finite")],
    ["Minimum", statsValue(stats, "minimum", "min")],
    ["Maximum", statsValue(stats, "maximum", "max")],
    ["Mean", statsValue(stats, "mean")],
    ["Std. deviation", statsValue(stats, "standard_deviation", "std", "stddev")],
    ["L1 norm", statsValue(stats, "l1_norm", "l1")],
    ["L2 norm", statsValue(stats, "l2_norm", "l2")],
    ["Zero fraction", statsValue(stats, "zero_fraction", "zeros")],
  ].filter(([, value]) => value !== undefined && value !== null);
  if (statEntries.length) {
    const statsGrid = document.createElement("div");
    statsGrid.className = "stats-grid";
    for (const [label, value] of statEntries) {
      const stat = document.createElement("div");
      stat.className = "stat";
      const statLabel = document.createElement("span");
      statLabel.textContent = label;
      const statValue = document.createElement("b");
      statValue.textContent = formatValue(value);
      stat.append(statLabel, statValue);
      statsGrid.append(stat);
    }
    elements.parameterDetail.append(statsGrid);
  } else {
    const statsMissing = document.createElement("p");
    statsMissing.className = "inline-availability";
    statsMissing.textContent = "Summary statistics were not supplied for this parameter.";
    elements.parameterDetail.append(statsMissing);
  }

  elements.parameterDetail.append(renderHistogram(parameter.histogram));
  elements.parameterDetail.append(renderSlice(parameter.slice));
  elements.parameterDetail.append(detailsBlock("View complete bounded parameter response", payload));
}

function histogramRows(histogram) {
  if (Array.isArray(histogram)) {
    return histogram.map((bin, index) => ({
      label: bin?.label ?? rangeLabel(bin?.min ?? bin?.lower ?? bin?.start, bin?.max ?? bin?.upper ?? bin?.end, index),
      count: bin?.count ?? bin?.value ?? bin,
    }));
  }
  if (!isObject(histogram)) return [];
  if (Array.isArray(histogram.bins)) {
    return histogram.bins.map((bin, index) => ({
      label: bin?.label ?? rangeLabel(bin?.min ?? bin?.lower ?? bin?.start, bin?.max ?? bin?.upper ?? bin?.end, index),
      count: bin?.count ?? bin?.value ?? bin,
    }));
  }
  if (Array.isArray(histogram.counts)) {
    return histogram.counts.map((count, index) => ({
      label: rangeLabel(histogram.edges?.[index], histogram.edges?.[index + 1], index),
      count,
    }));
  }
  return [];
}

function rangeLabel(lower, upper, index) {
  if (lower === undefined && upper === undefined) return `bin ${index}`;
  if (upper === undefined) return `≥ ${formatValue(lower)}`;
  if (lower === undefined) return `≤ ${formatValue(upper)}`;
  return `${formatValue(lower)} – ${formatValue(upper)}`;
}

function renderHistogram(histogram) {
  const section = document.createElement("section");
  section.className = "histogram-section";
  const heading = document.createElement("div");
  heading.className = "subsection-title";
  const title = document.createElement("h4");
  title.textContent = "Bounded histogram";
  heading.append(title);
  section.append(heading);
  if (isUnavailable(histogram)) {
    const message = document.createElement("p");
    message.className = "inline-availability";
    message.textContent = reasonFrom(histogram, "Histogram is unavailable for this parameter.");
    section.append(message);
    return section;
  }
  const rows = histogramRows(histogram);
  if (!rows.length) {
    const message = document.createElement("p");
    message.className = "inline-availability";
    message.textContent = "A bounded histogram was not supplied for this parameter.";
    section.append(message);
    return section;
  }
  const counts = rows.map((row) => (typeof row.count === "number" && Number.isFinite(row.count) ? row.count : 0));
  const maximum = Math.max(...counts, 0);
  const bars = document.createElement("div");
  bars.className = "histogram-bars";
  for (const row of rows) {
    const line = document.createElement("div");
    line.className = "histogram-row";
    const range = document.createElement("span");
    range.className = "histogram-range";
    range.textContent = row.label;
    const track = document.createElement("div");
    track.className = "histogram-track";
    const fill = document.createElement("div");
    fill.className = "histogram-fill";
    const count = typeof row.count === "number" && Number.isFinite(row.count) ? row.count : 0;
    fill.style.width = `${maximum > 0 ? (count / maximum) * 100 : 0}%`;
    track.append(fill);
    const countLabel = document.createElement("span");
    countLabel.className = "histogram-count";
    countLabel.textContent = formatValue(row.count);
    line.append(range, track, countLabel);
    bars.append(line);
  }
  section.append(bars);
  return section;
}

function renderSlice(slice) {
  const section = document.createElement("section");
  section.className = "slice-section";
  const heading = document.createElement("div");
  heading.className = "subsection-title";
  const title = document.createElement("h4");
  title.textContent = "Bounded tensor slice";
  heading.append(title);
  section.append(heading);
  if (isUnavailable(slice)) {
    const message = document.createElement("p");
    message.className = "inline-availability";
    message.textContent = reasonFrom(slice, "Tensor slice is unavailable for this parameter.");
    section.append(message);
    return section;
  }
  if (!slice || typeof slice !== "object") {
    const message = document.createElement("p");
    message.className = "inline-availability";
    message.textContent = "A bounded tensor slice was not supplied for this parameter.";
    section.append(message);
    return section;
  }
  const values = firstDefined(slice.values, slice.data, slice.elements);
  const offset = Number(firstDefined(slice.offset, state.parameterOffset, 0));
  const total = Number(firstDefined(slice.total, slice.total_elements));
  const length = Array.isArray(values) ? values.length : 0;
  const controls = document.createElement("div");
  controls.className = "slice-controls";
  const status = document.createElement("span");
  status.className = "slice-status";
  status.textContent = Number.isFinite(total)
    ? `offset ${formatNumber(offset)} · ${formatNumber(length)} returned · total ${formatNumber(total)}`
    : `offset ${formatNumber(offset)} · ${formatNumber(length)} returned`;
  const actions = document.createElement("div");
  actions.className = "slice-actions";
  const previous = document.createElement("button");
  previous.type = "button";
  previous.className = "button button-quiet";
  previous.dataset.sliceOffset = String(Math.max(0, offset - SLICE_LIMIT));
  previous.disabled = offset <= 0;
  previous.textContent = "Previous";
  const next = document.createElement("button");
  next.type = "button";
  next.className = "button button-quiet";
  next.dataset.sliceOffset = String(offset + Math.max(length, SLICE_LIMIT));
  next.disabled = length === 0 || (Number.isFinite(total) && offset + length >= total);
  next.textContent = "Next";
  actions.append(previous, next);
  controls.append(status, actions);
  section.append(controls);
  if (values === undefined || values === null) {
    const message = document.createElement("p");
    message.className = "inline-availability";
    message.textContent = "The source returned slice metadata without slice values.";
    section.append(message);
  } else {
    const pre = document.createElement("pre");
    pre.className = "json-block slice-values";
    pre.textContent = fullJson(values);
    section.append(pre);
  }
  return section;
}

async function refreshIndex() {
  const token = ++state.requests.index;
  elements.refreshButton.disabled = true;
  setReplayLinkControlsEnabled(false);
  setReplayLinkStatus("Refreshing the replay index…", "working");
  setGlobalNotice("Refreshing inspector index…", "working");
  const [healthResult, submissionsResult] = await Promise.allSettled([
    fetchJson(`${API}/health`),
    fetchJson(`${API}/submissions`),
  ]);
  if (token !== state.requests.index) return;
  elements.refreshButton.disabled = false;
  if (healthResult.status === "fulfilled") {
    state.health = healthResult.value;
    state.healthError = null;
  } else {
    state.health = null;
    state.healthError = healthResult.reason instanceof Error ? healthResult.reason : new Error("Health request failed.");
  }
  renderHealth();
  if (submissionsResult.status !== "fulfilled") {
    state.submissions = [];
    state.submissionId = "";
    resetSelect(elements.submissionSelect, "Submission index unavailable", true);
    resetSubmissionFilter("Submission index unavailable", true);
    setReplayLinkStatus("Quick find is unavailable because the submission index could not be loaded.", "error");
    resetSelect(elements.gameSelect, "Choose a submission first", true);
    resetGameFilter("Submission index unavailable", true);
    resetSelect(elements.stepSelect, "Choose a game first", true);
    resetSelect(elements.stageSelect, "Choose a decision step first", true);
    clearTrace("The submission index could not be loaded.");
    state.parametersPayload = null;
    state.parameters = [];
    renderWeights();
    const error = submissionsResult.reason instanceof Error ? submissionsResult.reason.message : "Submission index request failed.";
    setGlobalNotice(error, "error");
    return;
  }
  const submissionsPayload = submissionsResult.value;
  state.submissions = collectionFrom(submissionsPayload, "submissions").sort(newestSubmissionFirst);
  const previous = state.submissionId;
  const nextSubmission = state.submissions.find((submission) => sameId(submissionIdOf(submission), previous)) || state.submissions[0];
  state.submissionId = nextSubmission ? submissionIdOf(nextSubmission) || "" : "";
  renderFilteredSubmissionSelect();
  if (!state.submissions.length) {
    setReplayLinkStatus("Quick find is unavailable because no submissions are indexed.", "error");
    resetSelect(elements.gameSelect, "No games available", true);
    resetGameFilter("No games are available to search.", true);
    resetSelect(elements.stepSelect, "No steps available", true);
    resetStepFilter("No decision steps are available.", true);
    resetSelect(elements.stageSelect, "No stages available", true);
    state.games = [];
    state.steps = [];
    state.parameters = [];
    state.parametersPayload = { available: false, reason: reasonFrom(submissionsPayload, "No indexed submissions were returned.") };
    clearTrace("No indexed submissions were returned by the source.");
    renderWeights();
    setGlobalNotice("No submissions are currently indexed.");
    return;
  }
  setReplayLinkControlsEnabled(true);
  setReplayLinkStatus("Paste the replay link; quick find uses its submissionId and episodeId.");
  setGlobalNotice("");
  updateSelectionContext();
  await selectSubmission(state.submissionId, { retainGame: true });
}

async function waitForReplaySync() {
  const started = Date.now();
  let sawRunning = false;
  while (Date.now() - started < 15 * 60 * 1000) {
    await new Promise((resolve) => window.setTimeout(resolve, 3000));
    try {
      const payload = await fetchJson(`${API}/sync-status`);
      const running = recordValue(payload, "running") === true;
      sawRunning ||= running;
      if (running) {
        setGlobalNotice("Checking Kaggle for new submissions and episodes…", "working");
        continue;
      }
      if (sawRunning || Date.now() - started >= 10000) {
        setGlobalNotice("Kaggle check finished. Refreshing the replay index…", "working");
        await refreshIndex();
        elements.syncNowButton.disabled = false;
        return;
      }
    } catch (_error) {
      // A successful sync may briefly restart the inspector. Keep polling the
      // same-origin gateway until it returns or the bounded wait expires.
    }
  }
  elements.syncNowButton.disabled = false;
  setGlobalNotice("The Kaggle check is still running. You can refresh the index later.");
}

async function requestReplaySync() {
  elements.syncNowButton.disabled = true;
  setGlobalNotice("Requesting an on-demand Kaggle replay check…", "working");
  try {
    const response = await fetch(`${API}/sync`, {
      method: "POST",
      headers: { "X-Replay-Sync-Intent": "manual" },
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Kaggle check request failed (${response.status})`);
    const payload = await response.json();
    if (recordValue(payload, "accepted") !== true) throw new Error("Kaggle check was not accepted");
    setGlobalNotice("Kaggle check accepted. Waiting for new replay downloads…", "working");
    void waitForReplaySync();
  } catch (error) {
    elements.syncNowButton.disabled = false;
    setGlobalNotice(error instanceof Error ? error.message : "Kaggle check request failed.", "error");
  }
}

async function selectSubmission(submissionId, { retainGame = false, targetGameId = "" } = {}) {
  clearPendingSubmissionFilterLoad();
  const submission = state.submissions.find((item) => sameId(submissionIdOf(item), submissionId));
  if (!submission) return "submission_not_found";
  const token = ++state.requests.submission;
  const retainedGameId = retainGame ? state.gameId : "";
  const requestedGameId = targetGameId === "" ? undefined : strictDecimalIdentifier(targetGameId);
  let requestedGameMissing = false;
  ++state.requests.game;
  ++state.requests.trace;
  ++state.requests.parameter;
  ++state.requests.trainingRecipe;
  state.trainingRecipe = null;
  state.trainingRecipeError = null;
  clearGameTraceCache();
  state.submissionId = submissionIdOf(submission) || "";
  elements.submissionSelect.value = state.submissionId;
  state.games = [];
  state.gameId = "";
  resetGameFilter("Loading games for the selected submission…", true);
  state.steps = [];
  state.stepIndex = "";
  state.stage = "";
  state.parameters = [];
  state.parametersPayload = null;
  resetSelect(elements.gameSelect, "Loading games…", true);
  resetSelect(elements.stepSelect, "Choose a game first", true);
  resetStepFilter("Choose a game to enter a step number.", true);
  resetSelect(elements.stageSelect, "Choose a decision step first", true);
  clearTrace("Loading games for the selected submission…");
  void loadTrainingRecipe(state.submissionId);
  clearParameterDetail();
  renderWeights();
  updateSelectionContext();
  setGlobalNotice("Loading replay and checkpoint indexes…", "working");
  const encodedSubmission = encodeURIComponent(state.submissionId);
  const [gamesResult, parametersResult] = await Promise.allSettled([
    fetchJson(`${API}/submissions/${encodedSubmission}/games`),
    fetchJson(`${API}/submissions/${encodedSubmission}/parameters`),
  ]);
  if (token !== state.requests.submission) return "superseded";

  if (gamesResult.status === "fulfilled") {
    state.games = collectionFrom(gamesResult.value, "games").sort(newestGameFirst);
    const desiredGameId = requestedGameId !== undefined ? requestedGameId : retainedGameId;
    const exactGame = desiredGameId === ""
      ? undefined
      : state.games.find((item) => sameId(gameIdOf(item), desiredGameId));
    requestedGameMissing = requestedGameId !== undefined && !exactGame;
    const game = requestedGameMissing ? undefined : exactGame || state.games[0];
    state.gameId = game ? gameIdOf(game) || "" : "";
    renderFilteredGameSelect();
  } else {
    state.games = [];
    state.gameId = "";
    resetSelect(elements.gameSelect, "Game index unavailable", true);
    resetGameFilter("Game index unavailable", true);
  }

  if (parametersResult.status === "fulfilled") {
    state.parametersPayload = parametersResult.value;
    state.parameters = collectionFrom(parametersResult.value, "parameters");
  } else {
    const message = parametersResult.reason instanceof Error ? parametersResult.reason.message : "Parameter index request failed.";
    state.parametersPayload = { available: false, reason: message };
    state.parameters = [];
  }
  renderWeights();
  updateSelectionContext();
  if (requestedGameMissing) {
    resetSelect(elements.stepSelect, "Linked episode is not indexed", true);
    resetStepFilter("The linked episode is not indexed for this submission.", true);
    resetSelect(elements.stageSelect, "No linked episode selected", true);
    clearTrace(`Episode ${requestedGameId} is not indexed for submission ${state.submissionId}.`);
    setGlobalNotice(`Episode ${requestedGameId} is not indexed for submission ${state.submissionId}.`, "error");
    return "game_not_found";
  }
  if (!state.games.length) {
    resetSelect(elements.stepSelect, "No games available", true);
    resetStepFilter("No games are available.", true);
    resetSelect(elements.stageSelect, "No stages available", true);
    resetGameFilter("No games are available to search.", true);
    clearTrace(
      gamesResult.status === "fulfilled"
        ? reasonFrom(gamesResult.value, "No games were indexed for this submission.")
        : (gamesResult.reason instanceof Error ? gamesResult.reason.message : "Game index request failed."),
    );
    setGlobalNotice("Loaded checkpoint index; no replay game is available for this submission.");
    return gamesResult.status === "fulfilled" ? "no_games" : "game_index_unavailable";
  }
  setGlobalNotice("");
  await selectGame(state.gameId, { retainStep: retainGame });
  return "selected";
}

async function quickFindReplay(rawLink) {
  const address = replayAddressFromLink(rawLink);
  if (!address) {
    setReplayLinkStatus("Paste a replay link containing decimal submissionId and episodeId values.", "error");
    return;
  }
  const submission = state.submissions.find((item) => sameId(submissionIdOf(item), address.submissionId));
  if (!submission) {
    setReplayLinkStatus(`Submission ${address.submissionId} is not indexed. Use Check Kaggle now, then retry.`, "error");
    return;
  }

  setReplayLinkControlsEnabled(false);
  setReplayLinkStatus(`Opening submission ${address.submissionId}, episode ${address.episodeId}…`, "working");
  resetSubmissionFilter("Opening linked submission…", false);
  renderFilteredSubmissionSelect();
  try {
    const outcome = await selectSubmission(address.submissionId, { targetGameId: address.episodeId });
    if (
      outcome === "selected"
      && sameId(state.submissionId, address.submissionId)
      && sameId(state.gameId, address.episodeId)
    ) {
      setReplayLinkStatus(`Opened submission ${address.submissionId}, episode ${address.episodeId}.`, "success");
    } else if (outcome === "game_not_found") {
      setReplayLinkStatus(`Episode ${address.episodeId} is not indexed for submission ${address.submissionId}. Use Check Kaggle now, then retry.`, "error");
    } else if (outcome !== "superseded") {
      setReplayLinkStatus(`Could not open episode ${address.episodeId} for submission ${address.submissionId}.`, "error");
    }
  } finally {
    setReplayLinkControlsEnabled(state.submissions.length > 0);
  }
}

async function selectGame(gameId, { retainStep = false } = {}) {
  clearPendingGameFilterLoad();
  const game = state.games.find((item) => sameId(gameIdOf(item), gameId));
  if (!game || !state.submissionId) return;
  const token = ++state.requests.game;
  clearGameTraceCache();
  ++state.requests.trace;
  state.gameId = gameIdOf(game) || "";
  elements.gameSelect.value = state.gameId;
  const previousStep = retainStep ? state.stepIndex : "";
  state.steps = [];
  state.stepIndex = "";
  state.stage = "";
  resetSelect(elements.stepSelect, "Loading decision steps…", true);
  resetStepFilter("Loading decision steps…", true);
  resetSelect(elements.stageSelect, "Choose a decision step first", true);
  clearTrace("Loading the selected game's decision index…");
  updateSelectionContext();
  setGlobalNotice("Loading decision steps…", "working");
  try {
    const path = `${API}/submissions/${encodeURIComponent(state.submissionId)}/games/${encodeURIComponent(state.gameId)}/steps`;
    const payload = await fetchJson(path);
    if (token !== state.requests.game) return;
    state.steps = collectionFrom(payload, "steps");
    const step = state.steps.find((item) => sameId(stepIdOf(item), previousStep)) || state.steps[0];
    state.stepIndex = step ? stepIdOf(step) || "" : "";
    renderFilteredStepSelect();
    updateSelectionContext();
    if (!state.steps.length) {
      resetSelect(elements.stageSelect, "No stages available", true);
      clearTrace(reasonFrom(payload, "No decision steps were indexed for this game."));
      setGlobalNotice("No decision steps are available for this game.");
      return;
    }
    setGlobalNotice("");
    await selectStep(state.stepIndex);
  } catch (error) {
    if (token !== state.requests.game) return;
    const message = error instanceof Error ? error.message : "Decision step index request failed.";
    state.steps = [];
    state.stepIndex = "";
    resetSelect(elements.stepSelect, "Decision index unavailable", true);
    resetStepFilter("Decision index unavailable", true);
    resetSelect(elements.stageSelect, "Stage unavailable", true);
    clearTrace(message);
    setGlobalNotice(message, "error");
  }
}

async function selectStep(stepIndex) {
  clearPendingStepFilterLoad();
  const step = state.steps.find((item) => sameId(stepIdOf(item), stepIndex));
  if (!step) return;
  const previousStage = state.stage;
  ++state.requests.trace;
  state.stepIndex = stepIdOf(step) || "";
  elements.stepSelect.value = state.stepIndex;
  populateStages(step, previousStage);
  updateSelectionContext();
  if (state.stage === "") {
    clearTrace("The selected step does not state a factorized stage, so it cannot be addressed by this inspector API.");
    return;
  }
  await loadTrace();
}

async function selectStage(stage) {
  if (stage === "") return;
  state.stage = String(stage);
  elements.stageSelect.value = state.stage;
  updateSelectionContext();
  await loadTrace();
}

async function loadTrace() {
  if (!state.submissionId || !state.gameId || state.stepIndex === "" || state.stage === "") return;
  const token = ++state.requests.trace;
  abortTraceFetchesExcept(baseTraceKey(state.stepIndex, state.stage));
  resetDecisionInfluenceState();
  state.trace = null;
  state.traceError = null;
  renderTrace("Reconstructing the selected model causally from replay state…");
  setGlobalNotice("Loading causal model re-evaluation…", "working");
  try {
    const trace = await fetchBaseTraceCached(state.stepIndex, state.stage);
    if (token !== state.requests.trace) return;
    state.trace = trace;
    state.traceError = null;
    renderTrace();
    setGlobalNotice(isUnavailable(trace) ? reasonFrom(trace, "Trace unavailable.") : "");
  } catch (error) {
    if (token !== state.requests.trace) return;
    state.trace = null;
    state.traceError = error instanceof Error ? error : new Error("Trace request failed.");
    renderTrace();
    setGlobalNotice(state.traceError.message, "error");
  }
}

async function loadParameter(name, offset = 0) {
  if (!state.submissionId || !name) return;
  const token = ++state.requests.parameter;
  state.parameterName = name;
  state.parameterOffset = Math.max(0, Number(offset) || 0);
  state.parameterDetail = null;
  state.parameterError = null;
  elements.parameterDetailStatus.textContent = "Loading bounded detail";
  elements.parameterDetail.className = "parameter-detail empty-state";
  elements.parameterDetail.textContent = "Loading parameter statistics, bounded histogram, and bounded slice…";
  renderParameterInventory();
  const path = `${API}/submissions/${encodeURIComponent(state.submissionId)}/parameters/${encodeURIComponent(name)}?offset=${encodeURIComponent(state.parameterOffset)}&limit=${SLICE_LIMIT}`;
  try {
    const payload = await fetchJson(path);
    if (token !== state.requests.parameter) return;
    state.parameterDetail = payload;
    state.parameterError = null;
    renderParameterDetail();
  } catch (error) {
    if (token !== state.requests.parameter) return;
    state.parameterDetail = null;
    state.parameterError = error instanceof Error ? error : new Error("Parameter detail request failed.");
    renderParameterDetail();
  }
}

function switchView(view) {
  state.activeView = view;
  const showingDecision = view === "decision";
  elements.decisionView.hidden = !showingDecision;
  elements.weightsView.hidden = showingDecision;
  for (const tab of elements.tabs) {
    const active = tab.dataset.view === view;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  }
  if (!showingDecision) renderWeights();
}

function bindEvents() {
  elements.replayLinkForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void quickFindReplay(elements.replayLinkInput.value);
  });
  elements.syncNowButton.addEventListener("click", () => requestReplaySync());
  elements.refreshButton.addEventListener("click", () => refreshIndex());
  elements.submissionSelect.addEventListener("change", (event) => selectSubmission(event.target.value));
  elements.submissionSearch.addEventListener("input", (event) => applySubmissionFilter(event.target.value));
  elements.gameSearch.addEventListener("input", (event) => applyGameFilter(event.target.value));
  elements.gameSelect.addEventListener("change", (event) => selectGame(event.target.value));
  elements.stepSelect.addEventListener("change", (event) => selectStep(event.target.value));
  elements.stepSearch.addEventListener("input", (event) => applyStepFilter(event.target.value));
  elements.stageSelect.addEventListener("change", (event) => selectStage(event.target.value));
  elements.guideShadowToggle.addEventListener("change", (event) => {
    state.guideShadowEnabled = event.target.checked;
    renderGuideShadow(state.trace || {});
  });
  elements.decisionInfluenceControls.addEventListener("input", (event) => {
    const range = event.target.closest("input[data-decision-head]");
    if (!range || range.disabled) return;
    const value = finiteNumber(range.value);
    if (value === undefined) return;
    state.decisionInfluence.scales[range.dataset.decisionHead] = value;
    state.decisionInfluence.payload = null;
    state.decisionInfluence.error = null;
    updateDecisionInfluenceOutput(range.dataset.decisionHead, value);
    scheduleDecisionInfluenceRecompute();
  });
  elements.decisionInfluenceReset.addEventListener("click", () => resetDecisionInfluence());
  elements.decisionInfluenceApply.addEventListener("click", () => {
    if (state.decisionInfluence.debounceTimer) clearTimeout(state.decisionInfluence.debounceTimer);
    state.decisionInfluence.debounceTimer = null;
    loadDecisionInfluence();
  });
  elements.tabs.forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
  elements.parameterSearch.addEventListener("input", () => renderParameterInventory());
  elements.parametersBody.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-parameter-name]");
    if (!button) return;
    loadParameter(button.dataset.parameterName);
  });
  elements.parameterDetail.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-slice-offset]");
    if (!button || button.disabled || !state.parameterName) return;
    loadParameter(state.parameterName, Number(button.dataset.sliceOffset));
  });
}

bindEvents();
renderTrace();
renderWeights();
refreshIndex();
