// Private evaluation-only V3 successor-arena overlay.
//
// This translation unit is compiled only beside a licensed private engine
// snapshot.  It intentionally has no connection to the competition C ABI.
// It advances only a cloned, selected action through the reviewed dynamic
// provenance seam.  Any private-card identity, unclassified random source,
// unsupported chance shape, or non-owned selection remains fail-closed.

#include "BattleData.h"
#include "CardImpl.h"
#include "InitializeCard.h"
#include "ToJson.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _MSC_VER
#  define RTP_PLANNER_V3_API __declspec(dllexport)
#else
#  define RTP_PLANNER_V3_API __attribute__((visibility("default")))
#endif

namespace {

// Version 4 appends chance-set fields to the transition result and adds the
// read-only finite-chance accessor.  No V3 binary is permitted to call it.
constexpr uint32_t kAbiVersion = 4;
constexpr size_t kDeckCardCount = DECK_SIZE * 2U;
constexpr size_t kMaximumActionItems = 256U;

enum Status : int32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kUnknownArena = 2,
  kUnknownNode = 3,
  kNotLiveNode = 4,
  kWrongBoundary = 5,
  kInvalidAction = 6,
  kAuditRequired = 7,
  kBufferTooSmall = 8,
  kInitializationFailed = 9,
};

enum Boundary : uint32_t {
  kBoundaryControlledSelection = 1,
  kBoundaryOpponentSelection = 2,
  kBoundaryTerminal = 3,
  kBoundaryUnsupported = 4,
  kBoundaryFiniteChance = 5,
};

enum ChanceOutcomeLabel : uint32_t {
  kChanceOutcomeHeads = 1,
  kChanceOutcomeTails = 2,
};

struct RtpPlannerV3NodeInfo {
  uint32_t struct_size;
  uint32_t boundary;
  int32_t select_player;
  int32_t terminal_result;
  uint32_t option_count;
  uint64_t generation;
};

struct RtpPlannerV3Transition {
  uint32_t struct_size;
  int32_t status;
  uint32_t boundary;
  int32_t select_player;
  int32_t terminal_result;
  uint64_t child_node;
  uint64_t chance_set;
  uint32_t chance_outcome_count;
};

struct RtpPlannerV3ChanceOutcome {
  uint32_t struct_size;
  uint32_t label;
  uint32_t probability_numerator;
  uint32_t probability_denominator;
  uint64_t child_node;
};

thread_local std::string g_last_error;
std::once_flag g_initialize_once;
std::atomic<int> g_initialize_status{0};  // 0 unrun, 1 ready, 2 failed
std::string g_initialize_error;

void ClearError() { g_last_error.clear(); }

void SetError(const std::string& value) { g_last_error = value; }

[[noreturn]] void Fail(const char* value) { throw std::runtime_error(value); }

class StatusFailure : public std::runtime_error {
 public:
  StatusFailure(Status status, const char* message)
      : std::runtime_error(message), status_(status) {}

  Status status() const { return status_; }

 private:
  Status status_;
};

[[noreturn]] void FailStatus(Status status, const char* value) {
  throw StatusFailure(status, value);
}

bool TablesLookValid() {
  return !CardTable.empty() && !SkillTable.empty() && !AttackTable.empty() &&
      !NameTable.empty() && !FunctionTable.empty() &&
      FunctionIndexTable.size() == FunctionTable.size();
}

void InitializePrivateEngine() {
  try {
    // The upstream initializer only asserts this condition.  Under release
    // builds a second initialization appends function table entries, so V3
    // must execute in a fresh native worker.
    if (!CardTable.empty() || !SkillTable.empty() || !AttackTable.empty() ||
        !NameTable.empty() || !FunctionTable.empty() ||
        !FunctionIndexTable.empty()) {
      Fail("V3 initialization requires a fresh engine process");
    }
    InitializeBattleFunction();
    CardImpl();
    InitializeCard();
    if (!TablesLookValid()) {
      Fail("V3 initialization produced invalid engine tables");
    }
    g_initialize_status.store(1, std::memory_order_release);
  } catch (const std::exception& error) {
    g_initialize_error = error.what();
    g_initialize_status.store(2, std::memory_order_release);
  } catch (...) {
    g_initialize_error = "V3 initialization failed with an unknown exception";
    g_initialize_status.store(2, std::memory_order_release);
  }
}

void RequireInitialized() {
  std::call_once(g_initialize_once, InitializePrivateEngine);
  if (g_initialize_status.load(std::memory_order_acquire) != 1) {
    FailStatus(kInitializationFailed, g_initialize_error.empty()
        ? "V3 initialization failed" : g_initialize_error.c_str());
  }
  if (!TablesLookValid()) {
    FailStatus(kInitializationFailed, "V3 engine tables changed after initialization");
  }
}

bool HasSafeCommonBoundaryFields(const BattleData& battle) {
  const Game& game = battle.game;
  return battle.state.game == &battle.game && !game.config.deviceRand &&
      game.config.timeLimit == 0 && !game.pushResponseFunc &&
      game.remainingTime[0] == 0 && game.remainingTime[1] == 0 &&
      game.plannerAudit.isClear();
}

bool IsSafeBoundary(const BattleData& battle) {
  return HasSafeCommonBoundaryFields(battle) && !battle.game.config.manualCoin;
}

bool IsSafeSingleCoinPause(const BattleData& battle) {
  return HasSafeCommonBoundaryFields(battle) && battle.game.config.manualCoin;
}

struct Node {
  BattleData battle;
  uint64_t generation = 0;
};

struct ChanceSet {
  std::array<uint64_t, 2> child_nodes = {};
};

struct Arena {
  int controlled_player = -1;
  uint64_t next_node = 1;
  uint64_t next_chance_set = 1;
  uint64_t live_node = 0;
  std::map<uint64_t, std::unique_ptr<Node>> nodes;
  std::map<uint64_t, ChanceSet> chance_sets;
};

std::mutex g_arenas_mutex;
std::map<uint64_t, std::unique_ptr<Arena>> g_arenas;
uint64_t g_next_arena = 1;

Arena& RequireArena(uint64_t arena_handle) {
  const auto arena = g_arenas.find(arena_handle);
  if (arena == g_arenas.end()) {
    FailStatus(kUnknownArena, "unknown V3 arena handle");
  }
  return *arena->second;
}

Node& RequireNode(Arena& arena, uint64_t node_handle) {
  const auto node = arena.nodes.find(node_handle);
  if (node == arena.nodes.end()) {
    FailStatus(kUnknownNode, "unknown V3 node handle");
  }
  return *node->second;
}

ChanceSet& RequireChanceSet(Arena& arena, uint64_t chance_set_handle) {
  const auto chance_set = arena.chance_sets.find(chance_set_handle);
  if (chance_set == arena.chance_sets.end()) {
    FailStatus(kUnknownNode, "unknown V3 chance-set handle");
  }
  return chance_set->second;
}

std::unique_ptr<Node> CopyNode(const Node& source, bool allow_single_coin_pause = false) {
  const bool source_is_safe = IsSafeBoundary(source.battle) ||
      (allow_single_coin_pause && IsSafeSingleCoinPause(source.battle));
  if (!source_is_safe) {
    FailStatus(kAuditRequired, "source node is not a safe opaque boundary");
  }
  auto output = std::make_unique<Node>();
  output->battle.game = source.battle.game;
  output->battle.state = source.battle.state;
  output->battle.state.game = &output->battle.game;
  output->battle.game.pushResponseFunc = {};
  const bool output_is_safe = IsSafeBoundary(output->battle) ||
      (allow_single_coin_pause && IsSafeSingleCoinPause(output->battle));
  if (!output_is_safe) {
    FailStatus(kAuditRequired, "opaque node copy failed boundary validation");
  }
  return output;
}

uint64_t AddNode(Arena& arena, std::unique_ptr<Node> node) {
  if (node == nullptr || !IsSafeBoundary(node->battle)) {
    FailStatus(kAuditRequired, "cannot publish an unsafe opaque child node");
  }
  if (arena.next_node == 0) {
    FailStatus(kInitializationFailed, "V3 node handle space exhausted");
  }
  const uint64_t handle = arena.next_node++;
  node->generation = handle;
  arena.nodes.emplace(handle, std::move(node));
  return handle;
}

uint64_t AddCopy(Arena& arena, const Node& source) {
  return AddNode(arena, CopyNode(source));
}

uint64_t AddChanceSet(Arena& arena, const ChanceSet& chance_set) {
  if (arena.next_chance_set == 0) {
    FailStatus(kInitializationFailed, "V3 chance-set handle space exhausted");
  }
  const uint64_t handle = arena.next_chance_set++;
  arena.chance_sets.emplace(handle, chance_set);
  return handle;
}

bool IsChanceChild(const Arena& arena, uint64_t node_handle) {
  for (const auto& [unused_handle, chance_set] : arena.chance_sets) {
    (void)unused_handle;
    for (uint64_t child_handle : chance_set.child_nodes) {
      if (child_handle == node_handle) {
        return true;
      }
    }
  }
  return false;
}

Boundary NodeBoundary(const Arena& arena, const Node& node) {
  const State& state = node.battle.state;
  if (state.isFinish()) {
    return kBoundaryTerminal;
  }
  if (state.selectType == SelectType::None) {
    return kBoundaryUnsupported;
  }
  if (state.selectPlayer != arena.controlled_player) {
    return kBoundaryOpponentSelection;
  }
  return kBoundaryControlledSelection;
}

void FillNodeInfo(const Arena& arena, const Node& node,
                  RtpPlannerV3NodeInfo* output) {
  if (output == nullptr || output->struct_size < sizeof(RtpPlannerV3NodeInfo)) {
    FailStatus(kInvalidArgument, "node info output has an incompatible size");
  }
  const State& state = node.battle.state;
  const Boundary boundary = NodeBoundary(arena, node);
  RtpPlannerV3NodeInfo value = {};
  value.struct_size = sizeof(value);
  value.boundary = boundary;
  value.select_player = state.selectPlayer;
  value.terminal_result = state.apiResult();
  value.option_count = boundary == kBoundaryControlledSelection
      ? static_cast<uint32_t>(state.options.size())
      : 0U;
  value.generation = node.generation;
  *output = value;
}

void WriteTransition(Status status, Boundary boundary, int select_player,
                     int terminal_result, uint64_t child_node,
                     uint64_t chance_set, uint32_t chance_outcome_count,
                     RtpPlannerV3Transition* output) {
  if (output == nullptr || output->struct_size < sizeof(RtpPlannerV3Transition)) {
    FailStatus(kInvalidArgument, "transition output has an incompatible size");
  }
  RtpPlannerV3Transition value = {};
  value.struct_size = sizeof(value);
  value.status = status;
  value.boundary = boundary;
  value.select_player = select_player;
  value.terminal_result = terminal_result;
  value.child_node = child_node;
  value.chance_set = chance_set;
  value.chance_outcome_count = chance_outcome_count;
  *output = value;
}

void FillTransitionForNode(const Arena& arena, const Node& node, Status status,
                           uint64_t child_node,
                           RtpPlannerV3Transition* output) {
  const State& state = node.battle.state;
  WriteTransition(status, NodeBoundary(arena, node), state.selectPlayer,
                  state.apiResult(), child_node, 0, 0, output);
}

void FillFiniteChanceTransition(uint64_t chance_set,
                                RtpPlannerV3Transition* output) {
  WriteTransition(kOk, kBoundaryFiniteChance, -1, 0, 0, chance_set, 2,
                  output);
}

bool ApplyExactSelection(BattleData& battle, const int* selected,
                         size_t selected_count) {
  State& state = battle.state;
  state.selected.clear();
  for (size_t index = 0; index < selected_count; ++index) {
    state.selected.push_back(selected[index]);
  }
  if (state.checkPlayerSelect() != 0) {
    return false;
  }
  battle.next();
  while (!state.isFinish() && state.selectMax == 0) {
    state.selected.clear();
    battle.next();
  }
  return true;
}

bool IsStrictSingleCoinPause(const Node& node) {
  const Game& game = node.battle.game;
  const State& state = node.battle.state;
  if (!game.plannerAudit.active || game.plannerAudit.hasUnsafeProvenance() ||
      !game.plannerAudit.singleCoinPending ||
      game.plannerAudit.forcedCoinActive || !game.config.manualCoin ||
      state.isFinish() || state.selectType != SelectType::YesNo ||
      state.selectContext != SelectContext::CoinHead || state.selectDeck ||
      state.selectPlayer != game.plannerAudit.singleCoinPlayer ||
      state.selectMin != 1 || state.selectMax != 1 ||
      state.options.size() != 2) {
    return false;
  }
  return state.options[0].type == SelectOptionType::Yes &&
      state.options[1].type == SelectOptionType::No;
}

std::unique_ptr<Node> BuildForcedSingleCoinChild(const Arena& arena,
                                                 const Node& single_coin_pause,
                                                 int coin_player,
                                                 bool heads) {
  if (coin_player != 0 && coin_player != 1) {
    return nullptr;
  }
  auto child = CopyNode(single_coin_pause, true);
  Game& game = child->battle.game;
  game.plannerAudit.begin(arena.controlled_player);
  game.plannerAudit.armSingleCoin(coin_player);
  game.plannerForceSingleCoin(heads);
  // The pending SelectedCoinSingle continuation remains valid after manual
  // mode is disabled. Any later coin therefore consumes plannerRandom() and
  // rejects this supposedly simple chance branch.
  game.config.manualCoin = false;
  const int selected = heads ? 0 : 1;
  if (!ApplyExactSelection(child->battle, &selected, 1) ||
      child->battle.game.plannerAudit.hasUnsafeProvenance() ||
      child->battle.game.plannerAudit.singleCoinPending ||
      child->battle.game.plannerAudit.forcedCoinActive) {
    return nullptr;
  }
  child->battle.game.plannerAudit.clear();
  if (!IsSafeBoundary(child->battle)) {
    return nullptr;
  }
  return child;
}

void BuildPolicyJson(const Arena& arena, const Node& node, std::u8string* output) {
  if (NodeBoundary(arena, node) != kBoundaryControlledSelection) {
    FailStatus(kWrongBoundary,
               "policy JSON is only available at the controlled selection boundary");
  }
  JsonBuilder builder;
  // Do not replay prior logs into a later observation.  The current selection
  // and current policy-visible board are sufficient for this foundation, and
  // an immutable log start avoids cross-turn observation coupling.
  ToJsonApi(node.battle.state, builder,
            static_cast<int>(node.battle.state.logs.size()));
  *output = std::move(builder.buf);
}

template <typename Callback>
int Invoke(Callback&& callback) {
  ClearError();
  try {
    return callback();
  } catch (const StatusFailure& error) {
    SetError(error.what());
    return error.status();
  } catch (const std::exception& error) {
    SetError(error.what());
    return kInitializationFailed;
  } catch (...) {
    SetError("V3 native operation failed with an unknown exception");
    return kInitializationFailed;
  }
}

int RequireOutHandle(uint64_t* output, const char* label) {
  if (output == nullptr) {
    SetError(label);
    return kInvalidArgument;
  }
  *output = 0;
  return kOk;
}

}  // namespace

extern "C" {

RTP_PLANNER_V3_API uint32_t RtpPlannerV3AbiVersion() { return kAbiVersion; }

RTP_PLANNER_V3_API const char* RtpPlannerV3LastError() {
  return g_last_error.c_str();
}

RTP_PLANNER_V3_API int RtpPlannerV3Initialize() {
  return Invoke([] {
    RequireInitialized();
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3ArenaStart(
    const int* cards, size_t card_count, uint32_t seed, int controlled_player,
    uint64_t* arena_output, uint64_t* node_output) {
  return Invoke([&] {
    RequireInitialized();
    if (RequireOutHandle(arena_output, "arena output is null") != kOk ||
        RequireOutHandle(node_output, "node output is null") != kOk) {
      return kInvalidArgument;
    }
    if (cards == nullptr || card_count != kDeckCardCount || seed == 0 ||
        (controlled_player != 0 && controlled_player != 1)) {
      return kInvalidArgument;
    }

    auto initial = std::make_unique<Node>();
    GameConfig config = {};
    config.seed = seed;
    config.timeLimit = 0;
    config.recordLog = true;
    config.manualCoin = false;
    config.sendDeck = false;
    config.deviceRand = false;
    for (size_t player = 0; player < 2; ++player) {
      for (size_t index = 0; index < DECK_SIZE; ++index) {
        const int card_id = cards[player * DECK_SIZE + index];
        if (!CardTable.contains(card_id)) {
          return kInvalidArgument;
        }
        config.decks[player].cards[index] = card_id;
      }
    }
    initial->battle.init(config);
    initial->battle.start();
    initial->battle.next();
    if (!IsSafeBoundary(initial->battle)) {
      FailStatus(kAuditRequired, "initial node is not a safe opaque boundary");
    }

    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    if (g_next_arena == 0) {
      FailStatus(kInitializationFailed, "V3 arena handle space exhausted");
    }
    const uint64_t arena_handle = g_next_arena++;
    auto arena = std::make_unique<Arena>();
    arena->controlled_player = controlled_player;
    initial->generation = arena->next_node;
    const uint64_t node_handle = arena->next_node++;
    arena->live_node = node_handle;
    arena->nodes.emplace(node_handle, std::move(initial));
    g_arenas.emplace(arena_handle, std::move(arena));
    *arena_output = arena_handle;
    *node_output = node_handle;
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3ArenaDestroy(uint64_t arena_handle) {
  return Invoke([&] {
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    const auto arena = g_arenas.find(arena_handle);
    if (arena == g_arenas.end()) {
      return kUnknownArena;
    }
    g_arenas.erase(arena);
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3CaptureLive(uint64_t arena_handle,
                                                 uint64_t* node_output) {
  return Invoke([&] {
    if (RequireOutHandle(node_output, "node output is null") != kOk) {
      return kInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    Node& live = RequireNode(arena, arena.live_node);
    *node_output = AddCopy(arena, live);
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3CloneNode(uint64_t arena_handle,
                                               uint64_t source_node,
                                               uint64_t* node_output) {
  return Invoke([&] {
    if (RequireOutHandle(node_output, "node output is null") != kOk) {
      return kInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    Node& source = RequireNode(arena, source_node);
    *node_output = AddCopy(arena, source);
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3RestoreLive(uint64_t arena_handle,
                                                 uint64_t source_node,
                                                 uint64_t* live_output) {
  return Invoke([&] {
    if (RequireOutHandle(live_output, "live output is null") != kOk) {
      return kInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    Node& source = RequireNode(arena, source_node);
    const uint64_t restored = AddCopy(arena, source);
    arena.live_node = restored;
    *live_output = restored;
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3ReleaseNode(uint64_t arena_handle,
                                                 uint64_t node_handle) {
  return Invoke([&] {
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    if (node_handle == arena.live_node) {
      return kNotLiveNode;
    }
    if (IsChanceChild(arena, node_handle)) {
      FailStatus(kNotLiveNode, "cannot release a node owned by a finite chance set");
    }
    const auto node = arena.nodes.find(node_handle);
    if (node == arena.nodes.end()) {
      return kUnknownNode;
    }
    arena.nodes.erase(node);
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3NodeInfo(uint64_t arena_handle,
                                             uint64_t node_handle,
                                             RtpPlannerV3NodeInfo* output) {
  return Invoke([&] {
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    Node& node = RequireNode(arena, node_handle);
    FillNodeInfo(arena, node, output);
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3PolicyJsonSize(uint64_t arena_handle,
                                                   uint64_t node_handle,
                                                   size_t* bytes_output) {
  return Invoke([&] {
    if (bytes_output == nullptr) {
      return kInvalidArgument;
    }
    *bytes_output = 0;
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    Node& node = RequireNode(arena, node_handle);
    std::u8string json;
    BuildPolicyJson(arena, node, &json);
    *bytes_output = json.size();
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3PolicyJson(uint64_t arena_handle,
                                               uint64_t node_handle,
                                               char* buffer, size_t buffer_size,
                                               size_t* bytes_output) {
  return Invoke([&] {
    if (bytes_output == nullptr) {
      return kInvalidArgument;
    }
    *bytes_output = 0;
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    Node& node = RequireNode(arena, node_handle);
    std::u8string json;
    BuildPolicyJson(arena, node, &json);
    *bytes_output = json.size();
    if (buffer == nullptr || buffer_size < json.size() + 1U) {
      return kBufferTooSmall;
    }
    std::memcpy(buffer, json.data(), json.size());
    buffer[json.size()] = '\0';
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3ExpandAction(
    uint64_t arena_handle, uint64_t node_handle, const int* selected,
    size_t selected_count, RtpPlannerV3Transition* transition_output) {
  return Invoke([&] {
    if (selected_count > kMaximumActionItems ||
        (selected_count != 0 && selected == nullptr)) {
      return kInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    Node& node = RequireNode(arena, node_handle);
    if (NodeBoundary(arena, node) != kBoundaryControlledSelection) {
      FillTransitionForNode(arena, node, kWrongBoundary, 0, transition_output);
      return kWrongBoundary;
    }
    auto probe = CopyNode(node);
    probe->battle.game.plannerAudit.begin(arena.controlled_player);
    // This is the ordinary selected-action/automatic-continuation sequence,
    // but it stays entirely inside the isolated V3 clone and never exposes an
    // engine state or replay byte string.
    if (!ApplyExactSelection(probe->battle, selected, selected_count)) {
      FillTransitionForNode(arena, node, kInvalidAction, 0, transition_output);
      return kInvalidAction;
    }
    if (IsStrictSingleCoinPause(*probe)) {
      const int coin_player = probe->battle.game.plannerAudit.singleCoinPlayer;
      probe->battle.game.plannerAudit.clear();
      auto heads = BuildForcedSingleCoinChild(arena, *probe, coin_player, true);
      auto tails = BuildForcedSingleCoinChild(arena, *probe, coin_player, false);
      if (heads == nullptr || tails == nullptr ||
          arena.next_node == 0 ||
          arena.next_node == std::numeric_limits<uint64_t>::max() ||
          arena.next_chance_set == 0) {
        FillTransitionForNode(arena, node, kAuditRequired, 0, transition_output);
        return kAuditRequired;
      }
      ChanceSet chance_set = {};
      chance_set.child_nodes[0] = AddNode(arena, std::move(heads));
      chance_set.child_nodes[1] = AddNode(arena, std::move(tails));
      const uint64_t chance_set_handle = AddChanceSet(arena, chance_set);
      FillFiniteChanceTransition(chance_set_handle, transition_output);
      return kOk;
    }
    if (probe->battle.game.plannerAudit.hasUnsafeProvenance() ||
        probe->battle.game.plannerAudit.singleCoinPending ||
        probe->battle.game.plannerAudit.forcedCoinActive ||
        probe->battle.game.config.manualCoin) {
      FillTransitionForNode(arena, node, kAuditRequired, 0, transition_output);
      return kAuditRequired;
    }
    probe->battle.game.plannerAudit.clear();
    if (!IsSafeBoundary(probe->battle)) {
      FillTransitionForNode(arena, node, kAuditRequired, 0, transition_output);
      return kAuditRequired;
    }
    const uint64_t child_handle = AddNode(arena, std::move(probe));
    Node& child = RequireNode(arena, child_handle);
    FillTransitionForNode(arena, child, kOk, child_handle, transition_output);
    return kOk;
  });
}

RTP_PLANNER_V3_API int RtpPlannerV3ChanceOutcome(
    uint64_t arena_handle, uint64_t chance_set_handle, size_t outcome_index,
    RtpPlannerV3ChanceOutcome* output) {
  return Invoke([&] {
    if (output == nullptr || output->struct_size < sizeof(RtpPlannerV3ChanceOutcome)) {
      return kInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(g_arenas_mutex);
    Arena& arena = RequireArena(arena_handle);
    ChanceSet& chance_set = RequireChanceSet(arena, chance_set_handle);
    if (outcome_index >= chance_set.child_nodes.size()) {
      return kInvalidArgument;
    }
    const uint64_t child_handle = chance_set.child_nodes[outcome_index];
    RequireNode(arena, child_handle);
    RtpPlannerV3ChanceOutcome value = {};
    value.struct_size = sizeof(value);
    value.label = outcome_index == 0 ? kChanceOutcomeHeads : kChanceOutcomeTails;
    value.probability_numerator = 1;
    value.probability_denominator = 2;
    value.child_node = child_handle;
    *output = value;
    return kOk;
  });
}

}  // extern "C"
