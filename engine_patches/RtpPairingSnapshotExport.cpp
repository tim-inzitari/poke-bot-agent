// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
//
// Private evaluation-only overlay for the Pokémon TCG AI Battle Challenge
// engine.  It is intentionally not part of a competition package.  Build this
// translation unit beside the appropriately licensed engine source; do not
// copy this file into a submitted runtime.
//
// The ABI below creates an opaque, post-BattleStart snapshot.  It deliberately
// does *not* expose a seed-only reconstruction path.  A snapshot serializes the
// complete State (through the engine's own State::serialize routine) plus the
// deterministic Game config/rng/counters.  v1 captures only the initial
// external-selection boundary and rejects any nonempty Game scratch container
// or API serialization buffer rather than copying a container or pointer.
// Restore always allocates a fresh ApiData object and rebinds State::game to
// that fresh Game.

#include "All.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <locale>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#  define RTP_PAIRING_API __declspec(dllexport)
#else
#  define RTP_PAIRING_API __attribute__((visibility("default")))
#endif

namespace {

constexpr uint32_t kSnapshotMagic = 0x52545053U;  // "RTPS"
// v2 adds the overlay-owned JSON observation out-parameter ABI.  It avoids
// depending on the upstream C++ struct-return ABI from Python ctypes.
constexpr uint32_t kSnapshotAbiVersion = 2;
constexpr uint32_t kCaptureBoundaryPostBattleStartFirstSelection = 1;
constexpr size_t kMaximumSnapshotBytes = 64U * 1024U * 1024U;
constexpr uint32_t kMaximumContainerItems = 65536U;

thread_local std::string g_last_error;
std::once_flag g_pairing_initialize_once;
std::atomic<int> g_pairing_initialize_status{0};  // 0=unrun, 1=ready, 2=failed
std::string g_pairing_initialize_error;

void ClearError() {
  g_last_error.clear();
}

void SetError(const std::string& message) {
  g_last_error = message;
}

[[noreturn]] void SnapshotError(const char* message) {
  throw std::runtime_error(message);
}

bool PrivatePairingTablesLookValid() {
  if (CardTable.empty() || SkillTable.empty() || AttackTable.empty() ||
      NameTable.empty() || FunctionTable.empty() ||
      FunctionIndexTable.size() != FunctionTable.size()) {
    return false;
  }
  for (const auto& [pointer_bits, function_index] : FunctionIndexTable) {
    if (function_index < 0 ||
        function_index >= static_cast<int>(FunctionTable.size()) ||
        reinterpret_cast<std::uintptr_t>(FunctionTable[function_index]) !=
            static_cast<std::uintptr_t>(pointer_bits)) {
      return false;
    }
  }
  for (const auto& [card_id, card] : CardTable) {
    if (card.cardId != card_id) {
      return false;
    }
  }
  for (const auto& [name, card_id] : NameTable) {
    const auto card = CardTable.find(card_id);
    if (card == CardTable.end() || card->second.name != name) {
      return false;
    }
  }
  for (const auto& [skill_id, skill] : SkillTable) {
    if (skill.skillId != skill_id || !CardTable.contains(skill.cardId)) {
      return false;
    }
  }
  for (const auto& [attack_id, attack] : AttackTable) {
    if (attack.attackId != attack_id || !CardTable.contains(attack.cardId)) {
      return false;
    }
  }
  return true;
}

void InitializePrivatePairingEngineOnce() {
  try {
    // The upstream GameInitialize relies on debug-only asserts.  A second call
    // under -DNDEBUG silently appends to FunctionTable, so private pairing
    // workers require a pristine process and fail closed if any table exists.
    if (!CardTable.empty() || !SkillTable.empty() || !AttackTable.empty() ||
        !NameTable.empty() || !FunctionTable.empty() ||
        !FunctionIndexTable.empty()) {
      SnapshotError("private pairing initialization requires a pristine engine process");
    }
    InitializeAll();
    if (!PrivatePairingTablesLookValid()) {
      SnapshotError("private pairing engine initialization produced invalid tables");
    }
    g_pairing_initialize_status.store(1, std::memory_order_release);
  } catch (const std::exception& exc) {
    g_pairing_initialize_error = exc.what();
    g_pairing_initialize_status.store(2, std::memory_order_release);
  } catch (...) {
    g_pairing_initialize_error = "unknown private pairing initialization error";
    g_pairing_initialize_status.store(2, std::memory_order_release);
  }
}

void RequirePrivatePairingEngineInitialized() {
  std::call_once(g_pairing_initialize_once, InitializePrivatePairingEngineOnce);
  if (g_pairing_initialize_status.load(std::memory_order_acquire) != 1) {
    SnapshotError(g_pairing_initialize_error.empty()
                      ? "private pairing engine initialization failed"
                      : g_pairing_initialize_error.c_str());
  }
  // Do this on every invocation.  A later accidental public GameInitialize()
  // call can mutate the inline engine tables even though std::call_once has
  // already completed; snapshot/replay must detect that drift rather than
  // continuing with changed function indices or metadata.
  if (!PrivatePairingTablesLookValid()) {
    SnapshotError("private pairing engine global tables changed after initialization");
  }
}

class SnapshotWriter {
 public:
  void u8(uint8_t value) { data_.push_back(value); }

  void boolean(bool value) { u8(value ? 1U : 0U); }

  void u32(uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
      u8(static_cast<uint8_t>((value >> shift) & 0xffU));
    }
  }

  void i32(int value) { u32(static_cast<uint32_t>(value)); }

  void u64(uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
      u8(static_cast<uint8_t>((value >> shift) & 0xffU));
    }
  }

  void i64(int64_t value) { u64(static_cast<uint64_t>(value)); }

  void f64(double value) {
    static_assert(sizeof(double) == sizeof(uint64_t));
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }

  void raw(const uint8_t* begin, size_t count) {
    if (count > kMaximumSnapshotBytes ||
        data_.size() > kMaximumSnapshotBytes - count) {
      SnapshotError("snapshot exceeds maximum size");
    }
    data_.insert(data_.end(), begin, begin + count);
  }

  void bytes(const std::vector<uint8_t>& value) {
    if (value.size() > kMaximumSnapshotBytes) {
      SnapshotError("byte vector exceeds maximum snapshot size");
    }
    u64(static_cast<uint64_t>(value.size()));
    if (!value.empty()) {
      raw(value.data(), value.size());
    }
  }

  void chars(const std::vector<char>& value) {
    if (value.size() > kMaximumSnapshotBytes) {
      SnapshotError("character vector exceeds maximum snapshot size");
    }
    u64(static_cast<uint64_t>(value.size()));
    if (!value.empty()) {
      raw(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    }
  }

  void text(const std::string& value) {
    if (value.size() > kMaximumSnapshotBytes) {
      SnapshotError("text exceeds maximum snapshot size");
    }
    u64(static_cast<uint64_t>(value.size()));
    if (!value.empty()) {
      raw(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    }
  }

  void u8text(const std::u8string& value) {
    if (value.size() > kMaximumSnapshotBytes) {
      SnapshotError("u8 text exceeds maximum snapshot size");
    }
    u64(static_cast<uint64_t>(value.size()));
    if (!value.empty()) {
      raw(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    }
  }

  const std::vector<uint8_t>& data() const { return data_; }
  std::vector<uint8_t>& data() { return data_; }

 private:
  std::vector<uint8_t> data_;
};

class SnapshotReader {
 public:
  explicit SnapshotReader(const std::vector<uint8_t>& data) : data_(data) {}

  uint8_t u8() {
    Require(1);
    return data_[position_++];
  }

  bool boolean() {
    const uint8_t value = u8();
    if (value > 1U) {
      SnapshotError("invalid boolean in snapshot");
    }
    return value != 0U;
  }

  uint32_t u32() {
    uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
      value |= static_cast<uint32_t>(u8()) << shift;
    }
    return value;
  }

  int i32() { return static_cast<int>(u32()); }

  uint64_t u64() {
    uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
      value |= static_cast<uint64_t>(u8()) << shift;
    }
    return value;
  }

  int64_t i64() { return static_cast<int64_t>(u64()); }

  double f64() {
    const uint64_t bits = u64();
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  std::vector<uint8_t> bytes() {
    const size_t count = ReadLength(kMaximumSnapshotBytes, "byte vector");
    std::vector<uint8_t> value(count);
    if (count) {
      Require(count);
      std::copy(data_.begin() + static_cast<std::ptrdiff_t>(position_),
                data_.begin() + static_cast<std::ptrdiff_t>(position_ + count),
                value.begin());
      position_ += count;
    }
    return value;
  }

  std::vector<char> chars() {
    const size_t count = ReadLength(kMaximumSnapshotBytes, "character vector");
    std::vector<char> value(count);
    if (count) {
      Require(count);
      std::memcpy(value.data(), data_.data() + position_, count);
      position_ += count;
    }
    return value;
  }

  std::string text() {
    const size_t count = ReadLength(kMaximumSnapshotBytes, "text");
    std::string value(count, '\0');
    if (count) {
      Require(count);
      std::memcpy(value.data(), data_.data() + position_, count);
      position_ += count;
    }
    return value;
  }

  std::u8string u8text() {
    const size_t count = ReadLength(kMaximumSnapshotBytes, "u8 text");
    std::u8string value(count, u8'\0');
    if (count) {
      Require(count);
      std::memcpy(value.data(), data_.data() + position_, count);
      position_ += count;
    }
    return value;
  }

  bool at_end() const { return position_ == data_.size(); }

 private:
  void Require(size_t count) const {
    if (count > data_.size() - position_) {
      SnapshotError("truncated pairing snapshot");
    }
  }

  size_t ReadLength(size_t maximum, const char* label) {
    const uint64_t raw = u64();
    if (raw > maximum || raw > std::numeric_limits<size_t>::max()) {
      SnapshotError(label);
    }
    return static_cast<size_t>(raw);
  }

  const std::vector<uint8_t>& data_;
  size_t position_ = 0;
};

uint64_t Fnv1a64(const uint8_t* input, size_t count) {
  uint64_t hash = 1469598103934665603ULL;
  for (size_t index = 0; index < count; ++index) {
    hash ^= static_cast<uint64_t>(input[index]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint32_t RotateRight(uint32_t value, int amount) {
  return (value >> amount) | (value << (32 - amount));
}

void Sha256Block(const uint8_t* block, std::array<uint32_t, 8>& state) {
  static constexpr std::array<uint32_t, 64> kRoundConstants = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
  };
  std::array<uint32_t, 64> words = {};
  for (size_t index = 0; index < 16; ++index) {
    const size_t offset = index * 4;
    words[index] = (static_cast<uint32_t>(block[offset]) << 24) |
                   (static_cast<uint32_t>(block[offset + 1]) << 16) |
                   (static_cast<uint32_t>(block[offset + 2]) << 8) |
                   static_cast<uint32_t>(block[offset + 3]);
  }
  for (size_t index = 16; index < words.size(); ++index) {
    const uint32_t s0 = RotateRight(words[index - 15], 7) ^
                        RotateRight(words[index - 15], 18) ^
                        (words[index - 15] >> 3);
    const uint32_t s1 = RotateRight(words[index - 2], 17) ^
                        RotateRight(words[index - 2], 19) ^
                        (words[index - 2] >> 10);
    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
  }
  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  uint32_t e = state[4];
  uint32_t f = state[5];
  uint32_t g = state[6];
  uint32_t h = state[7];
  for (size_t index = 0; index < words.size(); ++index) {
    const uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
    const uint32_t choose = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + s1 + choose + kRoundConstants[index] + words[index];
    const uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = s0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

std::array<uint8_t, 32> Sha256(const uint8_t* input, size_t count) {
  std::array<uint32_t, 8> state = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };
  const size_t complete_bytes = count - (count % 64U);
  for (size_t offset = 0; offset < complete_bytes; offset += 64U) {
    Sha256Block(input + offset, state);
  }
  std::array<uint8_t, 128> tail = {};
  const size_t remainder = count % 64U;
  if (remainder != 0) {
    std::memcpy(tail.data(), input + complete_bytes, remainder);
  }
  tail[remainder] = 0x80U;
  const size_t padded_bytes = remainder + 1U <= 56U ? 64U : 128U;
  const uint64_t bit_count = static_cast<uint64_t>(count) * 8U;
  for (int byte = 0; byte < 8; ++byte) {
    tail[padded_bytes - 1U - static_cast<size_t>(byte)] =
        static_cast<uint8_t>(bit_count >> (byte * 8));
  }
  for (size_t offset = 0; offset < padded_bytes; offset += 64U) {
    Sha256Block(tail.data() + offset, state);
  }
  std::array<uint8_t, 32> digest = {};
  for (size_t index = 0; index < state.size(); ++index) {
    const uint32_t word = state[index];
    digest[index * 4] = static_cast<uint8_t>(word >> 24);
    digest[index * 4 + 1] = static_cast<uint8_t>(word >> 16);
    digest[index * 4 + 2] = static_cast<uint8_t>(word >> 8);
    digest[index * 4 + 3] = static_cast<uint8_t>(word);
  }
  return digest;
}

void WriteCardRef(SnapshotWriter& writer, CardRef value) {
  writer.u8(value.cardIndex);
}

CardRef ReadCardRef(SnapshotReader& reader) {
  return CardRef(reader.u8());
}

void WriteAreaRef(SnapshotWriter& writer, const AreaRef& value) {
  WriteCardRef(writer, value.card);
  writer.i32(value.moveCounter);
}

AreaRef ReadAreaRef(SnapshotReader& reader) {
  AreaRef value = {};
  value.card = ReadCardRef(reader);
  value.moveCounter = reader.i32();
  return value;
}

template <typename T, typename WriteOne>
void WriteBoundedVector(SnapshotWriter& writer,
                        const std::vector<T>& values,
                        WriteOne write_one) {
  if (values.size() > kMaximumContainerItems) {
    SnapshotError("snapshot container exceeds limit");
  }
  writer.u32(static_cast<uint32_t>(values.size()));
  for (const T& value : values) {
    write_one(value);
  }
}

template <typename T, typename ReadOne>
std::vector<T> ReadBoundedVector(SnapshotReader& reader, ReadOne read_one) {
  const uint32_t count = reader.u32();
  if (count > kMaximumContainerItems) {
    SnapshotError("snapshot container exceeds limit");
  }
  std::vector<T> result;
  result.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    result.push_back(read_one());
  }
  return result;
}

void WriteConfig(SnapshotWriter& writer, const GameConfig& config) {
  for (const Deck& deck : config.decks) {
    for (CardId card_id : deck.cards) {
      writer.i32(static_cast<int>(card_id));
    }
  }
  for (const std::string& deck_name : config.deckNames) {
    writer.text(deck_name);
  }
  writer.u32(config.seed);
  writer.i32(config.timeLimit);
  writer.boolean(config.recordLog);
  writer.boolean(config.manualCoin);
  writer.boolean(config.sendDeck);
  writer.boolean(config.deviceRand);
}

GameConfig ReadConfig(SnapshotReader& reader) {
  GameConfig config = {};
  for (Deck& deck : config.decks) {
    for (CardId& card_id : deck.cards) {
      card_id = static_cast<CardId>(reader.i32());
    }
  }
  for (std::string& deck_name : config.deckNames) {
    deck_name = reader.text();
  }
  config.seed = reader.u32();
  config.timeLimit = reader.i32();
  config.recordLog = reader.boolean();
  config.manualCoin = reader.boolean();
  config.sendDeck = reader.boolean();
  config.deviceRand = reader.boolean();
  return config;
}

void WriteRng(SnapshotWriter& writer, const std::mt19937& rng) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << rng;
  if (!stream.good()) {
    SnapshotError("cannot serialize mt19937 state");
  }
  writer.text(stream.str());
}

std::mt19937 ReadRng(SnapshotReader& reader) {
  const std::string serialized = reader.text();
  std::istringstream stream(serialized);
  stream.imbue(std::locale::classic());
  std::mt19937 rng;
  stream >> rng;
  if (stream.fail()) {
    SnapshotError("cannot deserialize mt19937 state");
  }
  // Reject non-whitespace trailer bytes without using std::ws at EOF: some
  // standard-library implementations set failbit when that manipulator's
  // sentry observes EOF, even though the mt19937 parse itself succeeded.
  if (!stream.eof()) {
    char trailing = '\0';
    while (stream.get(trailing)) {
      if (!std::isspace(static_cast<unsigned char>(trailing))) {
        SnapshotError("mt19937 state has trailing data");
      }
    }
    if (!stream.eof()) {
      SnapshotError("cannot validate mt19937 state trailer");
    }
  }
  return rng;
}

void WriteCardEffects(SnapshotWriter& writer,
                      const std::vector<CardEffect>& values) {
  WriteBoundedVector<CardEffect>(writer, values, [&writer](const CardEffect& value) {
    WriteCardRef(writer, value.ref);
    writer.i32(static_cast<int>(value.priority));
    writer.i32(value.skillOrder);
    writer.i32(value.moveCounter);
  });
}

std::vector<CardEffect> ReadCardEffects(SnapshotReader& reader) {
  return ReadBoundedVector<CardEffect>(reader, [&reader]() {
    CardEffect value = {};
    value.ref = ReadCardRef(reader);
    value.priority = static_cast<signed char>(reader.i32());
    value.skillOrder = reader.i32();
    value.moveCounter = reader.i32();
    return value;
  });
}

void WriteAttackEnergies(SnapshotWriter& writer,
                         const std::vector<AttackEnergy>& values) {
  WriteBoundedVector<AttackEnergy>(writer, values,
      [&writer](const AttackEnergy& value) {
        writer.boolean(value.attack != nullptr);
        if (value.attack != nullptr) {
          writer.i32(value.attack->attackId);
        }
        writer.i32(value.insufficientEnergy);
        writer.i32(value.srcAttackId);
      });
}

std::vector<AttackEnergy> ReadAttackEnergies(SnapshotReader& reader) {
  return ReadBoundedVector<AttackEnergy>(reader, [&reader]() {
    AttackEnergy value = {};
    const bool has_attack = reader.boolean();
    if (has_attack) {
      const int attack_id = reader.i32();
      const auto found = AttackTable.find(attack_id);
      if (found == AttackTable.end()) {
        SnapshotError("snapshot references an unknown attack");
      }
      value.attack = &found->second;
    }
    value.insufficientEnergy = reader.i32();
    value.srcAttackId = reader.i32();
    return value;
  });
}

void WriteAbilitySets(
    SnapshotWriter& writer,
    const std::array<std::unordered_set<int>, 2>& values) {
  for (const std::unordered_set<int>& source : values) {
    if (source.size() > kMaximumContainerItems) {
      SnapshotError("snapshot ability set exceeds limit");
    }
    std::vector<int> sorted(source.begin(), source.end());
    std::sort(sorted.begin(), sorted.end());
    writer.u32(static_cast<uint32_t>(sorted.size()));
    for (int value : sorted) {
      writer.i32(value);
    }
  }
}

std::array<std::unordered_set<int>, 2> ReadAbilitySets(SnapshotReader& reader) {
  std::array<std::unordered_set<int>, 2> values;
  for (std::unordered_set<int>& target : values) {
    const uint32_t count = reader.u32();
    if (count > kMaximumContainerItems) {
      SnapshotError("snapshot ability set exceeds limit");
    }
    for (uint32_t index = 0; index < count; ++index) {
      target.insert(reader.i32());
    }
  }
  return values;
}

void WriteGame(SnapshotWriter& writer, const Game& game) {
  // A callback can close over source-side state.  Copying it would violate
  // clone independence, so snapshots reject it rather than guessing.
  if (game.pushResponseFunc) {
    SnapshotError("snapshot capture rejects a nonempty Game response callback");
  }
  if (game.config.timeLimit != 0) {
    SnapshotError("pairing snapshot requires GameConfig.timeLimit == 0");
  }
  if (game.config.manualCoin || game.config.sendDeck) {
    SnapshotError("pairing snapshot requires manualCoin/sendDeck disabled");
  }
  writer.boolean(game.selecting);
  writer.i32(game.actionCount);
  WriteConfig(writer, game.config);
  WriteRng(writer, game.rng);
  for (double remaining : game.remainingTime) {
    writer.f64(remaining);
  }
  // This ABI permits only timeLimit == 0, so wall-clock startTime is inert.
  // Serializing it would create unfair A/B/C timeout drift after a delay.
}

Game ReadGame(SnapshotReader& reader) {
  Game game = {};
  game.selecting = reader.boolean();
  game.actionCount = reader.i32();
  game.config = ReadConfig(reader);
  if (game.config.deviceRand) {
    SnapshotError("snapshot has deviceRand enabled");
  }
  if (game.config.timeLimit != 0) {
    SnapshotError("snapshot has a nonzero time limit");
  }
  if (game.config.manualCoin || game.config.sendDeck) {
    SnapshotError("snapshot has manualCoin/sendDeck enabled");
  }
  if (!game.config.recordLog) {
    SnapshotError("snapshot has recordLog disabled");
  }
  game.rng = ReadRng(reader);
  for (double& remaining : game.remainingTime) {
    remaining = reader.f64();
    if (!std::isfinite(remaining) || remaining != 0.0) {
      SnapshotError("time-disabled snapshot has nonzero remaining time");
    }
  }
  game.startTime = std::chrono::high_resolution_clock::time_point{};
  game.pushResponseFunc = nullptr;
  return game;
}

void WriteApiData(SnapshotWriter& writer, const ApiData& data) {
  if (data.apiDataType != 1) {
    SnapshotError("snapshot capture requires battle ApiData");
  }
  if (data.state.game != &data.game) {
    SnapshotError("source State game pointer is not self-bound");
  }
  writer.i32(data.apiDataType);
  writer.i32(data.selectCount);
  writer.i32(data.preGetSelectCount);
  writer.i32(data.initializeError);
  // State::serialize writes fields and each dynamic vector's contents, never a
  // std::vector object or State::game pointer.  The pointer is rebound on
  // restore below.
  BinaryWriter state_writer;
  data.state.serialize(state_writer);
  writer.bytes(state_writer.buf);
}

void RequirePostBattleStartBoundary(const ApiData& data) {
  if (data.apiDataType != 1 || data.state.game != &data.game) {
    SnapshotError("pairing snapshot requires self-bound battle ApiData");
  }
  if (data.game.config.deviceRand) {
    SnapshotError("snapshot capture rejects deviceRand=true; seed-only is not pairing");
  }
  // This v1 ABI intentionally supports one narrow, auditable capture point:
  // immediately after RtpPairingBattleStartSeededOut returns and before any
  // external Select/GetBattleData call.  It is not a general mid-game clone
  // facility.  The State function stack preserves setup continuation.
  if (data.selectCount != 0 || data.preGetSelectCount != -1 ||
      !data.selected.empty() || !data.state.selected.empty() || !data.visData.empty() ||
      !data.jsonBuilder.buf.empty() || !data.writer.buf.empty() ||
      !data.writer.base64.empty() || !data.reader.buf.empty() ||
      !data.reader.base64.empty() || !data.reader.base64Dest.empty() ||
      data.reader.pos != 0 || data.writer.countA != 0 || data.writer.countSrc != 0) {
    SnapshotError("snapshot capture is not at the post-BattleStart boundary");
  }
  // Do not accept a merely selectable state.  SetupGame has one uniquely
  // auditable external boundary: before the first-player decision, it leaves
  // exactly the SelectedIsFirst continuation and a Yes/No IsFirst selection.
  // This makes the v2 bytes semantically meaningful after a fresh-process
  // restore and prevents a hand-built mid-game ApiData from being relabelled
  // as a post-BattleStart snapshot.
  const State& state = data.state;
  if (data.game.selecting || data.game.actionCount != 1 || state.isFinish() ||
      state.turn != 0 || state.phase != GamePhase::Setup ||
      state.firstPlayer != -1 || state.selectType != SelectType::YesNo ||
      state.selectContext != SelectContext::IsFirst || state.selectPlayer != 0 ||
      state.selectMin != 1 || state.selectMax != 1 ||
      state.options.size() != 2 ||
      state.options[0].type != SelectOptionType::Yes ||
      state.options[1].type != SelectOptionType::No) {
    SnapshotError("snapshot capture is not the exact post-BattleStart IsFirst boundary");
  }
  if (state.functionStack.size() != 1) {
    SnapshotError("snapshot capture has an unexpected setup continuation stack");
  }
  const GameFunction& continuation = state.functionStack[0];
  int selected_is_first_index = -1;
  for (size_t index = 0; index < FunctionTable.size(); ++index) {
    if (FunctionTable[index] == (void*)SelectedIsFirst) {
      selected_is_first_index = static_cast<int>(index);
      break;
    }
  }
  if (selected_is_first_index < 0 ||
      continuation.functionIndex != selected_is_first_index ||
      continuation.functionIndex < 0 ||
      continuation.functionIndex >= static_cast<int>(FunctionTable.size()) ||
      FunctionTable[continuation.functionIndex] != (void*)SelectedIsFirst ||
      continuation.argType != ArgType::None || continuation.callCount != 1 ||
      continuation.calledCount != 0) {
    SnapshotError("snapshot capture has an invalid SelectedIsFirst continuation");
  }
  if (!data.game.energyList.empty() || !data.game.energyList2.empty() ||
      !data.game.cardList.empty() || !data.game.cardEffectList.empty() ||
      !data.game.targetList.empty() || !data.game.attackEnergyList.empty() ||
      !data.game.abilitySet[0].empty() || !data.game.abilitySet[1].empty() ||
      !data.game.jsonBuilder.buf.empty()) {
    SnapshotError("snapshot capture has nonempty Game scratch state");
  }
  if (data.game.config.timeLimit != 0) {
    SnapshotError("snapshot capture requires an evaluator with no time limit");
  }
  if (data.game.config.manualCoin || data.game.config.sendDeck ||
      !data.game.config.recordLog) {
    SnapshotError("snapshot capture has an unsupported pairing GameConfig");
  }
  for (double remaining : data.game.remainingTime) {
    if (!std::isfinite(remaining) || remaining != 0.0) {
      SnapshotError("time-disabled snapshot has nonzero remaining time");
    }
  }
}

void ReadApiData(SnapshotReader& reader, ApiData& data) {
  data.apiDataType = reader.i32();
  if (data.apiDataType != 1) {
    SnapshotError("snapshot is not battle ApiData");
  }
  data.selectCount = reader.i32();
  data.preGetSelectCount = reader.i32();
  data.initializeError = reader.i32();
  const std::vector<uint8_t> state_bytes = reader.bytes();
  BinaryReader state_reader;
  state_reader.buf = state_bytes;
  state_reader.pos = 0;
  data.state.clear();
  data.state.deserialize(state_reader);
  if (state_reader.pos != state_reader.buf.size()) {
    SnapshotError("snapshot State payload has trailing data");
  }
}

void ValidateRestoredStateFunctionStack(const State& state) {
  for (const GameFunction& function : state.functionStack) {
    if (function.functionIndex < 0 ||
        function.functionIndex >= static_cast<int>(FunctionTable.size())) {
      SnapshotError("snapshot State has an invalid function-stack index");
    }
    const int arg_type = static_cast<int>(function.argType);
    if (arg_type < static_cast<int>(ArgType::None) ||
        arg_type > static_cast<int>(ArgType::III) || function.callCount == 0 ||
        function.calledCount > function.callCount) {
      SnapshotError("snapshot State has an invalid function-stack entry");
    }
  }
}

std::vector<uint8_t> SerializeSnapshot(const ApiData& data) {
  RequirePostBattleStartBoundary(data);
  SnapshotWriter writer;
  writer.u32(kSnapshotMagic);
  writer.u32(kSnapshotAbiVersion);
  writer.u32(kCaptureBoundaryPostBattleStartFirstSelection);
  WriteGame(writer, data.game);
  WriteApiData(writer, data);
  const std::vector<uint8_t>& before_checksum = writer.data();
  writer.u64(Fnv1a64(before_checksum.data(), before_checksum.size()));
  if (writer.data().size() > kMaximumSnapshotBytes) {
    SnapshotError("snapshot exceeds maximum size");
  }
  return std::move(writer.data());
}

uint32_t ReadLe32(const uint8_t* source) {
  return static_cast<uint32_t>(source[0]) |
         (static_cast<uint32_t>(source[1]) << 8) |
         (static_cast<uint32_t>(source[2]) << 16) |
         (static_cast<uint32_t>(source[3]) << 24);
}

uint64_t ReadLe64(const uint8_t* source) {
  uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(source[shift / 8]) << shift;
  }
  return value;
}

uint64_t ValidateSnapshotEnvelope(const uint8_t* bytes, size_t count) {
  constexpr size_t kHeaderBytes = sizeof(uint32_t) * 3;
  constexpr size_t kChecksumBytes = sizeof(uint64_t);
  if (bytes == nullptr || count < kHeaderBytes + kChecksumBytes ||
      count > kMaximumSnapshotBytes) {
    SnapshotError("invalid snapshot size");
  }
  if (ReadLe32(bytes) != kSnapshotMagic ||
      ReadLe32(bytes + sizeof(uint32_t)) != kSnapshotAbiVersion ||
      ReadLe32(bytes + sizeof(uint32_t) * 2) !=
          kCaptureBoundaryPostBattleStartFirstSelection) {
    SnapshotError("unsupported pairing snapshot ABI");
  }
  const size_t checksum_offset = count - kChecksumBytes;
  const uint64_t expected_checksum = ReadLe64(bytes + checksum_offset);
  const uint64_t actual_checksum = Fnv1a64(bytes, checksum_offset);
  if (actual_checksum != expected_checksum) {
    SnapshotError("snapshot integrity check failed");
  }
  return expected_checksum;
}

std::unique_ptr<ApiData> DeserializeSnapshot(const std::vector<uint8_t>& blob) {
  const uint64_t expected_checksum =
      ValidateSnapshotEnvelope(blob.data(), blob.size());

  SnapshotReader reader(blob);
  if (reader.u32() != kSnapshotMagic || reader.u32() != kSnapshotAbiVersion ||
      reader.u32() != kCaptureBoundaryPostBattleStartFirstSelection) {
    SnapshotError("unsupported pairing snapshot ABI");
  }
  std::unique_ptr<ApiData> restored(new ApiData());
  restored->game = ReadGame(reader);
  ReadApiData(reader, *restored);
  restored->state.game = &restored->game;
  // Search belongs only to AgentStart / apiDataType=2.  It is intentionally
  // empty on a restored battle object and never aliases source-side memory.
  if (reader.u64() != expected_checksum || !reader.at_end()) {
    SnapshotError("snapshot payload does not terminate at checksum");
  }
  ValidateRestoredStateFunctionStack(restored->state);
  RequirePostBattleStartBoundary(*restored);
  return restored;
}

struct RtpPairingSnapshot {
  std::vector<uint8_t> blob;
};

int CopyBlob(const RtpPairingSnapshot* snapshot,
             unsigned char* output,
             int capacity) {
  if (snapshot == nullptr) {
    SetError("snapshot is null");
    return -1;
  }
  if (snapshot->blob.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    SetError("snapshot is too large for ABI");
    return -1;
  }
  const int required = static_cast<int>(snapshot->blob.size());
  if (output == nullptr || capacity < required) {
    return required;
  }
  if (required > 0) {
    std::memcpy(output, snapshot->blob.data(), snapshot->blob.size());
  }
  return required;
}

StartData PairingBattleStartSeeded(const int* cards, uint32_t requested_seed) {
  if (cards == nullptr) {
    return {nullptr, -1, 30};
  }
  std::unique_ptr<ApiData> data(new ApiData());
  data->apiDataType = 1;

  GameConfig config = {};
  // Game::init treats zero as a request for random_device.  Keep its temporary
  // initialization nonzero, then restore the exact requested zero seed before
  // any game transition.  This extension never obtains entropy from the host.
  config.seed = requested_seed == 0U ? 1U : requested_seed;
  config.recordLog = true;
  config.deviceRand = false;
  for (int player = 0; player < 2; ++player) {
    std::unordered_map<std::u8string, int> name_count;
    bool ace_spec = false;
    bool has_basic = false;
    for (int slot = 0; slot < DECK_SIZE; ++slot) {
      const CardId id = cards[player * DECK_SIZE + slot];
      const auto found = CardTable.find(id);
      if (found == CardTable.end()) {
        return {nullptr, player, 1};
      }
      const CardMaster& master = found->second;
      if (master.aceSpec) {
        if (ace_spec) {
          return {nullptr, player, 4};
        }
        ace_spec = true;
      }
      if (master.cardType == CardType::Pokemon &&
          master.evolutionType == EvolutionType::Basic) {
        has_basic = true;
      }
      int& count = name_count[master.name];
      ++count;
      if (count > DECK_SAME_CARD_MAX && master.cardType != CardType::BasicEnergy) {
        return {nullptr, player, 2};
      }
      config.decks[player].cards[slot] = id;
    }
    if (!has_basic) {
      return {nullptr, player, 3};
    }
  }

  data->init(config);
  data->game.config.seed = requested_seed;
  data->game.config.deviceRand = false;
  data->game.rng = std::mt19937(requested_seed);
  data->start();
  data->next();
  return {data.release(), -1, 0};
}

}  // namespace

extern "C" {

RTP_PAIRING_API int RtpPairingSnapshotAbiVersion() {
  return static_cast<int>(kSnapshotAbiVersion);
}

RTP_PAIRING_API const char* RtpPairingSnapshotLastError() {
  return g_last_error.c_str();
}

// Private idempotent initialization.  It must be the first initialization in
// an isolated evaluation worker; public GameInitialize is intentionally not
// reused because its upstream duplicate-call protection is assert-only.
RTP_PAIRING_API int RtpPairingSnapshotInitialize() {
  ClearError();
  try {
    RequirePrivatePairingEngineInitialized();
    return 0;
  } catch (const std::exception& exc) {
    SetError(exc.what());
    return 2;
  } catch (...) {
    SetError("unknown private pairing initialization error");
    return 99;
  }
}

// Stable overlay-owned start ABI.  New callers use out parameters rather than
// relying on compiler-specific struct-return conventions for StartData.
RTP_PAIRING_API int RtpPairingBattleStartSeededOut(
    const int* cards,
    unsigned int seed,
    ApiData** out_battle,
    int* out_error_player,
    int* out_error_type) {
  ClearError();
  if (cards == nullptr || out_battle == nullptr || out_error_player == nullptr ||
      out_error_type == nullptr) {
    SetError("seeded pairing start requires non-null arguments");
    return 1;
  }
  *out_battle = nullptr;
  *out_error_player = -1;
  *out_error_type = 30;
  try {
    RequirePrivatePairingEngineInitialized();
    StartData started = PairingBattleStartSeeded(cards, static_cast<uint32_t>(seed));
    *out_battle = started.battlePtr;
    *out_error_player = started.errorPlayer;
    *out_error_type = started.errorType;
    return started.battlePtr == nullptr ? 2 : 0;
  } catch (const std::exception& exc) {
    SetError(exc.what());
    *out_error_type = 99;
    return 99;
  } catch (...) {
    SetError("unknown seeded pairing-start error");
    *out_error_type = 99;
    return 99;
  }
}

// Overlay-owned observation ABI.  ``out_json`` remains valid only until the
// next API call that mutates this ApiData (Select, another observation call,
// or BattleFinish).  The caller must copy exactly ``out_json_count`` bytes
// before then.  This deliberately keeps Python out of the upstream
// compiler-specific SerialData struct-return ABI.
RTP_PAIRING_API int RtpPairingSnapshotGetBattleJsonOut(
    ApiData* data,
    const char** out_json,
    int* out_json_count,
    int* out_select_player) {
  ClearError();
  if (data == nullptr || out_json == nullptr || out_json_count == nullptr ||
      out_select_player == nullptr) {
    SetError("observation requires non-null battle and output pointers");
    return 1;
  }
  *out_json = nullptr;
  *out_json_count = 0;
  *out_select_player = -1;
  try {
    RequirePrivatePairingEngineInitialized();
    if (data->apiDataType != 1) {
      SetError("observation requires battle ApiData");
      return 30;
    }
    if (data->preGetSelectCount != data->selectCount) {
      const State& state = data->state;
      const int index = std::max(state.logIndex[0], state.logIndex[1]);
      const std::vector<int>* selected = nullptr;
      if (!data->visData.empty()) {
        selected = &data->selected;
      }
      ToJsonVis(data->state, data->jsonBuilder, index, selected);
      data->visData.push_back(data->jsonBuilder.buf);
      data->preGetSelectCount = data->selectCount;
    }
    const SerialData serial = ApiGetBattleData(data);
    if (serial.json == nullptr || data->jsonBuilder.buf.size() >
                                      static_cast<size_t>(std::numeric_limits<int>::max())) {
      SnapshotError("observation JSON is unavailable or too large");
    }
    *out_json = reinterpret_cast<const char*>(serial.json);
    *out_json_count = static_cast<int>(data->jsonBuilder.buf.size());
    *out_select_player = serial.selectPlayer;
    return 0;
  } catch (const std::exception& exc) {
    SetError(exc.what());
    return 2;
  } catch (...) {
    SetError("unknown pairing observation error");
    return 99;
  }
}

RTP_PAIRING_API int RtpPairingSnapshotCapture(ApiData* source, void** out_snapshot) {
  ClearError();
  if (source == nullptr || out_snapshot == nullptr) {
    SetError("capture requires non-null source and output pointer");
    return 1;
  }
  *out_snapshot = nullptr;
  try {
    RequirePrivatePairingEngineInitialized();
    std::unique_ptr<RtpPairingSnapshot> snapshot(new RtpPairingSnapshot());
    snapshot->blob = SerializeSnapshot(*source);
    *out_snapshot = snapshot.release();
    return 0;
  } catch (const std::exception& exc) {
    SetError(exc.what());
    return 2;
  } catch (...) {
    SetError("unknown snapshot capture error");
    return 99;
  }
}

RTP_PAIRING_API int RtpPairingSnapshotRestore(const void* raw_snapshot,
                                               ApiData** out_battle) {
  ClearError();
  if (raw_snapshot == nullptr || out_battle == nullptr) {
    SetError("restore requires non-null snapshot and output pointer");
    return 1;
  }
  *out_battle = nullptr;
  try {
    RequirePrivatePairingEngineInitialized();
    const RtpPairingSnapshot* snapshot =
        static_cast<const RtpPairingSnapshot*>(raw_snapshot);
    std::unique_ptr<ApiData> restored = DeserializeSnapshot(snapshot->blob);
    *out_battle = restored.release();
    return 0;
  } catch (const std::exception& exc) {
    SetError(exc.what());
    return 2;
  } catch (...) {
    SetError("unknown snapshot restore error");
    return 99;
  }
}

// Cross-process restore entry point for private, externally sealed,
// manifest-SHA-256-verified same-build bytes only.  A caller must supply the
// exact raw 32-byte SHA-256 from its immutable snapshot-artifact identity;
// this native check occurs before the upstream State decoder sees any bytes.
// The SHA binding detects a swapped/corrupted blob but is not an authorization
// mechanism for arbitrary callers of this private ABI.  Untrusted network or
// user-supplied bytes are outside this ABI and must never reach it.
RTP_PAIRING_API int RtpPairingSnapshotRestoreSerialized(
    const unsigned char* serialized,
    int serialized_count,
    const unsigned char* sealed_sha256,
    int sealed_sha256_count,
    ApiData** out_battle) {
  ClearError();
  if (serialized == nullptr || sealed_sha256 == nullptr || out_battle == nullptr ||
      serialized_count <= 0 || sealed_sha256_count != 32) {
    SetError("serialized restore requires bytes, a 32-byte sealed SHA-256, and output pointer");
    return 1;
  }
  *out_battle = nullptr;
  try {
    RequirePrivatePairingEngineInitialized();
    const size_t count = static_cast<size_t>(serialized_count);
    if (count > kMaximumSnapshotBytes) {
      SnapshotError("serialized restore exceeds maximum snapshot size");
    }
    const std::array<uint8_t, 32> actual_sha256 = Sha256(serialized, count);
    if (!std::equal(actual_sha256.begin(), actual_sha256.end(), sealed_sha256)) {
      SnapshotError("serialized restore does not match the sealed snapshot SHA-256");
    }
    ValidateSnapshotEnvelope(serialized, count);
    std::vector<uint8_t> blob(serialized, serialized + count);
    std::unique_ptr<ApiData> restored = DeserializeSnapshot(blob);
    *out_battle = restored.release();
    return 0;
  } catch (const std::exception& exc) {
    SetError(exc.what());
    return 2;
  } catch (...) {
    SetError("unknown serialized snapshot restore error");
    return 99;
  }
}

RTP_PAIRING_API void RtpPairingSnapshotRelease(void* raw_snapshot) {
  delete static_cast<RtpPairingSnapshot*>(raw_snapshot);
}

RTP_PAIRING_API int RtpPairingSnapshotSerializedSize(const void* raw_snapshot) {
  ClearError();
  return CopyBlob(static_cast<const RtpPairingSnapshot*>(raw_snapshot), nullptr, 0);
}

RTP_PAIRING_API int RtpPairingSnapshotSerialize(const void* raw_snapshot,
                                                 unsigned char* output,
                                                 int capacity) {
  ClearError();
  if (capacity < 0) {
    SetError("snapshot output capacity is negative");
    return -1;
  }
  return CopyBlob(static_cast<const RtpPairingSnapshot*>(raw_snapshot), output, capacity);
}

// The canonical fingerprint bytes are the versioned snapshot payload itself.
// Python computes SHA-256 over these bytes and binds it to the evaluation cell.
RTP_PAIRING_API int RtpPairingSnapshotFingerprintSize(const void* raw_snapshot) {
  ClearError();
  return CopyBlob(static_cast<const RtpPairingSnapshot*>(raw_snapshot), nullptr, 0);
}

RTP_PAIRING_API int RtpPairingSnapshotFingerprint(const void* raw_snapshot,
                                                   unsigned char* output,
                                                   int capacity) {
  ClearError();
  if (capacity < 0) {
    SetError("snapshot fingerprint capacity is negative");
    return -1;
  }
  return CopyBlob(static_cast<const RtpPairingSnapshot*>(raw_snapshot), output, capacity);
}

}  // extern "C"
