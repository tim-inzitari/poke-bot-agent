// Additive StepBatch shim over stock competition libcg.so.
//
// Does NOT require ptcg_engine C++ sources. dlopens the official shared library
// and runs Select + GetBattleData for N handles in one C call so Python pays
// one ctypes round-trip instead of N.
//
// Stock ABI is unchanged; this .so only adds StepBatch / StepBatchFreeJsons.
// BattleStartSeeded needs an in-tree fork of Export.cpp (random_device) — not
// implemented in this shim.
//
// License: competition-use-only (derived from CABT / ptcg_engine terms). Keep
// LICENSE beside any redistributed binary.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct StartData {
  void* battlePtr;
  int errorPlayer;
  int errorType;
};

struct SerialData {
  const char* json;
  unsigned char* data;
  int count;
  int selectPlayer;
};

using BattleStartFn = StartData (*)(int*);
using SelectFn = int (*)(void*, int*, int);
using GetBattleDataFn = SerialData (*)(void*);
using BattleFinishFn = void (*)(void*);
using GameInitializeFn = void (*)();

struct StockLib {
  void* handle = nullptr;
  BattleStartFn BattleStart = nullptr;
  SelectFn Select = nullptr;
  GetBattleDataFn GetBattleData = nullptr;
  BattleFinishFn BattleFinish = nullptr;
  GameInitializeFn GameInitialize = nullptr;
  bool ok = false;
  std::string error;
};

StockLib g_stock;
std::once_flag g_once;

const char* env_or(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v && v[0]) ? v : fallback;
}

void load_stock() {
  const char* path = env_or("LIBCG_SO", "libcg.so");
  dlerror();
  g_stock.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!g_stock.handle) {
    const char* err = dlerror();
    g_stock.error = err ? err : "dlopen failed";
    return;
  }
  auto sym = [&](const char* name) -> void* {
    dlerror();
    void* p = dlsym(g_stock.handle, name);
    const char* err = dlerror();
    if (err || !p) {
      g_stock.error = std::string(name) + ": " + (err ? err : "null");
      return nullptr;
    }
    return p;
  };
  g_stock.BattleStart = reinterpret_cast<BattleStartFn>(sym("BattleStart"));
  g_stock.Select = reinterpret_cast<SelectFn>(sym("Select"));
  g_stock.GetBattleData = reinterpret_cast<GetBattleDataFn>(sym("GetBattleData"));
  g_stock.BattleFinish = reinterpret_cast<BattleFinishFn>(sym("BattleFinish"));
  g_stock.GameInitialize = reinterpret_cast<GameInitializeFn>(sym("GameInitialize"));
  if (!g_stock.BattleStart || !g_stock.Select || !g_stock.GetBattleData ||
      !g_stock.BattleFinish || !g_stock.GameInitialize) {
    return;
  }
  // GameInitialize is idempotent only if card tables are empty; stock cg.sim
  // already calls it. We do NOT call it here to avoid double-init asserts.
  g_stock.ok = true;
}

bool ensure_stock() {
  std::call_once(g_once, load_stock);
  return g_stock.ok;
}

char* dup_cstr(const char* s) {
  if (!s) {
    return nullptr;
  }
  size_t n = std::strlen(s);
  char* out = static_cast<char*>(std::malloc(n + 1));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, s, n + 1);
  return out;
}

}  // namespace

extern "C" {

// Probe: 1 if stock lib resolved, 0 otherwise. Optional error string via
// StepBatchLastError().
int StepBatchReady(void) { return ensure_stock() ? 1 : 0; }

const char* StepBatchLastError(void) {
  ensure_stock();
  return g_stock.error.empty() ? "" : g_stock.error.c_str();
}

// StepBatch — one C call for N envs.
//
// handles[i]          : ApiData* (void*)
// n                   : env count
// action_flat         : concatenated option-index lists
// action_offsets[i]   : start index into action_flat for env i
// action_lens[i]      : length for env i; 0 => skip Select (still may fetch JSON
//                       if fetch_obs_on_skip != 0)
// fetch_obs_on_skip   : if non-zero, GetBattleData even when action_lens[i]==0
// copy_json           : if non-zero, malloc-copy JSON (free with
//                       StepBatchFreeJsons). If 0, return borrowed pointers from
//                       stock GetBattleData (valid until next Select/GetBattleData/
//                       StepBatch on that handle — same as stock ABI).
// out_errors[i]       : Select return code; 0 ok; -1 skipped; -2 bad args /
//                       stock not loaded; -3 malloc failure on json copy
// out_jsons[i]        : JSON UTF-8 or nullptr
// out_select_players[i]: SerialData.selectPlayer (or -1 if no fetch)
//
// Returns 0 on ABI success (per-env errors still in out_errors). Non-zero only
// if the call itself is invalid (null required buffers, n<0, stock missing).
int StepBatch(
    void** handles,
    int n,
    const int* action_flat,
    const int* action_offsets,
    const int* action_lens,
    int fetch_obs_on_skip,
    int copy_json,
    int* out_errors,
    char** out_jsons,
    int* out_select_players) {
  if (n < 0 || !out_errors || !out_jsons) {
    return 1;
  }
  if (!ensure_stock()) {
    for (int i = 0; i < n; ++i) {
      out_errors[i] = -2;
      out_jsons[i] = nullptr;
      if (out_select_players) {
        out_select_players[i] = -1;
      }
    }
    return 2;
  }
  if (n > 0 && (!handles || !action_offsets || !action_lens)) {
    return 1;
  }

  for (int i = 0; i < n; ++i) {
    out_jsons[i] = nullptr;
    if (out_select_players) {
      out_select_players[i] = -1;
    }
    void* h = handles[i];
    int len = action_lens[i];
    if (!h) {
      out_errors[i] = -1;
      continue;
    }
    if (len < 0) {
      out_errors[i] = -2;
      continue;
    }
    if (len == 0) {
      out_errors[i] = -1;
      if (!fetch_obs_on_skip) {
        continue;
      }
    } else {
      if (!action_flat) {
        out_errors[i] = -2;
        continue;
      }
      int off = action_offsets[i];
      // Select mutates the int* buffer in some engines; copy to scratch.
      std::vector<int> scratch(static_cast<size_t>(len));
      std::memcpy(scratch.data(), action_flat + off, sizeof(int) * static_cast<size_t>(len));
      int err = g_stock.Select(h, scratch.data(), len);
      out_errors[i] = err;
      if (err != 0) {
        continue;
      }
    }
    SerialData sd = g_stock.GetBattleData(h);
    if (out_select_players) {
      out_select_players[i] = sd.selectPlayer;
    }
    if (copy_json) {
      char* copy = dup_cstr(sd.json);
      if (sd.json && !copy) {
        out_errors[i] = -3;
        continue;
      }
      out_jsons[i] = copy;
    } else {
      // Borrowed — cast away const for ctypes c_char_p symmetry with stock.
      out_jsons[i] = const_cast<char*>(sd.json);
    }
  }
  return 0;
}

void StepBatchFreeJsons(char** jsons, int n) {
  if (!jsons || n <= 0) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    std::free(jsons[i]);
    jsons[i] = nullptr;
  }
}

// Forwarders so a single CDLL can expose stock + batch (optional convenience).
// Prefer loading stock via cg.sim and this .so for StepBatch only.

StartData BattleStart(int* cards) {
  if (!ensure_stock()) {
    return StartData{nullptr, -1, -1};
  }
  return g_stock.BattleStart(cards);
}

int Select(void* data, int* select, int count) {
  if (!ensure_stock()) {
    return -2;
  }
  return g_stock.Select(data, select, count);
}

SerialData GetBattleData(void* data) {
  if (!ensure_stock()) {
    return SerialData{nullptr, nullptr, 0, -1};
  }
  return g_stock.GetBattleData(data);
}

void BattleFinish(void* data) {
  if (!ensure_stock()) {
    return;
  }
  g_stock.BattleFinish(data);
}

void GameInitialize(void) {
  if (!ensure_stock()) {
    return;
  }
  g_stock.GameInitialize();
}

}  // extern "C"
