#pragma once

#include <cstdint>

#define WAVE_DISPATCH_VERSION_MAJOR 0
#define WAVE_DISPATCH_VERSION_MINOR 1
#define WAVE_DISPATCH_VERSION_PATCH 0
#define WAVE_DISPATCH_VERSION_STRING "0.1.0"

namespace wave_dispatch {

inline constexpr int kProtoVersion = 1;
inline constexpr std::uint32_t kMaxFrameBytes = 256u * 1024u * 1024u;  // 256 MiB
inline constexpr int kDefaultPort = 8765;

}  // namespace wave_dispatch
