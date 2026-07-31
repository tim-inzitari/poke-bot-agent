#pragma once

#include <cstdint>

#define WAVE_DISPATCH_VERSION_MAJOR 0
#define WAVE_DISPATCH_VERSION_MINOR 2
#define WAVE_DISPATCH_VERSION_PATCH 0
#define WAVE_DISPATCH_VERSION_STRING "0.2.0"

namespace wave_dispatch {

inline constexpr int kProtoVersion = 1;
/** Binary frame magic: 'W''D''B''1' */
inline constexpr std::uint32_t kBinaryMagic = 0x31424457u;  // little-endian "WDB1"
inline constexpr std::uint32_t kMaxFrameBytes = 256u * 1024u * 1024u;
inline constexpr int kDefaultPort = 8765;

}  // namespace wave_dispatch
