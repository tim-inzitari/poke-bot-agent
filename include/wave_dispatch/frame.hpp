#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "wave_dispatch/error.hpp"
#include "wave_dispatch/version.hpp"

namespace wave_dispatch {

using Json = nlohmann::json;

/** Wire message: JSON meta + optional opaque binary blob (zero-copy friendly). */
struct Message {
  Json meta;
  std::vector<std::uint8_t> blob;

  bool has_blob() const { return !blob.empty(); }
};

/** Encode pure JSON frame (!I + UTF-8). v1-compatible. */
std::vector<std::uint8_t> encode_frame(const Json& payload);

/**
 * Encode binary frame:
 *   !I total_len | 'WDB1' | !I meta_len | meta_json | blob
 * Meta is small; blob is the fast path for trajectories.
 */
std::vector<std::uint8_t> encode_message(const Message& msg);

/** Decode one complete frame buffer into Message (JSON or WDB1). */
Message decode_message(const std::uint8_t* data, std::size_t n);

/** Convenience: decode JSON-only frame (blob empty). */
Json decode_frame(const std::uint8_t* data, std::size_t n);

/** Fast JSON parse (simdjson → nlohmann). */
Json parse_json_fast(std::string_view utf8);

/** Blocking POSIX helpers (used by tests / fallback). */
void recv_exact(int fd, void* buf, std::size_t n);
Message read_message(int fd);
Json read_frame(int fd);
void send_frame(int fd, const Json& payload);
void send_message(int fd, const Message& msg);

}  // namespace wave_dispatch
