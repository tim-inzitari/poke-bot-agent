#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "wave_dispatch/error.hpp"
#include "wave_dispatch/version.hpp"

namespace wave_dispatch {

using Json = nlohmann::json;

/** Encode a JSON object as !I big-endian length + UTF-8 body. */
std::vector<std::uint8_t> encode_frame(const Json& payload);

/** Decode one frame from a complete buffer (header + body). */
Json decode_frame(const std::uint8_t* data, std::size_t n);

/** Read exactly n bytes from a POSIX socket; throws on hangup/timeout. */
void recv_exact(int fd, void* buf, std::size_t n);

/** Read one length-prefixed JSON frame from a socket. */
Json read_frame(int fd);

/** Write one length-prefixed JSON frame to a socket. */
void send_frame(int fd, const Json& payload);

}  // namespace wave_dispatch
