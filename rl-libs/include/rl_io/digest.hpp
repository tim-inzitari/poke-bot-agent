#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rl_io {

/** Hex SHA-256 without algorithm prefix. */
std::string sha256_hex(const std::uint8_t* data, std::size_t n);
std::string sha256_hex(std::string_view data);
std::string sha256_hex(const std::vector<std::uint8_t>& data);

/** Prefixed digest: ``sha256:<hex>``. */
std::string sha256_digest(const std::uint8_t* data, std::size_t n);
std::string sha256_digest(std::string_view data);
std::string sha256_file(const std::string& path, std::size_t chunk = 1 << 20);

}  // namespace rl_io
