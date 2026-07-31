#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace rl_io {

using Json = nlohmann::json;

/**
 * Generic named-blob pack with mmap reader and atomic writer.
 *
 * On-disk layout (little-endian):
 *   magic "RLPK" (4) | version u32 | manifest_len u32 | index_len u32 |
 *   payload_len u64 | reserved u32
 *   manifest JSON bytes
 *   index JSON array of {name, offset, length, sha256}
 *   payload bytes (concatenated blobs)
 *
 * Manifest is opaque JSON owned by the caller (competition-specific keys).
 */
class BlobPackWriter {
 public:
  void set_manifest(Json manifest);
  void add(std::string name, std::vector<std::uint8_t> bytes);
  void add(std::string name, std::string_view bytes);
  void commit(const std::string& path) const;

 private:
  Json manifest_ = Json::object();
  std::vector<std::pair<std::string, std::vector<std::uint8_t>>> blobs_;
};

class BlobPackReader {
 public:
  explicit BlobPackReader(const std::string& path, bool verify = true);
  ~BlobPackReader();

  BlobPackReader(const BlobPackReader&) = delete;
  BlobPackReader& operator=(const BlobPackReader&) = delete;
  BlobPackReader(BlobPackReader&& other) noexcept;
  BlobPackReader& operator=(BlobPackReader&& other) noexcept;

  const Json& manifest() const { return manifest_; }
  std::vector<std::string> names() const;
  bool contains(const std::string& name) const;
  std::string_view view(const std::string& name) const;
  std::vector<std::uint8_t> get(const std::string& name) const;
  std::string blob_sha256(const std::string& name) const;

 private:
  void close_();
  void open_(const std::string& path, bool verify);

  int fd_ = -1;
  void* map_ = nullptr;
  std::size_t map_size_ = 0;
  Json manifest_;
  struct Entry {
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::string sha256;
  };
  std::unordered_map<std::string, Entry> index_;
  const std::uint8_t* payload_ = nullptr;
  std::uint64_t payload_len_ = 0;
};

}  // namespace rl_io
