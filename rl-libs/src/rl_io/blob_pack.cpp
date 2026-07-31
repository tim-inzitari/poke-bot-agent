#include "rl_io/blob_pack.hpp"

#include "rl_io/digest.hpp"
#include "rl_io/error.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace rl_io {
namespace {

constexpr char kMagic[4] = {'R', 'L', 'P', 'K'};
constexpr std::uint32_t kVersion = 1;

void write_u32(std::ostream& out, std::uint32_t v) {
  char b[4] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff),
               static_cast<char>((v >> 16) & 0xff),
               static_cast<char>((v >> 24) & 0xff)};
  out.write(b, 4);
}

void write_u64(std::ostream& out, std::uint64_t v) {
  write_u32(out, static_cast<std::uint32_t>(v & 0xffffffffu));
  write_u32(out, static_cast<std::uint32_t>((v >> 32) & 0xffffffffu));
}

std::uint32_t read_u32(const std::uint8_t* p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
         (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}

std::uint64_t read_u64(const std::uint8_t* p) {
  return std::uint64_t(read_u32(p)) | (std::uint64_t(read_u32(p + 4)) << 32);
}

}  // namespace

void BlobPackWriter::set_manifest(Json manifest) { manifest_ = std::move(manifest); }

void BlobPackWriter::add(std::string name, std::vector<std::uint8_t> bytes) {
  blobs_.emplace_back(std::move(name), std::move(bytes));
}

void BlobPackWriter::add(std::string name, std::string_view bytes) {
  blobs_.emplace_back(
      std::move(name),
      std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
}

void BlobPackWriter::commit(const std::string& path) const {
  Json index = Json::array();
  std::vector<std::uint8_t> payload;
  std::uint64_t offset = 0;
  for (const auto& [name, bytes] : blobs_) {
    index.push_back({
        {"name", name},
        {"offset", offset},
        {"length", bytes.size()},
        {"sha256", sha256_hex(bytes)},
    });
    payload.insert(payload.end(), bytes.begin(), bytes.end());
    offset += bytes.size();
  }
  const std::string manifest = manifest_.dump();
  const std::string index_s = index.dump();
  fs::path dest(path);
  fs::create_directories(dest.parent_path());
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream out(tmp, std::ios::binary);
    if (!out) throw IoError("cannot open pack temp: " + tmp);
    out.write(kMagic, 4);
    write_u32(out, kVersion);
    write_u32(out, static_cast<std::uint32_t>(manifest.size()));
    write_u32(out, static_cast<std::uint32_t>(index_s.size()));
    write_u64(out, payload.size());
    write_u32(out, 0);  // reserved
    out.write(manifest.data(), static_cast<std::streamsize>(manifest.size()));
    out.write(index_s.data(), static_cast<std::streamsize>(index_s.size()));
    if (!payload.empty()) {
      out.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
    if (!out) throw IoError("pack write failed");
  }
  {
    FILE* fp = std::fopen(tmp.c_str(), "r+b");
    if (!fp) throw IoError("pack fsync open failed");
    std::fflush(fp);
    ::fsync(::fileno(fp));
    std::fclose(fp);
  }
  fs::rename(tmp, path);
}

BlobPackReader::BlobPackReader(const std::string& path, bool verify) {
  open_(path, verify);
}

BlobPackReader::~BlobPackReader() { close_(); }

BlobPackReader::BlobPackReader(BlobPackReader&& other) noexcept {
  *this = std::move(other);
}

BlobPackReader& BlobPackReader::operator=(BlobPackReader&& other) noexcept {
  if (this != &other) {
    close_();
    fd_ = other.fd_;
    map_ = other.map_;
    map_size_ = other.map_size_;
    manifest_ = std::move(other.manifest_);
    index_ = std::move(other.index_);
    payload_ = other.payload_;
    payload_len_ = other.payload_len_;
    other.fd_ = -1;
    other.map_ = nullptr;
    other.map_size_ = 0;
    other.payload_ = nullptr;
    other.payload_len_ = 0;
  }
  return *this;
}

void BlobPackReader::close_() {
  if (map_ && map_ != MAP_FAILED) {
    ::munmap(map_, map_size_);
  }
  map_ = nullptr;
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
}

void BlobPackReader::open_(const std::string& path, bool verify) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw IoError("cannot open pack: " + path);
  struct stat st {};
  if (::fstat(fd_, &st) != 0) throw IoError("fstat failed: " + path);
  map_size_ = static_cast<std::size_t>(st.st_size);
  if (map_size_ < 28) throw ProtocolError("pack too small");
  map_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (map_ == MAP_FAILED) throw IoError("mmap failed: " + path);
  const auto* base = static_cast<const std::uint8_t*>(map_);
  if (std::memcmp(base, kMagic, 4) != 0) throw ProtocolError("bad pack magic");
  const auto version = read_u32(base + 4);
  if (version != kVersion) throw ProtocolError("unsupported pack version");
  const auto manifest_len = read_u32(base + 8);
  const auto index_len = read_u32(base + 12);
  payload_len_ = read_u64(base + 16);
  const std::size_t header = 28;
  if (header + manifest_len + index_len + payload_len_ > map_size_) {
    throw ProtocolError("pack truncated");
  }
  const char* man_ptr = reinterpret_cast<const char*>(base + header);
  manifest_ = Json::parse(std::string(man_ptr, manifest_len));
  const char* idx_ptr = man_ptr + manifest_len;
  Json index = Json::parse(std::string(idx_ptr, index_len));
  payload_ = reinterpret_cast<const std::uint8_t*>(idx_ptr + index_len);
  for (const auto& row : index) {
    Entry e;
    e.offset = row.at("offset").get<std::uint64_t>();
    e.length = row.at("length").get<std::uint64_t>();
    e.sha256 = row.at("sha256").get<std::string>();
    if (e.offset + e.length > payload_len_) throw ProtocolError("blob OOB");
    const std::string name = row.at("name").get<std::string>();
    if (verify) {
      const auto hex = sha256_hex(payload_ + e.offset, static_cast<std::size_t>(e.length));
      if (hex != e.sha256) throw ProtocolError("blob checksum mismatch: " + name);
    }
    index_.emplace(name, std::move(e));
  }
}

std::vector<std::string> BlobPackReader::names() const {
  std::vector<std::string> out;
  out.reserve(index_.size());
  for (const auto& [k, _] : index_) out.push_back(k);
  return out;
}

bool BlobPackReader::contains(const std::string& name) const {
  return index_.count(name) != 0;
}

std::string_view BlobPackReader::view(const std::string& name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) throw ProtocolError("missing blob: " + name);
  return std::string_view(reinterpret_cast<const char*>(payload_ + it->second.offset),
                          static_cast<std::size_t>(it->second.length));
}

std::vector<std::uint8_t> BlobPackReader::get(const std::string& name) const {
  auto v = view(name);
  return std::vector<std::uint8_t>(v.begin(), v.end());
}

std::string BlobPackReader::blob_sha256(const std::string& name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) throw ProtocolError("missing blob: " + name);
  return it->second.sha256;
}

}  // namespace rl_io
