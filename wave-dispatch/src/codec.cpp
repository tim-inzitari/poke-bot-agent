#include "wave_dispatch/codec.hpp"

#include <lz4.h>

#include "wave_dispatch/version.hpp"

namespace wave_dispatch {

const char* blob_codec_name(BlobCodec c) {
  switch (c) {
    case BlobCodec::kLz4:
      return "lz4";
    default:
      return "none";
  }
}

BlobCodec blob_codec_from_name(const std::string& name) {
  if (name == "lz4" || name == "LZ4") return BlobCodec::kLz4;
  return BlobCodec::kNone;
}

bool lz4_available() { return true; }

BlobCodec compress_blob(const std::vector<std::uint8_t>& in,
                        std::vector<std::uint8_t>& out, BlobCodec prefer) {
  out.clear();
  if (prefer != BlobCodec::kLz4 || in.empty()) {
    out = in;
    return BlobCodec::kNone;
  }
  const int bound = LZ4_compressBound(static_cast<int>(in.size()));
  if (bound <= 0) {
    out = in;
    return BlobCodec::kNone;
  }
  out.resize(static_cast<std::size_t>(bound) + 4);
  // Store original size as LE u32 prefix for decompress
  const std::uint32_t orig = static_cast<std::uint32_t>(in.size());
  out[0] = static_cast<std::uint8_t>(orig);
  out[1] = static_cast<std::uint8_t>(orig >> 8);
  out[2] = static_cast<std::uint8_t>(orig >> 16);
  out[3] = static_cast<std::uint8_t>(orig >> 24);
  const int n = LZ4_compress_default(
      reinterpret_cast<const char*>(in.data()),
      reinterpret_cast<char*>(out.data() + 4), static_cast<int>(in.size()), bound);
  if (n <= 0 || static_cast<std::size_t>(n) + 4 >= in.size()) {
    // Not worth it
    out = in;
    return BlobCodec::kNone;
  }
  out.resize(static_cast<std::size_t>(n) + 4);
  return BlobCodec::kLz4;
}

void decompress_blob(const std::vector<std::uint8_t>& in,
                     std::vector<std::uint8_t>& out, BlobCodec codec) {
  if (codec == BlobCodec::kNone) {
    out = in;
    return;
  }
  if (codec != BlobCodec::kLz4 || in.size() < 4) {
    throw ProtocolError("invalid lz4 blob");
  }
  const std::uint32_t orig = static_cast<std::uint32_t>(in[0]) |
                             (static_cast<std::uint32_t>(in[1]) << 8) |
                             (static_cast<std::uint32_t>(in[2]) << 16) |
                             (static_cast<std::uint32_t>(in[3]) << 24);
  if (orig == 0 || orig > kMaxFrameBytes) {
    throw ProtocolError("lz4 original size invalid");
  }
  out.resize(orig);
  const int n = LZ4_decompress_safe(
      reinterpret_cast<const char*>(in.data() + 4),
      reinterpret_cast<char*>(out.data()), static_cast<int>(in.size() - 4),
      static_cast<int>(orig));
  if (n < 0 || static_cast<std::uint32_t>(n) != orig) {
    throw ProtocolError("lz4 decompress failed");
  }
}

}  // namespace wave_dispatch
