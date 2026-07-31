#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "wave_dispatch/error.hpp"

namespace wave_dispatch {

enum class BlobCodec {
  kNone = 0,
  kLz4 = 1,
};

const char* blob_codec_name(BlobCodec c);
BlobCodec blob_codec_from_name(const std::string& name);

/** Compress blob; returns codec actually used (may fall back to none). */
BlobCodec compress_blob(const std::vector<std::uint8_t>& in,
                        std::vector<std::uint8_t>& out, BlobCodec prefer);

/** Decompress blob into out. */
void decompress_blob(const std::vector<std::uint8_t>& in,
                     std::vector<std::uint8_t>& out, BlobCodec codec);

bool lz4_available();

}  // namespace wave_dispatch
