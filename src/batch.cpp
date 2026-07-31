#include "wave_dispatch/batch.hpp"

#include "wave_dispatch/codec.hpp"

namespace wave_dispatch {

Message pack_batch(const std::string& type, const std::string& kind,
                   const std::vector<Message>& items, bool compress) {
  Message out;
  Json items_meta = Json::array();
  Json lens = Json::array();
  std::vector<std::uint8_t> blob;
  blob.reserve(4096);
  bool any_lz4 = false;

  for (const auto& it : items) {
    items_meta.push_back(it.meta);
    if (it.blob.empty()) {
      lens.push_back(0);
      continue;
    }
    std::vector<std::uint8_t> piece;
    BlobCodec used = BlobCodec::kNone;
    if (compress) {
      used = compress_blob(it.blob, piece, BlobCodec::kLz4);
    } else {
      piece = it.blob;
    }
    if (used == BlobCodec::kLz4) any_lz4 = true;
    // Per-item codec byte: 0=none 1=lz4, then bytes
    blob.push_back(static_cast<std::uint8_t>(used));
    blob.insert(blob.end(), piece.begin(), piece.end());
    lens.push_back(static_cast<int>(1 + piece.size()));
  }

  out.meta = {{"type", type},
              {"n", static_cast<int>(items.size())},
              {"items", std::move(items_meta)},
              {"blob_lens", std::move(lens)},
              {"blob_codec", any_lz4 ? "mixed" : "none"}};
  if (!kind.empty()) {
    out.meta["kind"] = kind;
  }
  out.blob = std::move(blob);
  return out;
}

std::vector<Message> unpack_batch(const Message& batch) {
  const auto& meta = batch.meta;
  if (!meta.contains("items") || !meta["items"].is_array()) {
    throw ProtocolError("batch missing items");
  }
  const auto& items_meta = meta["items"];
  const auto& lens = meta.value("blob_lens", Json::array());
  std::vector<Message> out;
  out.reserve(items_meta.size());
  std::size_t off = 0;
  for (std::size_t i = 0; i < items_meta.size(); ++i) {
    Message m;
    m.meta = items_meta[i];
    int len = 0;
    if (i < lens.size()) len = lens[i].get<int>();
    if (len > 0) {
      if (off + static_cast<std::size_t>(len) > batch.blob.size()) {
        throw ProtocolError("batch blob truncated");
      }
      const std::uint8_t codec_b = batch.blob[off];
      const auto* p = batch.blob.data() + off + 1;
      const std::size_t n = static_cast<std::size_t>(len) - 1;
      std::vector<std::uint8_t> piece(p, p + n);
      if (codec_b == static_cast<std::uint8_t>(BlobCodec::kLz4)) {
        decompress_blob(piece, m.blob, BlobCodec::kLz4);
      } else {
        m.blob = std::move(piece);
      }
      off += static_cast<std::size_t>(len);
    }
    out.push_back(std::move(m));
  }
  return out;
}

}  // namespace wave_dispatch
