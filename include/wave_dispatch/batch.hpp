#pragma once

#include <vector>

#include "wave_dispatch/frame.hpp"

namespace wave_dispatch {

/**
 * Pack many messages into one jobs/results frame.
 * meta: {type, kind?, n, items:[meta...], blob_lens:[...], blob_codec?}
 * blob: concatenation of (possibly compressed) item blobs.
 */
Message pack_batch(const std::string& type, const std::string& kind,
                   const std::vector<Message>& items, bool compress = false);

/** Unpack a jobs/results batch message into items. */
std::vector<Message> unpack_batch(const Message& batch);

}  // namespace wave_dispatch
