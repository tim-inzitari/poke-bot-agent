#include "wave_dispatch/buffer_pool.hpp"

#include <algorithm>

namespace wave_dispatch {

BufferPool& BufferPool::instance() {
  static BufferPool pool;
  return pool;
}

std::vector<std::uint8_t> BufferPool::acquire(std::size_t min_size) {
  std::lock_guard<std::mutex> lock(mu_);
  ++stats_.acquires;
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->capacity() >= min_size) {
      std::vector<std::uint8_t> out = std::move(*it);
      free_.erase(it);
      out.clear();
      if (out.capacity() < min_size) {
        out.reserve(min_size);
      }
      ++stats_.hits;
      return out;
    }
  }
  std::vector<std::uint8_t> out;
  out.reserve(std::max(min_size, std::size_t{4096}));
  return out;
}

void BufferPool::release(std::vector<std::uint8_t>&& buf) {
  std::lock_guard<std::mutex> lock(mu_);
  ++stats_.releases;
  buf.clear();
  if (buf.capacity() == 0) {
    return;
  }
  // Cap pool size to avoid unbounded RSS
  if (free_.size() >= 64) {
    // Drop smallest
    auto it = std::min_element(
        free_.begin(), free_.end(),
        [](const auto& a, const auto& b) { return a.capacity() < b.capacity(); });
    if (it != free_.end() && it->capacity() < buf.capacity()) {
      *it = std::move(buf);
    }
    return;
  }
  free_.push_back(std::move(buf));
}

BufferPool::Stats BufferPool::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

}  // namespace wave_dispatch
