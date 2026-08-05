#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace wave_dispatch {

/** Recycled byte buffers — avoids alloc churn on the frame hot path. */
class BufferPool {
 public:
  static BufferPool& instance();

  std::vector<std::uint8_t> acquire(std::size_t min_size);
  void release(std::vector<std::uint8_t>&& buf);

  struct Stats {
    std::size_t acquires = 0;
    std::size_t releases = 0;
    std::size_t hits = 0;
  };
  Stats stats() const;

 private:
  BufferPool() = default;
  mutable std::mutex mu_;
  std::vector<std::vector<std::uint8_t>> free_;
  Stats stats_;
};

/** RAII return of a buffer to the pool. */
class PooledBuffer {
 public:
  explicit PooledBuffer(std::size_t min_size = 0)
      : buf_(BufferPool::instance().acquire(min_size)) {}
  ~PooledBuffer() {
    if (!released_) {
      BufferPool::instance().release(std::move(buf_));
    }
  }
  PooledBuffer(const PooledBuffer&) = delete;
  PooledBuffer& operator=(const PooledBuffer&) = delete;
  PooledBuffer(PooledBuffer&& o) noexcept
      : buf_(std::move(o.buf_)), released_(o.released_) {
    o.released_ = true;
  }

  std::vector<std::uint8_t>& get() { return buf_; }
  std::vector<std::uint8_t> take() {
    released_ = true;
    return std::move(buf_);
  }

 private:
  std::vector<std::uint8_t> buf_;
  bool released_ = false;
};

}  // namespace wave_dispatch
