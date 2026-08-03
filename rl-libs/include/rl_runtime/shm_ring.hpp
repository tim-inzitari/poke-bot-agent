#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rl_runtime {

struct RingConfig {
  std::string name;                 // POSIX shm name, e.g. "/rl_leaf"
  std::uint32_t slot_count = 8;     // response lanes
  std::uint32_t request_slots = 256;
  std::uint32_t max_payload = 1 << 20;
  std::uint64_t generation = 1;
};

struct Request {
  std::uint32_t slot = 0;
  std::uint64_t rid = 0;
  std::vector<std::uint8_t> payload;
};

/**
 * POSIX shared-memory request/response rings for GPU leaf farms.
 *
 * Opaque payloads only. Model forward stays in the caller's process via a
 * coalesce loop: pop requests → batch → push responses.
 */
class ShmRing {
 public:
  static ShmRing create(const RingConfig& cfg);
  static ShmRing open(const std::string& name);
  ~ShmRing();

  ShmRing(const ShmRing&) = delete;
  ShmRing& operator=(const ShmRing&) = delete;
  ShmRing(ShmRing&& other) noexcept;
  ShmRing& operator=(ShmRing&& other) noexcept;

  const RingConfig& config() const { return cfg_; }
  std::uint64_t generation() const;

  /** Client: enqueue request; returns rid. */
  std::uint64_t submit(std::uint32_t slot,
                       const std::uint8_t* data,
                       std::size_t n,
                       double timeout_s = 30.0);

  /** Client: wait for response for rid on slot. */
  std::vector<std::uint8_t> wait(std::uint32_t slot,
                                 std::uint64_t rid,
                                 double timeout_s = 30.0);

  /** Server: non-blocking pop. */
  std::optional<Request> try_pop();

  /** Server: block until a request or timeout. */
  std::optional<Request> pop(double timeout_s);

  /**
   * Server: coalesce up to max_batch requests, waiting up to timeout_ms for the
   * first item and coalescing_ms for additional items.
   */
  std::vector<Request> coalesce(std::size_t max_batch,
                                double first_timeout_s,
                                double coalesce_s);

  void respond(std::uint32_t slot,
               std::uint64_t rid,
               const std::uint8_t* data,
               std::size_t n);

  void set_alive(bool alive);
  bool alive() const;
  void unlink();

 private:
  ShmRing() = default;
  void map_memory_();
  void close_();

  RingConfig cfg_;
  int fd_ = -1;
  void* mapped_ = nullptr;
  std::size_t map_size_ = 0;
  bool owner_ = false;
  std::atomic<std::uint64_t>* next_rid_ = nullptr;
};

}  // namespace rl_runtime
