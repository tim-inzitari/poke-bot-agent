#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "wave_dispatch/client.hpp"
#include "wave_dispatch/frame.hpp"
#include "wave_dispatch/pool.hpp"
#include "wave_dispatch/scheduler.hpp"

namespace wave_dispatch {

class ClaimLedger {
 public:
  explicit ClaimLedger(int total_jobs);
  /** ``targets`` are max **in-flight** jobs per endpoint, not lifetime quotas. */
  void set_endpoint_targets(const std::unordered_map<std::string, int>& targets);
  bool try_claim_remote(const std::string& endpoint, int n);
  void release_remote(const std::string& endpoint, int n);
  int claimed(const std::string& endpoint) const;
  int target(const std::string& endpoint) const;
  int total_jobs() const { return total_jobs_; }

 private:
  struct Slot {
    std::atomic<int> target{0};
    std::atomic<int> in_flight{0};
  };
  int total_jobs_ = 0;
  std::unordered_map<std::string, std::unique_ptr<Slot>> slots_;
};

using LocalSubmitFn = std::function<Json(const Json& job)>;
using LocalMessageFn = std::function<Message(const Message& job)>;
using ResultCallback = std::function<void(const Json& result)>;
using MessageResultCallback = std::function<void(const Message& result)>;

struct CollectConfig {
  int local_workers = 1;
  int remote_chunk = 128;
  std::string kind = "play";
  bool prefer_binary = true;
  /** Proto v2 multi-job frames per socket round-trip. */
  int batch_size = 16;
  bool compress_blobs = true;
  /** Reuse warm connections across the wave (and across waves if pool given). */
  bool use_connection_pool = true;
  /** Prefer Unix domain sockets for localhost. */
  bool prefer_uds = true;
  /** Async io worker threads for remote submits (0 → hardware_concurrency). */
  int async_threads = 0;
};

int run_scheduled_wave(const std::vector<Json>& jobs, LocalSubmitFn local_submit,
                       std::vector<JobClient*>& remote_clients,
                       MidWaveScheduler& scheduler, CollectConfig config,
                       ResultCallback on_result,
                       ConnectionPool* pool = nullptr);

int run_scheduled_wave_bin(const std::vector<Message>& jobs,
                           LocalMessageFn local_submit,
                           std::vector<JobClient*>& remote_clients,
                           MidWaveScheduler& scheduler, CollectConfig config,
                           MessageResultCallback on_result,
                           ConnectionPool* pool = nullptr);

}  // namespace wave_dispatch
