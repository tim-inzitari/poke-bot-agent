#pragma once

#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "wave_dispatch/client.hpp"
#include "wave_dispatch/frame.hpp"
#include "wave_dispatch/scheduler.hpp"

namespace wave_dispatch {

/** Per-endpoint claim credit ledger used by scheduled collectors. */
class ClaimLedger {
 public:
  explicit ClaimLedger(int total_jobs);

  void set_endpoint_targets(const std::unordered_map<std::string, int>& targets);
  bool try_claim_remote(const std::string& endpoint, int n);
  void note_remote_claimed(const std::string& endpoint, int n);
  int claimed(const std::string& endpoint) const;
  int target(const std::string& endpoint) const;
  int total_jobs() const { return total_jobs_; }

 private:
  int total_jobs_ = 0;
  mutable std::mutex mu_;
  std::unordered_map<std::string, int> targets_;
  std::unordered_map<std::string, int> claimed_;
};

/** Endpoint-owned job queues so remotes cannot starve each other. */
class EndpointQueues {
 public:
  void ensure(const std::string& endpoint);
  void push(const std::string& endpoint, Json job);
  bool pop(const std::string& endpoint, Json& out);
  std::size_t size(const std::string& endpoint) const;
  std::vector<std::string> endpoints() const;

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::deque<Json>> queues_;
};

using LocalSubmitFn = std::function<Json(const Json& job)>;
using ResultCallback = std::function<void(const Json& result)>;

struct CollectConfig {
  int local_workers = 1;
  int remote_chunk = 128;
  std::string kind = "play";
};

/**
 * Run a scheduled additive wave: local callback + remote JobClients.
 * Rebalances new claims via MidWaveScheduler; never cancels in-flight work.
 *
 * Returns number of successful results delivered to on_result.
 */
int run_scheduled_wave(const std::vector<Json>& jobs, LocalSubmitFn local_submit,
                       std::vector<JobClient*>& remote_clients,
                       MidWaveScheduler& scheduler, CollectConfig config,
                       ResultCallback on_result);

}  // namespace wave_dispatch
