#include "wave_dispatch/collector.hpp"

#include <atomic>
#include <chrono>
#include <thread>

namespace wave_dispatch {

ClaimLedger::ClaimLedger(int total_jobs) : total_jobs_(std::max(0, total_jobs)) {}

void ClaimLedger::set_endpoint_targets(
    const std::unordered_map<std::string, int>& targets) {
  std::lock_guard<std::mutex> lock(mu_);
  targets_ = targets;
}

bool ClaimLedger::try_claim_remote(const std::string& endpoint, int n) {
  std::lock_guard<std::mutex> lock(mu_);
  const int tgt = targets_.count(endpoint) ? targets_.at(endpoint) : 0;
  const int have = claimed_.count(endpoint) ? claimed_.at(endpoint) : 0;
  if (have + n > tgt && tgt > 0) {
    return false;
  }
  claimed_[endpoint] = have + n;
  return true;
}

void ClaimLedger::note_remote_claimed(const std::string& endpoint, int n) {
  std::lock_guard<std::mutex> lock(mu_);
  claimed_[endpoint] = (claimed_.count(endpoint) ? claimed_.at(endpoint) : 0) + n;
}

int ClaimLedger::claimed(const std::string& endpoint) const {
  std::lock_guard<std::mutex> lock(mu_);
  return claimed_.count(endpoint) ? claimed_.at(endpoint) : 0;
}

int ClaimLedger::target(const std::string& endpoint) const {
  std::lock_guard<std::mutex> lock(mu_);
  return targets_.count(endpoint) ? targets_.at(endpoint) : 0;
}

void EndpointQueues::ensure(const std::string& endpoint) {
  std::lock_guard<std::mutex> lock(mu_);
  queues_.emplace(endpoint, std::deque<Json>{});
}

void EndpointQueues::push(const std::string& endpoint, Json job) {
  std::lock_guard<std::mutex> lock(mu_);
  queues_[endpoint].push_back(std::move(job));
}

bool EndpointQueues::pop(const std::string& endpoint, Json& out) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = queues_.find(endpoint);
  if (it == queues_.end() || it->second.empty()) {
    return false;
  }
  out = std::move(it->second.front());
  it->second.pop_front();
  return true;
}

std::size_t EndpointQueues::size(const std::string& endpoint) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = queues_.find(endpoint);
  return it == queues_.end() ? 0 : it->second.size();
}

std::vector<std::string> EndpointQueues::endpoints() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> out;
  out.reserve(queues_.size());
  for (const auto& [ep, _] : queues_) {
    out.push_back(ep);
  }
  return out;
}

int run_scheduled_wave(const std::vector<Json>& jobs, LocalSubmitFn local_submit,
                       std::vector<JobClient*>& remote_clients,
                       MidWaveScheduler& scheduler, CollectConfig config,
                       ResultCallback on_result) {
  if (jobs.empty()) {
    return 0;
  }

  std::deque<Json> remaining(jobs.begin(), jobs.end());
  std::mutex rem_mu;
  std::atomic<int> completed{0};
  std::atomic<bool> stop{false};
  std::vector<std::string> errors;
  std::mutex err_mu;

  // Initial forced tick
  scheduler.maybe_tick(static_cast<int>(jobs.size()), true);
  auto dec = scheduler.decision();

  EndpointQueues queues;
  std::unordered_map<std::string, int> demand = dec.remote_demand;
  ClaimLedger ledger(static_cast<int>(jobs.size()));
  std::unordered_map<std::string, int> targets;
  for (const auto& [ep, n] : demand) {
    queues.ensure(ep);
    // Target work ≈ demand sockets * chunk waves
    targets[ep] = std::max(n, 1) * std::max(1, config.remote_chunk);
  }
  // If no demand map, assign equal share across clients
  if (targets.empty()) {
    for (auto* c : remote_clients) {
      if (!c) continue;
      queues.ensure(c->endpoint());
      targets[c->endpoint()] =
          std::max(1, static_cast<int>(jobs.size()) /
                          std::max<int>(1, static_cast<int>(remote_clients.size())));
    }
  }
  ledger.set_endpoint_targets(targets);

  auto claim_batch = [&](int n) -> std::vector<Json> {
    std::lock_guard<std::mutex> lock(rem_mu);
    std::vector<Json> out;
    while (!remaining.empty() && static_cast<int>(out.size()) < n) {
      out.push_back(std::move(remaining.front()));
      remaining.pop_front();
    }
    return out;
  };

  auto remaining_count = [&]() {
    std::lock_guard<std::mutex> lock(rem_mu);
    return static_cast<int>(remaining.size());
  };

  // Local workers
  std::vector<std::thread> threads;
  const int local_n = std::max(0, config.local_workers);
  for (int i = 0; i < local_n; ++i) {
    threads.emplace_back([&, i]() {
      (void)i;
      while (!stop.load()) {
        auto batch = claim_batch(1);
        if (batch.empty()) {
          break;
        }
        try {
          Json result = local_submit(batch[0]);
          scheduler.note_completed("local", 1, 1);
          completed.fetch_add(1);
          if (on_result) {
            on_result(result);
          }
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(err_mu);
          errors.push_back(e.what());
        }
        scheduler.maybe_tick(remaining_count(), false);
      }
    });
  }

  // Remote socket workers: one thread per demand slot (capped)
  for (auto* client : remote_clients) {
    if (!client) continue;
    const std::string ep = client->endpoint();
    int slots = 1;
    if (demand.count(ep)) {
      slots = std::max(1, demand[ep]);
    }
    for (int s = 0; s < slots; ++s) {
      threads.emplace_back([&, client, ep]() {
        // Clone connection via reconnect on the shared client is unsafe;
        // use the provided client sequentially per slot by connecting a temp.
        JobClient slot(client->host(), client->port(), 30.0);
        try {
          slot.connect();
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(err_mu);
          errors.push_back(ep + " connect: " + e.what());
          return;
        }
        while (!stop.load()) {
          // Refill endpoint queue from global remaining under credit
          if (queues.size(ep) == 0) {
            const int chunk = std::max(1, scheduler.decision().remote_chunk);
            if (ledger.try_claim_remote(ep, chunk) ||
                ledger.target(ep) == 0) {
              auto batch = claim_batch(chunk);
              if (batch.empty() && queues.size(ep) == 0) {
                break;
              }
              for (auto& job : batch) {
                queues.push(ep, std::move(job));
              }
            } else {
              // Over credit — try steal tiny tail if others idle
              auto batch = claim_batch(1);
              if (batch.empty()) {
                // Wait briefly for more or exit
                if (remaining_count() == 0 && queues.size(ep) == 0) {
                  break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
              }
              for (auto& job : batch) {
                queues.push(ep, std::move(job));
              }
            }
          }
          Json job;
          if (!queues.pop(ep, job)) {
            if (remaining_count() == 0) {
              break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
          }
          try {
            Json result = slot.submit_job(job, config.kind);
            scheduler.note_completed("remote", 1, 1);
            completed.fetch_add(1);
            if (on_result) {
              on_result(result);
            }
          } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(err_mu);
            errors.push_back(ep + ": " + e.what());
          }
          scheduler.maybe_tick(remaining_count(), false);
        }
        slot.close();
      });
    }
  }

  for (auto& t : threads) {
    t.join();
  }
  stop.store(true);

  if (!errors.empty() && completed.load() == 0) {
    throw TransportError("scheduled wave failed: " + errors.front());
  }
  return completed.load();
}

}  // namespace wave_dispatch
