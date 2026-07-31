#include "wave_dispatch/collector.hpp"

#include <concurrentqueue.h>

#include <chrono>
#include <mutex>
#include <thread>

namespace wave_dispatch {

ClaimLedger::ClaimLedger(int total_jobs) : total_jobs_(std::max(0, total_jobs)) {}

void ClaimLedger::set_endpoint_targets(
    const std::unordered_map<std::string, int>& targets) {
  slots_.clear();
  for (const auto& [ep, n] : targets) {
    auto slot = std::make_unique<Slot>();
    slot->target.store(n, std::memory_order_relaxed);
    slot->claimed.store(0, std::memory_order_relaxed);
    slots_.emplace(ep, std::move(slot));
  }
}

bool ClaimLedger::try_claim_remote(const std::string& endpoint, int n) {
  auto it = slots_.find(endpoint);
  if (it == slots_.end()) {
    return true;  // unconstrained
  }
  Slot& s = *it->second;
  const int tgt = s.target.load(std::memory_order_relaxed);
  while (true) {
    int have = s.claimed.load(std::memory_order_relaxed);
    if (tgt > 0 && have + n > tgt) {
      return false;
    }
    if (s.claimed.compare_exchange_weak(have, have + n, std::memory_order_acq_rel)) {
      return true;
    }
  }
}

int ClaimLedger::claimed(const std::string& endpoint) const {
  auto it = slots_.find(endpoint);
  return it == slots_.end() ? 0 : it->second->claimed.load(std::memory_order_relaxed);
}

int ClaimLedger::target(const std::string& endpoint) const {
  auto it = slots_.find(endpoint);
  return it == slots_.end() ? 0 : it->second->target.load(std::memory_order_relaxed);
}

namespace {

struct JobItem {
  Message msg;
};

int run_wave_impl(std::vector<Message> jobs, LocalMessageFn local_submit,
                  std::vector<JobClient*>& remote_clients,
                  MidWaveScheduler& scheduler, CollectConfig config,
                  MessageResultCallback on_result) {
  if (jobs.empty()) {
    return 0;
  }

  moodycamel::ConcurrentQueue<JobItem> global_q;
  for (auto& j : jobs) {
    global_q.enqueue(JobItem{std::move(j)});
  }
  std::atomic<int> remaining{static_cast<int>(jobs.size())};
  // jobs vector emptied into queue; size remembered
  const int total = remaining.load();
  (void)total;

  std::atomic<int> completed{0};
  std::vector<std::string> errors;
  std::mutex err_mu;

  scheduler.maybe_tick(remaining.load(), true);
  auto dec = scheduler.decision();

  ClaimLedger ledger(remaining.load());
  std::unordered_map<std::string, int> targets;
  auto demand = dec.remote_demand;
  for (const auto& [ep, n] : demand) {
    targets[ep] = std::max(n, 1) * std::max(1, config.remote_chunk);
  }
  if (targets.empty()) {
    for (auto* c : remote_clients) {
      if (!c) continue;
      targets[c->endpoint()] =
          std::max(1, remaining.load() /
                          std::max(1, static_cast<int>(remote_clients.size())));
    }
  }
  ledger.set_endpoint_targets(targets);

  // Per-endpoint lock-free queues
  std::unordered_map<std::string, std::unique_ptr<moodycamel::ConcurrentQueue<JobItem>>>
      ep_qs;
  for (const auto& [ep, _] : targets) {
    ep_qs.emplace(ep, std::make_unique<moodycamel::ConcurrentQueue<JobItem>>());
  }
  for (auto* c : remote_clients) {
    if (c) ep_qs.emplace(c->endpoint(),
                         std::make_unique<moodycamel::ConcurrentQueue<JobItem>>());
  }

  auto pop_global = [&](JobItem& out) -> bool {
    if (global_q.try_dequeue(out)) {
      remaining.fetch_sub(1, std::memory_order_relaxed);
      return true;
    }
    return false;
  };

  std::vector<std::thread> threads;

  const int local_n = std::max(0, config.local_workers);
  for (int i = 0; i < local_n; ++i) {
    threads.emplace_back([&]() {
      JobItem item;
      while (pop_global(item)) {
        try {
          Message result = local_submit(item.msg);
          scheduler.note_completed("local", 1, 1);
          completed.fetch_add(1, std::memory_order_relaxed);
          if (on_result) on_result(result);
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(err_mu);
          errors.push_back(e.what());
        }
        scheduler.maybe_tick(remaining.load(std::memory_order_relaxed), false);
      }
    });
  }

  for (auto* client : remote_clients) {
    if (!client) continue;
    const std::string ep = client->endpoint();
    int slots = 1;
    if (demand.count(ep)) slots = std::max(1, demand[ep]);
    auto* q = ep_qs[ep].get();

    for (int s = 0; s < slots; ++s) {
      threads.emplace_back([&, client, ep, q]() {
        JobClient slot(client->host(), client->port(), 30.0);
        try {
          slot.connect();
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(err_mu);
          errors.push_back(ep + " connect: " + e.what());
          return;
        }

        JobItem item;
        while (true) {
          // Prefer endpoint queue, else claim from global
          if (!q->try_dequeue(item)) {
            const int chunk = std::max(1, scheduler.decision().remote_chunk);
            if (ledger.try_claim_remote(ep, chunk)) {
              int got = 0;
              JobItem tmp;
              while (got < chunk && pop_global(tmp)) {
                if (got == 0) {
                  item = std::move(tmp);
                  ++got;
                } else {
                  q->enqueue(JobItem{std::move(tmp.msg)});
                  ++got;
                }
              }
              if (got == 0) {
                // Nothing left
                if (remaining.load(std::memory_order_relaxed) <= 0 &&
                    q->size_approx() == 0) {
                  break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
              }
            } else {
              if (remaining.load(std::memory_order_relaxed) <= 0 &&
                  q->size_approx() == 0) {
                break;
              }
              // Soft steal one
              if (!pop_global(item)) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
              }
            }
          }

          try {
            Message req = item.msg;
            if (!req.meta.contains("type")) {
              // Treat as job body
              Json job = req.meta;
              req.meta = {{"type", "job"}, {"kind", config.kind}, {"job", job}};
            }
            Message result = slot.submit_message(req, config.kind);
            // Unwrap nested result for JSON-shaped replies
            if (result.meta.contains("result") && result.blob.empty()) {
              Message unwrapped;
              unwrapped.meta = result.meta["result"];
              unwrapped.blob = std::move(result.blob);
              result = std::move(unwrapped);
            }
            scheduler.note_completed("remote", 1, 1);
            completed.fetch_add(1, std::memory_order_relaxed);
            if (on_result) on_result(result);
          } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(err_mu);
            errors.push_back(ep + ": " + e.what());
          }
          scheduler.maybe_tick(remaining.load(std::memory_order_relaxed), false);
        }
        slot.close();
      });
    }
  }

  for (auto& t : threads) t.join();

  if (!errors.empty() && completed.load() == 0) {
    throw TransportError("scheduled wave failed: " + errors.front());
  }
  return completed.load();
}

}  // namespace

int run_scheduled_wave_bin(const std::vector<Message>& jobs,
                           LocalMessageFn local_submit,
                           std::vector<JobClient*>& remote_clients,
                           MidWaveScheduler& scheduler, CollectConfig config,
                           MessageResultCallback on_result) {
  return run_wave_impl(jobs, std::move(local_submit), remote_clients, scheduler,
                       std::move(config), std::move(on_result));
}

int run_scheduled_wave(const std::vector<Json>& jobs, LocalSubmitFn local_submit,
                       std::vector<JobClient*>& remote_clients,
                       MidWaveScheduler& scheduler, CollectConfig config,
                       ResultCallback on_result) {
  std::vector<Message> msgs;
  msgs.reserve(jobs.size());
  for (const auto& j : jobs) {
    msgs.push_back(Message{j, {}});
  }
  LocalMessageFn lm = [local_submit](const Message& m) {
    Json job = m.meta;
    if (m.meta.contains("job")) job = m.meta["job"];
    Message out;
    out.meta = local_submit(job);
    return out;
  };
  MessageResultCallback cb;
  if (on_result) {
    cb = [on_result](const Message& m) { on_result(m.meta); };
  }
  return run_wave_impl(std::move(msgs), std::move(lm), remote_clients, scheduler,
                       std::move(config), std::move(cb));
}

}  // namespace wave_dispatch
