#include "wave_dispatch/collector.hpp"

// concurrentqueue before linux headers (io_uring pulls in BLOCK_SIZE macro)
#include <concurrentqueue.h>
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#include "wave_dispatch/asio_config.hpp"
#include <asio.hpp>

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
  if (it == slots_.end()) return true;
  Slot& s = *it->second;
  const int tgt = s.target.load(std::memory_order_relaxed);
  while (true) {
    int have = s.claimed.load(std::memory_order_relaxed);
    if (tgt > 0 && have + n > tgt) return false;
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
                  MessageResultCallback on_result, ConnectionPool* pool) {
  if (jobs.empty()) return 0;

  moodycamel::ConcurrentQueue<JobItem> global_q;
  for (auto& j : jobs) global_q.enqueue(JobItem{std::move(j)});
  std::atomic<int> remaining{static_cast<int>(jobs.size())};
  std::atomic<int> completed{0};
  std::vector<std::string> errors;
  std::mutex err_mu;

  scheduler.maybe_tick(remaining.load(), true);
  auto dec = scheduler.decision();

  ClaimLedger ledger(remaining.load());
  std::unordered_map<std::string, int> targets;
  auto demand = dec.remote_demand;

  // Build endpoint list from clients
  std::vector<std::string> endpoints;
  for (auto* c : remote_clients) {
    if (!c) continue;
    endpoints.push_back(c->endpoint());
  }

  for (const auto& [ep, n] : demand) {
    targets[ep] = std::max(n, 1) * std::max(1, config.remote_chunk);
  }
  if (targets.empty()) {
    for (const auto& ep : endpoints) {
      targets[ep] =
          std::max(1, remaining.load() / std::max(1, static_cast<int>(endpoints.size())));
    }
  }
  ledger.set_endpoint_targets(targets);

  // Optional owned pool if caller didn't pass one
  std::unique_ptr<ConnectionPool> owned_pool;
  ConnectionPool* active_pool = pool;
  if (config.use_connection_pool && active_pool == nullptr && !endpoints.empty()) {
    ConnectionPool::Options opt;
    opt.prefer_uds = config.prefer_uds;
    owned_pool = std::make_unique<ConnectionPool>(opt);
    active_pool = owned_pool.get();
  }
  if (active_pool) {
    for (const auto& ep : endpoints) {
      int n = demand.count(ep) ? std::max(1, demand[ep]) : 1;
      // Map TCP endpoint key to pool key (localhost → uds)
      std::string key = ep;
      active_pool->ensure(key, n);
    }
  }

  auto pop_global = [&](JobItem& out) -> bool {
    if (global_q.try_dequeue(out)) {
      remaining.fetch_sub(1, std::memory_order_relaxed);
      return true;
    }
    return false;
  };

  auto pop_many = [&](int n, std::vector<Message>& out) {
    out.clear();
    JobItem item;
    while (static_cast<int>(out.size()) < n && pop_global(item)) {
      out.push_back(std::move(item.msg));
    }
  };

  // Local workers (still threads — CPU job work)
  std::vector<std::thread> local_threads;
  const int local_n = std::max(0, config.local_workers);
  for (int i = 0; i < local_n; ++i) {
    local_threads.emplace_back([&]() {
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

  // Async remote path via asio thread pool + connection pool + batching
  int async_n = config.async_threads;
  if (async_n <= 0) {
    async_n = static_cast<int>(std::max(2u, std::thread::hardware_concurrency()));
  }
  asio::io_context ioc(async_n);
  auto work_guard = asio::make_work_guard(ioc);
  std::vector<std::thread> async_threads;
  for (int i = 0; i < async_n; ++i) {
    async_threads.emplace_back([&ioc]() { ioc.run(); });
  }

  std::atomic<int> remote_inflight{0};
  std::atomic<bool> remote_done{false};

  auto spawn_endpoint_workers = [&](const std::string& ep, int slots) {
    for (int s = 0; s < slots; ++s) {
      remote_inflight.fetch_add(1, std::memory_order_relaxed);
      asio::post(ioc, [&, ep]() {
        // Drain until empty
        while (true) {
          const int batch_n = std::max(1, config.batch_size);
          std::vector<Message> batch;
          // Claim credit then pull
          if (!ledger.try_claim_remote(ep, batch_n)) {
            // try single
            if (!ledger.try_claim_remote(ep, 1)) {
              if (remaining.load(std::memory_order_relaxed) <= 0) break;
              std::this_thread::sleep_for(std::chrono::microseconds(100));
              continue;
            }
            pop_many(1, batch);
          } else {
            pop_many(batch_n, batch);
          }
          if (batch.empty()) {
            if (remaining.load(std::memory_order_relaxed) <= 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
          }

          // Normalize metas
          for (auto& m : batch) {
            if (!m.meta.contains("type")) {
              Json job = m.meta;
              m.meta = {{"type", "job"}, {"kind", config.kind}, {"job", job}};
            }
          }

          JobClient* client = nullptr;
          bool from_pool = false;
          std::unique_ptr<JobClient> ephemeral;
          if (active_pool) {
            client = active_pool->acquire(ep, true);
            from_pool = client != nullptr;
          }
          if (!client) {
            // Fall back: find template client with matching endpoint
            for (auto* c : remote_clients) {
              if (c && c->endpoint() == ep) {
                ephemeral = std::make_unique<JobClient>(c->host(), c->port(), 30.0);
                try {
                  ephemeral->connect();
                  client = ephemeral.get();
                } catch (const std::exception& e) {
                  std::lock_guard<std::mutex> lock(err_mu);
                  errors.push_back(ep + " connect: " + e.what());
                  client = nullptr;
                }
                break;
              }
            }
          }
          if (!client) continue;

          bool healthy = true;
          try {
            auto results =
                client->submit_batch(batch, config.kind, config.compress_blobs);
            for (auto& result : results) {
              if (result.meta.contains("result") && result.blob.empty()) {
                Message unwrapped;
                unwrapped.meta = result.meta["result"];
                unwrapped.blob = std::move(result.blob);
                result = std::move(unwrapped);
              }
              scheduler.note_completed("remote", 1, 1);
              completed.fetch_add(1, std::memory_order_relaxed);
              if (on_result) on_result(result);
            }
          } catch (const std::exception& e) {
            healthy = false;
            std::lock_guard<std::mutex> lock(err_mu);
            errors.push_back(ep + ": " + e.what());
          }

          if (from_pool && active_pool) {
            active_pool->release(ep, client, healthy);
          } else if (ephemeral) {
            ephemeral->close();
          }
          scheduler.maybe_tick(remaining.load(std::memory_order_relaxed), false);
        }
        remote_inflight.fetch_sub(1, std::memory_order_relaxed);
      });
    }
  };

  if (endpoints.empty()) {
    remote_done.store(true);
  } else {
    for (const auto& ep : endpoints) {
      int slots = demand.count(ep) ? std::max(1, demand[ep]) : 1;
      spawn_endpoint_workers(ep, slots);
    }
  }

  for (auto& t : local_threads) t.join();

  // Wait for remote async work
  while (remote_inflight.load(std::memory_order_relaxed) > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  work_guard.reset();
  ioc.stop();
  for (auto& t : async_threads) t.join();

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
                           MessageResultCallback on_result, ConnectionPool* pool) {
  return run_wave_impl(jobs, std::move(local_submit), remote_clients, scheduler,
                       std::move(config), std::move(on_result), pool);
}

int run_scheduled_wave(const std::vector<Json>& jobs, LocalSubmitFn local_submit,
                       std::vector<JobClient*>& remote_clients,
                       MidWaveScheduler& scheduler, CollectConfig config,
                       ResultCallback on_result, ConnectionPool* pool) {
  std::vector<Message> msgs;
  msgs.reserve(jobs.size());
  for (const auto& j : jobs) msgs.push_back(Message{j, {}});
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
                       std::move(config), std::move(cb), pool);
}

}  // namespace wave_dispatch
