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
    slot->target.store(std::max(1, n), std::memory_order_relaxed);
    slot->in_flight.store(0, std::memory_order_relaxed);
    slots_.emplace(ep, std::move(slot));
  }
}

bool ClaimLedger::try_claim_remote(const std::string& endpoint, int n) {
  if (n <= 0) return false;
  auto it = slots_.find(endpoint);
  if (it == slots_.end()) return true;
  Slot& s = *it->second;
  const int tgt = s.target.load(std::memory_order_relaxed);
  while (true) {
    int have = s.in_flight.load(std::memory_order_relaxed);
    if (tgt > 0 && have + n > tgt) return false;
    if (s.in_flight.compare_exchange_weak(have, have + n, std::memory_order_acq_rel)) {
      return true;
    }
  }
}

void ClaimLedger::release_remote(const std::string& endpoint, int n) {
  if (n <= 0) return;
  auto it = slots_.find(endpoint);
  if (it == slots_.end()) return;
  it->second->in_flight.fetch_sub(n, std::memory_order_acq_rel);
}

int ClaimLedger::claimed(const std::string& endpoint) const {
  auto it = slots_.find(endpoint);
  return it == slots_.end() ? 0 : it->second->in_flight.load(std::memory_order_relaxed);
}

int ClaimLedger::target(const std::string& endpoint) const {
  auto it = slots_.find(endpoint);
  return it == slots_.end() ? 0 : it->second->target.load(std::memory_order_relaxed);
}

namespace {

struct JobItem {
  Message msg;
};

bool result_ok(const Message& result) {
  if (result.meta.contains("ok") && !result.meta.value("ok", true)) return false;
  if (result.meta.value("type", "") == "error") return false;
  return true;
}

int run_wave_impl(std::vector<Message> jobs, LocalMessageFn local_submit,
                  std::vector<JobClient*>& remote_clients,
                  MidWaveScheduler& scheduler, CollectConfig config,
                  MessageResultCallback on_result, ConnectionPool* pool) {
  if (jobs.empty()) return 0;
  const int expected = static_cast<int>(jobs.size());

  moodycamel::ConcurrentQueue<JobItem> global_q;
  for (auto& j : jobs) global_q.enqueue(JobItem{std::move(j)});
  std::atomic<int> remaining{expected};
  std::atomic<int> completed{0};
  std::atomic<bool> fail_closed{false};
  std::vector<std::string> errors;
  std::mutex err_mu;

  scheduler.maybe_tick(remaining.load(), true);
  auto dec = scheduler.decision();

  ClaimLedger ledger(expected);
  std::unordered_map<std::string, int> targets;
  auto demand = dec.remote_demand;

  std::vector<std::string> endpoints;
  for (auto* c : remote_clients) {
    if (!c) continue;
    endpoints.push_back(c->endpoint());
  }

  // In-flight caps (not lifetime quotas).
  for (const auto& [ep, n] : demand) {
    targets[ep] = std::max(1, n) * std::max(1, config.batch_size);
  }
  if (targets.empty()) {
    for (const auto& ep : endpoints) {
      targets[ep] = std::max(1, config.batch_size);
    }
  }
  ledger.set_endpoint_targets(targets);

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
      active_pool->ensure(ep, n);
    }
  }

  auto push_global = [&](Message msg) {
    global_q.enqueue(JobItem{std::move(msg)});
    remaining.fetch_add(1, std::memory_order_relaxed);
  };

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

  auto note_error = [&](const std::string& msg) {
    std::lock_guard<std::mutex> lock(err_mu);
    errors.push_back(msg);
    fail_closed.store(true, std::memory_order_relaxed);
  };

  std::vector<std::thread> local_threads;
  const int local_n = std::max(0, config.local_workers);
  for (int i = 0; i < local_n; ++i) {
    local_threads.emplace_back([&]() {
      JobItem item;
      while (!fail_closed.load(std::memory_order_relaxed) && pop_global(item)) {
        try {
          Message result = local_submit(item.msg);
          if (!result_ok(result)) {
            note_error("local job returned ok=false");
            push_global(std::move(item.msg));
          } else {
            scheduler.note_completed("local", 1, 1);
            completed.fetch_add(1, std::memory_order_relaxed);
            if (on_result) on_result(result);
          }
        } catch (const std::exception& e) {
          note_error(std::string("local: ") + e.what());
          push_global(std::move(item.msg));
        }
        scheduler.maybe_tick(remaining.load(std::memory_order_relaxed), false);
      }
    });
  }

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

  auto spawn_endpoint_workers = [&](const std::string& ep, int slots) {
    for (int s = 0; s < slots; ++s) {
      remote_inflight.fetch_add(1, std::memory_order_relaxed);
      asio::post(ioc, [&, ep]() {
        while (!fail_closed.load(std::memory_order_relaxed)) {
          const int batch_n = std::max(1, config.batch_size);
          int claimed = 0;
          std::vector<Message> batch;
          if (ledger.try_claim_remote(ep, batch_n)) {
            claimed = batch_n;
            pop_many(batch_n, batch);
          } else if (ledger.try_claim_remote(ep, 1)) {
            claimed = 1;
            pop_many(1, batch);
          } else {
            if (remaining.load(std::memory_order_relaxed) <= 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
          }
          if (static_cast<int>(batch.size()) < claimed) {
            ledger.release_remote(ep, claimed - static_cast<int>(batch.size()));
            claimed = static_cast<int>(batch.size());
          }
          if (batch.empty()) {
            if (claimed) ledger.release_remote(ep, claimed);
            if (remaining.load(std::memory_order_relaxed) <= 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
          }

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
            for (auto* c : remote_clients) {
              if (c && c->endpoint() == ep) {
                ephemeral = std::make_unique<JobClient>(c->host(), c->port(), 30.0);
                try {
                  ephemeral->connect();
                  client = ephemeral.get();
                } catch (const std::exception& e) {
                  note_error(ep + " connect: " + e.what());
                  client = nullptr;
                }
                break;
              }
            }
          }
          if (!client) {
            for (auto& m : batch) push_global(std::move(m));
            ledger.release_remote(ep, claimed);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }

          bool healthy = true;
          bool batch_ok = false;
          try {
            auto results =
                client->submit_batch(batch, config.kind, config.compress_blobs);
            if (static_cast<int>(results.size()) != static_cast<int>(batch.size())) {
              throw ProtocolError("batch result cardinality mismatch");
            }
            for (auto& result : results) {
              if (result.meta.contains("result") && result.blob.empty()) {
                Message unwrapped;
                unwrapped.meta = result.meta["result"];
                unwrapped.blob = std::move(result.blob);
                result = std::move(unwrapped);
              }
              if (!result_ok(result)) {
                throw ProtocolError(result.meta.value("error", "remote ok=false"));
              }
            }
            for (auto& result : results) {
              scheduler.note_completed("remote", 1, 1);
              completed.fetch_add(1, std::memory_order_relaxed);
              if (on_result) on_result(result);
            }
            batch_ok = true;
          } catch (const std::exception& e) {
            healthy = false;
            note_error(ep + ": " + e.what());
            for (auto& m : batch) push_global(std::move(m));
          }

          ledger.release_remote(ep, claimed);
          if (from_pool && active_pool) {
            active_pool->release(ep, client, healthy);
          } else if (ephemeral) {
            ephemeral->close();
          }
          (void)batch_ok;
          scheduler.maybe_tick(remaining.load(std::memory_order_relaxed), false);
          if (fail_closed.load(std::memory_order_relaxed)) break;
        }
        remote_inflight.fetch_sub(1, std::memory_order_relaxed);
      });
    }
  };

  if (!endpoints.empty()) {
    for (const auto& ep : endpoints) {
      int slots = demand.count(ep) ? std::max(1, demand[ep]) : 1;
      spawn_endpoint_workers(ep, slots);
    }
  }

  for (auto& t : local_threads) t.join();

  while (remote_inflight.load(std::memory_order_relaxed) > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  work_guard.reset();
  ioc.stop();
  for (auto& t : async_threads) t.join();

  const int done = completed.load();
  if (!errors.empty() || done != expected) {
    std::string detail = errors.empty() ? "incomplete wave" : errors.front();
    throw TransportError("scheduled wave failed (" + std::to_string(done) + "/" +
                         std::to_string(expected) + "): " + detail);
  }
  return done;
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
