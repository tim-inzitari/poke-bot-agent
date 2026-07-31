#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "wave_dispatch/wave_dispatch.hpp"

int main(int argc, char** argv) {
  using namespace wave_dispatch;
  int port = 19999;
  int jobs = 4000;
  int blob_bytes = 8192;
  int local = 2;
  int remote_slots = 8;
  int batch = 16;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--jobs") == 0 && i + 1 < argc) jobs = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--blob") == 0 && i + 1 < argc)
      blob_bytes = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--local") == 0 && i + 1 < argc)
      local = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
      port = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
      batch = std::atoi(argv[++i]);
  }

  std::atomic<bool> stop{false};
  ServerConfig scfg;
  scfg.host = "127.0.0.1";
  scfg.port = port;
  scfg.auto_uds = true;
  scfg.io_threads = 4;
  scfg.idle_timeout_s = 30;

  std::thread server([&]() {
    serve_forever(
        [](const Message& msg) {
          Message out;
          out.meta = {{"type", "result"},
                      {"ok", true},
                      {"result", {{"ok", true}, {"id", msg.meta["job"].value("id", 0)}}}};
          out.blob = msg.blob;
          return out;
        },
        scfg,
        []() {
          return Json{{"workers", 8},
                      {"max_workers", 16},
                      {"default_workers", 8},
                      {"hostname", "bench"}};
        },
        &stop);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  const std::string ep = "127.0.0.1:" + std::to_string(port);
  ConnectionPool::Options popts;
  popts.prefer_uds = true;
  ConnectionPool pool(popts);
  pool.ensure(ep, remote_slots);

  JobClient client("127.0.0.1", port, 30.0);
  client.connect();  // TCP template for endpoint identity

  SchedulerConfig cfg;
  cfg.tick_s = 0.5;
  cfg.min_gps_window_s = 0.2;
  cfg.demand_settle_s = 0.5;
  cfg.remote_defaults[ep] = remote_slots;
  cfg.remote_maxima[ep] = remote_slots;
  MidWaveScheduler sched(cfg);

  std::vector<std::uint8_t> blob(static_cast<std::size_t>(blob_bytes), 0xAB);
  std::vector<Message> wave;
  wave.reserve(jobs);
  for (int i = 0; i < jobs; ++i) {
    Message m;
    m.meta = {{"id", i}};
    m.blob = blob;
    wave.push_back(std::move(m));
  }

  CollectConfig ccfg;
  ccfg.local_workers = local;
  ccfg.remote_chunk = 64;
  ccfg.batch_size = batch;
  ccfg.compress_blobs = true;
  ccfg.use_connection_pool = true;
  ccfg.prefer_uds = true;
  ccfg.kind = "echo";
  std::vector<JobClient*> remotes{&client};

  const auto t0 = std::chrono::steady_clock::now();
  const int n = run_scheduled_wave_bin(
      wave,
      [&](const Message& m) {
        Message out;
        out.meta = {{"ok", true}, {"id", m.meta.value("id", 0)}, {"src", "local"}};
        out.blob = m.blob;
        return out;
      },
      remotes, sched, ccfg, {}, &pool);
  const double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

  auto bstats = BufferPool::instance().stats();
  std::cout << "bench v0.3 completed=" << n << " jobs=" << jobs
            << " blob_bytes=" << blob_bytes << " batch=" << batch
            << " elapsed_s=" << dt << " gps=" << (n / std::max(dt, 1e-9))
            << " GBps_payload="
            << ((static_cast<double>(n) * blob_bytes) / dt) /
                   (1024.0 * 1024.0 * 1024.0)
            << " pool_hits=" << bstats.hits << "/" << bstats.acquires << "\n";

  stop.store(true);
  try {
    JobClient wake("127.0.0.1", port, 1.0, 1.0, 1.0);
    wake.connect();
    wake.close();
  } catch (...) {
  }
  server.join();
  return n == jobs ? 0 : 1;
}
