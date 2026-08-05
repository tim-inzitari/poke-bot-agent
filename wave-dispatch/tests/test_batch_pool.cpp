#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "wave_dispatch/wave_dispatch.hpp"

static int expect(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL batch_pool: " << msg << "\n";
    return 1;
  }
  return 0;
}

int test_batch_pool() {
  using namespace wave_dispatch;
  int f = 0;

  // pack/unpack
  std::vector<Message> items;
  for (int i = 0; i < 4; ++i) {
    Message m;
    m.meta = {{"id", i}};
    m.blob.assign(64, static_cast<std::uint8_t>(i));
    items.push_back(std::move(m));
  }
  Message packed = pack_batch("jobs", "echo", items, true);
  f += expect(packed.meta.value("type", "") == "jobs", "type");
  auto back = unpack_batch(packed);
  f += expect(back.size() == 4, "count");
  f += expect(back[2].meta["id"] == 2, "id");
  f += expect(back[2].blob.size() == 64, "blob");

  // LZ4 roundtrip
  std::vector<std::uint8_t> raw(4096, 0x11);
  std::vector<std::uint8_t> comp;
  auto used = compress_blob(raw, comp, BlobCodec::kLz4);
  f += expect(used == BlobCodec::kLz4 || used == BlobCodec::kNone, "codec");
  if (used == BlobCodec::kLz4) {
    std::vector<std::uint8_t> out;
    decompress_blob(comp, out, BlobCodec::kLz4);
    f += expect(out == raw, "lz4 roundtrip");
  }

  // Live UDS + pool + batch wave
  std::atomic<bool> stop{false};
  const int port = 19880;
  ServerConfig scfg;
  scfg.host = "127.0.0.1";
  scfg.port = port;
  scfg.auto_uds = true;
  scfg.io_threads = 2;
  scfg.idle_timeout_s = 5;

  std::thread server([&]() {
    serve_forever(
        [](const Message& msg) {
          Message out;
          out.meta = {{"type", "result"},
                      {"ok", true},
                      {"result", {{"ok", true}, {"id", msg.meta["job"].value("id", -1)}}}};
          out.blob = msg.blob;
          return out;
        },
        scfg,
        []() {
          return Json{{"workers", 4},
                      {"max_workers", 8},
                      {"default_workers", 4},
                      {"hostname", "batch-test"}};
        },
        &stop);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  try {
    ConnectionPool::Options popts;
    popts.prefer_uds = true;
    ConnectionPool pool(popts);
    const std::string ep = "127.0.0.1:" + std::to_string(port);
    pool.ensure(ep, 2);
    f += expect(pool.live_count(ep) >= 1, "pool live");

    JobClient tmpl("127.0.0.1", port, 5.0, 5.0, 5.0);
    // TCP fallback template (pool uses UDS)
    tmpl.connect();

    SchedulerConfig cfg;
    cfg.tick_s = 0.0;
    cfg.remote_defaults[ep] = 2;
    cfg.remote_maxima[ep] = 2;
    MidWaveScheduler sched(cfg);

    std::vector<Message> wave;
    for (int i = 0; i < 32; ++i) {
      Message m;
      m.meta = {{"id", i}};
      m.blob.assign(128, 0xAA);
      wave.push_back(std::move(m));
    }
    CollectConfig ccfg;
    ccfg.local_workers = 1;
    ccfg.batch_size = 8;
    ccfg.compress_blobs = true;
    ccfg.use_connection_pool = true;
    ccfg.prefer_uds = true;
    std::vector<JobClient*> remotes{&tmpl};
    int n = run_scheduled_wave_bin(wave, [](const Message& m) {
      Message o;
      o.meta = {{"ok", true}, {"id", m.meta.value("id", 0)}};
      o.blob = m.blob;
      return o;
    }, remotes, sched, ccfg, {}, &pool);
    f += expect(n == 32, "wave completed");
    tmpl.close();
  } catch (const std::exception& e) {
    std::cerr << "FAIL batch_pool exception: " << e.what() << "\n";
    f += 1;
  }

  stop.store(true);
  try {
    JobClient wake("127.0.0.1", port, 1.0, 1.0, 1.0);
    wake.connect();
    wake.close();
  } catch (...) {
  }
  server.join();

  if (f == 0) std::cout << "OK test_batch_pool\n";
  return f;
}
