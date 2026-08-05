#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "wave_dispatch/wave_dispatch.hpp"

static int expect(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL fail_closed: " << msg << "\n";
    return 1;
  }
  return 0;
}

int test_fail_closed() {
  using namespace wave_dispatch;
  int f = 0;

  // Strict batch validation
  try {
    Message bad;
    bad.meta = {{"type", "jobs"}, {"n", 1}, {"items", Json::array({{{"id", 1}}})}};
    // missing blob_lens
    unpack_batch(bad);
    f += expect(false, "missing blob_lens should throw");
  } catch (const ProtocolError&) {
    // expected
  }

  Message packed = pack_batch("jobs", "echo", {Message{Json{{"id", 1}}, {}}}, false);
  packed.blob.push_back(0x7f);  // trailing
  try {
    unpack_batch(packed);
    f += expect(false, "trailing blob should throw");
  } catch (const ProtocolError&) {
  }

  // Live wave: remote returns ok=false → wave must fail closed
  std::atomic<bool> stop{false};
  const int port = 19891;
  ServerConfig scfg;
  scfg.host = "127.0.0.1";
  scfg.port = port;
  scfg.auto_uds = true;
  scfg.io_threads = 1;
  scfg.idle_timeout_s = 2.0;

  std::thread server([&]() {
    serve_forever(
        [](const Message& msg) {
          (void)msg;
          Message out;
          out.meta = {{"type", "result"}, {"ok", false}, {"error", "boom"}};
          return out;
        },
        scfg,
        []() {
          return Json{{"workers", 1},
                      {"max_workers", 1},
                      {"default_workers", 1},
                      {"hostname", "fail-test"}};
        },
        &stop);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  try {
    JobClient client("127.0.0.1", port, 5.0);
    client.connect();
    std::vector<JobClient*> remotes{&client};
    SchedulerConfig cfg;
    cfg.tick_s = 0;
    MidWaveScheduler sched(cfg);
    CollectConfig cc;
    cc.local_workers = 0;
    cc.batch_size = 1;
    cc.use_connection_pool = false;
    std::vector<Json> jobs = {{{"id", 1}}, {{"id", 2}}};
    bool threw = false;
    try {
      run_scheduled_wave(
          jobs, [](const Json& j) { return j; }, remotes, sched, cc, nullptr);
    } catch (const TransportError&) {
      threw = true;
    }
    f += expect(threw, "ok=false wave must throw");
    client.close();
  } catch (const std::exception& e) {
    std::cerr << "fail_closed setup: " << e.what() << "\n";
    f += 1;
  }

  stop.store(true);
  server.join();

  if (f == 0) std::cout << "OK test_fail_closed\n";
  return f;
}
