#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "wave_dispatch/wave_dispatch.hpp"

static int expect(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL roundtrip: " << msg << "\n";
    return 1;
  }
  return 0;
}

int test_roundtrip() {
  using namespace wave_dispatch;
  int f = 0;

  std::atomic<bool> stop{false};
  ServerConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 18765;
  cfg.idle_timeout_s = 5.0;

  std::thread server([&]() {
    try {
      serve_forever(
          [](const Json& msg) -> Json {
            if (msg.value("type", "") == "job") {
              return Json{{"type", "result"},
                          {"ok", true},
                          {"result",
                           {{"ok", true}, {"id", msg["job"].value("id", -1)}}}};
            }
            return Json{{"type", "error"}, {"error", "bad"}};
          },
          cfg,
          []() {
            return Json{{"workers", 2},
                        {"max_workers", 4},
                        {"default_workers", 2},
                        {"hostname", "test"}};
          },
          &stop);
    } catch (const std::exception& e) {
      std::cerr << "server: " << e.what() << "\n";
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  try {
    JobClient client("127.0.0.1", 18765, 5.0, 5.0, 5.0);
    auto info = client.connect();
    f += expect(info.workers == 2, "hello workers");
    auto pong = client.ping();
    f += expect(pong.value("type", "") == "pong", "pong");
    auto result = client.submit_job(Json{{"id", 42}}, "echo");
    f += expect(result.value("id", 0) == 42, "job id echo");
    client.close();
  } catch (const std::exception& e) {
    std::cerr << "FAIL roundtrip exception: " << e.what() << "\n";
    f += 1;
  }

  stop.store(true);
  // Connect once to wake accept loop
  try {
    JobClient wake("127.0.0.1", 18765, 1.0, 1.0, 1.0);
    wake.connect();
    wake.close();
  } catch (...) {
  }
  server.join();

  if (f == 0) {
    std::cout << "OK test_roundtrip\n";
  }
  return f;
}
