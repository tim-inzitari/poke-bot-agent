#include "rl_runtime/shm_ring.hpp"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

int test_shm_ring() {
  const std::string name = "/rl_test_shm_" + std::to_string(::getpid());
  rl_runtime::RingConfig cfg;
  cfg.name = name;
  cfg.slot_count = 4;
  cfg.request_slots = 64;
  cfg.max_payload = 1024;
  auto server = rl_runtime::ShmRing::create(cfg);
  auto client = rl_runtime::ShmRing::open(name);

  std::thread t([&] {
    for (;;) {
      auto batch = server.coalesce(8, 0.5, 0.002);
      if (batch.empty()) {
        if (!server.alive()) break;
        continue;
      }
      for (auto& req : batch) {
        std::string out = "echo:";
        out.append(reinterpret_cast<const char*>(req.payload.data()),
                   req.payload.size());
        server.respond(req.slot, req.rid,
                       reinterpret_cast<const std::uint8_t*>(out.data()), out.size());
      }
    }
  });

  const std::string msg = "hi";
  const auto rid =
      client.submit(0, reinterpret_cast<const std::uint8_t*>(msg.data()), msg.size());
  auto resp = client.wait(0, rid, 2.0);
  const std::string got(resp.begin(), resp.end());
  if (got != "echo:hi") {
    std::fprintf(stderr, "shm got %s\n", got.c_str());
    server.set_alive(false);
    t.join();
    server.unlink();
    return 1;
  }

  // Multi-producer pressure.
  std::atomic<int> errors{0};
  std::vector<std::thread> producers;
  for (int p = 0; p < 4; ++p) {
    producers.emplace_back([&, p] {
      auto c = rl_runtime::ShmRing::open(name);
      for (int i = 0; i < 50; ++i) {
        const std::string m = "p" + std::to_string(p) + "-" + std::to_string(i);
        try {
          const auto r =
              c.submit(static_cast<std::uint32_t>(p),
                       reinterpret_cast<const std::uint8_t*>(m.data()), m.size());
          auto out = c.wait(static_cast<std::uint32_t>(p), r, 5.0);
          const std::string s(out.begin(), out.end());
          if (s != "echo:" + m) errors.fetch_add(1);
        } catch (...) {
          errors.fetch_add(1);
        }
      }
    });
  }
  for (auto& th : producers) th.join();
  server.set_alive(false);
  t.join();
  server.unlink();
  if (errors.load() != 0) {
    std::fprintf(stderr, "multi-producer errors=%d\n", errors.load());
    return 1;
  }
  return 0;
}
