#include "rl_runtime/rl_runtime.hpp"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

int main() {
  const std::string name = "/rl_shm_bench_" + std::to_string(::getpid());
  rl_runtime::RingConfig cfg;
  cfg.name = name;
  cfg.slot_count = 4;
  cfg.request_slots = 128;
  cfg.max_payload = 4096;
  auto server = rl_runtime::ShmRing::create(cfg);
  auto client = rl_runtime::ShmRing::open(name);

  std::thread srv([&] {
    for (;;) {
      auto batch = server.coalesce(16, 1.0, 0.0005);
      if (batch.empty()) {
        if (!server.alive()) break;
        continue;
      }
      for (auto& req : batch) {
        server.respond(req.slot, req.rid, req.payload.data(), req.payload.size());
      }
    }
  });

  constexpr int N = 5000;
  std::vector<std::uint8_t> payload(256, 7);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) {
    const auto rid = client.submit(i % 4, payload.data(), payload.size());
    auto resp = client.wait(i % 4, rid);
    if (resp.size() != payload.size()) {
      std::fprintf(stderr, "bad response size\n");
      server.set_alive(false);
      srv.join();
      server.unlink();
      return 1;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  server.set_alive(false);
  srv.join();
  server.unlink();
  const double s = std::chrono::duration<double>(t1 - t0).count();
  std::printf("shm_ring: %d roundtrips in %.3fs (%.0f rt/s)\n", N, s, N / s);
  return 0;
}
