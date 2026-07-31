#include "rl_runtime/shm_ring.hpp"

#include <cstdio>
#include <string>
#include <thread>
#include <unistd.h>

int test_shm_ring() {
  const std::string name = "/rl_test_shm_" + std::to_string(::getpid());
  rl_runtime::RingConfig cfg;
  cfg.name = name;
  cfg.slot_count = 2;
  cfg.request_slots = 16;
  cfg.max_payload = 1024;
  auto server = rl_runtime::ShmRing::create(cfg);
  auto client = rl_runtime::ShmRing::open(name);
  std::thread t([&] {
    auto batch = server.coalesce(4, 2.0, 0.01);
    for (auto& req : batch) {
      std::string out = "echo:";
      out.append(reinterpret_cast<const char*>(req.payload.data()),
                 req.payload.size());
      server.respond(req.slot, req.rid,
                     reinterpret_cast<const std::uint8_t*>(out.data()), out.size());
    }
  });
  const std::string msg = "hi";
  const auto rid =
      client.submit(0, reinterpret_cast<const std::uint8_t*>(msg.data()), msg.size());
  auto resp = client.wait(0, rid, 2.0);
  t.join();
  const std::string got(resp.begin(), resp.end());
  server.unlink();
  if (got != "echo:hi") {
    std::fprintf(stderr, "shm got %s\n", got.c_str());
    return 1;
  }
  return 0;
}
