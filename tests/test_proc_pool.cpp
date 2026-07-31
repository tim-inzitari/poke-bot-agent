#include "proc_pool/supervisor.hpp"

#include <cstdio>
#include <filesystem>
#include <string>

int test_proc_pool() {
  // Prefer built echo worker next to test binary; fall back to /bin/cat-like custom.
  std::filesystem::path worker = std::filesystem::path("apps") / "rl_echo_worker";
  if (!std::filesystem::exists(worker)) {
    worker = std::filesystem::current_path() / "rl_echo_worker";
  }
  if (!std::filesystem::exists(worker)) {
    // Build tree: build/rl_echo_worker when ctest runs from build/
    worker = "rl_echo_worker";
  }
  if (!std::filesystem::exists(worker)) {
    std::fprintf(stderr, "skip proc_pool: echo worker missing\n");
    return 0;
  }
  proc_pool::WorkerSpec spec;
  spec.argv = {worker.string()};
  spec.num_workers = 2;
  spec.recycle_tasks = 10;
  proc_pool::Supervisor sup(spec);
  const std::string payload = "ping";
  const auto id = sup.submit(reinterpret_cast<const std::uint8_t*>(payload.data()),
                             payload.size());
  auto tr = sup.get(5.0);
  sup.request_stop("done");
  sup.join();
  if (!tr.ok || tr.task_id != id) {
    std::fprintf(stderr, "proc_pool failed: ok=%d err=%s\n", int(tr.ok),
                 tr.error.c_str());
    return 1;
  }
  const std::string got(tr.payload.begin(), tr.payload.end());
  if (got != payload) {
    std::fprintf(stderr, "proc_pool echo mismatch: %s\n", got.c_str());
    return 1;
  }
  return 0;
}
