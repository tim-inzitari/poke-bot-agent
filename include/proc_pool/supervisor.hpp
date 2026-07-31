#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace proc_pool {

struct WorkerSpec {
  std::vector<std::string> argv;
  std::uint32_t num_workers = 1;
  std::uint32_t recycle_tasks = 0;  // 0 = never
  double capacity_grace_s = 60.0;
};

struct TaskResult {
  std::uint64_t task_id = 0;
  std::int32_t worker_slot = -1;
  bool ok = false;
  std::string error;
  std::vector<std::uint8_t> payload;
};

/**
 * Recyclable process supervisor with length-prefixed stdin/stdout framing.
 *
 * Protocol (worker):
 *   read u32 BE length + task bytes
 *   write u32 BE length + result bytes
 *   empty length (0) means shutdown
 *
 * Domain work stays in the worker executable; this library only supervises.
 */
class Supervisor {
 public:
  explicit Supervisor(WorkerSpec spec);
  ~Supervisor();

  Supervisor(const Supervisor&) = delete;
  Supervisor& operator=(const Supervisor&) = delete;

  std::uint64_t submit(const std::uint8_t* data, std::size_t n);
  std::uint64_t submit(const std::vector<std::uint8_t>& data) {
    return submit(data.data(), data.size());
  }

  std::optional<TaskResult> try_get();
  TaskResult get(double timeout_s = 30.0);

  void request_stop(const std::string& reason = "");
  void join();
  bool healthy() const;
  std::string stop_reason() const;

 private:
  struct Worker;

  void ensure_workers_();
  void monitor_();
  void recycle_(Worker& w);
  void kill_worker_(Worker& w);

  WorkerSpec spec_;
  mutable std::mutex mu_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::atomic<std::uint64_t> next_task_{1};
  std::atomic<bool> stop_{false};
  std::string stop_reason_;
  std::thread monitor_thread_;
  std::vector<TaskResult> results_;
};

}  // namespace proc_pool
