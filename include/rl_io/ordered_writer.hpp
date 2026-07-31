#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

namespace rl_io {

using Json = nlohmann::json;

/**
 * Crash-safe ordered append writer.
 *
 * Workers may submit opaque records out of order. The writer commits contiguous
 * job indices with fsynced replay + journal streams, then an atomic JSON
 * checkpoint. Recovery truncates both streams to the last checkpoint.
 *
 * Payload and journal metadata are opaque bytes/JSON — no domain schema.
 */
class OrderedWriter {
 public:
  struct Config {
    std::string replay_partial;
    std::uint64_t expected_jobs = 0;
    std::size_t queue_depth = 64;
    std::size_t fsync_batch = 8;
  };

  explicit OrderedWriter(Config cfg);
  ~OrderedWriter();

  OrderedWriter(const OrderedWriter&) = delete;
  OrderedWriter& operator=(const OrderedWriter&) = delete;

  std::uint64_t resume_index() const;
  std::uint64_t written_records() const;

  /** Return false if already durably committed before resume. */
  bool submit(std::uint64_t job_index,
              const std::optional<std::string>& record_bytes,
              const Json& result_metadata,
              double timeout_s = 30.0);

  Json close();
  Json abort(const std::string& reason);
  Json telemetry() const;

  void finalize(const std::string& final_path);
  std::string quarantine(const std::string& suffix);

 private:
  enum class Cmd { None, Close, Abort };

  struct Item {
    std::uint64_t index = 0;
    std::optional<std::string> record;
    Json metadata;
  };

  void run_();
  void drain_ready_(bool force);
  void commit_(const std::vector<Item>& batch);
  Json load_or_create_state_();
  void save_state_(const Json& state);
  Json finish_(Cmd cmd, const std::string& reason);

  Config cfg_;
  std::string journal_path_;
  std::string state_path_;
  Json state_;
  std::uint64_t next_index_ = 0;
  std::uint64_t written_records_ = 0;

  mutable std::mutex mu_;
  std::condition_variable cv_space_;
  std::condition_variable cv_item_;
  std::deque<Item> queue_;
  Cmd cmd_ = Cmd::None;
  std::string abort_reason_;
  std::unordered_map<std::uint64_t, Item> pending_;
  std::unordered_set<std::uint64_t> submitted_;
  std::exception_ptr error_;
  bool closed_ = false;
  bool aborted_ = false;
  double queue_wait_total_ = 0.0;
  double queue_wait_max_ = 0.0;
  std::size_t max_queue_depth_ = 0;
  double started_ = 0.0;

  std::fstream replay_;
  std::fstream journal_;
  std::thread thread_;
};

}  // namespace rl_io
