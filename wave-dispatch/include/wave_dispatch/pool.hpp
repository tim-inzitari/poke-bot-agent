#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "wave_dispatch/client.hpp"

namespace wave_dispatch {

/**
 * Persistent warm JobClient pool — connections survive across waves.
 * acquire/release are thread-safe.
 */
class ConnectionPool {
 public:
  struct Options {
    double timeout_s = 30.0;
    double connect_timeout_s = 60.0;
    double control_timeout_s = 300.0;
    /** Prefer Unix domain sockets for localhost endpoints. */
    bool prefer_uds = true;
  };

  ConnectionPool();
  explicit ConnectionPool(Options opt);
  ~ConnectionPool();

  void ensure(const std::string& endpoint, int n);
  JobClient* acquire(const std::string& endpoint, bool create = true);
  void release(const std::string& endpoint, JobClient* client, bool healthy = true);
  void close();

  int idle_count(const std::string& endpoint) const;
  int live_count(const std::string& endpoint) const;

 private:
  struct Bucket {
    std::vector<std::unique_ptr<JobClient>> idle;
    int live = 0;
  };

  std::unique_ptr<JobClient> make_client(const std::string& endpoint);
  static bool is_localhost(const std::string& host);

  Options opt_;
  mutable std::mutex mu_;
  std::unordered_map<std::string, Bucket> buckets_;
  std::unordered_map<JobClient*, std::unique_ptr<JobClient>> in_use_;
};

}  // namespace wave_dispatch
