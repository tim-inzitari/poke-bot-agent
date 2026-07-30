#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "wave_dispatch/frame.hpp"

namespace wave_dispatch {

struct WorkerInfo {
  std::string endpoint;
  int workers = 0;
  int max_workers = 0;
  int default_workers = 0;
  std::string hostname;
  std::string device;
  Json raw_hello;
};

/** One TCP session to a remote worker (opaque JSON jobs). */
class JobClient {
 public:
  JobClient(std::string host, int port = kDefaultPort, double timeout_s = 30.0,
            double connect_timeout_s = 60.0, double control_timeout_s = 300.0);
  ~JobClient();

  JobClient(const JobClient&) = delete;
  JobClient& operator=(const JobClient&) = delete;
  JobClient(JobClient&& other) noexcept;
  JobClient& operator=(JobClient&& other) noexcept;

  const std::string& host() const { return host_; }
  int port() const { return port_; }
  std::string endpoint() const { return host_ + ":" + std::to_string(port_); }
  const WorkerInfo* info() const { return info_ ? &*info_ : nullptr; }
  bool connected() const { return fd_ >= 0; }

  WorkerInfo connect();
  void close() noexcept;
  WorkerInfo reconnect();

  Json ping();
  /** Submit one opaque job; blocks until result frame. */
  Json submit_job(const Json& job, const std::string& kind = "play");
  /** Generic control-plane call (reload/pin/health/…); returns reply frame. */
  Json control(const Json& msg);

 private:
  void set_timeout(double seconds);
  int require_fd() const;
  bool is_hangup(const std::exception& e) const;

  std::string host_;
  int port_ = kDefaultPort;
  double timeout_s_ = 30.0;
  double connect_timeout_s_ = 60.0;
  double control_timeout_s_ = 300.0;
  int fd_ = -1;
  std::optional<WorkerInfo> info_;
};

struct EndpointSpec {
  std::string host;
  int port = kDefaultPort;
};

EndpointSpec parse_endpoint(const std::string& spec);

/** Multi-endpoint farm: soft-drop unreachable peers by default. */
class WorkerFarm {
 public:
  explicit WorkerFarm(std::vector<std::string> endpoints, double timeout_s = 30.0);

  std::vector<WorkerInfo> connect(bool require_all = false);
  void close() noexcept;
  std::vector<JobClient>& clients() { return clients_; }
  const std::vector<JobClient>& clients() const { return clients_; }
  int total_workers() const;

 private:
  std::vector<std::string> endpoints_;
  double timeout_s_ = 30.0;
  std::vector<JobClient> clients_;
};

}  // namespace wave_dispatch
