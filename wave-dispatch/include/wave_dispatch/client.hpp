#pragma once

#include <cstdint>
#include <memory>
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

/** One TCP or Unix-domain session to a remote worker. */
class JobClient {
 public:
  /**
   * host: hostname, IPv4, or "unix:/path/to.sock"
   * port: ignored for unix: endpoints
   */
  JobClient(std::string host, int port = kDefaultPort, double timeout_s = 30.0,
            double connect_timeout_s = 60.0, double control_timeout_s = 300.0);
  ~JobClient();

  JobClient(const JobClient&) = delete;
  JobClient& operator=(const JobClient&) = delete;
  JobClient(JobClient&& other) noexcept;
  JobClient& operator=(JobClient&& other) noexcept;

  const std::string& host() const { return host_; }
  int port() const { return port_; }
  bool is_unix() const { return unix_path_.has_value(); }
  std::string endpoint() const;
  const WorkerInfo* info() const { return info_ ? &*info_ : nullptr; }
  bool connected() const;

  WorkerInfo connect();
  void close() noexcept;
  WorkerInfo reconnect();

  Json ping();
  Json submit_job(const Json& job, const std::string& kind = "play");
  Message submit_message(const Message& msg, const std::string& kind = "play");
  /** Proto v2: many jobs in one round-trip. */
  std::vector<Message> submit_batch(const std::vector<Message>& jobs,
                                    const std::string& kind = "play",
                                    bool compress = true);
  Json control(const Json& msg);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  std::string host_;
  int port_ = kDefaultPort;
  std::optional<std::string> unix_path_;
  double timeout_s_ = 30.0;
  double connect_timeout_s_ = 60.0;
  double control_timeout_s_ = 300.0;
  std::optional<WorkerInfo> info_;
};

struct EndpointSpec {
  std::string host;
  int port = kDefaultPort;
};

EndpointSpec parse_endpoint(const std::string& spec);

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
