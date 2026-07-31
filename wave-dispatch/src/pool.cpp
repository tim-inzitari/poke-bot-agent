#include "wave_dispatch/pool.hpp"

#include <algorithm>

namespace wave_dispatch {

ConnectionPool::ConnectionPool() : ConnectionPool(Options{}) {}
ConnectionPool::ConnectionPool(Options opt) : opt_(std::move(opt)) {}
ConnectionPool::~ConnectionPool() { close(); }

bool ConnectionPool::is_localhost(const std::string& host) {
  return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

std::unique_ptr<JobClient> ConnectionPool::make_client(const std::string& endpoint) {
  std::string host;
  int port = kDefaultPort;
  if (endpoint.rfind("unix:", 0) == 0) {
    host = endpoint;
    port = 0;
  } else {
    auto spec = parse_endpoint(endpoint);
    host = spec.host;
    port = spec.port;
    if (opt_.prefer_uds && is_localhost(host)) {
      host = "unix:/tmp/wave_dispatch_" + std::to_string(port) + ".sock";
      port = 0;
    }
  }
  auto c = std::make_unique<JobClient>(host, port, opt_.timeout_s,
                                       opt_.connect_timeout_s,
                                       opt_.control_timeout_s);
  c->connect();
  return c;
}

void ConnectionPool::ensure(const std::string& endpoint, int n) {
  int need = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto& b = buckets_[endpoint];
    need = std::max(0, n - b.live);
  }
  std::vector<std::unique_ptr<JobClient>> fresh;
  fresh.reserve(static_cast<std::size_t>(need));
  for (int i = 0; i < need; ++i) {
    try {
      fresh.push_back(make_client(endpoint));
    } catch (...) {
      break;
    }
  }
  std::lock_guard<std::mutex> lock(mu_);
  auto& b = buckets_[endpoint];
  for (auto& c : fresh) {
    b.idle.push_back(std::move(c));
    ++b.live;
  }
}

JobClient* ConnectionPool::acquire(const std::string& endpoint, bool create) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto& b = buckets_[endpoint];
    while (!b.idle.empty()) {
      auto holder = std::move(b.idle.back());
      b.idle.pop_back();
      if (holder && holder->connected()) {
        JobClient* raw = holder.get();
        in_use_.emplace(raw, std::move(holder));
        return raw;
      }
      --b.live;
    }
  }
  if (!create) return nullptr;
  std::unique_ptr<JobClient> created;
  try {
    created = make_client(endpoint);
  } catch (...) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mu_);
  JobClient* raw = created.get();
  in_use_.emplace(raw, std::move(created));
  buckets_[endpoint].live += 1;
  return raw;
}

void ConnectionPool::release(const std::string& endpoint, JobClient* client,
                             bool healthy) {
  if (!client) return;
  std::lock_guard<std::mutex> lock(mu_);
  auto it = in_use_.find(client);
  if (it == in_use_.end()) return;
  auto holder = std::move(it->second);
  in_use_.erase(it);
  auto& b = buckets_[endpoint];
  if (!healthy || !holder || !holder->connected()) {
    --b.live;
    return;
  }
  b.idle.push_back(std::move(holder));
}

void ConnectionPool::close() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& [_, b] : buckets_) {
    b.idle.clear();
    b.live = 0;
  }
  in_use_.clear();
}

int ConnectionPool::idle_count(const std::string& endpoint) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = buckets_.find(endpoint);
  return it == buckets_.end() ? 0 : static_cast<int>(it->second.idle.size());
}

int ConnectionPool::live_count(const std::string& endpoint) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = buckets_.find(endpoint);
  return it == buckets_.end() ? 0 : it->second.live;
}

}  // namespace wave_dispatch
