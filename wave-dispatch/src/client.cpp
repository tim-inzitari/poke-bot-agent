#include "wave_dispatch/client.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <ctime>
#include <cstring>
#include <exception>

namespace wave_dispatch {
namespace {

void set_fd_timeout(int fd, double seconds) {
  if (fd < 0) {
    return;
  }
  timeval tv{};
  if (seconds <= 0) {
    tv.tv_sec = 0;
    tv.tv_usec = 0;
  } else {
    tv.tv_sec = static_cast<time_t>(seconds);
    tv.tv_usec = static_cast<suseconds_t>((seconds - tv.tv_sec) * 1e6);
  }
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

int connect_tcp(const std::string& host, int port, double timeout_s) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string port_s = std::to_string(port);
  const int rc = getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
  if (rc != 0) {
    throw TransportError(std::string("getaddrinfo: ") + gai_strerror(rc));
  }
  int fd = -1;
  std::string last_err;
  for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
    fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) {
      last_err = std::strerror(errno);
      continue;
    }
    set_fd_timeout(fd, timeout_s);
    if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
      break;
    }
    last_err = std::strerror(errno);
    ::close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0) {
    throw TransportError("connect failed: " + last_err);
  }
  return fd;
}

}  // namespace

EndpointSpec parse_endpoint(const std::string& spec) {
  const auto pos = spec.rfind(':');
  if (pos == std::string::npos) {
    return EndpointSpec{spec, kDefaultPort};
  }
  EndpointSpec out;
  out.host = spec.substr(0, pos);
  out.port = std::stoi(spec.substr(pos + 1));
  return out;
}

JobClient::JobClient(std::string host, int port, double timeout_s,
                     double connect_timeout_s, double control_timeout_s)
    : host_(std::move(host)),
      port_(port),
      timeout_s_(timeout_s),
      connect_timeout_s_(connect_timeout_s),
      control_timeout_s_(control_timeout_s) {}

JobClient::~JobClient() { close(); }

JobClient::JobClient(JobClient&& other) noexcept
    : host_(std::move(other.host_)),
      port_(other.port_),
      timeout_s_(other.timeout_s_),
      connect_timeout_s_(other.connect_timeout_s_),
      control_timeout_s_(other.control_timeout_s_),
      fd_(other.fd_),
      info_(std::move(other.info_)) {
  other.fd_ = -1;
}

JobClient& JobClient::operator=(JobClient&& other) noexcept {
  if (this != &other) {
    close();
    host_ = std::move(other.host_);
    port_ = other.port_;
    timeout_s_ = other.timeout_s_;
    connect_timeout_s_ = other.connect_timeout_s_;
    control_timeout_s_ = other.control_timeout_s_;
    fd_ = other.fd_;
    info_ = std::move(other.info_);
    other.fd_ = -1;
  }
  return *this;
}

void JobClient::set_timeout(double seconds) { set_fd_timeout(fd_, seconds); }

int JobClient::require_fd() const {
  if (fd_ < 0) {
    throw TransportError("not connected");
  }
  return fd_;
}

bool JobClient::is_hangup(const std::exception& e) const {
  const std::string m = e.what();
  return m.find("connection closed") != std::string::npos ||
         m.find("Broken pipe") != std::string::npos ||
         m.find("Connection reset") != std::string::npos;
}

WorkerInfo JobClient::connect() {
  close();
  fd_ = connect_tcp(host_, port_, connect_timeout_s_);
  set_timeout(control_timeout_s_);
  send_frame(fd_, Json{{"type", "hello"},
                       {"proto", kProtoVersion},
                       {"client", "wave-dispatch"}});
  const Json reply = read_frame(fd_);
  if (reply.value("type", "") != "hello_ok" ||
      reply.value("proto", -1) != kProtoVersion) {
    throw ProtocolError("unexpected hello reply: " + reply.dump());
  }
  WorkerInfo info;
  info.endpoint = endpoint();
  info.workers = reply.value("workers", 0);
  info.max_workers = reply.value("max_workers", 0);
  info.default_workers = reply.value("default_workers", 0);
  info.hostname = reply.value("hostname", host_);
  info.device = reply.value("device", "");
  info.raw_hello = reply;
  info_ = info;
  return info;
}

void JobClient::close() noexcept {
  if (fd_ >= 0) {
    try {
      send_frame(fd_, Json{{"type", "bye"}});
    } catch (...) {
    }
    ::close(fd_);
    fd_ = -1;
  }
}

WorkerInfo JobClient::reconnect() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  return connect();
}

Json JobClient::ping() {
  const int fd = require_fd();
  set_timeout(control_timeout_s_);
  send_frame(fd, Json{{"type", "ping"}, {"t0", std::time(nullptr)}});
  const Json reply = read_frame(fd);
  if (reply.value("type", "") != "pong") {
    throw ProtocolError("unexpected ping reply");
  }
  return reply;
}

Json JobClient::submit_job(const Json& job, const std::string& kind) {
  std::exception_ptr last;
  for (int attempt = 0; attempt < 2; ++attempt) {
    try {
      const int fd = require_fd();
      set_timeout(timeout_s_);
      send_frame(fd, Json{{"type", "job"}, {"kind", kind}, {"job", job}});
      const Json reply = read_frame(fd);
      if (reply.value("type", "") != "result") {
        throw ProtocolError("unexpected job reply: " + reply.dump());
      }
      if (!reply.value("ok", false)) {
        throw ProtocolError(reply.value("error", "remote job failed"));
      }
      if (!reply.contains("result") || !reply["result"].is_object()) {
        throw ProtocolError("remote result missing body");
      }
      return reply["result"];
    } catch (const TimeoutError&) {
      throw;
    } catch (const std::exception& e) {
      last = std::current_exception();
      if (attempt == 0 && is_hangup(e)) {
        reconnect();
        continue;
      }
      throw;
    }
  }
  if (last) {
    std::rethrow_exception(last);
  }
  throw TransportError("submit_job failed");
}

Json JobClient::control(const Json& msg) {
  std::exception_ptr last;
  for (int attempt = 0; attempt < 2; ++attempt) {
    try {
      const int fd = require_fd();
      set_timeout(control_timeout_s_);
      send_frame(fd, msg);
      return read_frame(fd);
    } catch (const TimeoutError&) {
      throw;
    } catch (const std::exception& e) {
      last = std::current_exception();
      if (attempt == 0 && is_hangup(e)) {
        reconnect();
        continue;
      }
      throw;
    }
  }
  if (last) {
    std::rethrow_exception(last);
  }
  throw TransportError("control failed");
}

WorkerFarm::WorkerFarm(std::vector<std::string> endpoints, double timeout_s)
    : endpoints_(std::move(endpoints)), timeout_s_(timeout_s) {}

std::vector<WorkerInfo> WorkerFarm::connect(bool require_all) {
  close();
  std::vector<WorkerInfo> infos;
  std::vector<std::string> errors;
  for (const auto& ep : endpoints_) {
    const auto spec = parse_endpoint(ep);
    JobClient client(spec.host, spec.port, timeout_s_);
    try {
      infos.push_back(client.connect());
      clients_.push_back(std::move(client));
    } catch (const std::exception& e) {
      errors.push_back(ep + ": " + e.what());
      if (require_all) {
        close();
        throw TransportError("require_all connect failed: " + errors.back());
      }
    }
  }
  if (infos.empty() && !endpoints_.empty()) {
    std::string msg = "no remote endpoints reachable";
    for (const auto& e : errors) {
      msg += "; " + e;
    }
    throw TransportError(msg);
  }
  return infos;
}

void WorkerFarm::close() noexcept {
  for (auto& c : clients_) {
    c.close();
  }
  clients_.clear();
}

int WorkerFarm::total_workers() const {
  int n = 0;
  for (const auto& c : clients_) {
    if (c.info()) {
      n += c.info()->workers;
    }
  }
  return n;
}

}  // namespace wave_dispatch
