#include "wave_dispatch/client.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>

#include <asio.hpp>

#include <ctime>
#include <cstring>
#include <utility>

namespace wave_dispatch {
namespace {

using tcp = asio::ip::tcp;

void apply_timeout(tcp::socket& sock, double seconds) {
  if (seconds <= 0) {
    return;
  }
  // Asio socket timeouts via SO_RCVTIMEO / SO_SNDTIMEO on native handle
  timeval tv{};
  tv.tv_sec = static_cast<time_t>(seconds);
  tv.tv_usec = static_cast<suseconds_t>((seconds - tv.tv_sec) * 1e6);
  ::setsockopt(sock.native_handle(), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(sock.native_handle(), SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

void tune_socket(tcp::socket& sock) {
  asio::error_code ec;
  sock.set_option(tcp::no_delay(true), ec);
  sock.set_option(asio::socket_base::keep_alive(true), ec);
  sock.set_option(asio::socket_base::receive_buffer_size(4 * 1024 * 1024), ec);
  sock.set_option(asio::socket_base::send_buffer_size(4 * 1024 * 1024), ec);
}

Message read_message_asio(tcp::socket& sock) {
  std::uint8_t hdr[4];
  asio::error_code ec;
  asio::read(sock, asio::buffer(hdr), ec);
  if (ec) {
    if (ec == asio::error::timed_out || ec == asio::error::would_block) {
      throw TimeoutError("socket recv timed out");
    }
    if (ec == asio::error::eof || ec == asio::error::connection_reset) {
      throw TransportError("connection closed while reading frame");
    }
    throw TransportError("read header: " + ec.message());
  }
  std::uint32_t be = 0;
  std::memcpy(&be, hdr, 4);
  const std::uint32_t length = ntohl(be);
  if (length > kMaxFrameBytes) {
    throw ProtocolError("frame length exceeds max");
  }
  std::vector<std::uint8_t> wire(4 + length);
  std::memcpy(wire.data(), hdr, 4);
  if (length) {
    asio::read(sock, asio::buffer(wire.data() + 4, length), ec);
    if (ec) {
      if (ec == asio::error::timed_out) {
        throw TimeoutError("socket recv timed out");
      }
      throw TransportError("read body: " + ec.message());
    }
  }
  return decode_message(wire.data(), wire.size());
}

void write_message_asio(tcp::socket& sock, const Message& msg) {
  const auto bytes = encode_message(msg);
  asio::error_code ec;
  asio::write(sock, asio::buffer(bytes), ec);
  if (ec) {
    if (ec == asio::error::timed_out) {
      throw TimeoutError("socket send timed out");
    }
    throw TransportError("write: " + ec.message());
  }
}

bool is_hangup_msg(const std::string& m) {
  return m.find("connection closed") != std::string::npos ||
         m.find("Connection reset") != std::string::npos ||
         m.find("Broken pipe") != std::string::npos ||
         m.find("End of file") != std::string::npos;
}

}  // namespace

struct JobClient::Impl {
  asio::io_context ioc{1};
  std::unique_ptr<tcp::socket> sock;
};

JobClient::JobClient(std::string host, int port, double timeout_s,
                     double connect_timeout_s, double control_timeout_s)
    : impl_(std::make_unique<Impl>()),
      host_(std::move(host)),
      port_(port),
      timeout_s_(timeout_s),
      connect_timeout_s_(connect_timeout_s),
      control_timeout_s_(control_timeout_s) {}

JobClient::~JobClient() { close(); }

JobClient::JobClient(JobClient&& other) noexcept
    : impl_(std::move(other.impl_)),
      host_(std::move(other.host_)),
      port_(other.port_),
      timeout_s_(other.timeout_s_),
      connect_timeout_s_(other.connect_timeout_s_),
      control_timeout_s_(other.control_timeout_s_),
      info_(std::move(other.info_)) {}

JobClient& JobClient::operator=(JobClient&& other) noexcept {
  if (this != &other) {
    close();
    impl_ = std::move(other.impl_);
    host_ = std::move(other.host_);
    port_ = other.port_;
    timeout_s_ = other.timeout_s_;
    connect_timeout_s_ = other.connect_timeout_s_;
    control_timeout_s_ = other.control_timeout_s_;
    info_ = std::move(other.info_);
  }
  return *this;
}

bool JobClient::connected() const {
  return impl_ && impl_->sock && impl_->sock->is_open();
}

WorkerInfo JobClient::connect() {
  close();
  if (!impl_) {
    impl_ = std::make_unique<Impl>();
  }
  tcp::resolver resolver(impl_->ioc);
  asio::error_code ec;
  auto endpoints = resolver.resolve(host_, std::to_string(port_), ec);
  if (ec) {
    throw TransportError("resolve: " + ec.message());
  }
  auto sock = std::make_unique<tcp::socket>(impl_->ioc);
  asio::connect(*sock, endpoints, ec);
  if (ec) {
    throw TransportError("connect: " + ec.message());
  }
  tune_socket(*sock);
  apply_timeout(*sock, connect_timeout_s_);
  impl_->sock = std::move(sock);

  apply_timeout(*impl_->sock, control_timeout_s_);
  write_message_asio(*impl_->sock,
                     Message{Json{{"type", "hello"},
                                  {"proto", kProtoVersion},
                                  {"client", "wave-dispatch"}},
                             {}});
  Message reply = read_message_asio(*impl_->sock);
  if (reply.meta.value("type", "") != "hello_ok" ||
      reply.meta.value("proto", -1) != kProtoVersion) {
    throw ProtocolError("unexpected hello reply: " + reply.meta.dump());
  }
  WorkerInfo info;
  info.endpoint = endpoint();
  info.workers = reply.meta.value("workers", 0);
  info.max_workers = reply.meta.value("max_workers", 0);
  info.default_workers = reply.meta.value("default_workers", 0);
  info.hostname = reply.meta.value("hostname", host_);
  info.device = reply.meta.value("device", "");
  info.raw_hello = reply.meta;
  info_ = info;
  return info;
}

void JobClient::close() noexcept {
  if (!impl_ || !impl_->sock) {
    return;
  }
  try {
    if (impl_->sock->is_open()) {
      write_message_asio(*impl_->sock, Message{Json{{"type", "bye"}}, {}});
    }
  } catch (...) {
  }
  asio::error_code ec;
  impl_->sock->shutdown(tcp::socket::shutdown_both, ec);
  impl_->sock->close(ec);
  impl_->sock.reset();
}

WorkerInfo JobClient::reconnect() {
  if (impl_ && impl_->sock) {
    asio::error_code ec;
    impl_->sock->close(ec);
    impl_->sock.reset();
  }
  return connect();
}

Json JobClient::ping() {
  if (!connected()) {
    throw TransportError("not connected");
  }
  apply_timeout(*impl_->sock, control_timeout_s_);
  write_message_asio(
      *impl_->sock,
      Message{Json{{"type", "ping"}, {"t0", std::time(nullptr)}}, {}});
  Message reply = read_message_asio(*impl_->sock);
  if (reply.meta.value("type", "") != "pong") {
    throw ProtocolError("unexpected ping reply");
  }
  return reply.meta;
}

Message JobClient::submit_message(const Message& req_in,
                                  const std::string& kind) {
  Message req = req_in;
  if (!req.meta.contains("type")) {
    req.meta["type"] = "job";
  }
  if (!req.meta.contains("kind")) {
    req.meta["kind"] = kind;
  }
  // If caller passed job fields at top level without nesting, leave as-is.
  // Conventional: meta has type/kind/job; blob is opaque payload.
  if (!req.meta.contains("job") && req.meta.contains("id")) {
    // already a job body used as meta — wrap
    Json job = req.meta;
    job.erase("type");
    job.erase("kind");
    req.meta = {{"type", "job"}, {"kind", kind}, {"job", job}};
  }

  std::exception_ptr last;
  for (int attempt = 0; attempt < 2; ++attempt) {
    try {
      if (!connected()) {
        throw TransportError("not connected");
      }
      apply_timeout(*impl_->sock, timeout_s_);
      write_message_asio(*impl_->sock, req);
      Message reply = read_message_asio(*impl_->sock);
      if (reply.meta.value("type", "") != "result") {
        throw ProtocolError("unexpected job reply: " + reply.meta.dump());
      }
      if (!reply.meta.value("ok", false)) {
        throw ProtocolError(reply.meta.value("error", "remote job failed"));
      }
      // Normalize: if result nested in meta and no blob, unwrap for JSON API.
      return reply;
    } catch (const TimeoutError&) {
      throw;
    } catch (const std::exception& e) {
      last = std::current_exception();
      if (attempt == 0 && is_hangup_msg(e.what())) {
        reconnect();
        continue;
      }
      throw;
    }
  }
  if (last) std::rethrow_exception(last);
  throw TransportError("submit_message failed");
}

Json JobClient::submit_job(const Json& job, const std::string& kind) {
  Message req;
  req.meta = {{"type", "job"}, {"kind", kind}, {"job", job}};
  Message reply = submit_message(req, kind);
  if (reply.meta.contains("result") && reply.meta["result"].is_object()) {
    return reply.meta["result"];
  }
  return reply.meta;
}

Json JobClient::control(const Json& msg) {
  std::exception_ptr last;
  for (int attempt = 0; attempt < 2; ++attempt) {
    try {
      if (!connected()) {
        throw TransportError("not connected");
      }
      apply_timeout(*impl_->sock, control_timeout_s_);
      write_message_asio(*impl_->sock, Message{msg, {}});
      return read_message_asio(*impl_->sock).meta;
    } catch (const TimeoutError&) {
      throw;
    } catch (const std::exception& e) {
      last = std::current_exception();
      if (attempt == 0 && is_hangup_msg(e.what())) {
        reconnect();
        continue;
      }
      throw;
    }
  }
  if (last) std::rethrow_exception(last);
  throw TransportError("control failed");
}

EndpointSpec parse_endpoint(const std::string& spec) {
  const auto pos = spec.rfind(':');
  if (pos == std::string::npos) {
    return EndpointSpec{spec, kDefaultPort};
  }
  return EndpointSpec{spec.substr(0, pos), std::stoi(spec.substr(pos + 1))};
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
    for (const auto& e : errors) msg += "; " + e;
    throw TransportError(msg);
  }
  return infos;
}

void WorkerFarm::close() noexcept {
  for (auto& c : clients_) c.close();
  clients_.clear();
}

int WorkerFarm::total_workers() const {
  int n = 0;
  for (const auto& c : clients_) {
    if (c.info()) n += c.info()->workers;
  }
  return n;
}

}  // namespace wave_dispatch
