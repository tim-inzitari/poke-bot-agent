#include "wave_dispatch/client.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "wave_dispatch/asio_config.hpp"
#include <asio.hpp>
#include <asio/local/stream_protocol.hpp>

#include <ctime>
#include <cstring>
#include <type_traits>
#include <utility>

#include "wave_dispatch/batch.hpp"
#include "wave_dispatch/buffer_pool.hpp"

namespace wave_dispatch {
namespace {

using tcp = asio::ip::tcp;
using uds = asio::local::stream_protocol;

void apply_timeout_fd(int fd, double seconds) {
  if (fd < 0 || seconds <= 0) return;
  timeval tv{};
  tv.tv_sec = static_cast<time_t>(seconds);
  tv.tv_usec = static_cast<suseconds_t>((seconds - tv.tv_sec) * 1e6);
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

template <typename Socket>
void tune_socket(Socket& sock) {
  asio::error_code ec;
  if constexpr (std::is_same_v<Socket, tcp::socket>) {
    sock.set_option(tcp::no_delay(true), ec);
  }
  sock.set_option(asio::socket_base::keep_alive(true), ec);
  sock.set_option(asio::socket_base::receive_buffer_size(4 * 1024 * 1024), ec);
  sock.set_option(asio::socket_base::send_buffer_size(4 * 1024 * 1024), ec);
}

template <typename Socket>
Message read_message_sock(Socket& sock) {
  std::uint8_t hdr[4];
  asio::error_code ec;
  asio::read(sock, asio::buffer(hdr), ec);
  if (ec) {
    if (ec == asio::error::timed_out || ec == asio::error::would_block)
      throw TimeoutError("socket recv timed out");
    if (ec == asio::error::eof || ec == asio::error::connection_reset)
      throw TransportError("connection closed while reading frame");
    throw TransportError("read header: " + ec.message());
  }
  std::uint32_t be = 0;
  std::memcpy(&be, hdr, 4);
  const std::uint32_t length = ntohl(be);
  if (length > kMaxFrameBytes) throw ProtocolError("frame length exceeds max");
  auto wire = BufferPool::instance().acquire(4 + length);
  wire.resize(4 + length);
  std::memcpy(wire.data(), hdr, 4);
  if (length) {
    asio::read(sock, asio::buffer(wire.data() + 4, length), ec);
    if (ec) {
      BufferPool::instance().release(std::move(wire));
      if (ec == asio::error::timed_out) throw TimeoutError("socket recv timed out");
      throw TransportError("read body: " + ec.message());
    }
  }
  Message msg = decode_message(wire.data(), wire.size());
  BufferPool::instance().release(std::move(wire));
  return msg;
}

template <typename Socket>
void write_message_sock(Socket& sock, const Message& msg) {
  auto bytes = encode_message(msg);
  asio::error_code ec;
  asio::write(sock, asio::buffer(bytes), ec);
  if (ec) {
    if (ec == asio::error::timed_out) throw TimeoutError("socket send timed out");
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
  std::unique_ptr<tcp::socket> tcp_sock;
  std::unique_ptr<uds::socket> uds_sock;

  bool open() const {
    return (tcp_sock && tcp_sock->is_open()) || (uds_sock && uds_sock->is_open());
  }
  int native() const {
    if (tcp_sock) return tcp_sock->native_handle();
    if (uds_sock) return uds_sock->native_handle();
    return -1;
  }
  void close_sock() {
    asio::error_code ec;
    if (tcp_sock) {
      tcp_sock->shutdown(tcp::socket::shutdown_both, ec);
      tcp_sock->close(ec);
      tcp_sock.reset();
    }
    if (uds_sock) {
      uds_sock->shutdown(uds::socket::shutdown_both, ec);
      uds_sock->close(ec);
      uds_sock.reset();
    }
  }
};

JobClient::JobClient(std::string host, int port, double timeout_s,
                     double connect_timeout_s, double control_timeout_s)
    : impl_(std::make_unique<Impl>()),
      host_(std::move(host)),
      port_(port),
      timeout_s_(timeout_s),
      connect_timeout_s_(connect_timeout_s),
      control_timeout_s_(control_timeout_s) {
  if (host_.rfind("unix:", 0) == 0) {
    unix_path_ = host_.substr(5);
  }
}

JobClient::~JobClient() { close(); }

JobClient::JobClient(JobClient&& other) noexcept
    : impl_(std::move(other.impl_)),
      host_(std::move(other.host_)),
      port_(other.port_),
      unix_path_(std::move(other.unix_path_)),
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
    unix_path_ = std::move(other.unix_path_);
    timeout_s_ = other.timeout_s_;
    connect_timeout_s_ = other.connect_timeout_s_;
    control_timeout_s_ = other.control_timeout_s_;
    info_ = std::move(other.info_);
  }
  return *this;
}

std::string JobClient::endpoint() const {
  if (unix_path_) return "unix:" + *unix_path_;
  return host_ + ":" + std::to_string(port_);
}

bool JobClient::connected() const { return impl_ && impl_->open(); }

WorkerInfo JobClient::connect() {
  close();
  if (!impl_) impl_ = std::make_unique<Impl>();
  asio::error_code ec;

  if (unix_path_) {
    auto sock = std::make_unique<uds::socket>(impl_->ioc);
    sock->connect(uds::endpoint(*unix_path_), ec);
    if (ec) throw TransportError("uds connect: " + ec.message());
    tune_socket(*sock);
    apply_timeout_fd(sock->native_handle(), connect_timeout_s_);
    impl_->uds_sock = std::move(sock);
  } else {
    tcp::resolver resolver(impl_->ioc);
    auto endpoints = resolver.resolve(host_, std::to_string(port_), ec);
    if (ec) throw TransportError("resolve: " + ec.message());
    auto sock = std::make_unique<tcp::socket>(impl_->ioc);
    asio::connect(*sock, endpoints, ec);
    if (ec) throw TransportError("connect: " + ec.message());
    tune_socket(*sock);
    apply_timeout_fd(sock->native_handle(), connect_timeout_s_);
    impl_->tcp_sock = std::move(sock);
  }

  apply_timeout_fd(impl_->native(), control_timeout_s_);
  Message hello{Json{{"type", "hello"},
                     {"proto", kProtoVersion},
                     {"client", "wave-dispatch"}},
                {}};
  Message reply;
  if (impl_->tcp_sock) {
    write_message_sock(*impl_->tcp_sock, hello);
    reply = read_message_sock(*impl_->tcp_sock);
  } else {
    write_message_sock(*impl_->uds_sock, hello);
    reply = read_message_sock(*impl_->uds_sock);
  }
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
  if (!impl_) return;
  try {
    if (impl_->open()) {
      Message bye{Json{{"type", "bye"}}, {}};
      if (impl_->tcp_sock) write_message_sock(*impl_->tcp_sock, bye);
      else if (impl_->uds_sock) write_message_sock(*impl_->uds_sock, bye);
    }
  } catch (...) {
  }
  impl_->close_sock();
}

WorkerInfo JobClient::reconnect() {
  if (impl_) impl_->close_sock();
  return connect();
}

Json JobClient::ping() {
  if (!connected()) throw TransportError("not connected");
  apply_timeout_fd(impl_->native(), control_timeout_s_);
  Message req{Json{{"type", "ping"}, {"t0", std::time(nullptr)}}, {}};
  Message reply;
  if (impl_->tcp_sock) {
    write_message_sock(*impl_->tcp_sock, req);
    reply = read_message_sock(*impl_->tcp_sock);
  } else {
    write_message_sock(*impl_->uds_sock, req);
    reply = read_message_sock(*impl_->uds_sock);
  }
  if (reply.meta.value("type", "") != "pong") throw ProtocolError("unexpected ping reply");
  return reply.meta;
}

Message JobClient::submit_message(const Message& req_in, const std::string& kind) {
  Message req = req_in;
  if (!req.meta.contains("type")) req.meta["type"] = "job";
  if (!req.meta.contains("kind")) req.meta["kind"] = kind;
  if (!req.meta.contains("job") && req.meta.contains("id")) {
    Json job = req.meta;
    job.erase("type");
    job.erase("kind");
    req.meta = {{"type", "job"}, {"kind", kind}, {"job", job}};
  }

  std::exception_ptr last;
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool wrote = false;
    try {
      if (!connected()) throw TransportError("not connected");
      apply_timeout_fd(impl_->native(), timeout_s_);
      Message reply;
      if (impl_->tcp_sock) {
        write_message_sock(*impl_->tcp_sock, req);
        wrote = true;
        reply = read_message_sock(*impl_->tcp_sock);
      } else {
        write_message_sock(*impl_->uds_sock, req);
        wrote = true;
        reply = read_message_sock(*impl_->uds_sock);
      }
      if (reply.meta.value("type", "") != "result" &&
          reply.meta.value("type", "") != "results") {
        throw ProtocolError("unexpected job reply: " + reply.meta.dump());
      }
      if (reply.meta.contains("ok") && !reply.meta.value("ok", false) &&
          reply.meta.value("type", "") == "result") {
        throw ProtocolError(reply.meta.value("error", "remote job failed"));
      }
      return reply;
    } catch (const TimeoutError&) {
      throw;
    } catch (const std::exception& e) {
      last = std::current_exception();
      // Retry only when the request may not have reached the server.
      if (attempt == 0 && !wrote && is_hangup_msg(e.what())) {
        reconnect();
        continue;
      }
      if (wrote) {
        throw TransportError(
            std::string("ambiguous submit after write (no automatic retry): ") +
            e.what());
      }
      throw;
    }
  }
  if (last) std::rethrow_exception(last);
  throw TransportError("submit_message failed");
}

std::vector<Message> JobClient::submit_batch(const std::vector<Message>& jobs,
                                             const std::string& kind,
                                             bool compress) {
  if (jobs.empty()) return {};
  if (jobs.size() == 1) {
    Message r = submit_message(jobs[0], kind);
    if (r.meta.value("type", "") == "results") return unpack_batch(r);
    if (r.meta.contains("result") && r.blob.empty()) {
      return {Message{r.meta["result"], {}}};
    }
    return {std::move(r)};
  }
  Message batch = pack_batch("jobs", kind, jobs, compress);
  Message reply = submit_message(batch, kind);
  if (reply.meta.value("type", "") != "results") {
    // Server may not support batch — shouldn't happen if capabilities ok
    throw ProtocolError("expected results batch, got " + reply.meta.value("type", ""));
  }
  return unpack_batch(reply);
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
      if (!connected()) throw TransportError("not connected");
      apply_timeout_fd(impl_->native(), control_timeout_s_);
      if (impl_->tcp_sock) {
        write_message_sock(*impl_->tcp_sock, Message{msg, {}});
        return read_message_sock(*impl_->tcp_sock).meta;
      }
      write_message_sock(*impl_->uds_sock, Message{msg, {}});
      return read_message_sock(*impl_->uds_sock).meta;
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
  if (spec.rfind("unix:", 0) == 0) {
    return EndpointSpec{spec, 0};
  }
  const auto pos = spec.rfind(':');
  if (pos == std::string::npos) return EndpointSpec{spec, kDefaultPort};
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
