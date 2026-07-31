#include "wave_dispatch/server.hpp"

#include <arpa/inet.h>

#include <asio.hpp>

#include <array>
#include <ctime>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace wave_dispatch {
namespace {

using tcp = asio::ip::tcp;

void set_socket_options(tcp::socket& sock, const ServerConfig& cfg) {
  asio::error_code ec;
  sock.set_option(tcp::no_delay(cfg.tcp_nodelay), ec);
  sock.set_option(asio::socket_base::keep_alive(true), ec);
  // Enlarge buffers for large trajectory frames
  sock.set_option(asio::socket_base::receive_buffer_size(4 * 1024 * 1024), ec);
  sock.set_option(asio::socket_base::send_buffer_size(4 * 1024 * 1024), ec);
}

class Session : public std::enable_shared_from_this<Session> {
 public:
  Session(tcp::socket sock, MessageHandler handler, HelloFn hello,
          ServerConfig cfg, std::atomic<int>* active,
          const std::atomic<bool>* stop)
      : sock_(std::move(sock)),
        handler_(std::move(handler)),
        hello_(std::move(hello)),
        cfg_(std::move(cfg)),
        active_(active),
        stop_(stop),
        timer_(sock_.get_executor()) {}

  ~Session() {
    if (active_) {
      active_->fetch_sub(1, std::memory_order_relaxed);
    }
  }

  void start() {
    set_socket_options(sock_, cfg_);
    read_header();
  }

 private:
  void arm_idle_timer() {
    timer_.expires_after(
        std::chrono::milliseconds(static_cast<int>(cfg_.idle_timeout_s * 1000)));
    auto self = shared_from_this();
    timer_.async_wait([self](const asio::error_code& ec) {
      if (!ec) {
        // Idle timeout: do not close — farm sockets sit between waves.
        // Just re-arm; actual hangup detected on read error.
        self->arm_idle_timer();
      }
    });
  }

  void read_header() {
    if (stop_ && stop_->load(std::memory_order_relaxed)) {
      return;
    }
    arm_idle_timer();
    auto self = shared_from_this();
    asio::async_read(
        sock_, asio::buffer(hdr_),
        [self](const asio::error_code& ec, std::size_t) {
          if (ec) {
            return;
          }
          std::uint32_t be = 0;
          std::memcpy(&be, self->hdr_.data(), 4);
          const std::uint32_t length = ntohl(be);
          if (length > kMaxFrameBytes) {
            return;
          }
          self->body_.resize(length);
          self->read_body(length);
        });
  }

  void read_body(std::uint32_t length) {
    auto self = shared_from_this();
    if (length == 0) {
      self->on_frame();
      return;
    }
    asio::async_read(
        sock_, asio::buffer(self->body_),
        [self](const asio::error_code& ec, std::size_t) {
          if (ec) {
            return;
          }
          self->on_frame();
        });
  }

  void on_frame() {
    timer_.cancel();
    std::vector<std::uint8_t> wire(4 + body_.size());
    std::memcpy(wire.data(), hdr_.data(), 4);
    if (!body_.empty()) {
      std::memcpy(wire.data() + 4, body_.data(), body_.size());
    }
    Message msg;
    try {
      msg = decode_message(wire.data(), wire.size());
    } catch (...) {
      return;
    }

    if (!hello_done_) {
      if (msg.meta.value("type", "") != "hello") {
        write_json({{"type", "error"}, {"error", "expected hello"}});
        return;
      }
      if (msg.meta.value("proto", -1) != kProtoVersion) {
        write_json({{"type", "error"}, {"error", "unsupported proto"}});
        return;
      }
      Json info = hello_ ? hello_() : Json::object();
      Json hello_ok = {{"type", "hello_ok"}, {"proto", kProtoVersion}};
      for (auto it = info.begin(); it != info.end(); ++it) {
        hello_ok[it.key()] = it.value();
      }
      hello_done_ = true;
      write_json(hello_ok);
      return;
    }

    const std::string type = msg.meta.value("type", "");
    if (type == "bye") {
      return;
    }
    if (type == "ping") {
      write_json({{"type", "pong"},
                  {"t0", msg.meta.value("t0", 0)},
                  {"t1", std::time(nullptr)}});
      return;
    }

    Message reply;
    try {
      reply = handler_(msg);
    } catch (const std::exception& e) {
      reply.meta = {{"type", type == "job" ? "result" : "error"},
                    {"ok", false},
                    {"error", e.what()}};
    }
    write_message(reply);
  }

  void write_json(Json j) {
    Message m;
    m.meta = std::move(j);
    write_message(m);
  }

  void write_message(const Message& msg) {
    auto bytes = std::make_shared<std::vector<std::uint8_t>>(encode_message(msg));
    auto self = shared_from_this();
    asio::async_write(
        sock_, asio::buffer(*bytes),
        [self, bytes](const asio::error_code& ec, std::size_t) {
          if (ec) {
            return;
          }
          self->read_header();
        });
  }

  tcp::socket sock_;
  MessageHandler handler_;
  HelloFn hello_;
  ServerConfig cfg_;
  std::atomic<int>* active_;
  const std::atomic<bool>* stop_;
  asio::steady_timer timer_;
  std::array<std::uint8_t, 4> hdr_{};
  std::vector<std::uint8_t> body_;
  bool hello_done_ = false;
};

}  // namespace

void serve_forever(MessageHandler handler, ServerConfig config, HelloFn hello,
                   const std::atomic<bool>* stop) {
  int nthreads = config.io_threads;
  if (nthreads <= 0) {
    nthreads = static_cast<int>(std::max(2u, std::thread::hardware_concurrency()));
  }

  asio::io_context ioc(nthreads);
  tcp::acceptor acceptor(ioc);

  tcp::endpoint ep;
  if (config.host == "0.0.0.0" || config.host.empty()) {
    ep = tcp::endpoint(tcp::v4(), static_cast<std::uint16_t>(config.port));
  } else {
    ep = tcp::endpoint(asio::ip::make_address(config.host),
                       static_cast<std::uint16_t>(config.port));
  }

  asio::error_code ec;
  acceptor.open(ep.protocol(), ec);
  acceptor.set_option(asio::socket_base::reuse_address(true), ec);
#ifdef SO_REUSEPORT
  if (config.reuse_port) {
    int one = 1;
    ::setsockopt(acceptor.native_handle(), SOL_SOCKET, SO_REUSEPORT, &one,
                 sizeof(one));
  }
#endif
  acceptor.bind(ep, ec);
  if (ec) {
    throw TransportError("bind: " + ec.message());
  }
  acceptor.listen(config.backlog, ec);
  if (ec) {
    throw TransportError("listen: " + ec.message());
  }

  std::atomic<int> active{0};

  std::function<void()> do_accept;
  do_accept = [&]() {
    acceptor.async_accept([&](const asio::error_code& aec, tcp::socket sock) {
      if (!aec) {
        if (active.load(std::memory_order_relaxed) < config.max_connections) {
          active.fetch_add(1, std::memory_order_relaxed);
          std::make_shared<Session>(std::move(sock), handler, hello, config,
                                    &active, stop)
              ->start();
        }
      }
      if (stop == nullptr || !stop->load(std::memory_order_relaxed)) {
        do_accept();
      }
    });
  };
  do_accept();

  // Periodic stop poll
  asio::steady_timer stop_timer(ioc);
  std::function<void()> arm_stop;
  arm_stop = [&]() {
    stop_timer.expires_after(std::chrono::milliseconds(50));
    stop_timer.async_wait([&](const asio::error_code&) {
      if (stop && stop->load(std::memory_order_relaxed)) {
        asio::error_code ignored;
        acceptor.close(ignored);
        ioc.stop();
        return;
      }
      arm_stop();
    });
  };
  arm_stop();

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(nthreads));
  for (int i = 0; i < nthreads; ++i) {
    pool.emplace_back([&ioc]() { ioc.run(); });
  }
  for (auto& t : pool) {
    t.join();
  }
}

void serve_forever(JobHandler handler, ServerConfig config, HelloFn hello,
                   const std::atomic<bool>* stop) {
  MessageHandler mh = [handler = std::move(handler)](const Message& msg) {
    Message out;
    out.meta = handler(msg.meta);
    // Echo blob through if handler returned pure JSON without blob —
    // JSON handlers ignore blobs.
    return out;
  };
  serve_forever(std::move(mh), std::move(config), std::move(hello), stop);
}

}  // namespace wave_dispatch
