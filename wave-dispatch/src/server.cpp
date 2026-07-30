#include "wave_dispatch/server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <ctime>
#include <cstring>
#include <thread>
#include <vector>

namespace wave_dispatch {
namespace {

void set_recv_timeout(int fd, double seconds) {
  timeval tv{};
  tv.tv_sec = static_cast<time_t>(seconds);
  tv.tv_usec = static_cast<suseconds_t>((seconds - tv.tv_sec) * 1e6);
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

void handle_connection(int fd, JobHandler handler, HelloFn hello,
                       double idle_timeout_s, const std::atomic<bool>* stop) {
  set_recv_timeout(fd, idle_timeout_s);
  try {
    Json first;
    try {
      first = read_frame(fd);
    } catch (...) {
      ::close(fd);
      return;
    }
    if (first.value("type", "") != "hello") {
      try {
        send_frame(fd, Json{{"type", "error"}, {"error", "expected hello"}});
      } catch (...) {
      }
      ::close(fd);
      return;
    }
    if (first.value("proto", -1) != kProtoVersion) {
      try {
        send_frame(fd, Json{{"type", "error"},
                            {"error", "unsupported proto"}});
      } catch (...) {
      }
      ::close(fd);
      return;
    }
    Json info = hello ? hello() : Json::object();
    Json hello_ok = {{"type", "hello_ok"}, {"proto", kProtoVersion}};
    for (auto it = info.begin(); it != info.end(); ++it) {
      hello_ok[it.key()] = it.value();
    }
    try {
      send_frame(fd, hello_ok);
    } catch (...) {
      ::close(fd);
      return;
    }

    while (stop == nullptr || !stop->load()) {
      Json msg;
      try {
        msg = read_frame(fd);
      } catch (const TimeoutError&) {
        continue;  // idle farm socket — keep waiting
      } catch (...) {
        break;
      }
      const std::string type = msg.value("type", "");
      if (type == "bye") {
        break;
      }
      if (type == "ping") {
        try {
          send_frame(fd, Json{{"type", "pong"},
                              {"t0", msg.value("t0", 0)},
                              {"t1", std::time(nullptr)}});
        } catch (...) {
          break;
        }
        continue;
      }
      Json reply;
      try {
        reply = handler(msg);
      } catch (const std::exception& e) {
        reply = Json{{"type", type == "job" ? "result" : "error"},
                     {"ok", false},
                     {"error", e.what()}};
      }
      try {
        send_frame(fd, reply);
      } catch (...) {
        break;
      }
    }
  } catch (...) {
  }
  ::close(fd);
}

}  // namespace

void serve_forever(JobHandler handler, ServerConfig config, HelloFn hello,
                   const std::atomic<bool>* stop) {
  const int server = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) {
    throw TransportError(std::string("socket: ") + std::strerror(errno));
  }
  int yes = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(config.port));
  if (inet_pton(AF_INET, config.host.c_str(), &addr.sin_addr) != 1) {
    // Allow "0.0.0.0"
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  }
  if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(server);
    throw TransportError(std::string("bind: ") + std::strerror(errno));
  }
  if (listen(server, config.backlog) < 0) {
    ::close(server);
    throw TransportError(std::string("listen: ") + std::strerror(errno));
  }

  timeval accept_tv{};
  accept_tv.tv_sec = 1;
  setsockopt(server, SOL_SOCKET, SO_RCVTIMEO, &accept_tv, sizeof(accept_tv));

  std::atomic<int> active{0};
  std::vector<std::thread> threads;
  // Detach workers; track count for admission.
  try {
    while (stop == nullptr || !stop->load()) {
      sockaddr_in peer{};
      socklen_t peer_len = sizeof(peer);
      const int fd =
          ::accept(server, reinterpret_cast<sockaddr*>(&peer), &peer_len);
      if (fd < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
            errno == ETIMEDOUT) {
          continue;
        }
        break;
      }
      if (active.load() >= config.max_connections) {
        ::close(fd);
        continue;
      }
      active.fetch_add(1);
      std::thread([&, fd]() {
        handle_connection(fd, handler, hello, config.idle_timeout_s, stop);
        active.fetch_sub(1);
      }).detach();
    }
  } catch (...) {
    ::close(server);
    throw;
  }
  ::close(server);
}

}  // namespace wave_dispatch
