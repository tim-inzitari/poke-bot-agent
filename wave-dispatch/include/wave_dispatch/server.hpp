#pragma once

#include <atomic>
#include <functional>
#include <string>

#include "wave_dispatch/frame.hpp"

namespace wave_dispatch {

using JobHandler = std::function<Json(const Json& msg)>;
using MessageHandler = std::function<Message(const Message& msg)>;
using HelloFn = std::function<Json()>;

struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = kDefaultPort;
  int backlog = 512;
  int max_connections = 4096;
  double idle_timeout_s = 60.0;
  /** Asio io_context threads (accept + socket reactor). */
  int io_threads = 0;  // 0 → hardware_concurrency
  bool tcp_nodelay = true;
  bool reuse_port = true;
};

/**
 * Asio multi-threaded accept loop. One in-flight job per socket.
 * Prefers MessageHandler (JSON+blob); JobHandler wrapped if only JSON given.
 */
void serve_forever(MessageHandler handler, ServerConfig config = {},
                   HelloFn hello = {},
                   const std::atomic<bool>* stop = nullptr);

void serve_forever(JobHandler handler, ServerConfig config = {},
                   HelloFn hello = {},
                   const std::atomic<bool>* stop = nullptr);

}  // namespace wave_dispatch
