#pragma once

#include <atomic>
#include <functional>
#include <string>

#include "wave_dispatch/frame.hpp"

namespace wave_dispatch {

using JobHandler = std::function<Json(const Json& msg)>;
using HelloFn = std::function<Json()>;

struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = kDefaultPort;
  int backlog = 64;
  int max_connections = 128;
  double idle_timeout_s = 60.0;
};

/**
 * Accept TCP connections; one in-flight job per socket.
 * Concurrent sockets = concurrent work. Idle timeouts are retried so farm
 * sockets survive gaps between waves.
 */
void serve_forever(JobHandler handler, ServerConfig config = {},
                   HelloFn hello = {},
                   const std::atomic<bool>* stop = nullptr);

}  // namespace wave_dispatch
