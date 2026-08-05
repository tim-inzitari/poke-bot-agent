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
  /** If set, also (or only) listen on this Unix domain socket path. */
  std::string unix_path;
  /** If true and unix_path empty, auto unix:/tmp/wave_dispatch_<port>.sock */
  bool auto_uds = true;
  int backlog = 512;
  int max_connections = 4096;
  double idle_timeout_s = 60.0;
  int io_threads = 0;
  bool tcp_nodelay = true;
  bool reuse_port = true;
  /** Prefer Linux io_uring when compiled with support. */
  bool use_io_uring = true;
};

void serve_forever(MessageHandler handler, ServerConfig config = {},
                   HelloFn hello = {},
                   const std::atomic<bool>* stop = nullptr);

void serve_forever(JobHandler handler, ServerConfig config = {},
                   HelloFn hello = {},
                   const std::atomic<bool>* stop = nullptr);

}  // namespace wave_dispatch
