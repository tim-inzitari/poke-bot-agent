#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "wave_dispatch/wave_dispatch.hpp"

int main(int argc, char** argv) {
  using namespace wave_dispatch;
  int port = kDefaultPort;
  int workers = 4;
  int io_threads = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      port = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
      workers = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--io-threads") == 0 && i + 1 < argc) {
      io_threads = std::atoi(argv[++i]);
    }
  }

  std::atomic<bool> stop{false};
  ServerConfig cfg;
  cfg.port = port;
  cfg.auto_uds = true;
  cfg.idle_timeout_s = 120.0;
  cfg.io_threads = io_threads;
  cfg.max_connections = 4096;
  cfg.backlog = 512;

  auto hello = [workers]() {
    return Json{{"workers", workers},
                {"max_workers", workers * 2},
                {"default_workers", workers},
                {"hostname", "echo-worker"},
                {"device", "cpu"},
                {"job_kinds", Json::array({"play", "echo"})},
                {"capabilities", Json::array({"echo_v1", "binary_v1"})}};
  };

  MessageHandler handler = [](const Message& msg) -> Message {
    const std::string type = msg.meta.value("type", "");
    if (type == "job") {
      Message out;
      out.meta = {{"type", "result"},
                  {"ok", true},
                  {"result",
                   {{"ok", true},
                    {"echo", msg.meta.value("job", Json::object())},
                    {"kind", msg.meta.value("kind", "play")},
                    {"blob_bytes", static_cast<int>(msg.blob.size())}}}};
      // Echo blob back — binary fast path
      out.blob = msg.blob;
      return out;
    }
    if (type == "health") {
      return Message{Json{{"type", "health_ok"}, {"ok", true}}, {}};
    }
    return Message{Json{{"type", "error"}, {"error", "unsupported message"}}, {}};
  };

  std::cout << "wave_echo_worker asio port=" << port << " workers=" << workers
            << "\n";
  serve_forever(handler, cfg, hello, &stop);
  return 0;
}
