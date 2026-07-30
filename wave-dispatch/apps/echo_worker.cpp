#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "wave_dispatch/wave_dispatch.hpp"

int main(int argc, char** argv) {
  using namespace wave_dispatch;
  int port = kDefaultPort;
  int workers = 4;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
      port = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
      workers = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::cout << "Usage: wave_echo_worker [--port N] [--workers N]\n";
      return 0;
    }
  }

  std::atomic<bool> stop{false};
  ServerConfig cfg;
  cfg.port = port;
  cfg.idle_timeout_s = 120.0;

  auto hello = [workers]() {
    return Json{{"workers", workers},
                {"max_workers", workers * 2},
                {"default_workers", workers},
                {"hostname", "echo-worker"},
                {"device", "cpu"},
                {"job_kinds", Json::array({"play", "echo"})},
                {"capabilities", Json::array({"echo_v1"})}};
  };

  auto handler = [](const Json& msg) -> Json {
    const std::string type = msg.value("type", "");
    if (type == "job") {
      Json job = msg.value("job", Json::object());
      Json result = {{"ok", true},
                     {"echo", job},
                     {"kind", msg.value("kind", "play")}};
      return Json{{"type", "result"}, {"ok", true}, {"result", result}};
    }
    if (type == "health") {
      return Json{{"type", "health_ok"}, {"ok", true}};
    }
    return Json{{"type", "error"}, {"error", "unsupported message"}};
  };

  std::cout << "wave_echo_worker listening on 0.0.0.0:" << port
            << " workers=" << workers << "\n";
  serve_forever(handler, cfg, hello, &stop);
  return 0;
}
