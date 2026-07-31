#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "wave_dispatch/wave_dispatch.hpp"

int main(int argc, char** argv) {
  using namespace wave_dispatch;
  std::string endpoint = "127.0.0.1:8765";
  int jobs = 32;
  int local_workers = 2;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--endpoint") == 0 && i + 1 < argc) {
      endpoint = argv[++i];
    } else if (std::strcmp(argv[i], "--jobs") == 0 && i + 1 < argc) {
      jobs = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--local") == 0 && i + 1 < argc) {
      local_workers = std::atoi(argv[++i]);
    }
  }

  const auto spec = parse_endpoint(endpoint);
  JobClient client(spec.host, spec.port, 30.0);
  const auto info = client.connect();
  std::cout << "connected " << info.endpoint << " workers=" << info.workers
            << "\n";

  SchedulerConfig scfg;
  scfg.tick_s = 1.0;
  scfg.min_gps_window_s = 0.5;
  scfg.demand_settle_s = 1.0;
  scfg.remote_defaults[endpoint] = std::max(1, info.default_workers > 0
                                                   ? info.default_workers
                                                   : info.workers);
  scfg.remote_maxima[endpoint] =
      std::max(scfg.remote_defaults[endpoint],
               info.max_workers > 0 ? info.max_workers : info.workers * 2);
  MidWaveScheduler sched(scfg);

  std::vector<Json> wave;
  wave.reserve(jobs);
  for (int i = 0; i < jobs; ++i) {
    wave.push_back(Json{{"id", i}, {"payload", "kaggle-episode"}});
  }

  auto local = [](const Json& job) {
    return Json{{"ok", true}, {"source", "local"}, {"echo", job}};
  };

  std::vector<JobClient*> remotes{&client};
  CollectConfig ccfg;
  ccfg.local_workers = local_workers;
  ccfg.remote_chunk = 8;
  ccfg.kind = "echo";

  const auto t0 = std::chrono::steady_clock::now();
  int n = 0;
  n = run_scheduled_wave(wave, local, remotes, sched, ccfg, [&](const Json&) {
    // count via return value
  });
  const auto dt = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  std::cout << "completed=" << n << " elapsed_s=" << dt
            << " gps=" << (n / std::max(dt, 1e-6)) << "\n";
  auto dec = sched.decision();
  std::cout << "scheduler reason=" << dec.reason
            << " local_share=" << dec.local_share
            << " remote_share=" << dec.remote_share << "\n";
  client.close();
  return n == jobs ? 0 : 1;
}
