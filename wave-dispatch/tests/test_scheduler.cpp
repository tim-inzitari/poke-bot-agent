#include <chrono>
#include <iostream>
#include <thread>

#include "wave_dispatch/scheduler.hpp"

static int expect(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL scheduler: " << msg << "\n";
    return 1;
  }
  return 0;
}

int test_scheduler() {
  using namespace wave_dispatch;
  int f = 0;

  WaveGpsTracker tr(0.05, 0.5);
  tr.note("local", 10, 10);
  tr.note("remote", 5, 5);
  f += expect(tr.wave_gps() > 0, "wave_gps");
  f += expect(tr.local_gps() > tr.remote_gps(), "local>remote gps");

  SchedulerConfig cfg;
  cfg.tick_s = 0.0;
  cfg.min_gps_window_s = 0.01;
  cfg.demand_settle_s = 0.05;
  cfg.demand_grow_cooldown_s = 0.0;
  cfg.remote_defaults = {{"a:1", 2}, {"b:1", 1}};
  cfg.remote_maxima = {{"a:1", 8}, {"b:1", 4}};
  MidWaveScheduler sched(cfg);
  auto d0 = sched.decision();
  f += expect(d0.remote_demand.at("a:1") == 2, "start at default a");
  f += expect(d0.remote_demand.at("b:1") == 1, "start at default b");

  for (int i = 0; i < 20; ++i) {
    sched.note_completed("local", 2, 2);
    sched.note_completed("remote", 2, 2);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  auto tick = sched.maybe_tick(100, true);
  f += expect(tick.has_value(), "forced tick");
  f += expect(tick->metrics.count("wave_gps") == 1, "metrics");

  if (f == 0) {
    std::cout << "OK test_scheduler\n";
  }
  return f;
}
