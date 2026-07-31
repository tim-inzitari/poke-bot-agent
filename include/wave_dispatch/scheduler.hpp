#pragma once

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "wave_dispatch/frame.hpp"

namespace wave_dispatch {

struct HardwareSignals {
  double cpu_idle_pct = 50.0;
  double load1 = 0.0;
  double mem_available_gb = 0.0;
  double mem_total_gb = 0.0;
  bool ok = false;
};

/** Best-effort host signals (never throws). */
HardwareSignals sample_hardware_signals();

struct SchedulerDecision {
  double local_share = 0.55;
  double remote_share = 0.45;
  int target_workers = 96;
  int remote_chunk = 128;
  std::string reason = "init";
  std::unordered_map<std::string, int> remote_demand;
  std::unordered_map<std::string, double> metrics;
};

struct SchedulerConfig {
  double min_local_frac = 0.40;
  double prefer_local_frac = 0.55;
  double max_remote_frac = 0.60;
  double min_remote_frac = 0.25;
  int target_workers = 96;
  int min_workers = 64;
  int max_workers = 160;
  int remote_chunk = 128;
  double tick_s = 15.0;
  double min_gps_window_s = 20.0;
  double ema_alpha = 0.35;
  double demand_settle_s = 20.0;
  double demand_degrade_frac = 0.08;
  double demand_improve_frac = 0.03;
  double demand_eff_collapse_frac = 0.55;
  double demand_grow_cooldown_s = 180.0;
  std::unordered_map<std::string, int> remote_defaults;
  std::unordered_map<std::string, int> remote_maxima;
};

class WaveGpsTracker {
 public:
  explicit WaveGpsTracker(double min_window_s = 20.0, double ema_alpha = 0.35);

  void note(const std::string& side, int n = 1, int decisions = 0);
  double elapsed() const;
  double wave_gps() const;
  double local_gps() const;
  double remote_gps() const;
  double ema_gps() const;
  std::unordered_map<std::string, double> snapshot() const;

 private:
  void maybe_roll_window() const;

  double min_window_s_;
  double ema_alpha_;
  std::chrono::steady_clock::time_point t0_;
  mutable std::chrono::steady_clock::time_point win_t0_;
  int done_total_ = 0;
  int done_local_ = 0;
  int done_remote_ = 0;
  int decisions_total_ = 0;
  mutable int win_done_ = 0;
  mutable double ema_gps_ = 0.0;
  mutable bool ema_inited_ = false;
};

/**
 * Mid-wave capacity controller: local/remote shares + per-endpoint demand.
 * Domain-agnostic — capacity maps are supplied by the caller.
 */
class MidWaveScheduler {
 public:
  explicit MidWaveScheduler(SchedulerConfig config);

  void bind_endpoints(const std::unordered_map<std::string, int>& defaults,
                      const std::unordered_map<std::string, int>& maxima);
  void note_completed(const std::string& side, int n = 1, int decisions = 0);
  std::optional<SchedulerDecision> maybe_tick(int remaining, bool force = false);
  SchedulerDecision decision() const;
  std::unordered_map<std::string, int> remote_demand() const;

  WaveGpsTracker& tracker() { return tracker_; }
  const WaveGpsTracker& tracker() const { return tracker_; }

 private:
  void grow_demand_unlocked();
  void shrink_demand_unlocked();
  int demand_sum_unlocked() const;
  SchedulerDecision build_decision_unlocked(const std::string& reason) const;

  SchedulerConfig cfg_;
  WaveGpsTracker tracker_;
  mutable std::mutex mu_;
  std::chrono::steady_clock::time_point last_tick_;
  SchedulerDecision decision_;
  std::unordered_map<std::string, int> remote_demand_;
  int ticks_ = 0;

  // Demand completion probe state
  std::string pending_;  // "grow" | "shrink" | ""
  std::chrono::steady_clock::time_point probe_mono_{};
  int probe_demand_sum_ = 0;
  int probe_pre_demand_sum_ = 0;
  double probe_wave_gps_ = 0.0;
  std::chrono::steady_clock::time_point grow_blocked_until_{};
  int grow_ceiling_ = 0;
  double best_default_gps_ = 0.0;
};

}  // namespace wave_dispatch
