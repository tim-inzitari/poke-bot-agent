#include "wave_dispatch/scheduler.hpp"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace wave_dispatch {
namespace {

double clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

}  // namespace

HardwareSignals sample_hardware_signals() {
  HardwareSignals sig;
  try {
    double load1 = 0, load5 = 0, load15 = 0;
    std::ifstream avg("/proc/loadavg");
    if (avg) {
      avg >> load1 >> load5 >> load15;
      sig.load1 = load1;
      const int ncpu = std::max(1, static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)));
      sig.cpu_idle_pct = clamp(100.0 * (1.0 - (load1 / ncpu)), 0.0, 100.0);
    }
  } catch (...) {
  }
  try {
    std::ifstream mem("/proc/meminfo");
    std::string key;
    long value = 0;
    std::string unit;
    while (mem >> key >> value >> unit) {
      if (key == "MemAvailable:") {
        sig.mem_available_gb = value / (1024.0 * 1024.0);
      } else if (key == "MemTotal:") {
        sig.mem_total_gb = value / (1024.0 * 1024.0);
      }
    }
    sig.ok = sig.mem_total_gb > 0;
  } catch (...) {
  }
  return sig;
}

WaveGpsTracker::WaveGpsTracker(double min_window_s, double ema_alpha)
    : min_window_s_(min_window_s),
      ema_alpha_(ema_alpha),
      t0_(std::chrono::steady_clock::now()),
      win_t0_(t0_) {}

void WaveGpsTracker::note(const std::string& side, int n, int decisions) {
  n = std::max(0, n);
  decisions = std::max(0, decisions);
  if (n <= 0) {
    return;
  }
  done_total_ += n;
  decisions_total_ += decisions;
  if (side == "remote") {
    done_remote_ += n;
  } else {
    done_local_ += n;
  }
  win_done_ += n;
  maybe_roll_window();
}

void WaveGpsTracker::maybe_roll_window() const {
  const auto now = std::chrono::steady_clock::now();
  const double dt =
      std::chrono::duration<double>(now - win_t0_).count();
  if (dt < min_window_s_) {
    return;
  }
  const double inst = win_done_ / std::max(dt, 1e-6);
  if (!ema_inited_) {
    ema_gps_ = inst;
    ema_inited_ = true;
  } else {
    ema_gps_ = ema_alpha_ * inst + (1.0 - ema_alpha_) * ema_gps_;
  }
  win_t0_ = now;
  win_done_ = 0;
}

double WaveGpsTracker::elapsed() const {
  return std::max(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_)
          .count(),
      1e-6);
}

double WaveGpsTracker::wave_gps() const {
  return static_cast<double>(done_total_) / elapsed();
}
double WaveGpsTracker::local_gps() const {
  return static_cast<double>(done_local_) / elapsed();
}
double WaveGpsTracker::remote_gps() const {
  return static_cast<double>(done_remote_) / elapsed();
}
double WaveGpsTracker::ema_gps() const {
  maybe_roll_window();
  return ema_inited_ ? ema_gps_ : wave_gps();
}

std::unordered_map<std::string, double> WaveGpsTracker::snapshot() const {
  maybe_roll_window();
  return {
      {"wave_gps", wave_gps()},
      {"local_gps", local_gps()},
      {"remote_gps", remote_gps()},
      {"ema_gps", ema_gps()},
      {"elapsed_s", elapsed()},
      {"done_total", static_cast<double>(done_total_)},
      {"done_local", static_cast<double>(done_local_)},
      {"done_remote", static_cast<double>(done_remote_)},
  };
}

MidWaveScheduler::MidWaveScheduler(SchedulerConfig config)
    : cfg_(std::move(config)),
      tracker_(cfg_.min_gps_window_s, cfg_.ema_alpha),
      last_tick_(std::chrono::steady_clock::now()) {
  cfg_.min_local_frac = clamp(cfg_.min_local_frac, 0.05, 0.95);
  cfg_.prefer_local_frac =
      clamp(cfg_.prefer_local_frac, cfg_.min_local_frac, 0.95);
  cfg_.max_remote_frac =
      clamp(cfg_.max_remote_frac, 0.05, 1.0 - cfg_.min_local_frac);
  cfg_.min_remote_frac = clamp(cfg_.min_remote_frac, 0.0, cfg_.max_remote_frac);
  cfg_.remote_chunk = std::max(8, cfg_.remote_chunk);

  bind_endpoints(cfg_.remote_defaults, cfg_.remote_maxima);
  decision_ = build_decision_unlocked("init");
}

void MidWaveScheduler::bind_endpoints(
    const std::unordered_map<std::string, int>& defaults,
    const std::unordered_map<std::string, int>& maxima) {
  std::lock_guard<std::mutex> lock(mu_);
  cfg_.remote_defaults = defaults;
  cfg_.remote_maxima = maxima;
  remote_demand_.clear();
  for (const auto& [ep, dflt] : defaults) {
    const int mx = maxima.count(ep) ? maxima.at(ep) : dflt;
    remote_demand_[ep] = std::max(1, std::min(dflt, mx));
  }
  for (const auto& [ep, mx] : maxima) {
    if (!remote_demand_.count(ep)) {
      const int dflt = defaults.count(ep) ? defaults.at(ep) : mx;
      remote_demand_[ep] = std::max(1, std::min(dflt, mx));
    }
  }
  decision_.remote_demand = remote_demand_;
}

void MidWaveScheduler::note_completed(const std::string& side, int n,
                                      int decisions) {
  std::lock_guard<std::mutex> lock(mu_);
  tracker_.note(side, n, decisions);
}

int MidWaveScheduler::demand_sum_unlocked() const {
  int s = 0;
  for (const auto& [_, n] : remote_demand_) {
    s += n;
  }
  return s;
}

void MidWaveScheduler::grow_demand_unlocked() {
  bool grew = false;
  const int pre = demand_sum_unlocked();
  for (auto& [ep, n] : remote_demand_) {
    const int mx = cfg_.remote_maxima.count(ep) ? cfg_.remote_maxima.at(ep) : n;
    if (grow_ceiling_ > 0 && demand_sum_unlocked() >= grow_ceiling_) {
      break;
    }
    if (n < mx) {
      const int step = std::max(1, (mx - n + 3) / 4);
      n = std::min(mx, n + step);
      grew = true;
    }
  }
  if (grew) {
    pending_ = "grow";
    probe_mono_ = std::chrono::steady_clock::now();
    probe_pre_demand_sum_ = pre;
    probe_demand_sum_ = demand_sum_unlocked();
    probe_wave_gps_ = tracker_.wave_gps();
  }
}

void MidWaveScheduler::shrink_demand_unlocked() {
  bool shrunk = false;
  for (auto& [ep, n] : remote_demand_) {
    const int dflt =
        cfg_.remote_defaults.count(ep) ? cfg_.remote_defaults.at(ep) : 1;
    if (n > dflt) {
      const int step = std::max(1, (n - dflt + 3) / 4);
      n = std::max(dflt, n - step);
      shrunk = true;
    }
  }
  if (shrunk) {
    pending_ = "shrink";
    probe_mono_ = std::chrono::steady_clock::now();
    probe_demand_sum_ = demand_sum_unlocked();
    probe_wave_gps_ = tracker_.wave_gps();
  }
}

SchedulerDecision MidWaveScheduler::build_decision_unlocked(
    const std::string& reason) const {
  SchedulerDecision d;
  d.local_share = cfg_.prefer_local_frac;
  d.remote_share = 1.0 - cfg_.prefer_local_frac;
  d.remote_share = clamp(d.remote_share, cfg_.min_remote_frac, cfg_.max_remote_frac);
  d.local_share = 1.0 - d.remote_share;
  if (d.local_share < cfg_.min_local_frac) {
    d.local_share = cfg_.min_local_frac;
    d.remote_share = 1.0 - d.local_share;
  }
  d.target_workers =
      std::max(cfg_.min_workers, std::min(cfg_.max_workers, cfg_.target_workers));
  d.remote_chunk = cfg_.remote_chunk;
  d.reason = reason;
  d.remote_demand = remote_demand_;
  d.metrics = tracker_.snapshot();
  return d;
}

std::optional<SchedulerDecision> MidWaveScheduler::maybe_tick(int remaining,
                                                              bool force) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto now = std::chrono::steady_clock::now();
  const double since =
      std::chrono::duration<double>(now - last_tick_).count();
  if (!force && since < cfg_.tick_s) {
    return std::nullopt;
  }
  last_tick_ = now;
  ++ticks_;

  std::string reason = "tick";
  const double gps = tracker_.wave_gps();
  const int dsum = demand_sum_unlocked();
  const int default_sum = [&]() {
    int s = 0;
    for (const auto& [ep, n] : cfg_.remote_defaults) {
      s += n;
    }
    return s;
  }();

  // Resolve pending probe
  if (!pending_.empty()) {
    const double settled =
        std::chrono::duration<double>(now - probe_mono_).count();
    if (settled >= cfg_.demand_settle_s) {
      if (pending_ == "grow") {
        if (gps + 1e-9 < probe_wave_gps_ * (1.0 - cfg_.demand_degrade_frac)) {
          // Hurt — roll back toward pre-grow and ratchet ceiling
          shrink_demand_unlocked();
          grow_ceiling_ = std::max(1, probe_pre_demand_sum_);
          grow_blocked_until_ =
              now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(cfg_.demand_grow_cooldown_s));
          reason = "grow_hurt_rollback";
        } else if (gps >= probe_wave_gps_ * (1.0 + cfg_.demand_improve_frac)) {
          reason = "grow_helped";
        } else {
          reason = "grow_neutral";
        }
      } else if (pending_ == "shrink") {
        reason = "shrink_settled";
      }
      pending_.clear();
    } else {
      reason = "probe_pending_" + pending_;
    }
  } else if (now >= grow_blocked_until_ && dsum <= default_sum &&
             gps > 0 && remaining > 0) {
    // Track best GPS near defaults; try grow when healthy
    if (dsum <= default_sum) {
      best_default_gps_ = std::max(best_default_gps_, gps);
    }
    bool below_max = false;
    for (const auto& [ep, n] : remote_demand_) {
      const int mx = cfg_.remote_maxima.count(ep) ? cfg_.remote_maxima.at(ep) : n;
      if (n < mx && (grow_ceiling_ <= 0 || dsum < grow_ceiling_)) {
        below_max = true;
        break;
      }
    }
    if (below_max && gps >= best_default_gps_ * 0.97) {
      grow_demand_unlocked();
      reason = "grow_attempt";
    }
  } else if (dsum > default_sum && best_default_gps_ > 0 &&
             gps < best_default_gps_ * (1.0 - cfg_.demand_degrade_frac)) {
    shrink_demand_unlocked();
    reason = "below_best_default_shrink";
  }

  // Soft share nudge from side GPS
  const double local_g = tracker_.local_gps();
  const double remote_g = tracker_.remote_gps();
  if (local_g + remote_g > 1e-6) {
    double remote_share = remote_g / (local_g + remote_g);
    remote_share = clamp(remote_share, cfg_.min_remote_frac, cfg_.max_remote_frac);
    decision_.local_share = std::max(cfg_.min_local_frac, 1.0 - remote_share);
    decision_.remote_share = 1.0 - decision_.local_share;
  }

  decision_ = build_decision_unlocked(reason);
  decision_.local_share = decision_.local_share;
  return decision_;
}

SchedulerDecision MidWaveScheduler::decision() const {
  std::lock_guard<std::mutex> lock(mu_);
  return decision_;
}

std::unordered_map<std::string, int> MidWaveScheduler::remote_demand() const {
  std::lock_guard<std::mutex> lock(mu_);
  return remote_demand_;
}

}  // namespace wave_dispatch
