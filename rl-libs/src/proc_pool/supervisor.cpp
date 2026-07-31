#include "proc_pool/supervisor.hpp"

#include "proc_pool/error.hpp"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace proc_pool {
namespace {

bool write_all(int fd, const void* data, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  while (n) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w;
    n -= static_cast<std::size_t>(w);
  }
  return true;
}

bool read_all(int fd, void* data, std::size_t n) {
  auto* p = static_cast<std::uint8_t*>(data);
  while (n) {
    const ssize_t r = ::read(fd, p, n);
    if (r == 0) return false;
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += r;
    n -= static_cast<std::size_t>(r);
  }
  return true;
}

bool write_frame(int fd, const std::uint8_t* data, std::size_t n) {
  const std::uint32_t len = static_cast<std::uint32_t>(n);
  const std::uint8_t hdr[4] = {
      static_cast<std::uint8_t>((len >> 24) & 0xff),
      static_cast<std::uint8_t>((len >> 16) & 0xff),
      static_cast<std::uint8_t>((len >> 8) & 0xff),
      static_cast<std::uint8_t>(len & 0xff),
  };
  return write_all(fd, hdr, 4) && (n == 0 || write_all(fd, data, n));
}

bool read_frame(int fd, std::vector<std::uint8_t>& out) {
  std::uint8_t hdr[4];
  if (!read_all(fd, hdr, 4)) return false;
  const std::uint32_t len = (std::uint32_t(hdr[0]) << 24) | (std::uint32_t(hdr[1]) << 16) |
                            (std::uint32_t(hdr[2]) << 8) | std::uint32_t(hdr[3]);
  out.resize(len);
  if (len == 0) return true;
  return read_all(fd, out.data(), len);
}

double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

struct Supervisor::Worker {
  int slot = 0;
  pid_t pid = -1;
  int to_child = -1;    // parent write
  int from_child = -1;  // parent read
  std::uint32_t tasks_done = 0;
  bool busy = false;
  std::uint64_t inflight_id = 0;
  std::thread io_thread;
};

Supervisor::Supervisor(WorkerSpec spec) : spec_(std::move(spec)) {
  if (spec_.argv.empty()) throw Error("worker argv required");
  if (spec_.num_workers == 0) throw Error("num_workers must be > 0");
  ensure_workers_();
  monitor_thread_ = std::thread([this] { monitor_(); });
}

Supervisor::~Supervisor() {
  request_stop("destructor");
  join();
}

void Supervisor::ensure_workers_() {
  std::lock_guard<std::mutex> lock(mu_);
  workers_.resize(spec_.num_workers);
  for (std::uint32_t i = 0; i < spec_.num_workers; ++i) {
    if (!workers_[i]) workers_[i] = std::make_unique<Worker>();
    if (workers_[i]->pid > 0) continue;
    int in_pipe[2];
    int out_pipe[2];
    if (::pipe(in_pipe) != 0 || ::pipe(out_pipe) != 0) throw Error("pipe failed");
    const pid_t pid = ::fork();
    if (pid < 0) throw Error("fork failed");
    if (pid == 0) {
      ::dup2(in_pipe[0], STDIN_FILENO);
      ::dup2(out_pipe[1], STDOUT_FILENO);
      ::close(in_pipe[0]);
      ::close(in_pipe[1]);
      ::close(out_pipe[0]);
      ::close(out_pipe[1]);
      std::vector<char*> args;
      for (auto& s : spec_.argv) args.push_back(const_cast<char*>(s.c_str()));
      args.push_back(nullptr);
      ::execvp(args[0], args.data());
      std::_Exit(127);
    }
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    workers_[i]->slot = static_cast<int>(i);
    workers_[i]->pid = pid;
    workers_[i]->to_child = in_pipe[1];
    workers_[i]->from_child = out_pipe[0];
    workers_[i]->tasks_done = 0;
    workers_[i]->busy = false;
  }
}

void Supervisor::recycle_(Worker& w) {
  kill_worker_(w);
}

void Supervisor::kill_worker_(Worker& w) {
  if (w.to_child >= 0) {
    write_frame(w.to_child, nullptr, 0);
    ::close(w.to_child);
    w.to_child = -1;
  }
  if (w.from_child >= 0) {
    ::close(w.from_child);
    w.from_child = -1;
  }
  if (w.pid > 0) {
    int status = 0;
    if (::waitpid(w.pid, &status, WNOHANG) == 0) {
      ::kill(w.pid, SIGTERM);
      ::waitpid(w.pid, &status, 0);
    }
    w.pid = -1;
  }
  w.busy = false;
  w.inflight_id = 0;
}

std::uint64_t Supervisor::submit(const std::uint8_t* data, std::size_t n) {
  if (stop_.load()) throw Error("supervisor stopped: " + stop_reason_);
  const std::uint64_t id = next_task_.fetch_add(1);
  const double deadline = now_s() + 30.0;
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (auto& wp : workers_) {
        auto& w = *wp;
        if (w.pid > 0 && !w.busy && w.to_child >= 0) {
          if (!write_frame(w.to_child, data, n)) {
            kill_worker_(w);
            continue;
          }
          w.busy = true;
          w.inflight_id = id;
          const int from = w.from_child;
          const int slot = w.slot;
          std::thread([this, from, slot, id] {
            std::vector<std::uint8_t> payload;
            TaskResult tr;
            tr.task_id = id;
            tr.worker_slot = slot;
            if (!read_frame(from, payload)) {
              tr.ok = false;
              tr.error = "worker read failed";
            } else {
              tr.ok = true;
              tr.payload = std::move(payload);
            }
            std::lock_guard<std::mutex> lock2(mu_);
            if (slot >= 0 && static_cast<std::size_t>(slot) < workers_.size() &&
                workers_[slot] && workers_[slot]->inflight_id == id) {
              workers_[slot]->busy = false;
              workers_[slot]->tasks_done += 1;
              if (spec_.recycle_tasks > 0 &&
                  workers_[slot]->tasks_done >= spec_.recycle_tasks) {
                recycle_(*workers_[slot]);
              }
            }
            results_.push_back(std::move(tr));
          }).detach();
          return id;
        }
      }
    }
    if (stop_.load()) throw Error("supervisor stopped");
    if (now_s() >= deadline) throw Error("submit timeout: no free worker");
    ensure_workers_();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

std::optional<TaskResult> Supervisor::try_get() {
  std::lock_guard<std::mutex> lock(mu_);
  if (results_.empty()) return std::nullopt;
  TaskResult tr = std::move(results_.front());
  results_.erase(results_.begin());
  return tr;
}

TaskResult Supervisor::get(double timeout_s) {
  const double deadline = now_s() + timeout_s;
  while (true) {
    auto tr = try_get();
    if (tr) return *tr;
    if (now_s() >= deadline) throw Error("get timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

void Supervisor::monitor_() {
  while (!stop_.load()) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (auto& wp : workers_) {
        if (!wp || wp->pid <= 0) continue;
        int status = 0;
        const pid_t r = ::waitpid(wp->pid, &status, WNOHANG);
        if (r == wp->pid) {
          if (wp->busy) {
            TaskResult tr;
            tr.task_id = wp->inflight_id;
            tr.worker_slot = wp->slot;
            tr.ok = false;
            tr.error = "worker exited";
            results_.push_back(std::move(tr));
          }
          wp->pid = -1;
          wp->busy = false;
          if (wp->to_child >= 0) {
            ::close(wp->to_child);
            wp->to_child = -1;
          }
          if (wp->from_child >= 0) {
            ::close(wp->from_child);
            wp->from_child = -1;
          }
        }
      }
    }
    if (!stop_.load()) ensure_workers_();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void Supervisor::request_stop(const std::string& reason) {
  stop_.store(true);
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_reason_.empty()) stop_reason_ = reason;
    for (auto& wp : workers_) {
      if (wp) kill_worker_(*wp);
    }
  }
}

void Supervisor::join() {
  if (monitor_thread_.joinable()) monitor_thread_.join();
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& wp : workers_) {
    if (wp) kill_worker_(*wp);
  }
}

bool Supervisor::healthy() const { return !stop_.load(); }

std::string Supervisor::stop_reason() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stop_reason_;
}

}  // namespace proc_pool
