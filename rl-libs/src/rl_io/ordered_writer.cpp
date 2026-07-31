#include "rl_io/ordered_writer.hpp"

#include "rl_io/digest.hpp"
#include "rl_io/error.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <utility>

namespace fs = std::filesystem;

namespace rl_io {
namespace {

double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void fsync_path(const std::string& path) {
  FILE* fp = std::fopen(path.c_str(), "r+b");
  if (!fp) throw IoError("fsync open failed: " + path);
  if (std::fflush(fp) != 0) {
    std::fclose(fp);
    throw IoError("fsync fflush failed: " + path);
  }
  if (::fsync(::fileno(fp)) != 0) {
    std::fclose(fp);
    throw IoError("fsync failed: " + path);
  }
  std::fclose(fp);
}

std::uint64_t file_size(std::fstream& f) {
  f.clear();
  f.seekg(0, std::ios::end);
  return static_cast<std::uint64_t>(f.tellg());
}

}  // namespace

OrderedWriter::OrderedWriter(Config cfg) : cfg_(std::move(cfg)) {
  if (cfg_.expected_jobs == 0) throw ProtocolError("expected_jobs must be > 0");
  if (cfg_.queue_depth == 0) cfg_.queue_depth = 1;
  if (cfg_.fsync_batch == 0) cfg_.fsync_batch = 1;
  fs::path partial(cfg_.replay_partial);
  fs::create_directories(partial.parent_path());
  journal_path_ = partial.string() + ".journal";
  state_path_ = partial.string() + ".writer.json";
  state_ = load_or_create_state_();
  next_index_ = state_.at("next_index").get<std::uint64_t>();
  written_records_ = state_.at("written_records").get<std::uint64_t>();
  replay_.open(cfg_.replay_partial, std::ios::in | std::ios::out | std::ios::binary);
  journal_.open(journal_path_, std::ios::in | std::ios::out | std::ios::binary);
  if (!replay_ || !journal_) throw IoError("cannot open writer streams");
  started_ = now_s();
  thread_ = std::thread([this] { run_(); });
}

OrderedWriter::~OrderedWriter() {
  try {
    if (!closed_) {
      abort("destructor");
    }
  } catch (...) {
  }
  if (thread_.joinable()) thread_.join();
}

std::uint64_t OrderedWriter::resume_index() const {
  std::lock_guard<std::mutex> lock(mu_);
  return next_index_;
}

std::uint64_t OrderedWriter::written_records() const {
  std::lock_guard<std::mutex> lock(mu_);
  return written_records_;
}

Json OrderedWriter::load_or_create_state_() {
  if (fs::is_regular_file(state_path_)) {
    std::ifstream in(state_path_);
    Json state;
    in >> state;
    if (state.value("expected_jobs", std::uint64_t(0)) != cfg_.expected_jobs) {
      throw ProtocolError("writer expected-job count changed on resume");
    }
    for (const auto& [path, key] : std::vector<std::pair<std::string, const char*>>{
             {cfg_.replay_partial, "replay_offset"},
             {journal_path_, "journal_offset"}}) {
      if (!fs::is_regular_file(path)) {
        throw ProtocolError("missing writer recovery file " + path);
      }
      const auto off = state.at(key).get<std::uint64_t>();
      fs::resize_file(path, off);
    }
    return state;
  }
  if (fs::exists(cfg_.replay_partial) || fs::exists(journal_path_)) {
    throw ProtocolError("writer partial exists without an atomic recovery checkpoint");
  }
  { std::ofstream(cfg_.replay_partial, std::ios::binary).close(); }
  { std::ofstream(journal_path_, std::ios::binary).close(); }
  Json state = {
      {"schema", 1},
      {"expected_jobs", cfg_.expected_jobs},
      {"next_index", 0},
      {"written_records", 0},
      {"replay_offset", 0},
      {"journal_offset", 0},
  };
  save_state_(state);
  return state;
}

void OrderedWriter::save_state_(const Json& state) {
  const std::string tmp = state_path_ + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream out(tmp);
    out << state.dump() << '\n';
    if (!out) throw IoError("state write failed");
  }
  fsync_path(tmp);
  fs::rename(tmp, state_path_);
}

bool OrderedWriter::submit(std::uint64_t job_index,
                           const std::optional<std::string>& record_bytes,
                           const Json& result_metadata,
                           double timeout_s) {
  std::unique_lock<std::mutex> lock(mu_);
  if (closed_) throw ProtocolError("submit after writer close");
  if (error_) std::rethrow_exception(error_);
  if (job_index >= cfg_.expected_jobs) {
    throw ProtocolError("job index out of range");
  }
  if (job_index < next_index_) return false;
  if (submitted_.count(job_index)) throw ProtocolError("duplicate job submission");
  submitted_.insert(job_index);
  const double started = now_s();
  if (!cv_space_.wait_for(lock, std::chrono::duration<double>(timeout_s), [&] {
        return queue_.size() < cfg_.queue_depth || closed_ || error_;
      })) {
    submitted_.erase(job_index);
    throw IoError("ordered writer submit timeout");
  }
  if (closed_) throw ProtocolError("submit after writer close");
  if (error_) std::rethrow_exception(error_);
  const double waited = now_s() - started;
  queue_wait_total_ += waited;
  queue_wait_max_ = std::max(queue_wait_max_, waited);
  Item item;
  item.index = job_index;
  item.record = record_bytes;
  item.metadata = result_metadata;
  queue_.push_back(std::move(item));
  max_queue_depth_ = std::max(max_queue_depth_, queue_.size());
  cv_item_.notify_one();
  return true;
}

void OrderedWriter::run_() {
  try {
    bool stopping = false;
    bool aborting = false;
    while (!stopping) {
      Item item;
      Cmd cmd = Cmd::None;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_item_.wait(lock, [&] { return !queue_.empty() || cmd_ != Cmd::None; });
        if (cmd_ != Cmd::None && queue_.empty()) {
          cmd = cmd_;
          stopping = true;
          aborting = (cmd == Cmd::Abort);
        } else {
          item = std::move(queue_.front());
          queue_.pop_front();
          cv_space_.notify_one();
          if (pending_.count(item.index)) {
            throw ProtocolError("duplicate pending job");
          }
          pending_[item.index] = item;
        }
      }
      drain_ready_(stopping);
      if (stopping && !aborting) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!pending_.empty()) {
          throw ProtocolError("writer closed with ordering gaps");
        }
      }
      if (stopping && aborting) {
        std::lock_guard<std::mutex> lock(mu_);
        pending_.clear();
      }
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(mu_);
    error_ = std::current_exception();
    cv_space_.notify_all();
  }
}

void OrderedWriter::drain_ready_(bool force) {
  while (true) {
    std::vector<Item> batch;
    {
      std::lock_guard<std::mutex> lock(mu_);
      while (pending_.count(next_index_)) {
        batch.push_back(std::move(pending_[next_index_]));
        pending_.erase(next_index_);
        ++next_index_;
        if (batch.size() >= cfg_.fsync_batch) break;
      }
      if (!batch.empty() && batch.size() < cfg_.fsync_batch && !force &&
          !queue_.empty()) {
        next_index_ -= batch.size();
        for (auto& item : batch) pending_[item.index] = item;
        batch.clear();
      }
    }
    if (batch.empty()) break;
    commit_(batch);
    if (!force && batch.size() < cfg_.fsync_batch) break;
  }
}

void OrderedWriter::commit_(const std::vector<Item>& batch) {
  replay_.clear();
  journal_.clear();
  replay_.seekp(0, std::ios::end);
  journal_.seekp(0, std::ios::end);
  std::uint64_t added = 0;
  for (const auto& item : batch) {
    if (item.record.has_value()) {
      const std::string line = *item.record + "\n";
      replay_.write(line.data(), static_cast<std::streamsize>(line.size()));
      ++added;
    }
    Json journal = {
        {"job_index", item.index},
        {"record_written", item.record.has_value()},
        {"record_sha256",
         item.record ? Json(sha256_hex(*item.record)) : Json(nullptr)},
        {"result", item.metadata},
    };
    const std::string jline = journal.dump() + "\n";
    journal_.write(jline.data(), static_cast<std::streamsize>(jline.size()));
  }
  if (!replay_ || !journal_) throw IoError("commit write failed");
  replay_.flush();
  journal_.flush();
  fsync_path(cfg_.replay_partial);
  fsync_path(journal_path_);
  std::lock_guard<std::mutex> lock(mu_);
  written_records_ += added;
  state_ = {
      {"schema", 1},
      {"expected_jobs", cfg_.expected_jobs},
      {"next_index", next_index_},
      {"written_records", written_records_},
      {"replay_offset", file_size(replay_)},
      {"journal_offset", file_size(journal_)},
  };
  save_state_(state_);
}

Json OrderedWriter::finish_(Cmd cmd, const std::string& reason) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) return telemetry();
    closed_ = true;
    aborted_ = (cmd == Cmd::Abort);
    abort_reason_ = reason;
    cmd_ = cmd;
  }
  cv_item_.notify_one();
  if (thread_.joinable()) thread_.join();
  replay_.close();
  journal_.close();
  if (error_) {
    try {
      std::rethrow_exception(error_);
    } catch (const std::exception& ex) {
      throw ProtocolError(std::string("writer failed: ") + ex.what());
    }
  }
  if (aborted_) {
    std::lock_guard<std::mutex> lock(mu_);
    state_["aborted_at"] = std::time(nullptr);
    state_["abort_reason"] = abort_reason_;
    save_state_(state_);
  } else if (next_index_ != cfg_.expected_jobs) {
    throw ProtocolError("writer committed incomplete job set");
  }
  return telemetry();
}

Json OrderedWriter::close() { return finish_(Cmd::Close, ""); }

Json OrderedWriter::abort(const std::string& reason) {
  return finish_(Cmd::Abort, reason);
}

Json OrderedWriter::telemetry() const {
  std::lock_guard<std::mutex> lock(mu_);
  const double elapsed = std::max(1e-9, now_s() - started_);
  return {
      {"next_index", next_index_},
      {"expected_jobs", cfg_.expected_jobs},
      {"written_records", written_records_},
      {"aborted", aborted_},
      {"abort_reason", aborted_ ? Json(abort_reason_) : Json(nullptr)},
      {"queue_depth", queue_.size()},
      {"max_queue_depth", max_queue_depth_},
      {"queue_put_wait_ms_total", queue_wait_total_ * 1000.0},
      {"queue_put_wait_ms_max", queue_wait_max_ * 1000.0},
      {"jobs_per_s", next_index_ / elapsed},
      {"records_per_s", written_records_ / elapsed},
  };
}

void OrderedWriter::finalize(const std::string& final_path) {
  if (!closed_) throw ProtocolError("close writer before finalize");
  if (aborted_ || next_index_ != cfg_.expected_jobs) {
    throw ProtocolError("cannot finalize an aborted/incomplete writer");
  }
  if (fs::exists(final_path)) {
    throw ProtocolError("refusing to overwrite replay " + final_path);
  }
  fs::rename(cfg_.replay_partial, final_path);
}

std::string OrderedWriter::quarantine(const std::string& suffix) {
  if (!closed_) throw ProtocolError("close writer before quarantine");
  std::string first;
  for (const auto& source :
       {cfg_.replay_partial, journal_path_, state_path_}) {
    if (!fs::exists(source)) continue;
    const std::string dest = source + suffix;
    fs::rename(source, dest);
    if (first.empty()) first = dest;
  }
  if (first.empty()) throw ProtocolError("no writer artifacts remain to quarantine");
  return first;
}

}  // namespace rl_io
