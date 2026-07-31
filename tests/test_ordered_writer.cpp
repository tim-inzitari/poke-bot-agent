#include "rl_io/ordered_writer.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

int test_ordered_writer() {
  const auto dir = fs::temp_directory_path() / "rl_io_ow_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const auto partial = (dir / "replay.jsonl").string();
  {
    rl_io::OrderedWriter::Config cfg;
    cfg.replay_partial = partial;
    cfg.expected_jobs = 4;
    cfg.fsync_batch = 2;
    rl_io::OrderedWriter w(cfg);
    w.submit(2, std::string(R"({"episode_id":"2"})"), {{"i", 2}});
    w.submit(0, std::string(R"({"episode_id":"0"})"), {{"i", 0}});
    w.submit(1, std::string(R"({"episode_id":"1"})"), {{"i", 1}});
    w.submit(3, std::nullopt, {{"i", 3}});
    auto tel = w.close();
    if (tel.at("next_index") != 4 || tel.at("written_records") != 3) {
      std::fprintf(stderr, "ordered writer telemetry bad: %s\n", tel.dump().c_str());
      return 1;
    }
    w.finalize((dir / "final.jsonl").string());
  }
  std::ifstream in(dir / "final.jsonl");
  std::string line;
  int lines = 0;
  while (std::getline(in, line)) ++lines;
  if (lines != 3) {
    std::fprintf(stderr, "expected 3 replay lines, got %d\n", lines);
    return 1;
  }

  // Crash-resume: abort after partial contiguous commit, reopen, finish.
  const auto partial2 = (dir / "resume.jsonl").string();
  {
    rl_io::OrderedWriter::Config cfg;
    cfg.replay_partial = partial2;
    cfg.expected_jobs = 3;
    cfg.fsync_batch = 1;
    rl_io::OrderedWriter w(cfg);
    w.submit(0, std::string(R"({"episode_id":"0"})"), {{"i", 0}});
    // Leave 1 missing; submit 2 out of order then abort.
    w.submit(2, std::string(R"({"episode_id":"2"})"), {{"i", 2}});
    auto tel = w.abort("test");
    if (tel.at("next_index") != 1) {
      std::fprintf(stderr, "abort should keep durable next_index=1, got %s\n",
                   tel.dump().c_str());
      return 1;
    }
  }
  {
    rl_io::OrderedWriter::Config cfg;
    cfg.replay_partial = partial2;
    cfg.expected_jobs = 3;
    cfg.fsync_batch = 1;
    rl_io::OrderedWriter w(cfg);
    if (w.resume_index() != 1) {
      std::fprintf(stderr, "resume_index expected 1 got %llu\n",
                   (unsigned long long)w.resume_index());
      return 1;
    }
    if (w.submit(0, std::string(R"({"episode_id":"0"})"), {{"i", 0}})) {
      std::fprintf(stderr, "re-submit of committed index should return false\n");
      return 1;
    }
    w.submit(1, std::string(R"({"episode_id":"1"})"), {{"i", 1}});
    w.submit(2, std::string(R"({"episode_id":"2"})"), {{"i", 2}});
    auto tel = w.close();
    if (tel.at("next_index") != 3 || tel.at("written_records") != 3) {
      std::fprintf(stderr, "resume close bad: %s\n", tel.dump().c_str());
      return 1;
    }
  }

  fs::remove_all(dir);
  return 0;
}
