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
  fs::remove_all(dir);
  return 0;
}
