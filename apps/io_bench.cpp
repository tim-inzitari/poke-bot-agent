#include "rl_io/rl_io.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

int main() {
  namespace fs = std::filesystem;
  const auto dir = fs::temp_directory_path() / "rl_io_bench";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const auto partial = (dir / "replay.jsonl").string();
  constexpr int N = 2000;
  rl_io::OrderedWriter::Config cfg;
  cfg.replay_partial = partial;
  cfg.expected_jobs = N;
  cfg.fsync_batch = 16;
  cfg.queue_depth = 128;
  const auto t0 = std::chrono::steady_clock::now();
  {
    rl_io::OrderedWriter w(cfg);
    for (int i = N - 1; i >= 0; --i) {
      w.submit(static_cast<std::uint64_t>(i),
               std::string(R"({"episode_id":")") + std::to_string(i) + "\"}",
               {{"ok", true}});
    }
    w.close();
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double s = std::chrono::duration<double>(t1 - t0).count();
  std::printf("ordered_writer: %d jobs in %.3fs (%.0f jobs/s)\n", N, s, N / s);

  rl_io::BlobPackWriter pack;
  pack.set_manifest({{"schema", "bench"}, {"n", N}});
  std::string blob(1 << 16, 'x');
  for (int i = 0; i < 64; ++i) {
    pack.add("b" + std::to_string(i), blob);
  }
  const auto pack_path = (dir / "bench.rlpk").string();
  const auto p0 = std::chrono::steady_clock::now();
  pack.commit(pack_path);
  const auto p1 = std::chrono::steady_clock::now();
  rl_io::BlobPackReader reader(pack_path, true);
  const auto p2 = std::chrono::steady_clock::now();
  std::printf("blob_pack write: %.3fs  verify+mmap: %.3fs  blobs=%zu\n",
              std::chrono::duration<double>(p1 - p0).count(),
              std::chrono::duration<double>(p2 - p1).count(),
              reader.names().size());
  fs::remove_all(dir);
  return 0;
}
