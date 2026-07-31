#include "rl_io/blob_pack.hpp"

#include <cstdio>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

int test_blob_pack() {
  const auto dir = fs::temp_directory_path() / "rl_io_pack_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const auto path = (dir / "t.rlpk").string();
  {
    rl_io::BlobPackWriter w;
    w.set_manifest({{"schema", "test"}, {"k", 1}});
    w.add("a", std::string("hello"));
    w.add("b", std::string("world!!!"));
    w.commit(path);
  }
  rl_io::BlobPackReader r(path, true);
  if (r.manifest().at("schema") != "test") return 1;
  if (r.view("a") != "hello") return 1;
  if (r.view("b") != "world!!!") return 1;
  fs::remove_all(dir);
  return 0;
}
