#include "rl_io/digest.hpp"

#include <cstdio>
#include <string>

int test_digest() {
  const std::string empty = rl_io::sha256_hex("");
  if (empty !=
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
    std::fprintf(stderr, "digest empty mismatch: %s\n", empty.c_str());
    return 1;
  }
  const auto d = rl_io::sha256_digest("abc");
  if (d !=
      "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
    std::fprintf(stderr, "digest abc mismatch: %s\n", d.c_str());
    return 1;
  }
  return 0;
}
