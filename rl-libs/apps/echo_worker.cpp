#include <cstdint>
#include <iostream>
#include <vector>

// Length-prefixed echo worker for proc_pool demos/tests.
static bool read_all(std::streambuf* in, char* p, std::size_t n) {
  while (n) {
    const auto r = in->sgetn(p, static_cast<std::streamsize>(n));
    if (r <= 0) return false;
    p += r;
    n -= static_cast<std::size_t>(r);
  }
  return true;
}

int main() {
  auto* in = std::cin.rdbuf();
  auto* out = std::cout.rdbuf();
  while (true) {
    char hdr[4];
    if (!read_all(in, hdr, 4)) return 0;
    const std::uint32_t len =
        (std::uint32_t(static_cast<unsigned char>(hdr[0])) << 24) |
        (std::uint32_t(static_cast<unsigned char>(hdr[1])) << 16) |
        (std::uint32_t(static_cast<unsigned char>(hdr[2])) << 8) |
        std::uint32_t(static_cast<unsigned char>(hdr[3]));
    if (len == 0) return 0;
    std::vector<char> buf(len);
    if (!read_all(in, buf.data(), len)) return 1;
    if (out->sputn(hdr, 4) != 4) return 1;
    if (out->sputn(buf.data(), static_cast<std::streamsize>(len)) !=
        static_cast<std::streamsize>(len)) {
      return 1;
    }
    out->pubsync();
  }
}
