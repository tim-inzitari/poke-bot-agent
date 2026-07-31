#include <iostream>
#include <vector>

#include "wave_dispatch/frame.hpp"

static int expect(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL frame: " << msg << "\n";
    return 1;
  }
  return 0;
}

int test_frame() {
  using namespace wave_dispatch;
  int f = 0;
  Json payload = {{"type", "hello"}, {"proto", kProtoVersion}};
  auto bytes = encode_frame(payload);
  f += expect(bytes.size() >= 4, "encoded size");
  auto decoded = decode_frame(bytes.data(), bytes.size());
  f += expect(decoded["type"] == "hello", "type");
  f += expect(decoded["proto"] == kProtoVersion, "proto");

  bool threw = false;
  try {
    Json arr = Json::array({1, 2, 3});
    encode_frame(arr);
  } catch (const ProtocolError&) {
    threw = true;
  }
  f += expect(threw, "reject non-object");
  if (f == 0) {
    std::cout << "OK test_frame\n";
  }
  return f;
}
