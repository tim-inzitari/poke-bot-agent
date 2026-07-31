#include <iostream>
#include <vector>

#include "wave_dispatch/frame.hpp"

static int expect(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL binary: " << msg << "\n";
    return 1;
  }
  return 0;
}

int test_binary() {
  using namespace wave_dispatch;
  int f = 0;

  Message msg;
  msg.meta = {{"type", "job"}, {"kind", "echo"}, {"job", {{"id", 9}}}};
  msg.blob.assign(1024, 0xCD);
  auto wire = encode_message(msg);
  f += expect(wire.size() > 4 + 4 + 4, "wire size");
  // body should start with WDB1
  f += expect(wire[4] == 'W' && wire[5] == 'D', "magic");

  Message back = decode_message(wire.data(), wire.size());
  f += expect(back.meta["job"]["id"] == 9, "meta id");
  f += expect(back.blob.size() == 1024, "blob size");
  f += expect(back.blob[0] == 0xCD, "blob byte");

  // JSON-only still works
  auto jwire = encode_frame(Json{{"type", "ping"}});
  auto jmsg = decode_message(jwire.data(), jwire.size());
  f += expect(jmsg.meta["type"] == "ping", "json path");
  f += expect(jmsg.blob.empty(), "no blob");

  if (f == 0) std::cout << "OK test_binary\n";
  return f;
}
