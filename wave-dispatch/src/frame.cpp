#include "wave_dispatch/frame.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstring>

namespace wave_dispatch {
namespace {

[[noreturn]] void throw_errno(const char* what) {
  throw TransportError(std::string(what) + ": " + std::strerror(errno));
}

}  // namespace

std::vector<std::uint8_t> encode_frame(const Json& payload) {
  if (!payload.is_object()) {
    throw ProtocolError("frame root must be a JSON object");
  }
  const std::string body = payload.dump();
  if (body.size() > kMaxFrameBytes) {
    throw ProtocolError("frame too large: " + std::to_string(body.size()) + " bytes");
  }
  std::vector<std::uint8_t> out(4 + body.size());
  const std::uint32_t be = htonl(static_cast<std::uint32_t>(body.size()));
  std::memcpy(out.data(), &be, 4);
  std::memcpy(out.data() + 4, body.data(), body.size());
  return out;
}

Json decode_frame(const std::uint8_t* data, std::size_t n) {
  if (n < 4) {
    throw ProtocolError("frame truncated header");
  }
  std::uint32_t be = 0;
  std::memcpy(&be, data, 4);
  const std::uint32_t length = ntohl(be);
  if (length > kMaxFrameBytes) {
    throw ProtocolError("frame length exceeds max");
  }
  if (n < 4u + length) {
    throw ProtocolError("frame truncated body");
  }
  try {
    auto payload = Json::parse(reinterpret_cast<const char*>(data + 4),
                               reinterpret_cast<const char*>(data + 4 + length));
    if (!payload.is_object()) {
      throw ProtocolError("frame root must be object");
    }
    return payload;
  } catch (const Json::parse_error& e) {
    throw ProtocolError(std::string("invalid JSON frame: ") + e.what());
  }
}

void recv_exact(int fd, void* buf, std::size_t n) {
  auto* p = static_cast<std::uint8_t*>(buf);
  std::size_t got = 0;
  while (got < n) {
    const ssize_t r = ::recv(fd, p + got, n - got, 0);
    if (r == 0) {
      throw TransportError("connection closed while reading frame");
    }
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
        throw TimeoutError("socket recv timed out");
      }
      throw_errno("recv");
    }
    got += static_cast<std::size_t>(r);
  }
}

Json read_frame(int fd) {
  std::uint8_t hdr[4];
  recv_exact(fd, hdr, 4);
  std::uint32_t be = 0;
  std::memcpy(&be, hdr, 4);
  const std::uint32_t length = ntohl(be);
  if (length > kMaxFrameBytes) {
    throw ProtocolError("frame length exceeds max");
  }
  std::vector<std::uint8_t> body(length);
  if (length > 0) {
    recv_exact(fd, body.data(), length);
  }
  try {
    auto payload = Json::parse(body.begin(), body.end());
    if (!payload.is_object()) {
      throw ProtocolError("frame root must be object");
    }
    return payload;
  } catch (const Json::parse_error& e) {
    throw ProtocolError(std::string("invalid JSON frame: ") + e.what());
  }
}

void send_frame(int fd, const Json& payload) {
  const auto bytes = encode_frame(payload);
  std::size_t sent = 0;
  while (sent < bytes.size()) {
    const ssize_t r = ::send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
        throw TimeoutError("socket send timed out");
      }
      throw_errno("send");
    }
    sent += static_cast<std::size_t>(r);
  }
}

}  // namespace wave_dispatch
