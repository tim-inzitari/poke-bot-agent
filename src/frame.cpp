#include "wave_dispatch/frame.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstring>

#include <simdjson.h>

namespace wave_dispatch {
namespace {

[[noreturn]] void throw_errno(const char* what) {
  throw TransportError(std::string(what) + ": " + std::strerror(errno));
}

void append_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  const std::uint32_t be = htonl(v);
  const auto* p = reinterpret_cast<const std::uint8_t*>(&be);
  out.insert(out.end(), p, p + 4);
}

std::uint32_t read_be32(const std::uint8_t* p) {
  std::uint32_t be = 0;
  std::memcpy(&be, p, 4);
  return ntohl(be);
}

Json simdjson_to_nlohmann(simdjson::dom::element el);

Json simdjson_to_nlohmann(simdjson::dom::element el) {
  using simdjson::dom::element_type;
  switch (el.type()) {
    case element_type::OBJECT: {
      Json j = Json::object();
      for (auto [k, v] : el.get_object()) {
        j[std::string(k)] = simdjson_to_nlohmann(v);
      }
      return j;
    }
    case element_type::ARRAY: {
      Json j = Json::array();
      for (auto v : el.get_array()) {
        j.push_back(simdjson_to_nlohmann(v));
      }
      return j;
    }
    case element_type::STRING:
      return std::string(el.get_string().value());
    case element_type::INT64:
      return el.get_int64().value();
    case element_type::UINT64:
      return el.get_uint64().value();
    case element_type::DOUBLE:
      return el.get_double().value();
    case element_type::BOOL:
      return el.get_bool().value();
    case element_type::NULL_VALUE:
      return nullptr;
    default:
      return nullptr;
  }
}

}  // namespace

Json parse_json_fast(std::string_view utf8) {
  thread_local simdjson::dom::parser parser;
  simdjson::padded_string padded(utf8.data(), utf8.size());
  simdjson::dom::element doc;
  const auto err = parser.parse(padded).get(doc);
  if (err) {
    // Fallback to nlohmann for edge cases
    try {
      return Json::parse(utf8);
    } catch (const Json::parse_error& e) {
      throw ProtocolError(std::string("invalid JSON: ") + e.what());
    }
  }
  Json out = simdjson_to_nlohmann(doc);
  if (!out.is_object() && !out.is_array() && !out.is_null() && !out.is_boolean() &&
      !out.is_number() && !out.is_string()) {
    throw ProtocolError("json parse produced unexpected type");
  }
  return out;
}

std::vector<std::uint8_t> encode_frame(const Json& payload) {
  if (!payload.is_object()) {
    throw ProtocolError("frame root must be a JSON object");
  }
  const std::string body = payload.dump();
  if (body.size() > kMaxFrameBytes) {
    throw ProtocolError("frame too large: " + std::to_string(body.size()));
  }
  std::vector<std::uint8_t> out;
  out.reserve(4 + body.size());
  append_be32(out, static_cast<std::uint32_t>(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::vector<std::uint8_t> encode_message(const Message& msg) {
  if (!msg.has_blob()) {
    return encode_frame(msg.meta);
  }
  if (!msg.meta.is_object()) {
    throw ProtocolError("binary frame meta must be a JSON object");
  }
  const std::string meta = msg.meta.dump();
  const std::size_t total = 4 + 4 + meta.size() + msg.blob.size();  // magic+meta_len+meta+blob
  if (total > kMaxFrameBytes) {
    throw ProtocolError("binary frame too large");
  }
  std::vector<std::uint8_t> out;
  out.reserve(4 + total);
  append_be32(out, static_cast<std::uint32_t>(total));
  // magic "WDB1" as little-endian u32 on wire as raw bytes W D B 1
  out.push_back('W');
  out.push_back('D');
  out.push_back('B');
  out.push_back('1');
  append_be32(out, static_cast<std::uint32_t>(meta.size()));
  out.insert(out.end(), meta.begin(), meta.end());
  out.insert(out.end(), msg.blob.begin(), msg.blob.end());
  return out;
}

Message decode_message(const std::uint8_t* data, std::size_t n) {
  if (n < 4) {
    throw ProtocolError("frame truncated header");
  }
  const std::uint32_t length = read_be32(data);
  if (length > kMaxFrameBytes) {
    throw ProtocolError("frame length exceeds max");
  }
  if (n < 4u + length) {
    throw ProtocolError("frame truncated body");
  }
  const std::uint8_t* body = data + 4;
  Message msg;
  if (length >= 4 && body[0] == 'W' && body[1] == 'D' && body[2] == 'B' &&
      body[3] == '1') {
    if (length < 8) {
      throw ProtocolError("binary frame truncated");
    }
    const std::uint32_t meta_len = read_be32(body + 4);
    if (8u + meta_len > length) {
      throw ProtocolError("binary meta length invalid");
    }
    msg.meta = parse_json_fast(
        std::string_view(reinterpret_cast<const char*>(body + 8), meta_len));
    if (!msg.meta.is_object()) {
      throw ProtocolError("binary meta must be object");
    }
    const std::size_t blob_off = 8u + meta_len;
    msg.blob.assign(body + blob_off, body + length);
    return msg;
  }
  // JSON path
  msg.meta = parse_json_fast(
      std::string_view(reinterpret_cast<const char*>(body), length));
  if (!msg.meta.is_object()) {
    throw ProtocolError("frame root must be object");
  }
  return msg;
}

Json decode_frame(const std::uint8_t* data, std::size_t n) {
  return decode_message(data, n).meta;
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
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
        throw TimeoutError("socket recv timed out");
      }
      throw_errno("recv");
    }
    got += static_cast<std::size_t>(r);
  }
}

Message read_message(int fd) {
  std::uint8_t hdr[4];
  recv_exact(fd, hdr, 4);
  const std::uint32_t length = read_be32(hdr);
  if (length > kMaxFrameBytes) {
    throw ProtocolError("frame length exceeds max");
  }
  std::vector<std::uint8_t> buf(4 + length);
  std::memcpy(buf.data(), hdr, 4);
  if (length) {
    recv_exact(fd, buf.data() + 4, length);
  }
  return decode_message(buf.data(), buf.size());
}

Json read_frame(int fd) { return read_message(fd).meta; }

void send_raw(int fd, const std::vector<std::uint8_t>& bytes) {
  std::size_t sent = 0;
  while (sent < bytes.size()) {
    const ssize_t r =
        ::send(fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
    if (r < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
        throw TimeoutError("socket send timed out");
      }
      throw_errno("send");
    }
    sent += static_cast<std::size_t>(r);
  }
}

void send_frame(int fd, const Json& payload) { send_raw(fd, encode_frame(payload)); }

void send_message(int fd, const Message& msg) {
  send_raw(fd, encode_message(msg));
}

}  // namespace wave_dispatch
