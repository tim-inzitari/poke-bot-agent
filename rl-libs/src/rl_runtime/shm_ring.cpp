#include "rl_runtime/shm_ring.hpp"

#include "rl_runtime/error.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <thread>

namespace rl_runtime {
namespace {

constexpr char kMagic[4] = {'R', 'L', 'S', 'M'};
constexpr std::uint32_t kVersion = 1;

struct alignas(64) SlotHeader {
  std::atomic<std::uint64_t> rid{0};
  std::atomic<std::uint32_t> length{0};
  std::atomic<std::uint32_t> ready{0};
};

struct alignas(64) ReqSlot {
  std::atomic<std::uint32_t> state{0};  // 0 empty, 1 filled
  std::uint32_t slot = 0;
  std::uint64_t rid = 0;
  std::uint32_t length = 0;
  std::uint32_t pad = 0;
};

struct Header {
  char magic[4];
  std::uint32_t version;
  std::uint32_t slot_count;
  std::uint32_t request_slots;
  std::uint32_t max_payload;
  std::uint64_t generation;
  std::atomic<std::uint32_t> req_head;
  std::atomic<std::uint32_t> req_tail;
  std::atomic<std::uint64_t> next_rid;
  std::atomic<std::uint32_t> alive;
  std::uint32_t pad;
};

std::size_t compute_map_size(const RingConfig& cfg) {
  const std::size_t header = sizeof(Header);
  const std::size_t req_meta = sizeof(ReqSlot) * cfg.request_slots;
  const std::size_t req_payload = std::size_t(cfg.max_payload) * cfg.request_slots;
  const std::size_t resp_meta = sizeof(SlotHeader) * cfg.slot_count;
  const std::size_t resp_payload = std::size_t(cfg.max_payload) * cfg.slot_count;
  return header + req_meta + req_payload + resp_meta + resp_payload;
}

double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

struct ShmLayout {
  Header* header = nullptr;
  ReqSlot* reqs = nullptr;
  std::uint8_t* req_payload = nullptr;
  SlotHeader* resps = nullptr;
  std::uint8_t* resp_payload = nullptr;
};

static ShmLayout layout_from(void* map, const RingConfig& cfg) {
  auto* base = static_cast<std::uint8_t*>(map);
  ShmLayout L;
  L.header = reinterpret_cast<Header*>(base);
  std::size_t off = sizeof(Header);
  L.reqs = reinterpret_cast<ReqSlot*>(base + off);
  off += sizeof(ReqSlot) * cfg.request_slots;
  L.req_payload = base + off;
  off += std::size_t(cfg.max_payload) * cfg.request_slots;
  L.resps = reinterpret_cast<SlotHeader*>(base + off);
  off += sizeof(SlotHeader) * cfg.slot_count;
  L.resp_payload = base + off;
  return L;
}

ShmRing ShmRing::create(const RingConfig& cfg) {
  if (cfg.name.empty() || cfg.slot_count == 0 || cfg.request_slots == 0 ||
      cfg.max_payload == 0) {
    throw Error("invalid ring config");
  }
  ShmRing ring;
  ring.cfg_ = cfg;
  ring.owner_ = true;
  ring.map_size_ = compute_map_size(cfg);
  ::shm_unlink(cfg.name.c_str());
  ring.fd_ = ::shm_open(cfg.name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0600);
  if (ring.fd_ < 0) throw Error("shm_open create failed: " + cfg.name);
  if (::ftruncate(ring.fd_, static_cast<off_t>(ring.map_size_)) != 0) {
    throw Error("ftruncate failed");
  }
  ring.map_memory_();
  auto L = layout_from(ring.mapped_, cfg);
  std::memset(ring.mapped_, 0, ring.map_size_);
  std::memcpy(L.header->magic, kMagic, 4);
  L.header->version = kVersion;
  L.header->slot_count = cfg.slot_count;
  L.header->request_slots = cfg.request_slots;
  L.header->max_payload = cfg.max_payload;
  L.header->generation = cfg.generation;
  L.header->req_head.store(0);
  L.header->req_tail.store(0);
  L.header->next_rid.store(1);
  L.header->alive.store(1);
  ring.next_rid_ = &L.header->next_rid;
  return ring;
}

ShmRing ShmRing::open(const std::string& name) {
  ShmRing ring;
  ring.cfg_.name = name;
  ring.fd_ = ::shm_open(name.c_str(), O_RDWR, 0600);
  if (ring.fd_ < 0) throw Error("shm_open failed: " + name);
  struct stat st {};
  if (::fstat(ring.fd_, &st) != 0) throw Error("fstat failed");
  ring.map_size_ = static_cast<std::size_t>(st.st_size);
  ring.map_memory_();
  auto* header = static_cast<Header*>(ring.mapped_);
  if (std::memcmp(header->magic, kMagic, 4) != 0) throw Error("bad shm magic");
  if (header->version != kVersion) throw Error("bad shm version");
  ring.cfg_.slot_count = header->slot_count;
  ring.cfg_.request_slots = header->request_slots;
  ring.cfg_.max_payload = header->max_payload;
  ring.cfg_.generation = header->generation;
  ring.next_rid_ = &header->next_rid;
  return ring;
}

void ShmRing::map_memory_() {
  mapped_ = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mapped_ == MAP_FAILED) throw Error("mmap failed");
}

ShmRing::~ShmRing() { close_(); }

ShmRing::ShmRing(ShmRing&& other) noexcept { *this = std::move(other); }

ShmRing& ShmRing::operator=(ShmRing&& other) noexcept {
  if (this != &other) {
    close_();
    cfg_ = other.cfg_;
    fd_ = other.fd_;
    mapped_ = other.mapped_;
    map_size_ = other.map_size_;
    owner_ = other.owner_;
    next_rid_ = other.next_rid_;
    other.fd_ = -1;
    other.mapped_ = nullptr;
    other.map_size_ = 0;
    other.owner_ = false;
    other.next_rid_ = nullptr;
  }
  return *this;
}

void ShmRing::close_() {
  if (mapped_ && mapped_ != MAP_FAILED) ::munmap(mapped_, map_size_);
  mapped_ = nullptr;
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
}

void ShmRing::unlink() {
  if (!cfg_.name.empty()) ::shm_unlink(cfg_.name.c_str());
}

std::uint64_t ShmRing::generation() const {
  return layout_from(mapped_, cfg_).header->generation;
}

void ShmRing::set_alive(bool alive) {
  layout_from(mapped_, cfg_).header->alive.store(alive ? 1u : 0u);
}

bool ShmRing::alive() const {
  return layout_from(mapped_, cfg_).header->alive.load() != 0;
}

std::uint64_t ShmRing::submit(std::uint32_t slot,
                              const std::uint8_t* data,
                              std::size_t n,
                              double timeout_s) {
  if (slot >= cfg_.slot_count) throw Error("slot out of range");
  if (n > cfg_.max_payload) throw Error("payload too large");
  auto L = layout_from(mapped_, cfg_);
  if (!L.header->alive.load()) throw CancelledError("ring not alive");
  const std::uint64_t rid = L.header->next_rid.fetch_add(1);
  const double deadline = now_s() + timeout_s;
  while (true) {
    const auto head = L.header->req_head.load(std::memory_order_acquire);
    const auto tail = L.header->req_tail.load(std::memory_order_acquire);
    if (head - tail < cfg_.request_slots) {
      const auto idx = head % cfg_.request_slots;
      auto& rs = L.reqs[idx];
      std::uint32_t expected = 0;
      if (rs.state.compare_exchange_strong(expected, 1u)) {
        rs.slot = slot;
        rs.rid = rid;
        rs.length = static_cast<std::uint32_t>(n);
        if (n) {
          std::memcpy(L.req_payload + std::size_t(idx) * cfg_.max_payload, data, n);
        }
        L.header->req_head.fetch_add(1, std::memory_order_release);
        return rid;
      }
    }
    if (now_s() >= deadline) throw TimeoutError("submit timeout");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

std::vector<std::uint8_t> ShmRing::wait(std::uint32_t slot,
                                        std::uint64_t rid,
                                        double timeout_s) {
  if (slot >= cfg_.slot_count) throw Error("slot out of range");
  auto L = layout_from(mapped_, cfg_);
  auto& resp = L.resps[slot];
  const double deadline = now_s() + timeout_s;
  while (true) {
    if (!L.header->alive.load()) throw CancelledError("ring not alive");
    if (resp.ready.load(std::memory_order_acquire) &&
        resp.rid.load(std::memory_order_acquire) == rid) {
      const auto len = resp.length.load(std::memory_order_acquire);
      std::vector<std::uint8_t> out(len);
      if (len) {
        std::memcpy(out.data(),
                    L.resp_payload + std::size_t(slot) * cfg_.max_payload, len);
      }
      resp.ready.store(0, std::memory_order_release);
      return out;
    }
    if (now_s() >= deadline) throw TimeoutError("wait timeout");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

std::optional<Request> ShmRing::try_pop() {
  auto L = layout_from(mapped_, cfg_);
  const auto tail = L.header->req_tail.load(std::memory_order_acquire);
  const auto head = L.header->req_head.load(std::memory_order_acquire);
  if (tail >= head) return std::nullopt;
  const auto idx = tail % cfg_.request_slots;
  auto& rs = L.reqs[idx];
  if (rs.state.load(std::memory_order_acquire) != 1u) return std::nullopt;
  Request req;
  req.slot = rs.slot;
  req.rid = rs.rid;
  req.payload.resize(rs.length);
  if (rs.length) {
    std::memcpy(req.payload.data(),
                L.req_payload + std::size_t(idx) * cfg_.max_payload, rs.length);
  }
  rs.state.store(0, std::memory_order_release);
  L.header->req_tail.fetch_add(1, std::memory_order_release);
  return req;
}

std::optional<Request> ShmRing::pop(double timeout_s) {
  const double deadline = now_s() + timeout_s;
  while (true) {
    auto req = try_pop();
    if (req) return req;
    if (now_s() >= deadline) return std::nullopt;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

std::vector<Request> ShmRing::coalesce(std::size_t max_batch,
                                       double first_timeout_s,
                                       double coalesce_s) {
  std::vector<Request> out;
  out.reserve(max_batch);
  auto first = pop(first_timeout_s);
  if (!first) return out;
  out.push_back(std::move(*first));
  const double deadline = now_s() + coalesce_s;
  while (out.size() < max_batch) {
    auto next = try_pop();
    if (!next) {
      if (now_s() >= deadline) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      continue;
    }
    out.push_back(std::move(*next));
  }
  return out;
}

void ShmRing::respond(std::uint32_t slot,
                      std::uint64_t rid,
                      const std::uint8_t* data,
                      std::size_t n) {
  if (slot >= cfg_.slot_count) throw Error("slot out of range");
  if (n > cfg_.max_payload) throw Error("response too large");
  auto L = layout_from(mapped_, cfg_);
  auto& resp = L.resps[slot];
  // Wait until client consumed previous response.
  const double deadline = now_s() + 30.0;
  while (resp.ready.load(std::memory_order_acquire) != 0) {
    if (now_s() >= deadline) throw TimeoutError("respond backpressure timeout");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  if (n) {
    std::memcpy(L.resp_payload + std::size_t(slot) * cfg_.max_payload, data, n);
  }
  resp.length.store(static_cast<std::uint32_t>(n), std::memory_order_release);
  resp.rid.store(rid, std::memory_order_release);
  resp.ready.store(1, std::memory_order_release);
}

}  // namespace rl_runtime
