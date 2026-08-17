#include <archive.h>
#include <archive_entry.h>
#include <simdjson.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Task {
  std::string name;
  std::string body;
};

struct Queue {
  explicit Queue(std::size_t capacity) : capacity(capacity) {}
  std::mutex mutex;
  std::condition_variable readable;
  std::condition_variable writable;
  std::deque<Task> tasks;
  std::size_t capacity;
  bool closed = false;
};

bool take(Queue& queue, Task& task) {
  std::unique_lock lock(queue.mutex);
  queue.readable.wait(lock, [&] { return queue.closed || !queue.tasks.empty(); });
  if (queue.tasks.empty()) return false;
  task = std::move(queue.tasks.front());
  queue.tasks.pop_front();
  queue.writable.notify_one();
  return true;
}

void put(Queue& queue, Task task) {
  std::unique_lock lock(queue.mutex);
  queue.writable.wait(lock, [&] { return queue.tasks.size() < queue.capacity; });
  queue.tasks.push_back(std::move(task));
  queue.readable.notify_one();
}

std::string json_string(std::string value) {
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '\\' || ch == '"') out.push_back('\\');
    out.push_back(ch);
  }
  out.push_back('"');
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: replay_ingest_probe ARCHIVE.zip [workers]\n";
    return 2;
  }
  const std::string archive_path = argv[1];
  const unsigned workers = argc == 3 ? std::max(1, std::atoi(argv[2]))
                                     : std::max(1u, std::thread::hardware_concurrency());
  Queue queue(std::max<std::size_t>(8, workers * 2));
  std::atomic<std::uint64_t> episodes{0}, rejected{0}, steps{0}, bytes{0};
  std::mutex first_error_mutex;
  std::string first_error;
  const auto started = std::chrono::steady_clock::now();

  std::vector<std::thread> pool;
  pool.reserve(workers);
  for (unsigned index = 0; index < workers; ++index) {
    pool.emplace_back([&] {
      simdjson::dom::parser parser;
      Task task;
      while (take(queue, task)) {
        bytes += task.body.size();
        try {
          simdjson::padded_string padded(task.body);
          simdjson::dom::element root = parser.parse(padded);
          auto module = root["module_version"].get_string();
          auto schema = root["schema_version"].get_int64();
          auto episode_steps = root["steps"].get_array();
          if (module.error() || schema.error() || episode_steps.error()) {
            throw simdjson::simdjson_error(simdjson::INCORRECT_TYPE);
          }
          std::uint64_t count = 0;
          for ([[maybe_unused]] auto step : episode_steps.value()) ++count;
          steps += count;
          episodes += 1;
        } catch (const std::exception& error) {
          rejected += 1;
          std::lock_guard lock(first_error_mutex);
          if (first_error.empty()) first_error = task.name + ": " + error.what();
        }
      }
    });
  }

  archive* input = archive_read_new();
  archive_read_support_filter_all(input);
  archive_read_support_format_zip(input);
  if (archive_read_open_filename(input, archive_path.c_str(), 1 << 20) != ARCHIVE_OK) {
    std::cerr << archive_error_string(input) << "\n";
    archive_read_free(input);
    return 1;
  }

  archive_entry* entry = nullptr;
  while (archive_read_next_header(input, &entry) == ARCHIVE_OK) {
    const char* raw_name = archive_entry_pathname(entry);
    std::string name = raw_name ? raw_name : "";
    if (archive_entry_filetype(entry) != AE_IFREG ||
        name.size() < 5 || name.substr(name.size() - 5) != ".json") {
      archive_read_data_skip(input);
      continue;
    }
    std::string body;
    const auto declared = archive_entry_size(entry);
    if (declared > 0) body.reserve(static_cast<std::size_t>(declared));
    char buffer[1 << 20];
    while (true) {
      const auto count = archive_read_data(input, buffer, sizeof(buffer));
      if (count == 0) break;
      if (count < 0) {
        std::cerr << archive_error_string(input) << "\n";
        archive_read_close(input);
        archive_read_free(input);
        return 1;
      }
      body.append(buffer, static_cast<std::size_t>(count));
    }
    put(queue, Task{std::move(name), std::move(body)});
  }
  archive_read_close(input);
  archive_read_free(input);
  {
    std::lock_guard lock(queue.mutex);
    queue.closed = true;
  }
  queue.readable.notify_all();
  for (auto& worker : pool) worker.join();

  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  const auto total = episodes.load() + rejected.load();
  std::cout << "{\"schema\":\"pokebot-native-replay-ingest/v1\","
            << "\"archive\":" << json_string(archive_path) << ','
            << "\"workers\":" << workers << ','
            << "\"episodes\":" << episodes << ','
            << "\"rejected\":" << rejected << ','
            << "\"steps\":" << steps << ','
            << "\"uncompressed_bytes\":" << bytes << ','
            << "\"elapsed_seconds\":" << elapsed << ','
            << "\"episodes_per_second\":" << (elapsed > 0 ? total / elapsed : 0.0) << ','
            << "\"first_error\":" << json_string(first_error) << "}\n";
  return rejected == 0 ? 0 : 3;
}
