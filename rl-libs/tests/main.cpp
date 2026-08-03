#include <cstdio>
#include <exception>

int test_digest();
int test_ordered_writer();
int test_blob_pack();
int test_shm_ring();
int test_proc_pool();

int main() {
  try {
    int failed = 0;
    failed += test_digest();
    failed += test_ordered_writer();
    failed += test_blob_pack();
    failed += test_shm_ring();
    failed += test_proc_pool();
    if (failed) {
      std::fprintf(stderr, "%d test group(s) failed\n", failed);
      return 1;
    }
    std::printf("all rl-libs tests passed\n");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "uncaught: %s\n", ex.what());
    return 1;
  }
}
