#include <iostream>

int test_frame();
int test_scheduler();
int test_roundtrip();

int main() {
  int failed = 0;
  failed += test_frame();
  failed += test_scheduler();
  failed += test_roundtrip();
  if (failed == 0) {
    std::cout << "ALL_TESTS_PASSED\n";
  } else {
    std::cout << "FAILED=" << failed << "\n";
  }
  return failed == 0 ? 0 : 1;
}
