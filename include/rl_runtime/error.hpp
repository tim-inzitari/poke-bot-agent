#pragma once

#include <stdexcept>

namespace rl_runtime {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class TimeoutError : public Error {
 public:
  using Error::Error;
};

class CancelledError : public Error {
 public:
  using Error::Error;
};

}  // namespace rl_runtime
