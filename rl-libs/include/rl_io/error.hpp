#pragma once

#include <stdexcept>
#include <string>

namespace rl_io {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class IoError : public Error {
 public:
  using Error::Error;
};

class ProtocolError : public Error {
 public:
  using Error::Error;
};

}  // namespace rl_io
