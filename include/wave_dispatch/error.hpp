#pragma once

#include <stdexcept>
#include <string>

namespace wave_dispatch {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ProtocolError : public Error {
 public:
  using Error::Error;
};

class TransportError : public Error {
 public:
  using Error::Error;
};

class TimeoutError : public Error {
 public:
  explicit TimeoutError(const std::string& what) : Error(what) {}
};

}  // namespace wave_dispatch
