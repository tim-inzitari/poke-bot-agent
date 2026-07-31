#pragma once

#include <stdexcept>

namespace proc_pool {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace proc_pool
