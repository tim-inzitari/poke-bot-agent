#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "rl_runtime/rl_runtime.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_rl_runtime, m) {
  m.doc() = "rl_runtime: shared-memory leaf IPC rings";
  m.attr("__version__") = RL_RUNTIME_VERSION_STRING;
  py::register_exception<rl_runtime::Error>(m, "RlRuntimeError");
  py::register_exception<rl_runtime::TimeoutError>(m, "TimeoutError");
  py::register_exception<rl_runtime::CancelledError>(m, "CancelledError");

  py::class_<rl_runtime::RingConfig>(m, "RingConfig")
      .def(py::init<>())
      .def_readwrite("name", &rl_runtime::RingConfig::name)
      .def_readwrite("slot_count", &rl_runtime::RingConfig::slot_count)
      .def_readwrite("request_slots", &rl_runtime::RingConfig::request_slots)
      .def_readwrite("max_payload", &rl_runtime::RingConfig::max_payload)
      .def_readwrite("generation", &rl_runtime::RingConfig::generation);

  py::class_<rl_runtime::Request>(m, "Request")
      .def_readonly("slot", &rl_runtime::Request::slot)
      .def_readonly("rid", &rl_runtime::Request::rid)
      .def_property_readonly("payload", [](const rl_runtime::Request& r) {
        return py::bytes(reinterpret_cast<const char*>(r.payload.data()), r.payload.size());
      });

  py::class_<rl_runtime::ShmRing>(m, "ShmRing")
      .def_static(
          "create",
          [](const rl_runtime::RingConfig& cfg) {
            return rl_runtime::ShmRing::create(cfg);
          })
      .def_static("open", &rl_runtime::ShmRing::open)
      .def("generation", &rl_runtime::ShmRing::generation)
      .def("set_alive", &rl_runtime::ShmRing::set_alive)
      .def("alive", &rl_runtime::ShmRing::alive)
      .def("unlink", &rl_runtime::ShmRing::unlink)
      .def(
          "submit",
          [](rl_runtime::ShmRing& self, std::uint32_t slot, py::bytes data,
             double timeout) {
            std::string s = data;
            return self.submit(slot, reinterpret_cast<const std::uint8_t*>(s.data()),
                               s.size(), timeout);
          },
          py::arg("slot"), py::arg("data"), py::arg("timeout") = 30.0)
      .def(
          "wait",
          [](rl_runtime::ShmRing& self, std::uint32_t slot, std::uint64_t rid,
             double timeout) {
            auto v = self.wait(slot, rid, timeout);
            return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
          },
          py::arg("slot"), py::arg("rid"), py::arg("timeout") = 30.0)
      .def("try_pop", &rl_runtime::ShmRing::try_pop)
      .def("pop", &rl_runtime::ShmRing::pop, py::arg("timeout_s") = 1.0)
      .def("coalesce", &rl_runtime::ShmRing::coalesce, py::arg("max_batch"),
           py::arg("first_timeout_s"), py::arg("coalesce_s"))
      .def(
          "respond",
          [](rl_runtime::ShmRing& self, std::uint32_t slot, std::uint64_t rid,
             py::bytes data) {
            std::string s = data;
            self.respond(slot, rid, reinterpret_cast<const std::uint8_t*>(s.data()),
                         s.size());
          },
          py::arg("slot"), py::arg("rid"), py::arg("data"));
}
