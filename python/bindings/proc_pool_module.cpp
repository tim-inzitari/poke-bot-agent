#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "proc_pool/proc_pool.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_proc_pool, m) {
  m.doc() = "proc_pool: recyclable process supervisor";
  m.attr("__version__") = PROC_POOL_VERSION_STRING;
  py::register_exception<proc_pool::Error>(m, "ProcPoolError");

  py::class_<proc_pool::WorkerSpec>(m, "WorkerSpec")
      .def(py::init<>())
      .def_readwrite("argv", &proc_pool::WorkerSpec::argv)
      .def_readwrite("num_workers", &proc_pool::WorkerSpec::num_workers)
      .def_readwrite("recycle_tasks", &proc_pool::WorkerSpec::recycle_tasks)
      .def_readwrite("capacity_grace_s", &proc_pool::WorkerSpec::capacity_grace_s);

  py::class_<proc_pool::TaskResult>(m, "TaskResult")
      .def_readonly("task_id", &proc_pool::TaskResult::task_id)
      .def_readonly("worker_slot", &proc_pool::TaskResult::worker_slot)
      .def_readonly("ok", &proc_pool::TaskResult::ok)
      .def_readonly("error", &proc_pool::TaskResult::error)
      .def_property_readonly("payload", [](const proc_pool::TaskResult& t) {
        return py::bytes(reinterpret_cast<const char*>(t.payload.data()),
                         t.payload.size());
      });

  py::class_<proc_pool::Supervisor>(m, "Supervisor")
      .def(py::init<proc_pool::WorkerSpec>())
      .def(
          "submit",
          [](proc_pool::Supervisor& self, py::bytes data) {
            std::string s = data;
            return self.submit(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
          })
      .def("try_get", &proc_pool::Supervisor::try_get)
      .def("get", &proc_pool::Supervisor::get, py::arg("timeout_s") = 30.0)
      .def("request_stop", &proc_pool::Supervisor::request_stop,
           py::arg("reason") = "")
      .def("join", &proc_pool::Supervisor::join)
      .def("healthy", &proc_pool::Supervisor::healthy)
      .def("stop_reason", &proc_pool::Supervisor::stop_reason);
}
