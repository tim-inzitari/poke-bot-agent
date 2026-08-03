#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rl_io/rl_io.hpp"

namespace py = pybind11;

namespace {

rl_io::Json py_to_json(const py::object& obj) {
  if (obj.is_none()) return nullptr;
  return rl_io::Json::parse(py::module_::import("json").attr("dumps")(obj).cast<std::string>());
}

py::object json_to_py(const rl_io::Json& j) {
  if (j.is_null()) return py::none();
  return py::module_::import("json").attr("loads")(j.dump());
}

}  // namespace

PYBIND11_MODULE(_rl_io, m) {
  m.doc() = "rl_io: crash-safe ordered writer + blob packs";
  m.attr("__version__") = RL_IO_VERSION_STRING;
  py::register_exception<rl_io::Error>(m, "RlIoError");

  m.def("sha256_hex", [](py::bytes data) {
    std::string s = data;
    return rl_io::sha256_hex(s);
  });
  m.def("sha256_digest", [](py::bytes data) {
    std::string s = data;
    return rl_io::sha256_digest(s);
  });
  m.def("sha256_file", &rl_io::sha256_file, py::arg("path"), py::arg("chunk") = 1 << 20);

  py::class_<rl_io::OrderedWriter>(m, "OrderedWriter")
      .def(py::init([](const std::string& path, std::uint64_t expected_jobs,
                       std::size_t queue_depth, std::size_t fsync_batch) {
             rl_io::OrderedWriter::Config cfg;
             cfg.replay_partial = path;
             cfg.expected_jobs = expected_jobs;
             cfg.queue_depth = queue_depth;
             cfg.fsync_batch = fsync_batch;
             return std::make_unique<rl_io::OrderedWriter>(cfg);
           }),
           py::arg("replay_partial"), py::arg("expected_jobs"),
           py::arg("queue_depth") = 64, py::arg("fsync_batch") = 8)
      .def_property_readonly("resume_index", &rl_io::OrderedWriter::resume_index)
      .def_property_readonly("written_records", &rl_io::OrderedWriter::written_records)
      .def(
          "submit",
          [](rl_io::OrderedWriter& self, std::uint64_t index, py::object record,
             py::object metadata, double timeout) {
            std::optional<std::string> rec;
            if (!record.is_none()) {
              if (py::isinstance<py::bytes>(record) || py::isinstance<py::str>(record)) {
                rec = record.cast<std::string>();
              } else {
                throw rl_io::ProtocolError("record must be str/bytes/None");
              }
            }
            return self.submit(index, rec, py_to_json(metadata), timeout);
          },
          py::arg("job_index"), py::arg("record"), py::arg("result_metadata"),
          py::arg("timeout") = 30.0)
      .def("close", [](rl_io::OrderedWriter& self) { return json_to_py(self.close()); })
      .def("abort",
           [](rl_io::OrderedWriter& self, const std::string& reason) {
             return json_to_py(self.abort(reason));
           })
      .def("telemetry",
           [](const rl_io::OrderedWriter& self) { return json_to_py(self.telemetry()); })
      .def("finalize", &rl_io::OrderedWriter::finalize)
      .def("quarantine", &rl_io::OrderedWriter::quarantine);

  py::class_<rl_io::BlobPackWriter>(m, "BlobPackWriter")
      .def(py::init<>())
      .def("set_manifest",
           [](rl_io::BlobPackWriter& self, py::object manifest) {
             self.set_manifest(py_to_json(manifest));
           })
      .def(
          "add",
          [](rl_io::BlobPackWriter& self, const std::string& name, py::bytes data) {
            std::string s = data;
            self.add(name, std::string_view(s));
          })
      .def("commit", &rl_io::BlobPackWriter::commit);

  py::class_<rl_io::BlobPackReader>(m, "BlobPackReader")
      .def(py::init<const std::string&, bool>(), py::arg("path"), py::arg("verify") = true)
      .def_property_readonly("manifest",
                             [](const rl_io::BlobPackReader& self) {
                               return json_to_py(self.manifest());
                             })
      .def("names", &rl_io::BlobPackReader::names)
      .def("contains", &rl_io::BlobPackReader::contains)
      .def("get",
           [](const rl_io::BlobPackReader& self, const std::string& name) {
             auto v = self.view(name);
             return py::bytes(v.data(), v.size());
           })
      .def("blob_sha256", &rl_io::BlobPackReader::blob_sha256);
}
