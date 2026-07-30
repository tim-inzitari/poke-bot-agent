#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

#include "wave_dispatch/wave_dispatch.hpp"

namespace py = pybind11;
using namespace wave_dispatch;

namespace {

Json py_to_json(const py::object& obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  py::object dumps = py::module_::import("json").attr("dumps");
  return Json::parse(dumps(obj).cast<std::string>());
}

py::object json_to_py(const Json& j) {
  if (j.is_null()) {
    return py::none();
  }
  py::object loads = py::module_::import("json").attr("loads");
  return loads(j.dump());
}

void register_exceptions(py::module_& m) {
  py::register_exception<Error>(m, "WaveDispatchError");
  py::register_exception<ProtocolError>(m, "ProtocolError");
  py::register_exception<TransportError>(m, "TransportError");
  py::register_exception<TimeoutError>(m, "TimeoutError");
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "wave_dispatch C++ bindings for multi-machine RL collect";
  m.attr("__version__") = WAVE_DISPATCH_VERSION_STRING;
  m.attr("PROTO_VERSION") = kProtoVersion;
  m.attr("DEFAULT_PORT") = kDefaultPort;
  m.attr("MAX_FRAME_BYTES") = kMaxFrameBytes;

  register_exceptions(m);

  m.def(
      "encode_frame",
      [](const py::object& payload) {
        auto bytes = encode_frame(py_to_json(payload));
        return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      },
      py::arg("payload"));

  m.def(
      "decode_frame",
      [](const py::bytes& data) {
        const std::string raw = data;
        return json_to_py(decode_frame(
            reinterpret_cast<const std::uint8_t*>(raw.data()), raw.size()));
      },
      py::arg("data"));

  py::class_<WorkerInfo>(m, "WorkerInfo")
      .def_readonly("endpoint", &WorkerInfo::endpoint)
      .def_readonly("workers", &WorkerInfo::workers)
      .def_readonly("max_workers", &WorkerInfo::max_workers)
      .def_readonly("default_workers", &WorkerInfo::default_workers)
      .def_readonly("hostname", &WorkerInfo::hostname)
      .def_readonly("device", &WorkerInfo::device)
      .def_property_readonly("raw_hello",
                             [](const WorkerInfo& w) { return json_to_py(w.raw_hello); })
      .def("__repr__", [](const WorkerInfo& w) {
        return "<WorkerInfo " + w.endpoint + " workers=" + std::to_string(w.workers) +
               ">";
      });

  py::class_<JobClient>(m, "JobClient")
      .def(py::init<std::string, int, double, double, double>(), py::arg("host"),
           py::arg("port") = kDefaultPort, py::arg("timeout_s") = 30.0,
           py::arg("connect_timeout_s") = 60.0, py::arg("control_timeout_s") = 300.0)
      .def_property_readonly("host", &JobClient::host)
      .def_property_readonly("port", &JobClient::port)
      .def_property_readonly("endpoint", &JobClient::endpoint)
      .def_property_readonly("connected", &JobClient::connected)
      .def_property_readonly(
          "info",
          [](const JobClient& c) -> py::object {
            if (!c.info()) {
              return py::none();
            }
            return py::cast(*c.info());
          })
      .def("connect", &JobClient::connect, py::call_guard<py::gil_scoped_release>())
      .def("close", &JobClient::close)
      .def("reconnect", &JobClient::reconnect, py::call_guard<py::gil_scoped_release>())
      .def("ping",
           [](JobClient& c) {
             py::gil_scoped_release release;
             Json reply = c.ping();
             py::gil_scoped_acquire acquire;
             return json_to_py(reply);
           })
      .def(
          "submit_job",
          [](JobClient& c, const py::object& job, const std::string& kind) {
            Json j = py_to_json(job);
            Json result;
            {
              py::gil_scoped_release release;
              result = c.submit_job(j, kind);
            }
            return json_to_py(result);
          },
          py::arg("job"), py::arg("kind") = "play")
      .def(
          "control",
          [](JobClient& c, const py::object& msg) {
            Json j = py_to_json(msg);
            Json reply;
            {
              py::gil_scoped_release release;
              reply = c.control(j);
            }
            return json_to_py(reply);
          },
          py::arg("msg"));

  m.def("parse_endpoint", [](const std::string& spec) {
    auto ep = parse_endpoint(spec);
    return py::make_tuple(ep.host, ep.port);
  });

  py::class_<WorkerFarm>(m, "WorkerFarm")
      .def(py::init<std::vector<std::string>, double>(), py::arg("endpoints"),
           py::arg("timeout_s") = 30.0)
      .def(
          "connect",
          [](WorkerFarm& farm, bool require_all) {
            py::gil_scoped_release release;
            return farm.connect(require_all);
          },
          py::arg("require_all") = false)
      .def("close", &WorkerFarm::close)
      .def_property_readonly("total_workers", &WorkerFarm::total_workers)
      .def(
          "clients",
          [](WorkerFarm& farm) -> py::list {
            py::list out;
            for (auto& c : farm.clients()) {
              out.append(py::cast(c, py::return_value_policy::reference));
            }
            return out;
          },
          py::return_value_policy::reference_internal);

  py::class_<ServerConfig>(m, "ServerConfig")
      .def(py::init<>())
      .def_readwrite("host", &ServerConfig::host)
      .def_readwrite("port", &ServerConfig::port)
      .def_readwrite("backlog", &ServerConfig::backlog)
      .def_readwrite("max_connections", &ServerConfig::max_connections)
      .def_readwrite("idle_timeout_s", &ServerConfig::idle_timeout_s);

  m.def(
      "serve_forever",
      [](py::function handler, ServerConfig config, py::object hello,
         py::object stop_event) {
        auto stop = std::make_shared<std::atomic<bool>>(false);
        JobHandler cpp_handler = [handler](const Json& msg) -> Json {
          py::gil_scoped_acquire acquire;
          py::object reply = handler(json_to_py(msg));
          return py_to_json(reply);
        };
        HelloFn cpp_hello;
        if (!hello.is_none()) {
          py::function hello_fn = hello.cast<py::function>();
          cpp_hello = [hello_fn]() -> Json {
            py::gil_scoped_acquire acquire;
            return py_to_json(hello_fn());
          };
        }
        // Optional threading.Event compatibility
        std::thread watcher;
        if (!stop_event.is_none()) {
          watcher = std::thread([stop, stop_event]() {
            while (!stop->load()) {
              bool set = false;
              {
                py::gil_scoped_acquire acquire;
                set = stop_event.attr("is_set")().cast<bool>();
              }
              if (set) {
                stop->store(true);
                break;
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
          });
        }
        {
          py::gil_scoped_release release;
          serve_forever(cpp_handler, config, cpp_hello, stop.get());
        }
        stop->store(true);
        if (watcher.joinable()) {
          watcher.join();
        }
      },
      py::arg("handler"), py::arg("config") = ServerConfig{},
      py::arg("hello") = py::none(), py::arg("stop_event") = py::none());

  py::class_<HardwareSignals>(m, "HardwareSignals")
      .def_readonly("cpu_idle_pct", &HardwareSignals::cpu_idle_pct)
      .def_readonly("load1", &HardwareSignals::load1)
      .def_readonly("mem_available_gb", &HardwareSignals::mem_available_gb)
      .def_readonly("mem_total_gb", &HardwareSignals::mem_total_gb)
      .def_readonly("ok", &HardwareSignals::ok);

  m.def("sample_hardware_signals", &sample_hardware_signals);

  py::class_<SchedulerDecision>(m, "SchedulerDecision")
      .def_readonly("local_share", &SchedulerDecision::local_share)
      .def_readonly("remote_share", &SchedulerDecision::remote_share)
      .def_readonly("target_workers", &SchedulerDecision::target_workers)
      .def_readonly("remote_chunk", &SchedulerDecision::remote_chunk)
      .def_readonly("reason", &SchedulerDecision::reason)
      .def_readonly("remote_demand", &SchedulerDecision::remote_demand)
      .def_readonly("metrics", &SchedulerDecision::metrics);

  py::class_<SchedulerConfig>(m, "SchedulerConfig")
      .def(py::init<>())
      .def_readwrite("min_local_frac", &SchedulerConfig::min_local_frac)
      .def_readwrite("prefer_local_frac", &SchedulerConfig::prefer_local_frac)
      .def_readwrite("max_remote_frac", &SchedulerConfig::max_remote_frac)
      .def_readwrite("min_remote_frac", &SchedulerConfig::min_remote_frac)
      .def_readwrite("target_workers", &SchedulerConfig::target_workers)
      .def_readwrite("min_workers", &SchedulerConfig::min_workers)
      .def_readwrite("max_workers", &SchedulerConfig::max_workers)
      .def_readwrite("remote_chunk", &SchedulerConfig::remote_chunk)
      .def_readwrite("tick_s", &SchedulerConfig::tick_s)
      .def_readwrite("min_gps_window_s", &SchedulerConfig::min_gps_window_s)
      .def_readwrite("ema_alpha", &SchedulerConfig::ema_alpha)
      .def_readwrite("demand_settle_s", &SchedulerConfig::demand_settle_s)
      .def_readwrite("demand_degrade_frac", &SchedulerConfig::demand_degrade_frac)
      .def_readwrite("demand_improve_frac", &SchedulerConfig::demand_improve_frac)
      .def_readwrite("demand_eff_collapse_frac",
                    &SchedulerConfig::demand_eff_collapse_frac)
      .def_readwrite("demand_grow_cooldown_s",
                    &SchedulerConfig::demand_grow_cooldown_s)
      .def_readwrite("remote_defaults", &SchedulerConfig::remote_defaults)
      .def_readwrite("remote_maxima", &SchedulerConfig::remote_maxima);

  py::class_<WaveGpsTracker>(m, "WaveGpsTracker")
      .def(py::init<double, double>(), py::arg("min_window_s") = 20.0,
           py::arg("ema_alpha") = 0.35)
      .def("note", &WaveGpsTracker::note, py::arg("side"), py::arg("n") = 1,
           py::arg("decisions") = 0)
      .def("elapsed", &WaveGpsTracker::elapsed)
      .def("wave_gps", &WaveGpsTracker::wave_gps)
      .def("local_gps", &WaveGpsTracker::local_gps)
      .def("remote_gps", &WaveGpsTracker::remote_gps)
      .def("ema_gps", &WaveGpsTracker::ema_gps)
      .def("snapshot", &WaveGpsTracker::snapshot);

  py::class_<MidWaveScheduler>(m, "MidWaveScheduler")
      .def(py::init<SchedulerConfig>(), py::arg("config") = SchedulerConfig{})
      .def("bind_endpoints", &MidWaveScheduler::bind_endpoints, py::arg("defaults"),
           py::arg("maxima"))
      .def("note_completed", &MidWaveScheduler::note_completed, py::arg("side"),
           py::arg("n") = 1, py::arg("decisions") = 0)
      .def(
          "maybe_tick",
          [](MidWaveScheduler& s, int remaining, bool force) -> py::object {
            auto d = s.maybe_tick(remaining, force);
            if (!d) {
              return py::none();
            }
            return py::cast(*d);
          },
          py::arg("remaining"), py::arg("force") = false)
      .def("decision", &MidWaveScheduler::decision)
      .def("remote_demand", &MidWaveScheduler::remote_demand);

  py::class_<CollectConfig>(m, "CollectConfig")
      .def(py::init<>())
      .def_readwrite("local_workers", &CollectConfig::local_workers)
      .def_readwrite("remote_chunk", &CollectConfig::remote_chunk)
      .def_readwrite("kind", &CollectConfig::kind);

  m.def(
      "run_scheduled_wave",
      [](const py::list& jobs, py::function local_submit, const py::list& remote_clients,
         MidWaveScheduler& scheduler, CollectConfig config, py::object on_result) {
        std::vector<Json> cpp_jobs;
        cpp_jobs.reserve(py::len(jobs));
        for (auto item : jobs) {
          cpp_jobs.push_back(py_to_json(item.cast<py::object>()));
        }
        LocalSubmitFn cpp_local = [local_submit](const Json& job) -> Json {
          py::gil_scoped_acquire acquire;
          return py_to_json(local_submit(json_to_py(job)));
        };
        std::vector<JobClient*> clients;
        clients.reserve(py::len(remote_clients));
        for (auto item : remote_clients) {
          clients.push_back(item.cast<JobClient*>());
        }
        ResultCallback cpp_on;
        if (!on_result.is_none()) {
          py::function cb = on_result.cast<py::function>();
          cpp_on = [cb](const Json& result) {
            py::gil_scoped_acquire acquire;
            cb(json_to_py(result));
          };
        }
        int n = 0;
        {
          py::gil_scoped_release release;
          n = run_scheduled_wave(cpp_jobs, cpp_local, clients, scheduler, config,
                                 cpp_on);
        }
        return n;
      },
      py::arg("jobs"), py::arg("local_submit"), py::arg("remote_clients"),
      py::arg("scheduler"), py::arg("config") = CollectConfig{},
      py::arg("on_result") = py::none());
}
