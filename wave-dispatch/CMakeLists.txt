cmake_minimum_required(VERSION 3.16)
project(wave_dispatch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WAVE_DISPATCH_BUILD_TESTS "Build unit tests" ON)
option(WAVE_DISPATCH_BUILD_APPS "Build echo worker/client apps" ON)
option(WAVE_DISPATCH_BUILD_PYTHON "Build pybind11 Python module" OFF)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(wave_dispatch
  src/frame.cpp
  src/client.cpp
  src/server.cpp
  src/scheduler.cpp
  src/collector.cpp
)
add_library(wave_dispatch::wave_dispatch ALIAS wave_dispatch)

target_include_directories(wave_dispatch
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(wave_dispatch PUBLIC nlohmann_json::nlohmann_json)
if(UNIX)
  target_link_libraries(wave_dispatch PUBLIC pthread)
endif()

target_compile_options(wave_dispatch PRIVATE -Wall -Wextra -Wpedantic)
# Required when linking the static lib into the Python shared module.
set_target_properties(wave_dispatch PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
install(TARGETS wave_dispatch
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/wave_dispatch DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(WAVE_DISPATCH_BUILD_APPS)
  add_executable(wave_echo_worker apps/echo_worker.cpp)
  target_link_libraries(wave_echo_worker PRIVATE wave_dispatch)

  add_executable(wave_echo_client apps/echo_client.cpp)
  target_link_libraries(wave_echo_client PRIVATE wave_dispatch)
endif()

if(WAVE_DISPATCH_BUILD_TESTS)
  enable_testing()
  add_executable(wave_dispatch_tests
    tests/test_frame.cpp
    tests/test_scheduler.cpp
    tests/test_roundtrip.cpp
    tests/main.cpp
  )
  target_link_libraries(wave_dispatch_tests PRIVATE wave_dispatch)
  add_test(NAME wave_dispatch_tests COMMAND wave_dispatch_tests)
endif()

if(WAVE_DISPATCH_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  # Prefer an already-installed pybind11 (venv / build-system), else fetch.
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  pybind11_add_module(_native MODULE python/bindings/module.cpp)
  target_link_libraries(_native PRIVATE wave_dispatch)
  target_compile_definitions(_native PRIVATE WAVE_DISPATCH_PYTHON=1)

  # scikit-build-core sets SKBUILD_PLATLIB_DIR when building wheels.
  if(DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS _native DESTINATION wave_dispatch)
  else()
    # Dev/cmake builds: drop the extension next to the pure-Python package.
    set_target_properties(_native PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/wave_dispatch"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/wave_dispatch"
    )
  endif()
endif()
