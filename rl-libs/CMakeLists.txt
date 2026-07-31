cmake_minimum_required(VERSION 3.16)
project(rl_libs VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RL_LIBS_BUILD_TESTS "Build unit tests" ON)
option(RL_LIBS_BUILD_APPS "Build demo/bench apps" ON)
option(RL_LIBS_BUILD_PYTHON "Build pybind11 modules" OFF)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(rl_io
  src/rl_io/digest.cpp
  src/rl_io/ordered_writer.cpp
  src/rl_io/blob_pack.cpp
)
add_library(rl_io::rl_io ALIAS rl_io)
target_include_directories(rl_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(rl_io PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(rl_io PRIVATE -Wall -Wextra -Wpedantic -O3)

add_library(rl_runtime src/rl_runtime/shm_ring.cpp)
add_library(rl_runtime::rl_runtime ALIAS rl_runtime)
target_include_directories(rl_runtime PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_options(rl_runtime PRIVATE -Wall -Wextra -Wpedantic -O3)
if(UNIX AND NOT APPLE)
  target_link_libraries(rl_runtime PUBLIC rt pthread)
elseif(UNIX)
  target_link_libraries(rl_runtime PUBLIC pthread)
endif()

add_library(proc_pool src/proc_pool/supervisor.cpp)
add_library(proc_pool::proc_pool ALIAS proc_pool)
target_include_directories(proc_pool PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_options(proc_pool PRIVATE -Wall -Wextra -Wpedantic -O3)
if(UNIX)
  target_link_libraries(proc_pool PUBLIC pthread)
endif()

include(GNUInstallDirs)
install(TARGETS rl_io rl_runtime proc_pool
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/rl_io include/rl_runtime include/proc_pool
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(RL_LIBS_BUILD_APPS)
  add_executable(rl_echo_worker apps/echo_worker.cpp)
  add_executable(rl_io_bench apps/io_bench.cpp)
  target_link_libraries(rl_io_bench PRIVATE rl_io)
  add_executable(rl_shm_bench apps/shm_bench.cpp)
  target_link_libraries(rl_shm_bench PRIVATE rl_runtime)
endif()

if(RL_LIBS_BUILD_TESTS)
  enable_testing()
  add_executable(rl_libs_tests
    tests/main.cpp
    tests/test_digest.cpp
    tests/test_ordered_writer.cpp
    tests/test_blob_pack.cpp
    tests/test_shm_ring.cpp
    tests/test_proc_pool.cpp
  )
  target_link_libraries(rl_libs_tests PRIVATE rl_io rl_runtime proc_pool)
  add_test(NAME rl_libs_tests COMMAND rl_libs_tests)
endif()

if(RL_LIBS_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
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
  pybind11_add_module(_rl_io MODULE python/bindings/rl_io_module.cpp)
  target_link_libraries(_rl_io PRIVATE rl_io)
  pybind11_add_module(_rl_runtime MODULE python/bindings/rl_runtime_module.cpp)
  target_link_libraries(_rl_runtime PRIVATE rl_runtime)
  pybind11_add_module(_proc_pool MODULE python/bindings/proc_pool_module.cpp)
  target_link_libraries(_proc_pool PRIVATE proc_pool)
  if(DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS _rl_io DESTINATION rl_io)
    install(TARGETS _rl_runtime DESTINATION rl_runtime)
    install(TARGETS _proc_pool DESTINATION proc_pool)
  else()
    set_target_properties(_rl_io PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/rl_io")
    set_target_properties(_rl_runtime PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/rl_runtime")
    set_target_properties(_proc_pool PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/proc_pool")
  endif()
endif()
