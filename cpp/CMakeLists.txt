# portopt: C++ port of the portfolio-construction toolkit.
#
# Targets: portopt::portopt (static library), portopt_demo, portopt_tests.
# Options:
#   PORTOPT_BUILD_TESTS  build the GoogleTest suite (default ON when top level)
#   PORTOPT_WERROR       treat warnings as errors (default OFF; CI turns it on)
#   PORTOPT_SANITIZE     build with AddressSanitizer + UBSan (GCC/Clang only)
#   PORTOPT_BUILD_BENCHMARKS  build portopt_bench (Google Benchmark, fetched;
#                        default OFF, never part of the test build)
# GoogleTest is taken from the system when available, otherwise fetched.
# Tests run with the working directory set to this source dir so the
# golden file resolves at ../data/golden/golden.json (per the contract).
cmake_minimum_required(VERSION 3.25)
project(portopt VERSION 1.2.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# C++26.  CMake has no C++26 level for MSVC (there is no /std:c++26), so
# there the newest standard is requested with /std:c++latest for every
# target, GoogleTest included.
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
  add_compile_options(/std:c++latest)
else()
  set(CMAKE_CXX_STANDARD 26)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PORTOPT_BUILD_TESTS "Build the portopt test suite" ${PROJECT_IS_TOP_LEVEL})
option(PORTOPT_WERROR "Treat compiler warnings as errors" OFF)
option(PORTOPT_SANITIZE "Build with ASan and UBSan (GCC/Clang)" OFF)
option(PORTOPT_BUILD_BENCHMARKS "Build the Google Benchmark micro-benchmarks" OFF)

# Warning and floating-point flags per compiler family (MSVC rejects
# -Wextra outright).  -ffp-contract=off pins IEEE-754 semantics (no
# implicit FMA fusion), so every a*b+c rounds the same way on every
# target; MSVC does not contract under its default /fp:precise.
add_library(portopt_flags INTERFACE)
if(MSVC)
  target_compile_options(portopt_flags INTERFACE /W4 /permissive- /utf-8
                         $<$<BOOL:${PORTOPT_WERROR}>:/WX>)
else()
  target_compile_options(portopt_flags INTERFACE -Wall -Wextra -Wpedantic
                         -ffp-contract=off
                         $<$<BOOL:${PORTOPT_WERROR}>:-Werror>)
endif()
if(PORTOPT_SANITIZE AND NOT MSVC)
  target_compile_options(portopt_flags INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(portopt_flags INTERFACE -fsanitize=address,undefined)
endif()

add_library(portopt
  src/linalg.cpp
  src/estimators.cpp
  src/markowitz.cpp
  src/black_litterman.cpp
  src/risk_parity.cpp
  src/backtest.cpp
  src/data_io.cpp)
add_library(portopt::portopt ALIAS portopt)
target_include_directories(portopt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(NOT MSVC)
  target_compile_features(portopt PUBLIC cxx_std_26)
endif()
target_compile_definitions(portopt PUBLIC PORTOPT_VERSION="${PROJECT_VERSION}")
target_link_libraries(portopt PRIVATE $<BUILD_INTERFACE:portopt_flags>)

add_executable(portopt_demo src/demo.cpp)
target_link_libraries(portopt_demo PRIVATE portopt portopt_flags)

include(GNUInstallDirs)
install(TARGETS portopt EXPORT portoptTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT portoptTargets NAMESPACE portopt:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/portopt)

if(PORTOPT_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(portopt_tests
    tests/test_golden.cpp
    tests/test_linalg.cpp
    tests/test_estimators.cpp
    tests/test_markowitz.cpp
    tests/test_black_litterman.cpp
    tests/test_risk_parity.cpp
    tests/test_backtest.cpp)
  target_include_directories(portopt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(portopt_tests PRIVATE portopt portopt_flags GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(portopt_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       DISCOVERY_TIMEOUT 60)
endif()

# Micro-benchmarks (opt-in): cmake -DPORTOPT_BUILD_BENCHMARKS=ON, then run
# portopt_bench from the build tree. Google Benchmark is fetched at a pinned
# release and hash; its own tests, install rules and -Werror are disabled
# so a third-party warning never fails a PORTOPT_WERROR build.
if(PORTOPT_BUILD_BENCHMARKS)
  include(FetchContent)
  FetchContent_Declare(googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    URL_HASH SHA256=32131c08ee31eeff2c8968d7e874f3cb648034377dfc32a4c377fa8796d84981
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)
  if(MSVC)
    # Google Benchmark pins C++17 for its own targets, which would put
    # /std:c++17 next to the global /std:c++latest (warning D9025); build it
    # with /std:c++latest like every other target.
    set_property(TARGET benchmark benchmark_main PROPERTY CXX_STANDARD)
  endif()

  add_executable(portopt_bench benchmarks/bench_kernels.cpp)
  target_link_libraries(portopt_bench PRIVATE portopt portopt_flags benchmark::benchmark)
endif()
