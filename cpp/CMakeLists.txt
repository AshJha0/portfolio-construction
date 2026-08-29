# portopt — C++17 port of the portfolio-construction toolkit.
# Targets: portopt (static lib), portopt_demo, portopt_tests (GoogleTest).
# Tests run with the working directory set to this source dir so the
# golden file resolves at ../data/golden/golden.json (per the contract).
cmake_minimum_required(VERSION 3.16)
project(portopt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -ffp-contract=off pins IEEE-754 semantics: no implicit FMA fusion, so
# every a*b+c rounds like the numpy reference (explicit std::fma is still
# used where the reference BLAS kernels fuse).
add_compile_options(-Wall -Wextra -ffp-contract=off)

add_library(portopt
  src/linalg.cpp
  src/estimators.cpp
  src/markowitz.cpp
  src/black_litterman.cpp
  src/risk_parity.cpp
  src/backtest.cpp
  src/data_io.cpp)
target_include_directories(portopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(portopt_demo src/demo.cpp)
target_link_libraries(portopt_demo PRIVATE portopt)

enable_testing()
find_package(GTest REQUIRED)

add_executable(portopt_tests
  tests/test_golden.cpp
  tests/test_linalg.cpp
  tests/test_estimators.cpp
  tests/test_markowitz.cpp
  tests/test_black_litterman.cpp
  tests/test_risk_parity.cpp
  tests/test_backtest.cpp)
target_include_directories(portopt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(portopt_tests PRIVATE portopt GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(portopt_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
