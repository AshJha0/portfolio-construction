#!/usr/bin/env bash
# Build the portopt C++ library, demo and tests (GoogleTest is fetched if
# it is not installed).  Run the tests afterwards with:
#   ctest --test-dir build -C Release --output-on-failure
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build --config Release --parallel
