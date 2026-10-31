#!/usr/bin/env bash
# Compiles and packages the portopt library (target/portopt-*.jar) without running tests.
set -euo pipefail
cd "$(dirname "$0")"
exec mvn -B -q -DskipTests package
