#!/usr/bin/env bash
# Builds and runs the JUnit 4 suite (golden values + unit/property tests).
# Surefire runs from java/, so the tests find ../data/golden/golden.json.
set -euo pipefail
cd "$(dirname "$0")"
exec mvn -B verify
