#!/usr/bin/env bash
# Runs the end-to-end demo (compiles first if needed; expects the bundled data at ../data).
set -euo pipefail
cd "$(dirname "$0")"
exec mvn -B -q compile exec:java
