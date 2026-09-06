#!/usr/bin/env bash
# Runs the end-to-end demo (expects the bundled data at ../data).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f out/com/quant/portopt/Demo.class ]; then
    bash build.sh
fi

java -cp out com.quant.portopt.Demo
