#!/usr/bin/env bash
# Runs the JUnit 4 suite (golden values + unit/property tests).
# Must be run from the java/ directory: tests load ../data/golden/golden.json.
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

./build.sh

java -cp "out:$JUNIT:$HAMCREST" org.junit.runner.JUnitCore \
    com.quant.portopt.LinalgTest \
    com.quant.portopt.EstimatorsTest \
    com.quant.portopt.MarkowitzTest \
    com.quant.portopt.BlackLittermanTest \
    com.quant.portopt.RiskParityTest \
    com.quant.portopt.BacktestTest \
    com.quant.portopt.GoldenTest
