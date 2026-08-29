#!/usr/bin/env bash
# Builds the portopt library, demo and tests into out/ (no Maven; plain javac).
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

rm -rf out
mkdir -p out

javac -Xlint:all -Werror -d out $(find src/main/java -name '*.java')
javac -Xlint:all -Werror -cp "out:$JUNIT:$HAMCREST" -d out $(find src/test/java -name '*.java')

echo "build OK"
