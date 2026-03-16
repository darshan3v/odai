#!/bin/bash

# Robust path detection
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"

# Configuration
INCLUDE_DIR="$ROOT_DIR/src/include"
LIB_DIR="$ROOT_DIR/out/linux/Release/lib"
TEST_SOURCE="$SCRIPT_DIR/main.cpp"
OUTPUT_BINARY="$SCRIPT_DIR/main"

echo "--- Compiling ODAI SDK Tests ---"
g++ "$TEST_SOURCE" -o "$OUTPUT_BINARY" -I "$INCLUDE_DIR" -L "$LIB_DIR" -l odai -Wl,-rpath,"$LIB_DIR"

if [ $? -eq 0 ]; then
    echo "--- Compilation Successful ---"
    echo "--- Running Tests ---"
    # Execute the binary
    "$OUTPUT_BINARY"
else
    echo "--- Compilation Failed ---"
    exit 1
fi
