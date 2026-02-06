#!/bin/bash
# ODAI SDK Code Formatter
# Uses clang-format to format C/C++ source files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_help() {
    cat << EOF
ODAI Code Formatter - Format C/C++ source files using clang-format

USAGE:
    format.sh [OPTIONS] [FILES...]

OPTIONS:
    --all       Format all project source files (src/ and main.cpp)
    --check     Check formatting without modifying files (for CI)
    --help      Show this help message

EXAMPLES:
    format.sh --all                     # Format all project files
    format.sh --all --check             # Check all files (CI mode)
    format.sh src/impl/odai_sdk.cpp     # Format specific file
    format.sh src/include/*.h           # Format all headers

FILES:
    If no --all flag and no files specified, shows this help.
    Only project files (src/, main.cpp) are processed.
EOF
}

# Parse arguments
CHECK_MODE=false
ALL_MODE=false
FILES=()

for arg in "$@"; do
    case $arg in
        --help|-h)
            show_help
            exit 0
            ;;
        --check)
            CHECK_MODE=true
            ;;
        --all)
            ALL_MODE=true
            ;;
        *)
            FILES+=("$arg")
            ;;
    esac
done

# Determine files to process
if [ "$ALL_MODE" = true ]; then
    FILES=($(find "$PROJECT_ROOT/src" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.c" \) 2>/dev/null))
    if [ -f "$PROJECT_ROOT/main.cpp" ]; then
        FILES+=("$PROJECT_ROOT/main.cpp")
    fi
elif [ ${#FILES[@]} -eq 0 ]; then
    show_help
    exit 0
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found to format."
    exit 0
fi

# Run clang-format
if [ "$CHECK_MODE" = true ]; then
    echo "Checking format for ${#FILES[@]} file(s)..."
    clang-format --dry-run --Werror "${FILES[@]}"
    echo "✓ All files are formatted correctly!"
else
    echo "Formatting ${#FILES[@]} file(s)..."
    clang-format -i "${FILES[@]}"
    echo "✓ Done!"
fi
