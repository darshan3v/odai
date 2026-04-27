#!/bin/bash
# General-purpose C/C++ code formatter using clang-format.
# Accepts individual files or directories to search recursively.

set -e

show_help() {
    cat << EOF
Code Formatter - Format C/C++ source files using clang-format

USAGE:
    format.sh [OPTIONS] [FILES_OR_DIRS...]

OPTIONS:
    --check     Check formatting without modifying files (for CI)
    --help      Show this help message

ARGUMENTS:
    Pass any combination of files and directories.
    Directories are searched recursively for .c, .cpp, .cc, .h, .hpp files.
    If no arguments are given, shows this help.

EXAMPLES:
    format.sh src/ tests/                   # Format all files in src/ and tests/
    format.sh src/ tests/ --check           # Check formatting (CI mode)
    format.sh src/impl/odai_sdk.cpp         # Format a specific file
    format.sh src/include/*.h               # Format matching headers
EOF
}

# Parse arguments
CHECK_MODE=false
INPUTS=()

for arg in "$@"; do
    case $arg in
        --help|-h)
            show_help
            exit 0
            ;;
        --check)
            CHECK_MODE=true
            ;;
        *)
            INPUTS+=("$arg")
            ;;
    esac
done

if [ ${#INPUTS[@]} -eq 0 ]; then
    show_help
    exit 0
fi

# Expand directories into file lists, pass files through as-is
FILES=()
for input in "${INPUTS[@]}"; do
    if [ -d "$input" ]; then
        while IFS= read -r -d '' file; do
            FILES+=("$file")
        done < <(find "$input" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.c" -o -name "*.cc" -o -name "*.hpp" \) -print0 2>/dev/null)
    elif [ -f "$input" ]; then
        FILES+=("$input")
    else
        echo "Warning: '$input' is not a file or directory, skipping."
    fi
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No C/C++ files found."
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
