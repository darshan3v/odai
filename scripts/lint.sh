#!/bin/bash
# ODAI SDK Code Linter
# Uses clang-tidy to enforce naming conventions and detect issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_help() {
    cat << EOF
ODAI Code Linter - Lint C/C++ source files using clang-tidy

USAGE:
    lint.sh [OPTIONS] [FILES...]

OPTIONS:
    --all       Lint all project source files (src/)
    --fix       Auto-fix issues where possible
    --help      Show this help message

EXAMPLES:
    lint.sh --all                       # Lint all project files
    lint.sh --all --fix                 # Lint and auto-fix all files
    lint.sh src/impl/odai_sdk.cpp       # Lint specific file
    lint.sh --fix src/include/odai_sdk.h  # Lint and fix specific file

REQUIREMENTS:
    - compile_commands.json must exist in build/
    - Run CMake first: cmake --preset debug

FILES:
    If no --all flag and no files specified, shows this help.
    Only project files (src/) are processed.
EOF
}

# Parse arguments
FIX_MODE=false
ALL_MODE=false
FILES=()

for arg in "$@"; do
    case $arg in
        --help|-h)
            show_help
            exit 0
            ;;
        --fix)
            FIX_MODE=true
            ;;
        --all)
            ALL_MODE=true
            ;;
        *)
            FILES+=("$arg")
            ;;
    esac
done

# run cmake so that compile_commands.json is generated
cmake --preset linux-release

# Symlink compile_commands.json to project root if not exists
ln -sf "$PROJECT_ROOT/build/compile_commands.json" "$PROJECT_ROOT/compile_commands.json"

# Determine files to process
if [ "$ALL_MODE" = true ]; then
    # Only grab source files to ensure we have valid compilation database entries
    FILES=$(find "$PROJECT_ROOT/src" -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.cc" \) 2>/dev/null)
elif [ ${#FILES[@]} -gt 0 ]; then
    # If user passed specific files, filter them to keep only source files
    VALID_SOURCES=""
    for f in "${FILES[@]}"; do
        if [[ "$f" =~ \.(cpp|c|cc)$ ]]; then
            VALID_SOURCES="$VALID_SOURCES $f"
        else
            # Optional: Warn the user why the header isn't being passed directly
            echo "Note: Skipping direct processing of $f (it will be checked via included source files)"
        fi
    done
    FILES="$VALID_SOURCES"
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found to lint."
    exit 0
fi

# Build clang-tidy command
# --header-filter: Only check headers in src/ (ignore dependencies)
TIDY_ARGS=(-p "$PROJECT_ROOT" "--header-filter=$PROJECT_ROOT/src/.*")

if [ "$FIX_MODE" = true ]; then
    TIDY_ARGS+=(--fix --fix-notes)
    echo "Linting and fixing ${#FILES[@]} file(s)..."
else
    echo "Linting ${#FILES[@]} file(s)..."
fi

clang-tidy "${TIDY_ARGS[@]}" "${FILES[@]}"
echo "✓ Done!"
