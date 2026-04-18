#!/bin/bash
# ODAI SDK Code Linter
# Uses run-clang-tidy to lint source files in parallel

set -e

# Ensure Ctrl+C kills all child processes (run-clang-tidy spawns a worker pool).
# Without this, interrupting during pre-commit leaves orphaned clang-tidy processes.
cleanup() {
    kill -- -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_help() {
    cat << EOF
ODAI Code Linter - Lint C/C++ source files using clang-tidy

USAGE:
    lint.sh [OPTIONS]

OPTIONS:
    --fix       Auto-fix issues where possible
    --help      Show this help message

EXAMPLES:
    lint.sh                       # Lint all project files
    lint.sh --fix                 # Lint and auto-fix all files

REQUIREMENTS:
    - compile_commands.json must exist in build/
    - Run CMake first: cmake --preset linux-default-debug
    - run-clang-tidy must be installed (ships with clang-tidy)
EOF
}

# Parse arguments
FIX_MODE=false

for arg in "$@"; do
    case $arg in
        --help|-h)
            show_help
            exit 0
            ;;
        --fix)
            FIX_MODE=true
            ;;
        *)
            echo "Error: Unknown argument '$arg'"
            show_help
            exit 1
            ;;
    esac
done

# Ensure run-clang-tidy is available
if ! command -v run-clang-tidy &>/dev/null; then
    echo "Error: run-clang-tidy not found."
    echo "It ships with clang-tidy. Install it via your package manager."
    exit 1
fi

# run cmake so that compile_commands.json is generated
cmake --preset linux-default-release >/dev/null

# Symlink compile_commands.json to project root if not exists
ln -sf "$PROJECT_ROOT/build/compile_commands.json" "$PROJECT_ROOT/compile_commands.json"

# Collect source files
FILES=($(find "$PROJECT_ROOT/src" -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.cc" \) 2>/dev/null))

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found to lint in src/"
    exit 0
fi

# Determine parallelism: use all available cores but cap at file count
NPROC=$(nproc 2>/dev/null || echo 4)
JOBS=$(( NPROC < ${#FILES[@]} ? NPROC : ${#FILES[@]} ))

# Build run-clang-tidy args
# -p: compilation database directory
# -header-filter: only check headers under src/ (ignore deps)
# -j: parallel jobs
# -quiet: suppress clang-tidy's per-file "N warnings generated" noise
RUN_ARGS=(
    -p "$PROJECT_ROOT"
    -header-filter "$PROJECT_ROOT/src/.*"
    -j "$JOBS"
    -quiet
)

if [ "$FIX_MODE" = true ]; then
    RUN_ARGS+=(-fix -format)
    echo "Linting and fixing ${#FILES[@]} file(s) with $JOBS parallel jobs..."
else
    echo "Linting ${#FILES[@]} file(s) with $JOBS parallel jobs..."
fi

# run-clang-tidy accepts file-path regex filter as a positional arg.
# Build a regex that matches exactly our source files under src/.
# Escape the project root for regex safety and anchor to src/.
ESCAPED_ROOT=$(printf '%s' "$PROJECT_ROOT" | sed 's/[.[\*^$()+?{|]/\\&/g')
FILE_FILTER="${ESCAPED_ROOT}/src/.*\.(cpp|c|cc)$"

run-clang-tidy "${RUN_ARGS[@]}" "$FILE_FILTER"

echo "✓ Done!"
