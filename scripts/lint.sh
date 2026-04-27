#!/bin/bash
# General-purpose C/C++ code linter using run-clang-tidy.
# Accepts directories to search recursively for source files.

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
Code Linter - Lint C/C++ source files using clang-tidy in parallel

USAGE:
    lint.sh [OPTIONS] [DIRS...]

OPTIONS:
    --fix       Auto-fix issues where possible
    --help      Show this help message

ARGUMENTS:
    Pass directories to search recursively for .c, .cpp, .cc files.

EXAMPLES:
    lint.sh src/                      # Lint files in src/
    lint.sh --fix src/                # Lint and auto-fix files in src/

REQUIREMENTS:
    - the linux-default-release preset must be able to generate build/compile_commands.json
    - run-clang-tidy must be installed (ships with clang-tidy)
EOF
}

# Parse arguments
FIX_MODE=false
DIRS=()

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
            if [ -d "$arg" ]; then
                DIRS+=("$arg")
            else
                echo "Error: '$arg' is not a directory"
                show_help
                exit 1
            fi
            ;;
    esac
done

# Require at least one directory
if [ ${#DIRS[@]} -eq 0 ]; then
    echo "Error: no directories specified."
    show_help
    exit 1
fi

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

# Collect source files from all specified directories
FILES=()
for dir in "${DIRS[@]}"; do
    while IFS= read -r -d '' file; do
        FILES+=("$file")
    done < <(find "$dir" -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.cc" \) -print0 2>/dev/null)
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found to lint in: ${DIRS[*]}"
    exit 0
fi

# Determine parallelism: use all available cores but cap at file count
NPROC=$(nproc 2>/dev/null || echo 4)
JOBS=$(( NPROC < ${#FILES[@]} ? NPROC : ${#FILES[@]} ))

# Build run-clang-tidy args
# -p: compilation database directory
# -header-filter: only check headers under the searched dirs (ignore deps)
# -j: parallel jobs
# -quiet: suppress clang-tidy's per-file "N warnings generated" noise

# Build header-filter regex from the provided directories
HEADER_FILTER_PARTS=()
for dir in "${DIRS[@]}"; do
    abs_dir="$(cd "$dir" && pwd)"
    escaped=$(printf '%s' "$abs_dir" | sed 's/[.[\*^$()+?{|]/\\&/g')
    HEADER_FILTER_PARTS+=("${escaped}/.*")
done
# Join parts with | for regex alternation
HEADER_FILTER=$(IFS='|'; echo "${HEADER_FILTER_PARTS[*]}")

RUN_ARGS=(
    -p "$PROJECT_ROOT"
    -header-filter "$HEADER_FILTER"
    -j "$JOBS"
    -quiet
)

if [ "$FIX_MODE" = true ]; then
    RUN_ARGS+=(-fix -format)
    echo "Linting and fixing ${#FILES[@]} file(s) with $JOBS parallel jobs..."
else
    echo "Linting ${#FILES[@]} file(s) with $JOBS parallel jobs..."
fi

# Build a file-filter regex that matches the provided directories
FILTER_PARTS=()
for dir in "${DIRS[@]}"; do
    abs_dir="$(cd "$dir" && pwd)"
    escaped=$(printf '%s' "$abs_dir" | sed 's/[.[\*^$()+?{|]/\\&/g')
    FILTER_PARTS+=("${escaped}/.*\\.(cpp|c|cc)$")
done
FILE_FILTER=$(IFS='|'; echo "(${FILTER_PARTS[*]})")

run-clang-tidy "${RUN_ARGS[@]}" "$FILE_FILTER"

echo "✓ Done!"
