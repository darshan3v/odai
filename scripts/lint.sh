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
    lint.sh [OPTIONS]

OPTIONS:
    --fix       Auto-fix issues where possible
    --help      Show this help message

EXAMPLES:
    lint.sh                       # Lint all project files
    lint.sh --fix                 # Lint and auto-fix all files

REQUIREMENTS:
    - compile_commands.json must exist in build/
    - Run CMake first: cmake --preset debug
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

# run cmake so that compile_commands.json is generated
cmake --preset linux-release >/dev/null

# Symlink compile_commands.json to project root if not exists
ln -sf "$PROJECT_ROOT/build/compile_commands.json" "$PROJECT_ROOT/compile_commands.json"

# Determine files to process
# Always grab source files
FILES=($(find "$PROJECT_ROOT/src" -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.cc" \) 2>/dev/null))

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found to lint in src/"
    exit 0
fi

# Build basic clang-tidy args
# --header-filter: Only check headers in src/ (ignore dependencies)
TIDY_ARGS=(-p "$PROJECT_ROOT" "--header-filter=$PROJECT_ROOT/src/.*")

if [ "$FIX_MODE" = true ]; then
    echo "Linting and fixing ${#FILES[@]} file(s)..."
    
    # Create temp directory for fixes
    FIX_DIR="$PROJECT_ROOT/.clang-tidy-fixes"
    mkdir -p "$FIX_DIR"
    # Ensure cleanup on exit
    trap 'rm -rf "$FIX_DIR"' EXIT

    # Process files individually
    for f in "${FILES[@]}"; do
        # Generate a unique yaml filename based on the source path
        REL_PATH="${f#$PROJECT_ROOT/}"
        SAFE_NAME=$(echo "$REL_PATH" | sed 's/[^a-zA-Z0-9]/_/g').yaml
        
        # Run clang-tidy
        # We use || true because clang-tidy returns non-zero if issues are found, 
        # but we want to continue processing other files.
        echo "Checking $REL_PATH..."
        clang-tidy "${TIDY_ARGS[@]}" -export-fixes="$FIX_DIR/$SAFE_NAME" "$f" || true
    done

    echo "Applying fixes..."
    # Apply replacements
    # -format: Formats changed code
    # -style=file: Uses .clang-format
    # -p: Compilation database path
    clang-apply-replacements -format -style=file -p "$PROJECT_ROOT" "$FIX_DIR"
    
else
    echo "Linting ${#FILES[@]} file(s)..."
    # In non-fix mode, we can just run them all at once or individually. 
    # Running individually gives better progress feedback but might be slower due to startup process? 
    # Actually, clang-tidy usually takes multiple files fine. 
    # Let's run all at once for speed in read-only mode, or keep it consistent?
    # The existing script ran them all at once. Let's stick to that for read-only mode for now unless user asked otherwise.
    # Wait, user said "apply clang tidy to file individually ... and apply the fixes then".
    # This implies the individual processing is for the FIX workflow primarily to handle export-fixes safely.
    # But for consistency, let's keep the existing bulk behavior for non-fix mode OR change it.
    # Actually, `clang-tidy [files...]` is standard.
    
    clang-tidy "${TIDY_ARGS[@]}" "${FILES[@]}"
fi

echo "✓ Done!"
