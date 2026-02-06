#!/bin/bash
# ODAI SDK Repository Setup Script
# Configures git hooks to use the shared .githooks directory.

set -e

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$PROJECT_ROOT/.githooks"

echo "Setting up repository..."

# Check if .githooks directory exists
if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: .githooks directory not found at $HOOKS_DIR"
    exit 1
fi

# Configure git to use the hooks directory
git config core.hooksPath .githooks
echo "✅ git config core.hooksPath set to .githooks"

# Make hooks executable
if [ -f "$HOOKS_DIR/pre-commit" ]; then
    chmod +x "$HOOKS_DIR/pre-commit"
    echo "✅ Made pre-commit hook executable"
else
    echo "⚠️  Warning: pre-commit hook not found in .githooks"
fi

echo "Repository setup complete!"
