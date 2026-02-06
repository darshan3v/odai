# ODAI SDK

> [!NOTE]
> 🚧 **Work in Progress**: Both the repository and this documentation are currently under active development. Significant changes may occur.


## Overview
The ODAI SDK (On-Device AI SDK) provides tools and libraries for integrating AI capabilities directly into your applications.

## Getting Started

### Prerequisites
- CMake
- Clang Compiler (supporting C++20 or later)
- Git

### Repository Setup
To ensure a consistent development environment, we use shared git hooks. Please run the setup script immediately after cloning the repository:

```bash
./setup_repo.sh
```

This script configures `core.hooksPath` to use the hooks in `.githooks`, ensuring that pre-commit checks (clang-format, clang-tidy) issues are caught early.

## AI-Assisted Development
This repository includes a `.agent/skills` directory designed to enhance AI-based coding workflows. These resources provide context, guidelines, and specific skills to help AI agents assist you more effectively in this codebase.