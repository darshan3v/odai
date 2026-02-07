# ODAI SDK (On-Device Artificial Intelligence SDK)

> [!NOTE]
> 🚧 **Work in Progress**: Both the repository and this documentation are currently under active development. Significant changes may occur.

## Overview
The ODAI SDK is a high-performance C++ library designed to integrate AI capabilities directly into applications running on edge devices. It enables on-device inference, retrieval-augmented generation (RAG), and persistent chat capabilities without relying on cloud services.

### Vision: Service vs. Library
While the SDK can be linked as a standard library, our vision is to deploy it as a **standalone system service**.
-   **Efficiency**: A single service instance manages the heavy AI models (LLMs, Embeddings) and vector database.
-   **Resource Management**: Multiple applications can query the service simultaneously without each app loading its own copy of a multi-gigabyte model, preventing memory exhaustion and resource contention.
-   **Unified Context**: A shared service allows for a unified knowledge base accessible across permitted applications.

## Key Features
-   **Privacy-First**: All processing happens on-device. No data leaves the user's hardware.
-   **Native Performance**: Built with C++20 for maximum efficiency and minimal overhead.
-   **Cross-Platform**: Designed to run on Linux, Android, and iOS.

## Core Concepts

Understanding these concepts is crucial for using the RAG and Memory features effectively.

### Semantic Space
A **Semantic Space** is a logical grouping of knowledge that shares the same embedding model and chunking strategy.
-   It represents a "knowledge domain" (e.g., "Medical Docs", "Codebase", "General Chat").
-   Documents added to a Semantic Space are vectorized using its specific configuration.

### Scope
A **Scope** provides granular filtering within a Semantic Space.
-   It allows you to partition data (e.g., by `User ID`, `Application ID`, or `Document Group`).
-   When retrieving context, you can restrict the search to a specific Scope, ensuring that users only access data they are authorized to see, even if it resides in the same physical vector store.

## Technical Architecture

### C ABI & Extensibility
The SDK exposes a **C ABI (Application Binary Interface)**.
-   **Stability**: Ensures binary compatibility across compiler versions and updates.
-   **Interoperability**: Makes it easy to write bindings for almost any language (Python, Rust, Dart, Java/Kotlin, Swift).
-   **Mobile Support**: We are actively developing JNI (Android) and C-Interdict/Swift (iOS) bindings to make this a first-class mobile library.

### Modular Backend
The architecture is interface-based, allowing core components to be swapped:
-   **Inference Engine**: Currently uses `llama.cpp`, but can be extended to support other backends (e.g., ONNX, MLC-LLM).
-   **Vector Database**: Currently uses `SQLite` (via `sqlite-vec` and `SQLiteCpp`), but can be extended to support other vector stores (e.g., FAISS, pgvector).

## Tech Stack
-   **Language**: C++ 20 (Core), C (API Surface)
-   **Build System**: CMake, Ninja
-   **Key Libraries**:
    -   `llama.cpp`: For LLM inference.
    -   `sqlite-vec`: For vector similarity search.
    -   `nlohmann/json`: For JSON manipulation.
    -   `SQLiteCpp`: C++ wrapper for SQLite.

## Build Instructions

We use `CMakePresets.json` to simplify the build process for different platforms.

### Prerequisites
-   CMake (3.23+)
-   Ninja Build System
-   Clang Compiler (supporting C++20)
-   Android NDK (for mobile builds)

### Building
Configure and build using the predefined presets:

#### Linux
```bash
# Debug Build
cmake --preset linux-debug
cmake --build --preset linux-debug

# Release Build
cmake --preset linux-release
cmake --build --preset linux-release
```

#### Android (ARM64)
Ensure `ANDROID_HOME` is set.
```bash
# Release Build
cmake --preset android-arm64-release
cmake --build --preset android-arm64-release
```

## Current Capabilities & Limitations

### Implemented
-   ✅ **Inference**: Basic LLM text generation.
-   ✅ **Chat**: Stateful chat sessions with history persistence and system prompts.
-   ✅ **RAG Core**: Vector store basis, semantic search, and context retrieval.
-   ✅ **Data Isolation**: Semantic Spaces and Scoping are functional.

### Limitations (Work in Progress)
-   ❌ **Document Ingestion**: Logic for parsing files (PDF, raw text) and chunking them is **not yet implemented**.
-   ❌ **Embedding Generation**: The pipeline to automatically compute embeddings from raw text input is currently under development. Developers currently must handle embedding generation or insert pre-computed vectors (for testing).

## Future Roadmap

1.  **Direct Object Passing**:
    -   Allow C++ developers to pass their own class implementations (implementing our interfaces) directly to the SDK, offering maximum flexibility without modifying the core library.

2.  **Task-Based RAG**:
    -   Simplify RAG for developers. Instead of manual configuration, developers will provide a `TaskType`, and the SDK will automatically determine the optimal Chunking, Parsing, Retrieval, and Generation strategies.

3.  **Multi-Modal Support**:
    -   **Vision API**: Support for image inputs (e.g., LLaVA, BakLLaVA).
    -   **Voice API**: Audio transcription (Whisper) and TTS integration.

4.  **Structured Output & Tool Calling**:
    -   Force the LLM to output valid JSON matching a schema.
    -   Enable the model to "call" external tools defined by the developer.

## Repository Setup
To ensure a consistent development environment, we use shared git hooks.
```bash
./setup_repo.sh
```
This configures extensive pre-commit checks (clang-format, clang-tidy) to ensure code quality.

## AI-Assisted Development
This repository includes a `.agent/skills` directory designed to enhance AI-based coding workflows. These resources provide context, guidelines, and specific skills to help AI agents assist you more effectively in this codebase.