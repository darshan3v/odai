---
name: ODAI Development Guidelines
description: Architectural patterns, type systems, and development workflows for the ODAI SDK.
---

# ODAI SDK Development Guidelines

This document outlines the architectural patterns, type systems, and development workflows for the ODAI SDK. It serves as a reference for maintaining consistency across the codebase.

## 1. High-Level Architecture

The SDK follows a layered architecture designed to expose a stable C API while leveraging modern C++ for implementation.

```mermaid
graph TD
    UserApp[User Application] --> C_API[C Public API (odai_public.h)]
    C_API --> CPP_SDK[C++ SDK Singleton (odai_sdk.h)]
    CPP_SDK --> DB[Database Layer]
    CPP_SDK --> Backend[Backend Engine (LLM)]
    CPP_SDK --> RAG[RAG Engine]
```

- **C Public API (`src/include/odai_public.h`)**: The external interface. Pure C functions with C-compatible types. Stable ABI.
- **C++ SDK (`src/include/odai_sdk.h`)**: The internal entry point. A singleton class (`ODAISdk`) that orchestrates components.
- **Internal Components**: `ODAIDb`, `ODAIBackendEngine`, `ODAIRagEngine` handle specific logic.

## 2. Type System & Data Flow

We maintain a strict separation between C types (API boundary) and C++ types (Internal logic).

### 2.1 Type Definitions
- **C Types (`src/include/types/odai_ctypes.h`)**: 
    - Structs strictly compatible with C.
    - Naming convention: `c_` prefix (e.g., `c_ChatConfig`).
    - Use `typedef char*` for opaque handles (e.g., `c_ChatId`) for type safety.
- **C++ Types (`src/include/types/odai_types.h`)**:
    - Modern C++ structs/classes using STL (`std::string`, `std::vector`).
    - Validation logic included via `is_sane()` member functions.

### 2.2 Enums and ABI Stability
- **Avoid C Enums in Public API**: 
    - C enums do not have a guaranteed size (can be `int`, `unsigned int`, `char`, etc., depending on compiler/flags). This breaks ABI stability.
    - **Solution**: Use `typedef uint8_t` or `typedef uint32_t` along with `#define` macros for constants in `odai_ctypes.h`.
    - Example: `typedef uint32_t c_ModelType; #define ODAI_MODEL_TYPE_LLM (c_ModelType)1`

### 2.3 Sanitization & Conversion
- **Sanitizers (`src/include/utils/odai_csanitizers.h`)**:
    - `is_sane(const c_Type*)` / `is_sane(c_ValueType value)`
    - **Purpose**: Low-level safety checks and **structural validation**.
        - Pointer validity (not null).
        - Value expectations (e.g., checking if an int representing an enum is within the valid range of defined macros).
        - It is acceptable to perform these checks here as they validate the "structure" or "encoding" of the data before it enters the C++ layer.
- **Converters (`src/include/types/odai_type_conversions.h`)**:
    - `toCpp(const c_Type&)`: Converts C structs/types to C++ objects.
    - **Expectation**: Converters **assume input data is safe**. They rely on `is_sane()` having been called previously. For example, `toCpp` for a model type assumes the input `uint32_t` corresponds to a valid `ModelType` enum value.
    - `toC(const CppType&)`: Converts C++ objects to C structs (handles memory allocation for C strings).

## 3. Implementation Patterns

### 3.1 Implementing a New Feature

When adding a new feature, follow this data flow:

#### Step 1: Public C API (`src/impl/odai_public.cpp`)
1.  **Receive Inputs**: Take C types as arguments.
2.  **Sanitize**: Call `is_sane()` from `odai_csanitizers.h` to ensure pointers are valid.
3.  **Log Errors**: specific error if sanitation fails.
4.  **Convert**: Use `toCpp()` to convert verified C inputs to C++ objects.
5.  **Forward**: Call the corresponding `ODAISdk` singleton method.
6.  **Return**: Convert result back to C types if necessary (or return `bool`/`int`).

**Example (`odai_public.cpp`):**
```cpp
bool odai_create_feature(const c_FeatureConfig *c_config)
{
    // 1. Sanitize
    if (!is_sane(c_config)) {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid config passed");
        return false;
    }
    
    // 2. Convert & Forward
    return ODAISdk::get_instance().create_feature(toCpp(*c_config));
}
```

#### Step 2: C++ SDK Layer (`src/impl/odai_sdk.cpp`)
1.  **Receive Inputs**: Take C++ types.
2.  **Validate**: Call `config.is_sane()` for business logic validation (e.g., checking empty strings, valid checking range).
3.  **Check State**: Ensure SDK is initialized (`m_sdkInitialized`).
4.  **Execute**: Call internal engines (DB, RAG, etc.) or perform logic.
5.  **Handle Exceptions**: Wrap broad logic in `try-catch` blocks to prevent crashes.

**Example (`odai_sdk.cpp`):**
```cpp
bool ODAISdk::create_feature(const FeatureConfig& config)
{
    try {
        if (!m_sdkInitialized) return false;

        // 1. Business Logic Validation
        if (!config.is_sane()) {
            ODAI_LOG(ODAI_LOG_ERROR, "Config missing required fields");
            return false;
        }

        // 2. Implementation
        return m_backend->process(config);
    }
    catch (...) {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
        return false;
    }
}
```

## 4. Key Directives

1.  **Sync Headers**: `odai_public.h` (C) and `odai_sdk.h` (C++) must be kept in sync regarding feature parity.
2.  **Memory Management**: 
    - C consumers own memory they allocate.
    - Functions returning arrays/strings in C API must have corresponding `free` functions (e.g., `odai_free_chat_messages`).
    - `toC` allocates memory; ensure there's a path to free it.
    - **`free_members` Helper**:
        - Located in `src/include/types/odai_ctypes.h`.
        - Use `free_members(c_Type*)` to free dynamically allocated members (e.g., `char*`) within a struct *before* freeing the struct itself.
        - Essential for types like `c_ChatMessage` or `c_SemanticSpaceConfig` where members are heap-allocated by `toC()`.
3.  **Error Handling**:
    - **C API**: Returns `bool` (success/fail) or `int` (error codes). logs errors.
    - **C++ SDK**: Catches exceptions and returns error status to C API. Never let exceptions propagate across the C boundary.

## 5. Directory Structure Reference

- `src/include/`
    - `odai_public.h`: **Public C Interface**
    - `odai_sdk.h`: **Public C++ Interface** (Internal singleton)
    - `types/`
        - `odai_ctypes.h`: C struct definitions.
        - `odai_types.h`: C++ struct definitions + logic validation.
        - `odai_common_types.h`: Shared enums/constants.
        - `odai_type_conversions.h`: Converters declaration.
    - `utils/odai_csanitizers.h`: C type safety checks.
- `src/impl/`
    - `odai_public.cpp`: C API implementation (Sanitization -> Conversion -> SDK Call).
    - `odai_sdk.cpp`: C++ SDK implementation (Business Logic -> Engine Call).

## 6. Naming Conventions

### 6.1 Member Variables
- **Prefix with `m_`**: All non-static member variables of a class or struct should be prefixed with `m_`.
    - **Reasoning**: This makes it immediately obvious when reading code which variables are private members of the class versus local variables or function arguments.
    - **Example**:
    ```cpp
    class MyClass {
    private:
        int m_itemCount; // Correct
        string name; // Incorrect, should be m_name
    public:
        void set_count(int count) {
            m_itemCount = count; // unambiguous
        }
    };
    ```

## 7. Code Style & Conventions

To ensure consistency between the C API and C++ implementation, and to maintain a clean codebase, the following style rules are strictly enforced.

### 7.1 Naming Conventions
- **Variables & Functions**: Use `snake_case`.
    - *Reasoning*: Matches the stable C API style and standard C++ STL conventions.
    - *Examples*: `db_path`, `odai_register_model`, `is_sane`, `user_data`.
    - *Incorrect*: `dbPath`, `registerModel`, `userData`.
- **Types (Classes, Structs, Enums, Typedefs)**: Use `PascalCase`.
    - *Examples*: `ChatConfig`, `BackendEngineInternal`, `OAIModelType`.
    - *Incorrect*: `chat_config`, `backend_engine`.
- **Constants & Macros**: Use `UPPER_SNAKE_CASE`.
    - *Examples*: `ODAI_MAX_PATH`, `DEFAULT_CHUNK_SIZE`.
- **Member Variables**: Prefix with `m_` followed by `camelBack`.
    - *Examples*: `m_dbPath`, `m_isInitialized`.

### 7.2 File Headers
- **Pragma Once**: Use `#pragma once` at the top of all header files.

### 7.3 Comments
- **Doxygen**: Use `///` for documentation comments on public APIs (classes, methods, functions).
- **Implementation**: Use `//` for logic explanations inside functions.

### 7.4 Code Style Tooling

Style is enforced via git pre-commit hook. Scripts are in `scripts/`:
- **`format.sh`** - Format code using clang-format (`--help` for usage)
- **`lint.sh`** - Enforce naming conventions using clang-tidy (`--help` for usage)
- **`pre-commit`** - Git hook (install with: `ln -sf ../../scripts/pre-commit .git/hooks/pre-commit`)

> **Maintenance**: When updating `format.sh` or `lint.sh`, also update this guideline if behavior changes.
