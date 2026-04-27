---
name: ODAI Development Guidelines
description: Coding patterns, conventions, and style rules for the ODAI SDK.
---

# ODAI SDK Development Guidelines

This document covers the **coding patterns, conventions, and style rules** for day-to-day development. For the system architecture (layers, swappable interfaces, data flow, type system), see [`docs/architecture/`](../../../docs/architecture/README.md).

## 1. Implementation Patterns

### 1.1 Implementing a New Feature

When adding a new feature, follow this data flow through the layers:

#### Step 1: Public C API (`src/impl/odai_public.cpp`)
1.  **Receive Inputs**: Take C types as arguments.
2.  **Sanitize**: Call `is_sane()` from `odai_csanitizers.h` to ensure pointers are valid.
3.  **Log Errors**: specific error if sanitation fails.
4.  **Convert**: Use `toCpp()` to convert verified C inputs to C++ objects.
5.  **Forward**: Call the corresponding `OdaiSdk` singleton method.
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
    return OdaiSdk::get_instance().create_feature(toCpp(*c_config));
}
```

#### Step 2: C++ SDK Layer (`src/impl/odai_sdk.cpp`)
1.  **Receive Inputs**: Take C++ types.
2.  **Validate**: Call `config.is_sane()` for business logic validation (e.g., checking empty strings and valid ranges).
3.  **Check State**: Ensure SDK is initialized (`m_sdkInitialized`).
4.  **Execute**: Call internal engines (DB, RAG, etc.) or perform logic.
5.  **Return Rich Results Where Appropriate**: Prefer `OdaiResult<void>` for operation-style methods with no payload, `OdaiResult<T>` for methods returning data, and `OdaiResult<bool>` for state queries that can fail operationally. Reserve plain `bool` for true predicates such as `is_sane()` and similarly narrow helpers.
6.  **Handle Exceptions**: Wrap broad logic in `try-catch` blocks to prevent crashes.

**Example (`odai_sdk.cpp`):**
```cpp
bool OdaiSdk::create_feature(const FeatureConfig& config)
{
    try {
        if (!m_sdkInitialized) return false;

        // 1. Business Logic Validation
        if (!config.is_sane()) {
            ODAI_LOG(ODAI_LOG_ERROR, "Config missing required fields");
            return false;
        }

        // 2. Implementation forwarded to RagEngine
        return m_ragEngine->process(config);
    }
    catch (...) {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
        return false;
    }
}
```

### 1.2 Interface Design

When defining pure virtual interfaces in the C++ backend:
1.  **Naming**: Interfaces must be prefixed with `I` (e.g., `IAudioDecoder`, `IOdaiDb`). This is enforced by `clang-tidy`.
2.  **Lifecycle**: Interfaces must declare a `virtual ~IInterfaceName() = default;` destructor to ensure proper cleanup of derived classes.
3.  **Non-Copyable & Non-Movable**: Interfaces must explicitly delete copy and move constructors/assignments to prevent slicing and accidental copying of polymorphic objects.
4.  **Avoid `const` on Interface Methods**: Do not mark pure virtual methods in interfaces as `const` unless strictly necessary. This gives implementers the flexibility to mutate internal state (e.g., for caching, lazy loading, or maintaining connection state) during method execution without needing `mutable` members or breaking the interface contract.

**Example (`IExampleInterface.h`):**
```cpp
class IExampleInterface {
public:
    virtual ~IExampleInterface() = default;
    IExampleInterface() = default;
    
    // Non-copyable
    IExampleInterface(const IExampleInterface&) = delete;
    IExampleInterface& operator=(const IExampleInterface&) = delete;
    
    // Non-movable
    IExampleInterface(IExampleInterface&&) = delete;
    IExampleInterface& operator=(IExampleInterface&&) = delete;

    // Not marked as const to allow implementers to cache or mutate internal state
    virtual void do_something() = 0;
};
```

## 2. Key Directives

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
    - **C++ SDK**: Never let exceptions propagate across the C boundary. Legacy APIs may still return `bool`, but operation-style methods should prefer `OdaiResult` so callers can distinguish validation, not-found, not-initialized, and internal failures.
    - **Exception Guards**: Prefer explicit boundary-local `try`/`catch` blocks so `ODAI_LOG` continues to report the real caller `__func__` and `__LINE__`.
      - A small catch-tail macro from `src/include/utils/odai_exception_macros.h` is acceptable for repeated exception translation, but keep the main function body written normally.
      - Keep validation and output initialization outside any shared catch-tail macro so invalid-argument paths remain explicit.
      - The shared macro should use caller-site `__func__` directly; do not require a separate context string unless the function name alone is ambiguous.
      - Standard fallback mapping remains:
        - `c_OdaiResult` -> `ODAI_INTERNAL_ERROR`
        - `int32_t` streaming APIs -> `-1`
        - `bool` -> `false`
        - `OdaiResult<T>` -> `unexpected_internal_error()`
        - `void` -> log and return
      - If a function needs additional exception-side cleanup, keep the full `catch` local instead of hiding cleanup behind a shared macro.
    - **`OdaiResult<T>` Usage** (defined in `src/include/types/odai_result.h`):
      - `OdaiResult<T>` is a type alias for `tl::expected<T, OdaiResultEnum>`. It either holds a success value of type `T` or an `OdaiResultEnum` error code.
      - **Error codes** (`OdaiResultEnum`): `ALREADY_EXISTS`, `NOT_FOUND`, `VALIDATION_FAILED`, `INVALID_ARGUMENT`, `INTERNAL_ERROR`, `NOT_INITIALIZED`.
      - **Returning success**: Return the value directly — `return some_value;` (implicit conversion).
      - **Returning failure**: Use the helpers `unexpected_internal_error()` or `unexpected_not_initialized()`, or construct manually — `return tl::unexpected(OdaiResultEnum::NOT_FOUND);`.
      - **Crossing the C boundary**: Use `to_c_result(error_code)` to cast `OdaiResultEnum` to `c_OdaiResult`.
      - **When to use which return type**: Prefer `OdaiResult<void>` for operations with no payload, `OdaiResult<T>` for methods returning data, `OdaiResult<bool>` for state queries that can fail operationally. Reserve plain `bool` for true predicates like `is_sane()`.

## 3. Naming Conventions

### 3.1 General
- **Variables & Functions**: Use `snake_case`.
    - *Reasoning*: Matches the stable C API style and standard C++ STL conventions.
    - *Examples*: `db_path`, `odai_register_model`, `is_sane`, `user_data`.
    - *Incorrect*: `dbPath`, `registerModel`, `userData`.
- **Types (Classes, Structs, Enums, Typedefs)**: Use `PascalCase`.
    - *Examples*: `ChatConfig`, `BackendEngineInternal`, `OAIModelType`.
- **Interfaces**: Abstract classes must be prefixed with `I` followed by `PascalCase`. Enforced by `clang-tidy` (`readability-identifier-naming.AbstractClassPrefix`).
    - *Examples*: `IAudioDecoder`, `IOdaiDb`.
- **Constants & Macros**: Use `UPPER_SNAKE_CASE`.
    - *Examples*: `ODAI_MAX_PATH`, `DEFAULT_CHUNK_SIZE`.

### 3.2 Member Variables
- **Classes**: Prefix with `m_` followed by `camelBack`. *Examples*: `m_dbPath`, `m_isInitialized`.
- **Structs**: Use `camelBack` (no prefix). *Examples*: `width`, `creationDate`.

## 4. Code Style & Conventions

### 4.1 Code Formatting
Enforced by `clang-format` (`.clang-format`):
- **Indentation**: 2 spaces (no tabs).
- **Line Length**: 120 columns.
- **Braces**: Allman style (break before braces).
    ```cpp
    void function()
    {
        if (condition)
        {
            // code
        }
    }
    ```
- **Includes**: Sorted case-sensitively.

### 4.2 File Headers
- **Pragma Once**: Use `#pragma once` at the top of all header files.

### 4.3 Documentation & Comments

#### Comment Styles
- **Doxygen**: Use `///` (triple slash) for documentation comments on public APIs (classes, methods, functions). Do not use `//` or `/* */` for doc comments.
- **Implementation**: Use `//` for logic explanations inside functions.

#### Function Documentation

The first line of a doc comment should be a concise, one-line summary. Add additional lines after for behavior details, edge cases, or important notes.

**Required elements:**
1.  **Brief description**: Start with a clear, concise description of the function's purpose.
2.  **Parameter documentation**: Use `@param` tags for each parameter:
    - Format: `/// @param parameter_name Description of the parameter`
    - Include type information if not obvious from the signature.
    - Mention if the parameter is modified in place.
    - Note if the parameter is optional or has special constraints.
3.  **Return value documentation**: Use `@return` tag:
    - Format: `/// @return Description of return value`
    - Always mention error conditions (e.g., "or -1 on error", "or nullptr on error", "or empty vector on error").
    - Specify what the return value represents.

**Additional guidelines:**
- Mention edge cases (e.g., "If the same model is already loaded, only updates the configuration").
- Note side effects and in-place mutations.
- Document ownership clearly — if a function allocates memory or returns heap-backed data, state who frees it and which matching free function to call.
- Mark reserved parameters as "currently unused, reserved for future use".
- Mark unimplemented functions with "ToDo: Implementation not yet defined."
- Avoid exposing implementation details — focus on what the function does from the user's perspective.

**Checklist** (verify before completing documentation):
- [ ] Brief description on first line
- [ ] All parameters documented with `@param`
- [ ] Return value documented with `@return` including error conditions
- [ ] Edge cases and special behaviors mentioned
- [ ] Side effects documented if any
- [ ] Ownership/allocation rules documented if the API returns or mutates heap-backed data
- [ ] Complex behavior explained in additional lines

### 4.4 Code Style Tooling

Style is enforced via git pre-commit hook. Scripts are in `scripts/`:
- **`format.sh`** - Format code using clang-format. Accepts files or directories as arguments (e.g. `format.sh src/ tests/`). Covers both production and test code.
- **`lint.sh`** - Enforce naming conventions using `run-clang-tidy` in parallel. Accepts directories as arguments (e.g. `lint.sh src/`). The script regenerates `build/compile_commands.json` with the `linux-default-release` preset and symlinks it at the repo root before linting.

> **Maintenance**: When updating `format.sh` or `lint.sh`, also update this guideline if behavior changes.

### 4.5 Header-Only Libraries

For STB-style or similar header-only libraries, keep implementation macros in a dedicated translation unit under `src/impl/headerOnlyLib/`.

- Feature code should include the plain library headers only.
- Each header-only library implementation must be compiled exactly once.
- This keeps generated implementation code out of feature files, prevents duplicate symbols, and reduces rebuild noise.

Current examples:
- `src/impl/headerOnlyLib/odai_miniaudio_impl.cpp`
- `src/impl/headerOnlyLib/odai_stb_image_impl.cpp`

## 5. Header Management

### 5.1 Prefer Forward Declarations
Whenever a header file only needs to know that a class exists (e.g., for pointers, references, or `std::unique_ptr`), use a forward declaration instead of including the full header.

```cpp
class IOdaiDb; // Correct: Forward declaration
```

### 5.2 Minimize Include Surface
- Only include what is strictly necessary for the header to compile independently.
- Move feature-specific includes to the `.cpp` file.
- Use the Pimpl pattern or abstract interfaces for complex internal components.

### 5.3 Macro/Singleton Decoupling
Avoid direct dependencies on the singleton `OdaiSdk` inside macros or common headers. Use bridge functions (like `GetOdaiLogger()`) declared in low-level headers and implemented in the SDK layer to provide access to global services.
