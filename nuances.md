# Development Nuances

This document outlines key technical reasoning, build system quirks, and "gotchas" encountered during the development of the ODAI SDK.

## Index
- [Build System (CMake)](#build-system-cmake)
    - [Third-Party Library Integration Quirks](#third-party-library-integration-quirks)
        - [Miniaudio (Header-Only STB-style)](#miniaudio-header-only-stb-style)
        - [Llama.cpp (Traditional C/C++ Library)](#llamacpp-traditional-c-c-library)

## Build System (CMake)

### Third-Party Library Integration Quirks

#### Miniaudio (Header-Only STB-style)
Miniaudio is a single-file header library (`miniaudio.h`). It requires a specific, two-part configuration when integrated via CMake's `FetchContent`.

1. **Before FetchContent (Target Configuration):**
   ```cmake
   set(MINIAUDIO_NO_LIBVORBIS ON CACHE BOOL "" FORCE)
   set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
   FetchContent_MakeAvailable(miniaudio)
   ```
   **Why:** Setting these variables *before* fetching prevents miniaudio's own `CMakeLists.txt` from spamming the `archives` output folder with standalone static libraries (`.a` files) for features we don't need (like the Vorbis or Opus decoders).

2. **After FetchContent (Compiler Configuration):**
   ```cmake
   target_compile_definitions(odai PRIVATE MA_NO_VORBIS)
   ```
   **Why:** Because miniaudio is implemented directly into our code via `#define MINIAUDIO_IMPLEMENTATION`, the actual C/C++ compiler needs to know to strip the Vorbis/Opus code out of the header file *while it is compiling our odai library target*. So we define these flags as compiler definitions for our odai target.

#### Llama.cpp (Traditional C/C++ Library)
Unlike miniaudio, `llama.cpp` builds as a traditional, separate library containing multiple `.cpp` and `.h` files.

* **Configuration:**
  ```cmake
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(llama)
  target_link_libraries(odai PRIVATE llama)
  ```
  **Why:** These flags instruct `llama.cpp`'s internal CMake script on *how* to compile its own `.cpp` files into a library. Once it is built, our `odai` library only needs to *link* to the resulting `llama` binary. We do not need to pass `target_compile_definitions` to our `odai` target for llama features, because our compiler is not compiling the llama source code directly—it is just pulling in the pre-compiled symbols.
