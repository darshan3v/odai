# Development Nuances

This document outlines key technical reasoning, build system quirks, and "gotchas" encountered during the development of the ODAI SDK.

## Index
- [Build System (CMake)](#build-system-cmake)
    - [Third-Party Library Integration Quirks](#third-party-library-integration-quirks)
        - [Miniaudio (Header-Only STB-style)](#miniaudio-header-only-stb-style)
        - [Llama.cpp (Traditional C/C++ Library)](#llamacpp-traditional-cc-library)
        - [mtmd (Explicit Subdirectory with LLAMA_INSTALL_VERSION)](#mtmd-explicit-subdirectory-with-llama_install_version)
        - [Duplicate Symbol Errors (Header-Only Libraries)](#duplicate-symbol-errors-header-only-libraries)
- [mtmd API Quirks](#mtmd-api-quirks)
    - [mtmd_get_audio_bitrate Returns Sample Rate](#mtmd_get_audio_bitrate-returns-sample-rate)

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

#### mtmd (Explicit Subdirectory with LLAMA_INSTALL_VERSION)
The `mtmd` (multimodal) library lives inside `llama.cpp` at `tools/mtmd/`. We set `LLAMA_BUILD_TOOLS OFF` to avoid building all of llama's CLI tools, but this also skips `mtmd`. To include it, we explicitly add its subdirectory:

```cmake
if(NOT DEFINED LLAMA_INSTALL_VERSION)
    set(LLAMA_INSTALL_VERSION "0.0.0")
endif()
add_subdirectory(${llama_SOURCE_DIR}/tools/mtmd ${llama_BINARY_DIR}/tools/mtmd)
```

**Why `LLAMA_INSTALL_VERSION`?** mtmd's `CMakeLists.txt` uses `VERSION ${LLAMA_INSTALL_VERSION}` in `set_target_properties`. This variable is normally set by llama.cpp's root CMake, but since `LLAMA_BUILD_TOOLS` is OFF, the tools subdirectory is never processed by llama's own build — so the variable is undefined. An undefined variable causes CMake's `set_target_properties` to fail with "incorrect number of arguments" (empty string = missing arg). Setting it to `"0.0.0"` is safe because we build mtmd as a **static library** (`BUILD_SHARED_LIBS` is OFF at that point), and the `VERSION` property on a static lib is a no-op — it only matters for shared library versioned symlinks (`libfoo.so.X.Y.Z`), which don't apply here.

#### Duplicate Symbol Errors (Header-Only Libraries)
When linking `libmtmd.a` and integrating `miniaudio.h` into your main codebase (`odai`), you might encounter duplicate symbol linker errors if both libraries try to compile the `MINIAUDIO_IMPLEMENTATION` or if one compiles it globally and the other tries to link to it.

*   **Symptom:** Linker errors like `ld.lld: error: duplicate symbol: ma_atomic_global_lock` defined in multiple objects, or undefined references when trying to call miniaudio functions after removing the implementation macro.
*   **Reason:** Header-only libraries like `miniaudio.h` generate their implementation code when `#define MINIAUDIO_IMPLEMENTATION` is set. However, if multiple translation units generate it globally, the linker complains about duplicate symbols. If you remove the macro, you'll get undefined reference errors because the compiler doesn't have the function implementations available for linking in your translation unit.
*   **Fix:** Force the library to compile its implementation internally to your specific C++ file by making everything `static`. You also need to rename any improperly-prefixed global locks to avoid collisions.
    ```cpp
    #define MINIAUDIO_IMPLEMENTATION
    // This forces ALL miniaudio functions and global variables to be 'static'
    // (internal to this specific C++ file). It completely hides them from the linker.
    #define MA_API static
    // The hack to hide the leaky global variable that miniaudio forgot to prefix
    #define ma_atomic_global_lock odai_ma_atomic_global_lock
    #include "miniaudio.h"
    #undef MA_API
    ```

## mtmd API Quirks

### mtmd_get_audio_bitrate Returns Sample Rate
`mtmd_get_audio_bitrate()` is a **misnomer** in llama.cpp's mtmd API. Despite its name, it returns the **sample rate** (Hz), not bitrate (bits/sec):

```cpp
// From mtmd.cpp:
int mtmd_get_audio_bitrate(mtmd_context * ctx) {
    return clip_get_hparams(ctx->ctx_a)->audio_sample_rate;  // ← returns sample rate!
}
```

Our `OdaiAudioTargetSpec` uses the correct terminology (`sampleRate`), so we map from this misleadingly-named function to a correctly-named field. Be aware of the naming discrepancy when reading mtmd code.
