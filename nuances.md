# Development Nuances

This document outlines key technical reasoning, build system quirks, and "gotchas" encountered during the development of the ODAI SDK.
It is intentionally for non-obvious rationale and workaround-heavy behavior, not for restating the stable architecture already documented under `docs/architecture/`.

## Index
- [Build System (CMake)](#build-system-cmake)
    - [Third-Party Library Integration Quirks](#third-party-library-integration-quirks)
        - [Miniaudio (Header-Only STB-style)](#miniaudio-header-only-stb-style)
        - [Llama.cpp (Traditional C/C++ Library)](#llamacpp-traditional-cc-library)
        - [Best Practice: Dedicated Implementation File (Header-Only)](#best-practice-dedicated-implementation-file-header-only)
        - [Duplicate Symbol Errors (Header-Only Libraries)](#duplicate-symbol-errors-header-only-libraries)
- [Runtime Behavior](#runtime-behavior)
    - [Explicit SDK Shutdown for Deterministic Backend Cleanup](#explicit-sdk-shutdown-for-deterministic-backend-cleanup)
    - [llama.cpp Vulkan Device Type on Android and Other Integrated GPUs](#llamacpp-vulkan-device-type-on-android-and-other-integrated-gpus)
    - [Desktop GPU Placement Uses Different dGPU and iGPU Rules](#desktop-gpu-placement-uses-different-dgpu-and-igpu-rules)
    - [LLM Reload Releases Old State Before Retry and Commit](#llm-reload-releases-old-state-before-retry-and-commit)
    - [llama.cpp Load Failures Must Collapse to Return Paths for Fallback](#llamacpp-load-failures-must-collapse-to-return-paths-for-fallback)

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

#### Best Practice: Dedicated Implementation File (Header-Only)
The preferred way to handle header-only libraries (STB-style) in this SDK is to use a **dedicated implementation file**. Instead of defining the implementation macro inside decoder or engine files, create a separate file in `src/impl/headerOnlyLib/`.

*   **Examples:**
    * `src/impl/headerOnlyLib/odai_miniaudio_impl.cpp`
    * `src/impl/headerOnlyLib/odai_stb_image_impl.cpp`
*   **Example (`src/impl/headerOnlyLib/odai_miniaudio_impl.cpp`):**
    ```cpp
    #define MINIAUDIO_IMPLEMENTATION
    #define MA_API static
    #define ma_atomic_global_lock odai_ma_atomic_global_lock
    #include "miniaudio.h"
    ```
*   **Why:**
    *   **Encapsulation:** The heavy implementation code is compiled once in its own translation unit.
    *   **Prevention of Duplicate Symbols:** Since the implementation exists in only one `.cpp` file, you won't accidentally generate multiple copies if you include the header elsewhere.
    *   **Faster Rebuilds:** Modifying your functional logic (e.g., `odai_miniaudio_decoder.cpp`) doesn't require the compiler to re-process the massive header implementation.
    *   **Cleaner Feature Code:** Decoder and engine files can include the library normally without carrying macro-heavy setup.

#### Duplicate Symbol Errors (Header-Only Libraries)
When linking `libmtmd.a` and integrating `miniaudio.h` into your main codebase (`odai`), you might encounter duplicate symbol linker errors if both libraries try to compile the `MINIAUDIO_IMPLEMENTATION` or if one compiles it globally and the other tries to link to it.

> [!TIP]
> Use the [Dedicated Implementation File](#best-practice-dedicated-implementation-file-header-only) approach mentioned above as the primary fix.

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

## Runtime Behavior

### Explicit SDK Shutdown for Deterministic Backend Cleanup
ODAI's singleton lifetime is not a safe place to rely on backend teardown, especially once GPU-backed model state exists.

* **What to do:** Applications should call `OdaiSdk::shutdown()` or `odai_shutdown()` during normal control flow before process exit or before reinitializing the SDK.
* **Why this matters:** Resetting SDK-owned orchestrator state explicitly gives deterministic release of backend-owned resources without depending on late global destruction order.
* **Observed failure mode before explicit shutdown:** `OdaiSdk` is a static singleton, so without an explicit shutdown path its owned runtime state was being torn down at the very end of process exit. In GPU-backed runs, model destruction could then race backend / driver unload during process teardown, which led to driver-level errors while freeing model resources.
* **Current scope:** The shutdown path clears the SDK's owned runtime state by resetting the RAG engine and SDK lifecycle members. More backend-specific teardown can be layered on top later without changing the public lifecycle direction.

### llama.cpp Vulkan Device Type on Android and Other Integrated GPUs
When ODAI uses llama.cpp / ggml's Vulkan backend, the backend family name (`"vulkan"`) and the device class (`GPU` vs `IGPU`) are **not** the same thing.

* **What ggml does:** In the local llama.cpp source (`build/_deps/llama-src/ggml/src/ggml-vulkan/ggml-vulkan.cpp`), Vulkan devices report `GGML_BACKEND_DEVICE_TYPE_IGPU` when Vulkan exposes the physical device as an integrated GPU, otherwise they report `GGML_BACKEND_DEVICE_TYPE_GPU`.
* **Why this matters for ODAI:** On Android, Vulkan is usually the compute API for the SoC's integrated GPU. That means probing `"vulkan"` with an ODAI target type of `GPU` can miss a valid Android Vulkan device if ggml classifies it as `IGPU`.
* **Design implication:** Treat Vulkan as a backend family only. Device-selection policy should rely on ggml's reported device type, and Android-specific selection logic should decide whether a Vulkan iGPU satisfies a user request for "GPU acceleration" or should remain distinct from explicit `IGPU`.

### Desktop GPU Placement Uses Different dGPU and iGPU Rules
Desktop accelerated placement should not treat discrete GPUs and integrated GPUs as the same kind of target.

* **dGPU rule:** Query fresh free VRAM, then greedily pick the minimum discrete-GPU subset whose combined free VRAM satisfies the estimate. If ODAI cannot prove a full fit, it still passes all discovered dGPUs to llama.cpp so layer offload can remain best-effort instead of becoming all-or-nothing.
* **iGPU rule:** Require a full-fit check against the integrated GPU's fresh free-memory reading. If that check fails, or the free-memory reading is unavailable, ODAI falls back to an explicit CPU-only plan.
* **Why the split exists:** Partial offload across dGPUs is still useful when a full fit is uncertain, but integrated GPUs are much more likely to compete with CPU/system memory and to regress badly when ODAI guesses wrong. Treating iGPU placement as a gate keeps CPU fallback explicit instead of silently leaving llama.cpp on an accelerator that did not actually have enough headroom.

### LLM Reload Releases Old State Before Retry and Commit
LLM reload is intentionally not an "old model stays live until new model fully succeeds" transaction.

* **Why the old state is released first:** GPU-backed models can consume enough VRAM that attempting to hold both the old and new models at once makes reload failures nondeterministic. Releasing the old state first gives the new plan a clean memory picture and makes CPU retry meaningful instead of competing with stale allocations.
* **Why cached config/files are also cleared before retry:** If the new load fails after the old state was released, keeping the previous `m_llmModelFiles` / `m_llmModelConfig` would make future equality checks think a model was still loaded when the engine was actually empty.

### llama.cpp Load Failures Must Collapse to Return Paths for Fallback
ODAI's CPU-retry logic only works if accelerated-load failures stay inside the normal `bool`/result flow.

* **Why local catch blocks are needed:** `llama_model_load_from_file()` does catch some internal `std::exception` cases in upstream `llama.cpp`, but not the entire wrapper path around model construction, device-list setup, or every downstream helper. `mtmd_init_from_file()` is also outside ODAI's control. If either load step throws past ODAI, the accelerated path can bypass the planned CPU retry entirely.
* **Implementation rule:** Keep exception guards directly around ODAI's load attempts and translate any thrown failure into a logged `false`/error result. That preserves fallback behavior and keeps exceptions from leaking toward higher layers or the C API boundary.
