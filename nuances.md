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
    - [Single-Accelerator Placement Keeps a Shared-Memory Reserve](#single-accelerator-placement-keeps-a-shared-memory-reserve)
    - [LLM Planner Rejects Context Shrink And May Disable KQV Offload](#llm-planner-rejects-context-shrink-and-may-disable-kqv-offload)
    - [LLM Reload Releases Old State Before Retry and Commit](#llm-reload-releases-old-state-before-retry-and-commit)
    - [mmproj Placement Rechecks Fresh Memory But Still Reuses The Base Model's Main GPU](#mmproj-placement-rechecks-fresh-memory-but-still-reuses-the-base-models-main-gpu)
    - [Experimental mmproj Flow Stays Behind Its Own Helper Boundary](#experimental-mmproj-flow-stays-behind-its-own-helper-boundary)
    - [mtmd Does Not Own A Separate Rolling Context Window](#mtmd-does-not-own-a-separate-rolling-context-window)
    - [LLM Load Preallocates One Reusable Context](#llm-load-preallocates-one-reusable-context)
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
   target_compile_definitions(odai PRIVATE
       MA_NO_VORBIS MA_NO_OPUS MA_NO_DEVICE_IO MA_NO_RESOURCE_MANAGER
       MA_NO_NODE_GRAPH MA_NO_ENGINE MA_NO_GENERATION)
   ```
   **Why:** Because miniaudio is implemented directly into our code via `#define MINIAUDIO_IMPLEMENTATION`, the actual C/C++ compiler needs to see the same feature-pruning defines while compiling the `odai` target. In practice ODAI strips the unused codec, device-I/O, resource-manager, node-graph, engine, and generation paths at compile time.

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
    #define ma_atomic_global_lock odai_ma_atomic_global_lock
    #include "miniaudio.h"
    #undef ma_atomic_global_lock
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
*   **Fix:** Keep one dedicated implementation translation unit for miniaudio and rename the unprefixed global lock symbol so ODAI's copy does not collide with other bundled code that may also embed miniaudio.
    ```cpp
    #define MINIAUDIO_IMPLEMENTATION
    // Rename the leaky global lock symbol that miniaudio leaves unprefixed.
    #define ma_atomic_global_lock odai_ma_atomic_global_lock
    #include "miniaudio.h"
    #undef ma_atomic_global_lock
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

### Single-Accelerator Placement Keeps a Shared-Memory Reserve
ODAI now prefers "as much accelerator offload as possible" across platform/device classes, but single-accelerator targets that share memory with the CPU still keep a reserve.

* **Behavior summary:** The exact shared-memory placement flow lives in [`docs/architecture/implementations/llamacpp-backend.md`](docs/architecture/implementations/llamacpp-backend.md). The non-obvious part is that ODAI intentionally keeps a fixed 2 GB reserve before allowing llama.cpp fit planning to decide the final offload level.
* **Why the reserve remains:** Shared-memory accelerators compete directly with the CPU and the rest of the process. The reserve prevents ODAI from consuming the last visible headroom just because llama.cpp can technically attempt partial offload.
* **Why missing telemetry is still a hard block here:** Shared-memory placement is intentionally reserve-aware. If ODAI cannot observe a fresh free-memory reading, it cannot prove that the reserve still exists, so it stays on CPU instead of guessing.

### LLM Planner Rejects Context Shrink And May Disable KQV Offload
ODAI now uses `llama_params_fit()` during accelerated LLM planning, but it treats a reduced context window as a rejected accelerated plan rather than a successful fit.

* **Behavior summary:** The architecture doc describes the planner flow. The non-obvious policy choices are that ODAI rejects any accelerated fit that shrinks `n_ctx`, and it retries once with `offload_kqv = false` before giving up on acceleration.
* **Why this matters:** The requested context window is part of the runtime contract and cache identity. Quietly accepting a smaller fitted context would make the loaded runtime differ from the caller's requested model configuration.
* **Why the KQV retry exists:** Moving KV/cache work off the accelerator can still preserve useful model-weight acceleration without changing the requested runtime contract.
* **Why this is still a planner decision:** Disabling KQV offload is a placement-policy fallback, not a semantic change to the requested model configuration. If neither accelerated variant preserves the full requested context, ODAI materializes an explicit CPU plan instead of shrinking context.

### LLM Reload Releases Old State Before Retry and Commit
LLM reload is intentionally not an "old model stays live until new model fully succeeds" transaction.

* **Why the old state is released first:** GPU-backed models can consume enough VRAM that attempting to hold both the old and new models at once makes reload failures nondeterministic. Releasing the old state first gives the new plan a clean memory picture and makes CPU retry meaningful instead of competing with stale allocations.
* **Why cached config/files are also cleared before retry:** If the new load fails after the old state was released, keeping the previous loaded-state metadata would make future equality checks think a model was still loaded when the engine was actually empty.

### mmproj Placement Rechecks Fresh Memory But Still Reuses The Base Model's Main GPU
ODAI treats the multimodal projector as a second admission decision, but not as an independent device-placement system.

* **Behavior summary:** The stable runtime flow lives in [`docs/architecture/implementations/llamacpp-backend.md`](docs/architecture/implementations/llamacpp-backend.md). The non-obvious rule is that projector loading re-queries free memory only after the base LLM is already loaded, then decides whether `mtmd` may reuse the base model's existing `main_gpu` path or should fall back to CPU.
* **Why this is narrower than the base LLM planner:** `mtmd_init_from_file()` only gets `use_gpu` and the already-loaded `llama_model`. ODAI does not get a separate device list or layer-split control for the projector, so it cannot move the projector to a different accelerator when the base model is on CPU or on some other GPU path.
* **Why fresh memory still matters:** Even though the projector cannot pick a new device, it still allocates after the base LLM is resident. Rechecking current free memory avoids assuming the remaining headroom from pre-LLM estimates.
* **Why shared-memory targets still keep the reserve:** When the reused path is a shared-memory target, the projector admission step preserves the same 2 GB reserve used by single-accelerator LLM planning so the smaller projector does not consume the last visible shared-memory cushion after the base model is already resident.

### Experimental mmproj Flow Stays Behind Its Own Helper Boundary
ODAI keeps the optional projector branch behind a dedicated helper instead of mixing projector retries directly into the main LLM reload body.

* **Why this split exists:** `mtmd` is still experimental, and its projector placement rules are narrower than normal llama.cpp placement. Treating it as an add-on step reduces the chance that projector-specific retry logic obscures or destabilizes the normal LLM path.
* **What should stay out of the base path:** Projector file discovery, post-LLM memory rechecks, and projector CPU retries belong in the helper boundary unless they become part of the core backend contract later.

### mtmd Does Not Own A Separate Rolling Context Window
The current `mtmd` integration can look like it has its own runtime context because ODAI loads a persistent `mtmd_context`, but request-time sequence state still lives in the normal `llama_context`.

* **Why there is no separate `n_ctx` policy for the projector today:** `mtmd_context_params` does not expose an independent context-window field. ODAI can tune image token bounds and GPU usage there, but the actual request context limit is still governed by the loaded `llama_context`.
* **Practical implication:** Any multimodal prompt that expands into too many text+media tokens still consumes the same LLM context budget. A future design could add preflight token budgeting, but that would still be enforcing the LLM context window rather than a second mtmd-specific window.

### LLM Load Preallocates One Reusable Context
ODAI now treats a loaded LLM as "usable" only when the model load also succeeds in creating one reusable `llama_context`.

* **Why load-time context allocation matters:** It moves context-allocation failures to the reload boundary instead of the first request, so cache hits represent a runtime that can actually serve generation immediately.
* **Why only one reusable context exists right now:** The current ownership shape keeps the failure boundary correct without adding a context pool or extra synchronization rules. That also means one backend instance effectively supports one in-flight LLM generation at a time until a later multi-context design lands.

### llama.cpp Load Failures Must Collapse to Return Paths for Fallback
ODAI's CPU-retry logic only works if accelerated-load failures stay inside the normal `bool`/result flow.

* **Why local catch blocks are needed:** `llama_model_load_from_file()` does catch some internal `std::exception` cases in upstream `llama.cpp`, but not the entire wrapper path around model construction, device-list setup, or every downstream helper. `mtmd_init_from_file()` is also outside ODAI's control. If either load step throws past ODAI, the accelerated path can bypass the planned CPU retry entirely.
* **Implementation rule:** Keep exception guards directly around ODAI's load attempts and translate any thrown failure into a logged `false`/error result. That preserves fallback behavior and keeps exceptions from leaking toward higher layers or the C API boundary.
