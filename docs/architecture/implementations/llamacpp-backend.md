# OdaiLlamaEngine — llama.cpp Backend Implementation

**Interface**: [`IOdaiBackendEngine`](../interfaces/backend-engine.md)  
**Header**: [`src/include/backendEngine/odai_llamacpp/odai_llama_backend_engine.h`](../../src/include/backendEngine/odai_llamacpp/odai_llama_backend_engine.h)  
**Implementation**: `src/impl/backendEngine/odai_llamacpp/`  
**CMake Guard**: `ODAI_ENABLE_LLAMA_BACKEND`

## Overview

Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM inference and the experimental `mtmd` (multimodal) API for image and audio inputs. Supports decoder-only LLMs in GGUF format.

## Hardware Discovery

During `initialize_engine()`, the engine discovers available hardware using GGML's backend system. Backend registration is done through `ggml_backend_load_all_from_path()` so ggml can score and load the best variant of each available backend family from the runtime backend directory.

Discovery is split into two internal concerns:

- **Device inventory**: store each accepted candidate as both public `BackendDevice` info and the private `ggml_backend_dev_t` handle needed by later placement and load decisions
- **Probe policy**: decide which backend families to query, in what order, for the requested device class

Current discovery policy is:

- **AUTO**: probe prioritized GPU backends, then fall back directly to CPU
- **GPU**: probe prioritized GPU backends only
- **IGPU**: probe prioritized iGPU backends only
- **CPU**: probe the CPU backend directly

Notes:

- **Desktop AUTO** does not run a second iGPU-only pass after GPU probing fails
- Discovery logs both the configured probe order and the per-backend probe result counts so later placement behavior can be explained without mixing discovery into load heuristics
- The Vulkan-on-Android device-class nuance is documented in [`nuances.md`](../../nuances.md#llamacpp-vulkan-device-type-on-android-and-other-integrated-gpus)

## LLM Placement Planning

Base LLM loading now runs through an internal `LlmLoadPlan` before calling `llama_model_load_from_file()`. The planner keeps placement policy separate from load execution and records the reason for the selected placement.

Current planning policy is:

- **Explicit CPU**: materialize a CPU-only plan directly
- **Apple / Android unified-memory platforms**: use the first accelerated candidate for full offload
- **Desktop dGPU**: query fresh free VRAM, choose the minimum fitting subset when possible, otherwise pass all discovered dGPUs for best-effort partial offload
- **Desktop iGPU**: require a full-fit free-memory check, otherwise fall back to CPU
- **No accelerator candidate**: materialize an explicit CPU-only plan instead of relying on llama.cpp defaults

The plan currently carries:

- placement mode (`CPU_ONLY`, `ACCELERATED_FULL`, `ACCELERATED_PARTIAL`)
- selected candidate-device indices
- `n_gpu_layers`, `split_mode`, `use_mmap`, and `use_mlock`
- a human-readable reason string used in runtime logs

The dGPU versus iGPU rationale for this split lives in [`nuances.md`](../../nuances.md#desktop-gpu-placement-uses-different-dgpu-and-igpu-rules).

## Load Execution and Reload

Base LLM loading now executes as an internal reload transaction instead of mutating live engine state inline.

Current execution behavior is:

- Convert `LlmLoadPlan` into transaction-local `llama_model_params`
- Materialize the NULL-terminated ggml device buffer only for the duration of the load call
- Release the previously committed LLM, vocab, and multimodal projector state before attempting the new load so GPU-backed allocations do not overlap
- Attempt the planned accelerated load first when applicable, then retry once with an explicit CPU-only plan if the planner allowed fallback
- Commit the newly loaded model, vocab, projector context, and cached config/files to engine state only after the transaction succeeds

This keeps the planner responsible for policy while the load step only executes one concrete plan at a time.

## Model Caching

The engine caches the currently loaded LLM state and embedding model. LLM cache state is held as one internal ownership unit containing the model, vocab, multimodal projector context, and the matching config/files. If a generation call requests the same model with the same config, it skips reloading.

### Expected Model Files

| Model Type | Required Entries | Optional Entries |
|---|---|---|
| **LLM** | `base_model_path` | `mmproj_model_path` (multimodal projector) |
| **Embedding** | `base_model_path` | _(none)_ |

## Multimodal Support

When `mmproj_model_path` is registered, the engine creates an `mtmd_context` for processing images and audio. During `process_input_items()`, the engine:

1. Creates audio/image decoder instances on-demand via `OdaiSdk::get_new_odai_audio_decoder_instance()` / `get_new_odai_image_decoder_instance()`
2. Decodes media files into raw pixel/PCM data using those decoders
3. Converts decoded data into `mtmd_bitmap` objects
4. The `mtmd` API then handles tokenizing text with media placeholders, encoding media into embeddings, and interleaving them in context

## Resource Management

All llama.cpp resources use `std::unique_ptr` with custom deleters (`LlamaModelDeleter`, `LlamaContextDeleter`, `LlamaSamplerDeleter`, `LlamaBatchDeleter`, `MtmdContextDeleter`) to ensure proper RAII cleanup.

Applications should release those resources through the normal SDK lifecycle (`OdaiSdk::shutdown()` / `odai_shutdown()`). The teardown rationale is documented in [`nuances.md`](../../nuances.md#explicit-sdk-shutdown-for-deterministic-backend-cleanup).

## Known Limitations

- Only supports **decoder-only** LLMs (no encoder-decoder models)
- Context window is currently hardcoded to 4096 tokens
- Reasoning tokens are not handled separately from normal tokens
- The `mtmd` API for multimodal is experimental
