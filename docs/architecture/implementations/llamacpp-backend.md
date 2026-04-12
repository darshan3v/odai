# OdaiLlamaEngine — llama.cpp Backend Implementation

**Interface**: [`IOdaiBackendEngine`](../interfaces/backend-engine.md)  
**Header**: [`src/include/backendEngine/odai_llamacpp/odai_llama_backend_engine.h`](../../src/include/backendEngine/odai_llamacpp/odai_llama_backend_engine.h)  
**Implementation**: `src/impl/backendEngine/odai_llamacpp/`  
**CMake Guard**: `ODAI_ENABLE_LLAMA_BACKEND`

## Overview

Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM inference and the experimental `mtmd` (multimodal) API for image and audio inputs. Supports decoder-only LLMs in GGUF format.

## Build Integration

llama.cpp is fetched via CMake `FetchContent` and built as a **traditional separate library** (not header-only). ODAI only links against the resulting `llama` target — no `target_compile_definitions` needed on `odai`. See [nuances.md](../../nuances.md) for details.

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

Platform-specific notes:

- **Desktop AUTO** does not run a second iGPU-only pass after GPU probing fails
- **Android GPU probing** accepts Vulkan devices that ggml reports as `IGPU`, because that still represents the accelerated SoC path ODAI wants for `AUTO`/`GPU`
- Discovery logs both the configured probe order and the per-backend probe result counts so later placement behavior can be explained without mixing discovery into load heuristics

> Backend libraries are **never unloaded** during the application lifecycle — doing so risks driver-level instability.

## Model Caching

The engine caches the currently loaded LLM and embedding model (`m_llmModel`, `m_embeddingModel`). If a generation call requests the same model that's already loaded, it skips reloading.

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

ODAI expects applications to trigger that cleanup through explicit SDK lifecycle control (`OdaiSdk::shutdown()` / `odai_shutdown()`) instead of relying on late singleton destruction during process teardown.

## Known Limitations

- Only supports **decoder-only** LLMs (no encoder-decoder models)
- Context window is currently hardcoded to 4096 tokens
- Reasoning tokens are not handled separately from normal tokens
- The `mtmd` API for multimodal is experimental
