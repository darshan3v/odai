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

Base LLM loading now runs through an internal `LlmLoadPlan` plus planner-owned runtime-param preparation before calling `llama_model_load_from_file()`. The planner keeps placement policy separate from load execution and records the reason for the selected placement.

Current planning policy is:

- **Explicit CPU**: materialize a CPU-only plan directly
- **Apple / Android unified-memory platforms**: use the first accelerated candidate, require fresh memory data, preserve a 2 GB shared-memory reserve as a precondition, and then rely on `llama_params_fit()` to keep as much acceleration as possible while preserving the requested context window
- **Desktop dGPU**: rank discovered dGPUs by fresh free VRAM, then try the largest ranked prefix first and shrink toward smaller prefixes with `llama_params_fit()` until one preserves the requested context window; once a smaller prefix fails after that downward search begins, stop shrinking further and fall back explicitly to CPU
- **Desktop iGPU**: use the first integrated GPU, require fresh memory data, preserve a 2 GB shared-memory reserve as a precondition, and then rely on `llama_params_fit()` to keep as much acceleration as possible while preserving the requested context window
- **No accelerator candidate**: materialize an explicit CPU-only plan instead of relying on llama.cpp defaults

The prepared load state currently carries:

- placement mode (`CPU_ONLY`, `ACCELERATED_FULL`, `ACCELERATED_PARTIAL`)
- selected candidate-device indices
- `n_gpu_layers`, `split_mode`, `use_mmap`, and `use_mlock`
- prepared `llama_model_params` / `llama_context_params`
- a fit-buffer bundle for `tensor_split`, `tensor_buft_overrides`, and per-device margins
- a human-readable reason string used in runtime logs

Accelerated plans are now validated with `llama_params_fit()` against the requested `LLMModelConfig.m_contextWindow`.

- Accelerated placement is accepted only if fit preserves the exact requested context window
- the planner may retry accelerated placement with `offload_kqv = false` when that keeps the full requested context window better than the default accelerated settings
- if acceleration cannot satisfy the requested context window exactly, the planner materializes an explicit CPU-only load plan instead of silently shrinking context

The shared-memory reserve rationale for single-accelerator targets lives in
[`nuances.md`](../../nuances.md#single-accelerator-placement-keeps-a-shared-memory-reserve).

## Thread Policy Hook

llama context creation now goes through an internal thread-policy helper instead of setting raw literals inline at
each call site.

Current behavior is intentionally conservative:

- the helper emits both `n_threads` and `n_threads_batch`
- the baseline still returns fixed values for now
- load placement and thread policy are separated so later platform-specific tuning can change one without reshaping the
  other

## Load Execution and Reload

Base LLM loading executes as an internal reload transaction instead of mutating live engine state inline.

Current execution behavior is:

- Convert `LlmLoadPlan` into transaction-local `llama_model_params`
- Set `main_gpu` explicitly to the first entry in the prepared llama.cpp `devices` array whenever the base LLM load is accelerated instead of relying on the implicit first-device default
- Preallocate one reusable `llama_context` from the loaded model during the same transaction and verify that
  `llama_n_ctx()` matches the requested `LLMModelConfig.m_contextWindow`
- Materialize the NULL-terminated ggml device buffer only for the duration of the load call
- Release the previously committed LLM, vocab, reusable context, and multimodal projector state before attempting the new
  load so GPU-backed allocations do not overlap
- Attempt the planned accelerated load first when applicable, then retry once with an explicit CPU-only plan if the planner allowed fallback
- After the base LLM is loaded, run a separate multimodal projector admission step against fresh post-LLM free-memory data on the base model's existing `main_gpu` path
- Retry the projector once on CPU if an accelerated projector attempt fails
- Commit the newly loaded model, vocab, reusable context, projector context, and cached config/files to engine state only after the full transaction succeeds

The key internal helper boundaries are:

- `plan_llm_load()` chooses placement policy and produces planner-owned runtime params
- `try_load_language_model_for_plan()` executes one prepared base-LLM load into transaction-local state
- `load_optional_mmproj_into_state()` owns the optional experimental projector branch so normal LLM reload sequencing stays separate from `mtmd`
- `plan_mmproj_load()` chooses whether the multimodal projector may reuse the base model's `main_gpu` path from the fresh post-LLM memory picture
- `try_load_mmproj_for_plan()` executes one prepared projector load into transaction-local state
- `load_language_model()` owns reload sequencing, state clearing, optional CPU retries, projector planning against the plan that actually loaded the base LLM, and final commit

This keeps the planners responsible for policy while each load step only executes one concrete plan at a time.

## Request-Time Context Reuse

Prompt and chat generation share one internal request-context preparation helper.

Current request-time behavior is:

- Validate or load the requested LLM state first
- Acquire the preallocated reusable llama context from that loaded state
- Call `prepare_reusable_llm_context_for_request()`, which clears the reusable context through `clear_llm_context()`
- Replay stored chat history into that same context only for chat generation
- Return the prepared context to the thin request handler, which then formats the new prompt and starts generation

This keeps context clearing and history replay behind one internal boundary while preserving detailed request-time
errors from reusable-context reset and chat-history reconstruction.

## Model Caching

The engine caches the currently loaded LLM state and embedding model. LLM cache state is held as one internal ownership
unit containing the model, vocab, reusable context, multimodal projector context, and the matching config/files. If a
generation call requests the same model with the same config, including the requested context window, it skips
reloading.

### Expected Model Files

| Model Type | Required Entries | Optional Entries |
|---|---|---|
| **LLM** | `base_model_path` | `mmproj_model_path` (multimodal projector) |
| **Embedding** | `base_model_path` | _(none)_ |

## Multimodal Support

When `mmproj_model_path` is registered, the engine creates an `mtmd_context` for processing images and audio after the
base LLM load succeeds. Projector placement is a separate admission decision, but its control surface is narrower than
the base LLM's:

- Explicit `CPU` runtime preference forces CPU projector placement
- If the base LLM ended up on CPU, the projector also stays on CPU because `mtmd` only gets `use_gpu` plus the already-loaded `llama_model`; it does not pick an independent device
- If the base LLM is accelerated, the projector may reuse that same `main_gpu` path after a fresh post-LLM headroom check
- Shared-memory accelerator targets preserve the same 2 GB reserve before enabling accelerated projector placement on that reused `main_gpu` path
- Accelerated projector loads retry once on CPU if `mtmd_init_from_file()` fails
- `mtmd` does not expose an independent `n_ctx` knob in `mtmd_context_params`; request-time multimodal tokens are still evaluated into the regular reusable `llama_context`, so the active context window is the same `LLMModelConfig.m_contextWindow` already used for the base LLM

Ownership and request-time behavior are split:

- `mtmd_context` is long-lived loaded model state, similar to the loaded projector weights and capability metadata
- It is not a second rolling inference context and does not hold the chat/prompt KV state
- Request-time multimodal evaluation still happens against the normal reusable `llama_context` passed into `mtmd_helper_eval_chunks()`

During `process_input_items()`, the engine:

1. Creates audio/image decoder instances on-demand via `OdaiSdk::get_new_odai_audio_decoder_instance()` / `get_new_odai_image_decoder_instance()`
2. Decodes media files into raw pixel/PCM data using those decoders
3. Converts decoded data into `mtmd_bitmap` objects
4. The `mtmd` API then handles tokenizing text with media placeholders, encoding media into embeddings, and interleaving them in context

## Resource Management

All llama.cpp resources use `std::unique_ptr` with custom deleters (`LlamaModelDeleter`, `LlamaContextDeleter`, `LlamaSamplerDeleter`, `LlamaBatchDeleter`, `MtmdContextDeleter`) to ensure proper RAII cleanup.

Applications should release those resources through the normal SDK lifecycle (`OdaiSdk::shutdown()` / `odai_shutdown()`). The teardown rationale is documented in [`nuances.md`](../../nuances.md#explicit-sdk-shutdown-for-deterministic-backend-cleanup).

## Known Limitations

- Only supports **decoder-only** LLMs (no encoder-decoder models)
- The backend currently reuses one preallocated llama context per loaded LLM, so concurrent LLM generations need a later
  multi-context or pooled design instead of sharing one backend instance
- Reasoning tokens are not handled separately from normal tokens
- The `mtmd` API for multimodal is experimental
