# IOdaiBackendEngine — Backend Engine Interface

**Header**: [`src/include/backendEngine/odai_backend_engine.h`](../../../src/include/backendEngine/odai_backend_engine.h)

## Purpose

Abstracts the LLM inference runtime. Any backend that can load models and generate tokens (llama.cpp, ONNX, MLC-LLM, etc.) implements this interface.

## Ownership

`OdaiRagEngine` owns a `std::unique_ptr<IOdaiBackendEngine>`. Created during SDK initialization and lives for the SDK's lifetime.

## Responsibilities

- **Hardware discovery** — detect available devices (GPU, iGPU, CPU) and select based on configured preferences.
- **Model validation** — verify that provided `ModelFiles` match what this engine expects (e.g. required file entries, correct engine type). The validation call now returns `OdaiResult<bool>` so callers can distinguish an invalid registration from an operational failure while checking it.
- **Streaming generation** — load models, generate tokens, stream output via callback. Supports both single-shot completion and chat-with-history modes, and returns `OdaiResult<StreamingStats>` so callers can distinguish cancellation from operational failure.

## Input Contract

Media items in prompts must be `FILE_PATH` type; text must be `MEMORY_BUFFER` type. The engine is responsible for decoding media items internally — it creates audio/image decoder instances on demand (via `OdaiSdk::get_new_odai_audio_decoder_instance()` / `get_new_odai_image_decoder_instance()`) and uses them to convert raw files into the format required by the underlying inference runtime.

## Config Types

- **`BackendEngineConfig`** — Engine type + preferred device type
- **`ModelFiles`** — Model type, engine type, and a `string→string` entries map for file paths
- **`LLMModelConfig`** — Model name plus requested context window (used for cache identity, placement validation, and load admission)
- **`SamplerConfig`** — `maxTokens`, `topP`, `topK`

## Design Notes

- Methods are intentionally **not `const`** — implementations may cache loaded models, maintain KV caches, or lazy-load backends.
- The engine manages its own model lifecycle (loading, unloading, caching between calls).
- Backends may define "loaded" state as more than just loaded weights. The current llama.cpp backend also preallocates one reusable generation context during load so request-time failures happen at the reload boundary instead of the first generation call.
- The engine is responsible for its own media decoding — it internally creates and uses `IOdaiAudioDecoder` / `IOdaiImageDecoder` instances to process multimodal inputs before feeding them to the inference runtime.

## Current Implementation

- [OdaiLlamaEngine (llama.cpp)](../implementations/llamacpp-backend.md)
