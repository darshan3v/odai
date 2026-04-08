# OdaiMiniAudioDecoder — miniaudio Audio Decoder Implementation

**Interface**: [`IOdaiAudioDecoder`](../interfaces/audio-decoder.md)  
**Header**: [`src/include/audioEngine/odai_miniaudio_decoder.h`](../../src/include/audioEngine/odai_miniaudio_decoder.h)  
**Implementation**: `src/impl/audioEngine/odai_miniaudio_decoder.cpp`  
**CMake Guard**: `ODAI_ENABLE_MINIAUDIO`

## Overview

Uses [miniaudio](https://github.com/mackron/miniaudio) (single-file header library) to decode audio files into raw float32 PCM samples. Supports WAV, MP3, and FLAC formats.

## Build Integration

miniaudio is a **header-only STB-style library** using the [dedicated implementation file pattern](../../nuances.md#best-practice-dedicated-implementation-file-header-only):

- **Implementation file**: `src/impl/headerOnlyLib/odai_miniaudio_impl.cpp`
- Feature code includes `miniaudio.h` normally — no `#define MINIAUDIO_IMPLEMENTATION`

Two-phase CMake setup is required (see [nuances.md](../../nuances.md#miniaudio-header-only-stb-style)) — disable Vorbis/Opus before fetch, strip via `target_compile_definitions` after. Static linkage (`#define MA_API static`) prevents duplicate symbol collisions with `libmtmd.a`.

## Design Notes

- Stateless — each decode call creates and destroys its own miniaudio decoder instance
- Handles both `FILE_PATH` and `MEMORY_BUFFER` input types
- Output is always float32 PCM, resampled and channel-mixed to match the target spec
