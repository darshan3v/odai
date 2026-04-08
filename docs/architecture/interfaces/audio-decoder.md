# IOdaiAudioDecoder — Audio Decoder Interface

**Header**: [`src/include/audioEngine/odai_audio_decoder.h`](../../src/include/audioEngine/odai_audio_decoder.h)

## Purpose

Abstracts audio file decoding. Decodes audio inputs (file path or memory buffer) into raw float32 PCM samples matching a target specification (sample rate, channel count).

## Ownership

Stateless — the backend engine creates a fresh instance on demand via `OdaiSdk::get_new_odai_audio_decoder_instance()` within its `process_input_items()` method. The SDK provides the factory, but the engine is the actual consumer.

## Design Pattern: Template Method

The base class uses the Template Method pattern — `decode_to_spec()` handles input validation (sanity, media type, empty data), then delegates to the protected `do_decode_to_spec()` which subclasses implement. This ensures consistent validation across all decoder implementations.

## Current Implementation

- [OdaiMiniAudioDecoder (miniaudio)](../implementations/miniaudio-decoder.md)
