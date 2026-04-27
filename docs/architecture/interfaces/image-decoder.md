# IOdaiImageDecoder — Image Decoder Interface

**Header**: [`src/include/imageEngine/odai_image_decoder.h`](../../../src/include/imageEngine/odai_image_decoder.h)

## Purpose

Abstracts image file decoding. Decodes image inputs (file path or memory buffer) into raw pixel buffers matching a target specification (max dimensions, channel count).

`is_supported()` accepts a file extension token without a leading dot and matches it case-insensitively.
`decode_to_spec()` accepts `InputItem` values whose MIME type has an `image/` prefix. Prefix matching is case-insensitive through `InputItem::get_media_type()`, and the original MIME string is not modified by that classification.

## Ownership

Stateless — the backend engine creates a fresh instance on demand via `OdaiSdk::get_new_odai_image_decoder_instance()` within its `process_input_items()` method. The SDK provides the factory, but the engine is the actual consumer.

## Design Pattern: Template Method

Same pattern as [`IOdaiAudioDecoder`](./audio-decoder.md) — base class `decode_to_spec()` validates input, media type, and non-empty data, then calls the protected `do_decode_to_spec()` implemented by subclasses. Both methods use `OdaiResult<void>` so image decode failures retain result-code detail.

## Current Implementation

- [OdaiStbImageDecoder (stb_image)](../implementations/stb-image-decoder.md)
