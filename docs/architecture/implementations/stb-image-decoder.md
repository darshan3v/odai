# OdaiStbImageDecoder — stb_image Image Decoder Implementation

**Interface**: [`IOdaiImageDecoder`](../interfaces/image-decoder.md)  
**Header**: [`src/include/imageEngine/odai_stb_image_decoder.h`](../../src/include/imageEngine/odai_stb_image_decoder.h)  
**Implementation**: `src/impl/imageEngine/odai_stb_image_decoder.cpp`  
**CMake Guard**: `ODAI_ENABLE_STB_IMAGE`

## Overview

Uses [stb_image](https://github.com/nothings/stb) (single-file header library) to decode image files into raw pixel buffers. Supports PNG, JPG, JPEG, BMP, TGA, GIF, HDR, PIC, and PNM formats.

## Build Integration

Same [dedicated implementation file pattern](../../nuances.md#best-practice-dedicated-implementation-file-header-only) as miniaudio:

- **Implementation file**: `src/impl/headerOnlyLib/odai_stb_image_impl.cpp`
- Feature code includes `stb_image.h` normally

## Design Notes

- Stateless — each decode is self-contained
- Handles both `FILE_PATH` and `MEMORY_BUFFER` input types
- Output: uint8 per channel; respects `m_maxWidth`/`m_maxHeight` constraints with aspect ratio preservation
