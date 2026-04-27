# Test Nuances

This document captures non-obvious testing rationale that does not belong in the stable test plan or architecture docs.

Use it for:
- why a test intentionally avoids exact assertions
- where third-party behavior makes a stricter assertion brittle
- what still needs manual verification because automation is impractical, environment-dependent, or too expensive

Do not use it for:
- basic test structure or routine commands
- stable architecture behavior
- generic testing advice that already belongs in `AGENTS.md` or the testing skill

## Index
- [Test Infrastructure](#test-infrastructure)
  - [Implementation Tests Need Both Feature Guards And Compile Definitions](#implementation-tests-need-both-feature-guards-and-compile-definitions)
  - [HDR Test Data Generation Requires imageio's OpenCV Plugin](#hdr-test-data-generation-requires-imageios-opencv-plugin)
  - [FLAC Test Data Generation Requires ffmpeg](#flac-test-data-generation-requires-ffmpeg)
- [Assertion Boundaries](#assertion-boundaries)
  - [Audio Decoder Compressed Fixtures Avoid Exact PCM And Frame Counts](#audio-decoder-compressed-fixtures-avoid-exact-pcm-and-frame-counts)
  - [Image Decoder Real-World Fixtures Avoid Exact Pixel Values](#image-decoder-real-world-fixtures-avoid-exact-pixel-values)
- [Manual Verification Register](#manual-verification-register)
  - [Current Manual Checks](#current-manual-checks)
  - [Audio Decoder Perceptual Sanity Check](#audio-decoder-perceptual-sanity-check)
  - [Image Decoder Visual Sanity Check](#image-decoder-visual-sanity-check)

## Test Infrastructure

### Implementation Tests Need Both Feature Guards And Compile Definitions
Concrete implementation headers are hidden behind the same `ODAI_ENABLE_*` macros that control implementation compilation.

Why:
- a test target can be correctly guarded in CMake but still fail to compile if the target itself does not receive the matching compile definition
- the header will be included, but the concrete class declaration will be skipped by the preprocessor

Rule:
- put each implementation-specific test target, and its matching contract instantiation target, under the implementation's CMake feature guard
- also pass the matching `ODAI_ENABLE_*` compile definition to those test targets when they include macro-gated concrete implementation headers

### HDR Test Data Generation Requires imageio's OpenCV Plugin
The Radiance HDR fixture is the reason the build-local test venv needs `opencv-python-headless` in addition to `imageio` and Pillow.

Why:
- recent plain `imageio` environments may not include an HDR-capable writer
- the `imageio[opencv]` extra is not provided by the package version currently resolved in the test venv, so the OpenCV backend package must be listed explicitly
- `imageio` uses Pillow as the backend for many common image formats, and PSD generation still creates a Pillow image before handing it to `psd-tools`

Rule:
- keep the optional backend dependency explicit in `tests/CMakeLists.txt` so HDR generation stays library-owned
- if a new generator dependency is added, update `tests/CMakeLists.txt` and `docs/architecture/testing.md` together

### FLAC Test Data Generation Requires ffmpeg
The generated FLAC fixture is exported through `pydub`, which shells out to `ffmpeg`.

Why:
- `pydub` can synthesize the audio segment in Python, but compressed/container exports are delegated to external encoder tools
- Python package installation alone is not enough to guarantee FLAC generation works on a clean machine
- failing during test-data generation hides the real environment requirement behind a later missing-fixture or decoder error

Rule:
- keep the configure-time `ffmpeg` check in `tests/CMakeLists.txt`
- if FLAC generation moves to a pure library-owned path, remove the configure check and this nuance together

## Assertion Boundaries

### Audio Decoder Compressed Fixtures Avoid Exact PCM And Frame Counts
Tests for bundled audio fixtures should assert the decoder contract that ODAI owns:
- decode succeeds
- output sample rate matches the requested target spec
- output channel count matches the requested target spec
- PCM buffer is non-empty, frame-aligned, and finite
- decoded content is meaningfully non-silent when the fixture is known to contain real audio

They should usually avoid exact sample-by-sample PCM comparisons and exact frame-count assertions for compressed fixtures such as MP3.

Why:
- compressed formats can include encoder delay and end padding
- miniaudio may expose slightly different decoded lengths depending on codec/container behavior
- exact PCM values after decode/resample/channel conversion are partly controlled by third-party library internals, so exact-value assertions become brittle without improving contract coverage

For generated uncompressed fixtures with deterministic properties, such as tiny WAV assets produced by `scripts/generate_test_data.py`, exact structural assertions are appropriate when they validate ODAI-owned behavior.

### Image Decoder Real-World Fixtures Avoid Exact Pixel Values
Tests for real-world image fixtures (e.g. `sample_chamaleon.jpg`) should assert the decoder contract that ODAI owns:
- decode succeeds
- output dimensions match expected width and height
- output channel count matches the requested target spec
- pixel buffer is non-empty and correctly sized (`width × height × channels`)

They should avoid asserting exact pixel values for real-world or compressed fixtures such as JPEG.

Why:
- JPEG is a lossy format; exact decoded values depend on the encoder used to produce the fixture
- stb_image's color space handling (e.g. JFIF vs. EXIF gamma, chroma subsampling) can differ between file origins
- exact pixel-value assertions become brittle without improving contract coverage of ODAI-owned behavior

For synthetic fixtures with deterministic pixel grids (PNG, BMP, TGA, PPM, PGM, PNM, PSD, HDR) produced by `scripts/generate_test_data.py`, exact dimension and channel assertions are appropriate. Exact per-pixel value assertions are still brittle for compressed or HDR formats due to third-party encode/decode round-trip behavior.

## Manual Verification Register

Record checks here when we intentionally leave coverage manual.

Each entry should say:
- what must be checked manually
- why it is not automated
- when the manual check should be repeated

### Current Manual Checks

This section tracks checks that are intentionally manual today.

### Audio Decoder Perceptual Sanity Check

Check:
- listen to the original fixture and the decoded or decoded-plus-resampled output
- confirm there is no obvious truncation, unintended silence, severe distortion, gross speed or pitch error, or unintended channel collapse beyond the requested conversion

Why manual:
- perceptual audio quality is difficult to assert robustly with simple automated checks
- compressed formats and third-party decoder or resampler behavior can still produce outputs that satisfy structural assertions while sounding obviously wrong

Re-run when:
- changing audio decode logic
- changing target-spec resampling or channel-conversion behavior
- changing miniaudio version or compile-time configuration
- replacing or modifying bundled audio fixtures

### Image Decoder Visual Sanity Check

Check:
- open the original fixture and the decoded output (e.g. by writing it to a PNG via stb_image_write)
- confirm there is no obvious corruption: wrong colors, swapped channels, mangled alpha, severe JPEG blocking artifacts beyond what the original exhibits, or incorrect aspect ratio

Why manual:
- visual image quality is difficult to assert robustly with simple automated checks
- stb_image's decode output for real-world fixtures can satisfy structural assertions (correct dimensions, buffer size) while still having wrong color interpretation or channel ordering
- HDR format tone-mapping and linear/gamma handling differences are particularly hard to assert automatically

Re-run when:
- changing image decode logic
- changing target-spec resize or channel-conversion behavior
- changing stb_image version or compile-time configuration
- replacing or modifying bundled image fixtures
