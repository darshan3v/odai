# ODAI SDK — Systematic Testing Strategy

> [!NOTE]
> This document describes the testing strategy for the ODAI SDK.
> Tests target the **public API of each architectural layer** using real implementations — no mocking.
> It is a living document — update it when new layers or testable surfaces are added.

## 1. Goals

- **Detect regressions** — Every change should have a fast way to know whether existing behavior broke.
- **Test real functionality** — No heavy mocking. Current tests use real SQLite, miniaudio, and stb_image implementations. Planned backend tests should use real llama.cpp instead of mocks.
- **Layer-level coverage** — Each architectural layer's public contract is tested independently. These are the tests that catch real bugs.
- **Per-layer execution** — Every implemented test layer has its own CTest label so you can run just the layer you're changing.
- **Agent-friendly** — Agents run current labels such as `ctest -L db`, `ctest -L image`, or `ctest -L audio` to verify the layer they touched. Backend/API/E2E labels are roadmap items until their targets exist.

## 2. What We're Not Doing

- **Unit tests for every helper/utility** — No tests for `is_sane()`, sanitizers, `to_cpp()`/`to_c()`, string utils, etc. These are internal plumbing. If they break, a layer-level test will catch it.
- **Mocking interfaces** — Current tests use real `OdaiSqliteDb`, real `OdaiMiniAudioDecoder`, and real `OdaiStbImageDecoder`; planned backend tests should use real `OdaiLlamaEngine`. The compile-time swapping architecture means mock implementations don't naturally fit.
- **Asserting on LLM output text** — Model output is non-deterministic. Assert on result codes, structure, non-emptiness, and data shapes.
- **Testing the OdaiSdk singleton in isolation** — The SDK is a thin forwarder. Its behavior should be covered by planned C API and E2E tests.

## 3. Classification: Integration vs E2E

| Category | What it tests | Scope |
|---|---|---|
| **Integration** | One layer's public API in isolation using its real implementation | Single layer boundary |
| **E2E** | Planned full multi-layer workflows through the C API: `C API → SDK → RAG → Backend → DB` | Multiple layers together |

Current GoogleTest infrastructure covers Phase 1 only: DB, image decoder, and audio decoder. Backend engine tests that need real models should still be integration tests when added — they test one layer's contract (device discovery, model validation, streaming generation) in isolation, without DB or RAG orchestration.

---

## 4. Test Layers

### Directory Structure

The test directory mirrors `src/impl/` so that each layer's tests live in a predictable location:

```
tests/
├── CMakeLists.txt                  ← test-data target + current layer subdirectories
├── db/
│   ├── CMakeLists.txt              ← SQLite-gated contract + implementation targets
│   ├── odai_db_contract_tests.h    ← reusable IOdaiDb typed contract suite
│   ├── odai_db_test_helpers.h
│   ├── odai_sqlite_db_contract_test.cpp
│   └── odai_sqlite_db_test.cpp
├── imageEngine/
│   ├── CMakeLists.txt              ← STB-gated contract + implementation targets
│   ├── odai_image_decoder_contract_test.cpp
│   └── odai_stb_image_decoder_test.cpp
├── audioEngine/
│   ├── CMakeLists.txt              ← miniaudio-gated contract + implementation targets
│   ├── odai_audio_decoder_contract_test.cpp
│   └── odai_miniaudio_decoder_test.cpp
└── data/
    ├── images/
    │   └── sample_chamaleon.jpg    ← real checked-in JPEG fixture
    └── audio/
        └── Echoes_of_Unseen_Light.mp3
```

Planned later directories are `tests/publicApi/`, `tests/backendEngine/`, `tests/support/`, and `tests/e2e/`. Do not document their labels as runnable until the CMake targets are added.

---

### Layer: Database — `OdaiSqliteDb` (label: `db`)

**What**: Test every public method of `IOdaiDb` using the real `OdaiSqliteDb` against a temp directory.

**Why**: The DB layer is the persistence backbone. Bugs here silently corrupt stored state. These tests are fast (SQLite temp file, no models needed) but exercise real SQL, real schema migrations, and real transaction logic.

**Fixture**: Each test creates a temp directory, constructs `OdaiSqliteDb(config)`, calls `initialize_db()`, and deletes the temp dir on teardown.

**Needs models**: No

#### Public methods to test:

| Method | Key scenarios |
|---|---|
| `initialize_db()` | Creates tables on fresh DB; idempotent on re-init |
| `register_model_files()` | Registers successfully; duplicate name → `ALREADY_EXISTS` |
| `get_model_files()` | Retrieves registered model; unknown name → `NOT_FOUND` |
| `get_model_checksums()` | Retrieves checksums; unknown name → `NOT_FOUND` |
| `update_model_files()` | Updates existing model; unknown name → `NOT_FOUND` |
| `create_semantic_space()` | Creates successfully; duplicate → `ALREADY_EXISTS` |
| `get_semantic_space_config()` | Retrieves existing; unknown → `NOT_FOUND` |
| `list_semantic_spaces()` | Empty DB → empty vec; after creates → correct count and content |
| `delete_semantic_space()` | Deletes existing, then get → `NOT_FOUND`; unknown → `NOT_FOUND` |
| `create_chat()` | Creates chat; verify via `chat_id_exists()` |
| `chat_id_exists()` | Existing → `true`; unknown → `false` |
| `get_chat_config()` | Returns correct config for existing chat; unknown → error |
| `get_chat_history()` | System prompt present after create; messages appear after insert |
| `insert_chat_messages()` | Messages persist in order; multi-turn history is retrievable |
| `store_media_item()` | Text items pass through unchanged; file/binary items stored to media cache |
| `begin/commit/rollback_transaction()` | Nested semantics work; rollback undoes changes |
| `close()` | Idempotent; no crash on double close |

---

### Layer: Image Decoder — `OdaiStbImageDecoder` (label: `image`)

**What**: Test the public API of `IOdaiImageDecoder` using the real `OdaiStbImageDecoder` with build-local generated fixtures plus checked-in real sample files.

**Why**: The image decoder does real pixel decoding and resizing. Bugs here produce corrupt pixel buffers that silently break multimodal inference.

**Test data**: Real JPEG input is checked into `tests/data/images/`. Synthetic PNG, BMP, TGA, GIF, HDR, PNM/PPM/PGM, and PSD fixtures are generated into `build/tests/data/images/`.

**Needs models**: No

#### Public methods to test:

| Method | Key scenarios |
|---|---|
| `is_supported()` | supported STB contract formats (`jpg`, `jpeg`, `png`, `bmp`, `tga`, `gif`, `hdr`, `pnm`, `ppm`, `pgm`, `psd`) → true; `"pic"`, `"pdf"`, `"mp3"`, `""` → false; case insensitivity |
| `decode_to_spec()` — file path | Decode a generated PNG from disk → valid pixels, correct dimensions, correct channels |
| `decode_to_spec()` — memory buffer | Load file bytes into `InputItem`, decode → same result as file path |
| `decode_to_spec()` — channel conversion | Request 3 channels from a 4-channel PNG → output has 3 channels |
| `decode_to_spec()` — resize down | Set `maxWidth`/`maxHeight` smaller than source → output within bounds, aspect ratio preserved |
| `decode_to_spec()` — no resize | Set max to 0 or larger than source → original dimensions preserved |
| `decode_to_spec()` — invalid input | Non-image mime type → `INVALID_ARGUMENT`; empty data → `INVALID_ARGUMENT`; corrupt bytes → error |
| `decode_to_spec()` — multiple formats | Decode JPEG and generated STB fixtures → all produce valid non-empty pixel buffers |

---

### Layer: Audio Decoder — `OdaiMiniAudioDecoder` (label: `audio`)

**What**: Test the public API of `IOdaiAudioDecoder` using the real `OdaiMiniAudioDecoder` with build-local generated fixtures plus checked-in real sample files.

**Why**: The audio decoder resamples and channel-converts audio for model whisper-style inputs. Wrong sample rates or channel counts produce garbage embeddings.

**Test data**: Real MP3 input is checked into `tests/data/audio/`. Synthetic WAV and FLAC fixtures are generated into `build/tests/data/audio/`.

**Needs models**: No

#### Public methods to test:

| Method | Key scenarios |
|---|---|
| `is_supported()` | `"wav"`, `"mp3"`, `"flac"` → true; `"ogg"`, `"aac"`, `""` → false; case insensitivity |
| `decode_to_spec()` — file path | Decode a generated WAV from disk → non-empty samples, correct sample rate and channel count |
| `decode_to_spec()` — memory buffer | Load file bytes into `InputItem`, decode → same valid output |
| `decode_to_spec()` — resampling | Source at 44100Hz, request 16000Hz → output has 16000Hz sample rate |
| `decode_to_spec()` — channel conversion | Stereo source, request mono → output has 1 channel |
| `decode_to_spec()` — invalid input | Non-audio mime type → `INVALID_ARGUMENT`; empty data → `INVALID_ARGUMENT` |
| `decode_to_spec()` — multiple formats | WAV, MP3, FLAC all produce valid non-empty sample buffers |

---

### Layer: Backend Engine — `OdaiLlamaEngine` (planned label: `backend`)

**What**: Model-backed integration tests for `IOdaiBackendEngine` using the real `OdaiLlamaEngine` and real GGUF fixtures. These tests cover device discovery, model validation, generation, placement, cache/reload behavior, and multimodal projector handling through the public API.

**Why**: The backend engine is where hardware-sensitive policy and real-model loading live. Real fixtures catch regressions in discovery, placement, caching, and multimodal loading that mocks would miss.

**Needs models**: Yes

The executable phase-3 plan lives in [`docs/plans/testing-strategy-phase-3-backend-breakdown.md`](./testing-strategy-phase-3-backend-breakdown.md). It splits backend testing into the following ordered subphases:

1. Harness and model discovery
2. Engine smoke, init, and validation
3. Text-generation baseline
4. Placement, cache, and reload behavior
5. Multimodal projector path
6. Documentation and nuance sync

**High-level acceptance**:
- backend tests configure cleanly under the model-backed test gate
- model-backed tests skip cleanly when model files are unavailable
- the suite passes on CPU-only hosts and on accelerated hosts when the relevant fixtures are present

> [!NOTE]
> The backend section above is intentionally high level. The detailed, executable step-by-step breakdown for Phase 3 lives in the companion plan linked above.

---

### Layer: C API — `odai_public.h` (planned label: `api`)

**What**: Planned boundary-only tests for the public C API — null/invalid argument guards, output reset behavior, free functions, and pre-initialization result mapping.

**Why**: The C API is the stable ABI consumed by all external users (Android JNI, Flutter, etc.). Fast boundary tests should catch ownership and validation regressions without requiring SDK initialization or model files. Fully initialized C API workflows belong in the planned E2E phase unless the SDK gains a reliable non-model initialization mode.

**Needs models**: No

#### Coverage:

| Scenario | Key assertions |
|---|---|
| **Not-initialized guards** | Call any API before init → `ODAI_NOT_INITIALIZED` |
| **Null/invalid argument rejection** | Null pointers to any function → `ODAI_INVALID_ARGUMENT` (no crash) |
| **Output reset** | Output parameters are initialized/reset on validation and not-initialized paths |
| **Memory ownership** | All `odai_free_*` functions work on valid data where fixtures can be built locally and don't crash on null |

---

### E2E Feature Tests (planned label: `e2e`)

**What**: Full multi-layer workflows through the C API with real models, real inference, real SQLite — testing the **interaction between layers**.

**Why**: This is the ultimate regression gate — it tests that the layers work together correctly through the full `C API → SDK → RAG → BackendEngine → DB` pipeline. Individual layer tests might pass but cross-layer integration could still break.

**Needs models**: Yes

#### Scenarios (from existing `tests/main.cpp`):

| Scenario | What it specifically tests across layers |
|---|---|
| **Init → register → generate → shutdown** | SDK lifecycle + RAG engine wiring + backend model loading + cleanup |
| **Model registration (embedding + LLM + multimodal)** | C API sanitization + RAG validation + backend file validation + DB persistence |
| **Streaming text generation** | RagEngine model resolution (DB lookup) + BackendEngine generation pipeline |
| **Streaming image generation** | RagEngine → BackendEngine multimodal pipeline (image decoder + mtmd) |
| **Streaming audio generation** | RagEngine → BackendEngine multimodal pipeline (audio decoder + whisper) |
| **Multi-turn chat** | C API → SDK → RagEngine chat orchestration + DB message persistence + BackendEngine context replay |
| **Chat history retrieval** | DB persistence round-trip through all layers; C API memory ownership (to_c/free) |
| **Shutdown → reinitialize** | Full lifecycle reset; proves cleanup is complete and reusable |

#### Non-assertions:
- Never assert on exact text output
- Never assert on timing
- Assert on: result codes, non-empty output, correct data shapes, persisted state consistency

---

## 5. Build Integration

### Test Data Path Management

Image and audio decoder tests reference build-local files under `build/tests/data`. To ensure test binaries can locate these files regardless of the working directory they are invoked from, use CMake's `target_compile_definitions` to inject a `TEST_DATA_DIR` macro at compile time:

```cmake
target_compile_definitions(odai_stb_image_decoder_tests PRIVATE
  TEST_DATA_DIR="${TEST_DATA_OUTPUT_DIR}"
)
```

Test code then resolves paths relative to `TEST_DATA_DIR`, for example `TEST_DATA_DIR/images/tiny_rgba.png`. This avoids fragile relative-path assumptions and works correctly under CTest, IDE runners, and manual invocation.

`tests/CMakeLists.txt` owns an `odai_test_data` target that:
- creates a build-local Python venv
- installs generator dependencies (`Pillow`, `pydub`, `audioop-lts`, `imageio`, `opencv-python-headless`, `psd-tools`, `numpy`)
- removes the old build-local fixture directory before regeneration
- runs `scripts/generate_test_data.py`
- copies the checked-in JPEG and MP3 fixtures alongside the generated assets

The generated and copied fixture files themselves are declared as custom-command outputs. There is no stamp file. If a fixture is removed, renamed, or added, update `TEST_DATA_OUTPUT_FILES`, the generator/copy command, and the tests in the same change.

### CMake Flags

| Flag | Purpose |
|---|---|
| `ODAI_BUILD_TESTS` | Enables GoogleTest fetch, CTest, and the current non-model-backed tests (`db`, `image`, `audio`) |

`ODAI_BUILD_E2E_TESTS`, API/backend/E2E targets, and sanitizer-specific test workflows are planned/deferred. Add and document their CMake options only when their targets are implemented.

### CTest Labels

Every test target gets a layer label and at least one category label. Reusable contract targets also add `contract`, and implementation-specific targets add an implementation label.

| Layer Label | Category | Needs Models? | What |
|---|---|---|---|
| `db` | integration; contract targets also add `contract`; SQLite targets add `sqlite` | No | Database contract coverage and SQLite-specific behavior |
| `image` | integration; contract targets also add `contract`; STB targets add `stb` | No | Image decoder contract coverage and STB-specific behavior |
| `audio` | integration; contract targets also add `contract`; miniaudio targets add `miniaudio` | No | Audio decoder contract coverage and miniaudio-specific behavior |

Current labels are: `db`, `image`, `audio`, `integration`, `contract`, `sqlite`, `stb`, and `miniaudio`.
Planned labels are: `api`, `backend`, and `e2e`.

### Running Tests Per-Layer

```bash
# Run only the layer you changed:
ctest --test-dir build -L db --output-on-failure
ctest --test-dir build -L image --output-on-failure
ctest --test-dir build -L audio --output-on-failure
ctest --test-dir build -L contract --output-on-failure
ctest --test-dir build -L sqlite --output-on-failure
ctest --test-dir build -L stb --output-on-failure
ctest --test-dir build -L miniaudio --output-on-failure
```

### Running All Fast Tests (No Models Needed)

```bash
cmake --preset linux-default-debug -DODAI_BUILD_TESTS=ON
cmake --build --preset linux-default-debug
ctest --test-dir build -L "db|image|audio" --output-on-failure
```

For a faster non-model verification loop, disable the llama backend while keeping the non-model implementations enabled:

```bash
cmake --preset linux-default-debug -DODAI_BUILD_TESTS=ON -DODAI_ENABLE_LLAMA_BACKEND=OFF
cmake --build --preset linux-default-debug
ctest --test-dir build -L "db|image|audio" --output-on-failure
```

### Planned Model-Backed Runs

```bash
cmake --preset linux-default-release -DODAI_BUILD_TESTS=ON -DODAI_BUILD_E2E_TESTS=ON
cmake --build --preset linux-default-release
ODAI_E2E_MODEL_ROOT=/path/to/models ctest --test-dir build --output-on-failure
```

This command shape is directional only until `ODAI_BUILD_E2E_TESTS` and the backend/E2E targets are added.

### Planned E2E Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `ODAI_E2E_MODEL_ROOT` | Root directory containing model subdirectories | `tests/` relative to binary |
| `ODAI_E2E_DEVICE` | Backend device preference (`CPU`, `GPU`, `IGPU`, `AUTO`) | `AUTO` |
| `ODAI_E2E_ARTIFACT_ROOT` | Where temp DBs, caches, and logs are written | System temp dir |

---

## 6. Phased Implementation Roadmap

Each phase is independently valuable and builds on the prior. The plan is designed so an agent (or developer) can pick up any phase and execute it without ambiguity.

### How to use this roadmap

1. Start with Phase 1 — it delivers the most test coverage for the least effort.
2. Each phase lists **prerequisites**, **deliverables** (exact files to create/modify), **implementation steps**, and **acceptance criteria**.
3. Mark phases complete in this document as you finish them.
4. If a phase introduces nuances or build quirks, update `dev_nuances.md`.

---

### Phase 1: Foundation + Database + Decoders

**Status**: `[x]` Completed

**Prerequisites**: None — this is the starting phase.

**Goal**: Set up GoogleTest infrastructure and deliver the three fastest, most valuable test suites (DB, Image, Audio). These run in seconds, need no models, and cover three of the four swappable interface implementations.

**Build assumptions for this phase**:

- Use the existing Linux presets first (`linux-default-debug` / `linux-default-release`). These already enable `ODAI_ENABLE_SQLITE_DB`, `ODAI_ENABLE_STB_IMAGE`, and `ODAI_ENABLE_MINIAUDIO`.
- Phase 1 tests should link against the existing `odai` target unless an internal test-support target is introduced first. There is currently no `odai_core` target.
- Test targets that include concrete implementation headers must receive the same feature macros as the library (`ODAI_ENABLE_SQLITE_DB`, `ODAI_ENABLE_STB_IMAGE`, `ODAI_ENABLE_MINIAUDIO`) so `#ifdef`-guarded classes are visible.
- If tests include public headers that expose third-party header-only types (`nlohmann/json.hpp`, `tl::expected`), make those include directories available to tests. Prefer fixing the `odai` target's public include requirements if the headers are part of the public compile surface; otherwise add a small CMake helper for test-only include wiring.

#### Step 1.1: GoogleTest + CTest infrastructure

**Deliverables**:

| File | Action | What |
|---|---|---|
| `CMakeLists.txt` (root) | MODIFY | Add `ODAI_BUILD_TESTS` option, GoogleTest `FetchContent`, `enable_testing()`, and `add_subdirectory(tests)` gated by the flag |
| `tests/CMakeLists.txt` | NEW | Top-level test CMake that prepares build-local test data and adds each current layer subdirectory |

**Implementation details**:

1. Add at the root `CMakeLists.txt`:
   ```cmake
   option(ODAI_BUILD_TESTS "Build test targets" OFF)

   if(ODAI_BUILD_TESTS)
     include(FetchContent)
     FetchContent_Declare(
       googletest
       GIT_REPOSITORY https://github.com/google/googletest.git
       GIT_TAG v1.17.0
     )
     FetchContent_MakeAvailable(googletest)
     enable_testing()
     add_subdirectory(tests)
   endif()
   ```

2. Keep the GoogleTest setup near the existing dependency setup. The root already calls `include(FetchContent)`, so do not add a second global dependency pattern unless needed for readability.

3. Create `tests/CMakeLists.txt` with the build-local test-data target before adding layer subdirectories. The target should declare the generated and copied fixtures as outputs, remove stale `build/tests/data` before regeneration, run `scripts/generate_test_data.py`, and copy checked-in sample files from `tests/data/`:
   ```cmake
   set(TEST_DATA_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
   set(TEST_DATA_OUTPUT_FILES
     "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgba.png"
     # keep this list in sync with scripts/generate_test_data.py and copied real fixtures
   )

   add_custom_command(
     OUTPUT ${TEST_DATA_OUTPUT_FILES}
     COMMAND ${CMAKE_COMMAND} -E rm -rf "${TEST_DATA_OUTPUT_DIR}"
     COMMAND "${VENV_PYTHON}" "${CMAKE_SOURCE_DIR}/scripts/generate_test_data.py" "${TEST_DATA_OUTPUT_DIR}"
     COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "${CMAKE_CURRENT_SOURCE_DIR}/data/images/sample_chamaleon.jpg"
       "${TEST_DATA_OUTPUT_DIR}/images/sample_chamaleon.jpg"
     DEPENDS
       "${CMAKE_SOURCE_DIR}/scripts/generate_test_data.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/data/images/sample_chamaleon.jpg"
   )

   add_custom_target(odai_test_data DEPENDS ${TEST_DATA_OUTPUT_FILES})
   ```

4. Add current layer subdirectories:
   ```cmake
   add_subdirectory(db)
   add_subdirectory(imageEngine)
   add_subdirectory(audioEngine)
   ```

5. Add a small helper in `tests/CMakeLists.txt` if repeated wiring becomes noisy:
   ```cmake
   function(odai_configure_integration_test target label)
     target_link_libraries(${target} PRIVATE odai GTest::gtest_main)
     target_include_directories(${target} PRIVATE
       ${nlohmann_json_SOURCE_DIR}
       ${tl_expected_SOURCE_DIR}/include
     )
     gtest_discover_tests(${target} PROPERTIES LABELS "${label};integration")
   endfunction()
   ```

   Extend this helper per layer only when needed (for example, SQLiteCpp include dirs for DB tests, STB include dirs for image tests, miniaudio include dirs for audio tests).

**Acceptance criteria**:
- [x] `cmake --preset linux-default-debug -DODAI_BUILD_TESTS=ON` configures successfully
- [x] GoogleTest is fetched and available
- [x] `ctest --test-dir build -N` lists the Phase 1 test executables and discovered cases

#### Step 1.2: SQLite DB integration tests

**Deliverables**:

| File | Action | What |
|---|---|---|
| `tests/db/CMakeLists.txt` | NEW | Test target `odai_sqlite_db_tests`, linked against the DB impl and GoogleTest, labeled `db` + `integration` |
| `tests/db/odai_sqlite_db_test.cpp` | NEW | GoogleTest file covering all `IOdaiDb` public methods from §4 DB table |

**Implementation details**:

1. **Fixture**: Create a `SqliteDbTestFixture` that:
   - `SetUp()`: creates a unique temp directory using a collision-resistant suffix (for example, timestamp + process id + atomic counter, or `mkdtemp` on POSIX), constructs `OdaiSqliteDb` with config pointing to that dir, calls `initialize_db()`
   - `TearDown()`: calls `close()`, recursively removes the temp directory

2. **Test naming**: Use `TEST_F(SqliteDbTestFixture, MethodName_Scenario_ExpectedOutcome)` pattern, e.g.:
   - `TEST_F(SqliteDbTestFixture, RegisterModelFiles_ValidModel_ReturnsSuccess)`
   - `TEST_F(SqliteDbTestFixture, RegisterModelFiles_DuplicateName_ReturnsAlreadyExists)`
   - `TEST_F(SqliteDbTestFixture, GetModelFiles_UnknownName_ReturnsNotFound)`

3. **Test all methods** from the DB table in §4. Each method needs at minimum:
   - A happy-path test (valid inputs → expected result)
   - An error-path test (invalid/edge case → expected error code)

4. **Scope control**: If this phase is being implemented incrementally, split the DB suite into two commits without changing the phase boundary:
   - First commit: initialization, model file CRUD, semantic-space CRUD, and close/idempotency
   - Second commit: chat lifecycle, chat history/messages, media storage, and transaction semantics

5. **CMake wiring**:
   ```cmake
   add_executable(odai_sqlite_db_tests odai_sqlite_db_test.cpp)
   target_link_libraries(odai_sqlite_db_tests PRIVATE odai GTest::gtest_main)
   target_compile_definitions(odai_sqlite_db_tests PRIVATE ODAI_ENABLE_SQLITE_DB)
   target_include_directories(odai_sqlite_db_tests PRIVATE
     ${nlohmann_json_SOURCE_DIR}
     ${tl_expected_SOURCE_DIR}/include
     ${SQLiteCpp_SOURCE_DIR}
   )
   include(GoogleTest)
   gtest_discover_tests(odai_sqlite_db_tests PROPERTIES LABELS "db;integration")
   ```

**Acceptance criteria**:
- [x] `cmake --build --preset linux-default-debug` compiles `odai_sqlite_db_tests`
- [x] `ctest --test-dir build -L db --output-on-failure` passes
- [x] All 17 method groups from the DB table have at least one test

#### Step 1.3: Image decoder integration tests

**Deliverables**:

| File | Action | What |
|---|---|---|
| `tests/imageEngine/CMakeLists.txt` | NEW | Test target `odai_stb_image_decoder_tests`, labeled `image` + `integration` |
| `tests/imageEngine/odai_stb_image_decoder_test.cpp` | NEW | GoogleTest file covering all `IOdaiImageDecoder` methods from §4 Image table |
| `scripts/generate_test_data.py` | NEW | Generates small deterministic image fixtures into `build/tests/data/images/` |
| `tests/CMakeLists.txt` | MODIFY | Adds generated/copied image fixtures to `TEST_DATA_OUTPUT_FILES` |

**Implementation details**:

1. **Test data**: Keep the existing `sample_chamaleon.jpg` checked in under `tests/data/images/`. Generate deterministic synthetic fixtures into the build directory instead of checking in extra binaries:
   - `tiny_rgba.png` for channel-conversion tests
   - `tiny_rgb.bmp`, `tiny_rgb.tga`, `tiny_rgb.gif`, `tiny_rgb.ppm`, `tiny_rgb.pnm`, `tiny_gray.pgm`, `tiny_rgb.psd`, and `tiny_rgb.hdr` for implementation-specific format coverage
   - PIC stays unsupported and should remain in the negative support-list test

2. **Fixture**: Create `StbImageDecoderTestFixture` that constructs an `OdaiStbImageDecoder` instance in `SetUp()`. No teardown needed (stateless decoder).

3. **Path resolution**: Use `TEST_DATA_DIR` macro injected via CMake:
   ```cmake
   target_compile_definitions(odai_stb_image_decoder_tests PRIVATE
     TEST_DATA_DIR="${TEST_DATA_OUTPUT_DIR}"
   )
   ```

4. **CMake wiring**:
   ```cmake
   add_executable(odai_stb_image_decoder_tests odai_stb_image_decoder_test.cpp)
   target_link_libraries(odai_stb_image_decoder_tests PRIVATE odai GTest::gtest_main)
   target_compile_definitions(odai_stb_image_decoder_tests PRIVATE
     ODAI_ENABLE_STB_IMAGE
     TEST_DATA_DIR="${TEST_DATA_OUTPUT_DIR}"
   )
   target_include_directories(odai_stb_image_decoder_tests PRIVATE
     ${tl_expected_SOURCE_DIR}/include
     ${stb_SOURCE_DIR}
   )
   include(GoogleTest)
   gtest_discover_tests(odai_stb_image_decoder_tests PROPERTIES LABELS "image;integration")
   ```

**Acceptance criteria**:
- [x] `ctest --test-dir build -L image --output-on-failure` passes
- [x] Tests cover: `is_supported()`, file path decode, memory buffer decode, channel conversion, resize, invalid input, multi-format
- [x] Test data files exist and are generated into the build-local fixture directory

#### Step 1.4: Audio decoder integration tests

**Deliverables**:

| File | Action | What |
|---|---|---|
| `tests/audioEngine/CMakeLists.txt` | NEW | Test target `odai_miniaudio_decoder_tests`, labeled `audio` + `integration` |
| `tests/audioEngine/odai_miniaudio_decoder_test.cpp` | NEW | GoogleTest file covering all `IOdaiAudioDecoder` methods from §4 Audio table |
| `scripts/generate_test_data.py` | MODIFY | Generates small deterministic audio fixtures into `build/tests/data/audio/` |
| `tests/CMakeLists.txt` | MODIFY | Adds generated/copied audio fixtures to `TEST_DATA_OUTPUT_FILES` |

**Implementation details**:

1. **Test data**: Keep the existing `Echoes_of_Unseen_Light.mp3` checked in under `tests/data/audio/`. Generate deterministic synthetic fixtures into the build directory:
   - `tiny_stereo_44100.wav` for baseline decode, channel conversion, and resampling tests
   - `tiny_stereo_44100.flac` for FLAC coverage

2. **Fixture**: Construct `OdaiMiniAudioDecoder` in `SetUp()`. Stateless, no teardown.

3. **CMake wiring**: Same pattern as image tests, with `TEST_DATA_DIR` macro and `audio;integration` labels:
   ```cmake
   add_executable(odai_miniaudio_decoder_tests odai_miniaudio_decoder_test.cpp)
   target_link_libraries(odai_miniaudio_decoder_tests PRIVATE odai GTest::gtest_main)
   target_compile_definitions(odai_miniaudio_decoder_tests PRIVATE
     ODAI_ENABLE_MINIAUDIO
     TEST_DATA_DIR="${TEST_DATA_OUTPUT_DIR}"
   )
   target_include_directories(odai_miniaudio_decoder_tests PRIVATE
     ${tl_expected_SOURCE_DIR}/include
     ${miniaudio_SOURCE_DIR}
   )
   include(GoogleTest)
   gtest_discover_tests(odai_miniaudio_decoder_tests PROPERTIES LABELS "audio;integration")
   ```

**Acceptance criteria**:
- [x] `ctest --test-dir build -L audio --output-on-failure` passes
- [x] Tests cover: `is_supported()`, file path decode, memory buffer decode, resampling, channel conversion, invalid input
- [x] Test data files exist and are generated into the build-local fixture directory

#### Phase 1 verification

After completing all steps in Phase 1, run:

```bash
cmake --preset linux-default-debug -DODAI_BUILD_TESTS=ON
cmake --build --preset linux-default-debug
ctest --test-dir build -L "db|image|audio" --output-on-failure
```

If the default Linux preset enables model backends that are not needed for Phase 1, the same Phase 1 suite can be
verified faster with:

```bash
cmake --preset linux-default-debug -DODAI_BUILD_TESTS=ON -DODAI_ENABLE_LLAMA_BACKEND=OFF
cmake --build --preset linux-default-debug
ctest --test-dir build -L "db|image|audio" --output-on-failure
```

All tests must pass. Total expected runtime: < 5 seconds.

Before marking Phase 1 complete:

- [x] Confirm whether the test-target include/compile-definition wiring exposed a durable build quirk. If yes, update `dev_nuances.md` and its index.
- [x] If the `odai` target's public/private include ownership changed, verify the architecture docs do not need updates. Pure test wiring should not require architecture changes.

---

### Phase 2: C API Boundary

**Status**: `[x]` Completed

**Prerequisites**: Phase 1 complete (GoogleTest infrastructure available).

**Goal**: Test the public C ABI boundary without requiring SDK initialization or model files — null/invalid argument guards, output reset behavior, pre-initialization result mapping, and memory ownership for `odai_free_*` functions.

#### Step 2.1: C API boundary tests

**Deliverables**:

| File | Action | What |
|---|---|---|
| `tests/publicApi/CMakeLists.txt` | NEW | Test target `odai_api_tests`, labeled `api` + `integration` |
| `tests/publicApi/odai_public_api_boundary_test.cpp` | NEW | GoogleTest file covering all C API boundary scenarios from §4 API table |
| `tests/CMakeLists.txt` | MODIFY | Add `add_subdirectory(publicApi)` |

**Implementation details**:

1. **Fixture**: Create a boundary fixture that calls `odai_shutdown()` before and after each test so singleton state from earlier tests cannot leak into boundary assertions. Do not initialize the SDK in this phase.

2. **Link against the shared library**: The C API tests should link against the full `odai` library (shared or static) since they test the public C surface:
   ```cmake
   add_executable(odai_api_tests odai_public_api_boundary_test.cpp)
   target_link_libraries(odai_api_tests PRIVATE odai GTest::gtest_main)
   ```

3. **Test boundary behavior only**: These tests specifically avoid model-backed generation and initialized CRUD workflows — those are E2E territory until a non-model SDK initialization path exists. Focus on:
   - Not-initialized result mapping
   - Null pointer rejection on every API function
   - Output pointer reset/initialization on failure paths
   - All `odai_free_*` functions on null inputs and on locally constructed data where no SDK allocation is required

4. **Memory ownership discipline**: Every `odai_free_*` call in tests must match its allocation. This phase is the natural validation layer for those contracts.

**Acceptance criteria**:
- [x] `ctest --test-dir build -L api --output-on-failure` passes
- [x] Tests cover not-initialized guards, null argument rejection, output reset behavior, and memory freeing
- [x] No SDK initialization or model files are required

#### Phase 2 verification

```bash
ctest --test-dir build -L "db|image|audio|api" --output-on-failure
```

All Phase 1 + Phase 2 tests pass. Total expected runtime: < 10 seconds.

---

### Phase 3: Backend Engine (Model-Backed)

**Status**: `[ ]` Not started

**Prerequisites**: Phase 1 complete. Phase 2 is NOT required — backend tests are independent of C API tests.

**Goal**: Add the backend-engine test harness first, then grow the suite in small subphases using real model fixtures. The detailed execution plan, acceptance criteria, and verification order live in [`testing-strategy-phase-3-backend-breakdown.md`](./testing-strategy-phase-3-backend-breakdown.md).

**Roadmap**:
- add a model-discovery harness and clean skip behavior first
- cover engine init, device discovery, and model validation next
- add text-generation coverage before placement and reload scenarios
- finish with multimodal projector coverage and documentation/nuance sync

---

### Phase 4: E2E Migration

**Status**: `[ ]` Not started

**Prerequisites**: Phase 1, Phase 2, and Phase 3 complete.

**Goal**: Migrate the existing `tests/main.cpp` scenarios into GoogleTest, retire `tests/run_tests.sh`, and establish the full E2E regression gate.

#### Step 4.1: E2E support utilities

**Deliverables**:

| File | Action | What |
|---|---|---|
| `tests/support/e2e_test_env.h` | NEW | Shared fixture base class for E2E tests: SDK init/shutdown, temp dir management, model path resolution, log callbacks |
| `tests/support/e2e_test_env.cpp` | NEW | Implementation of the shared fixture |
| `tests/CMakeLists.txt` | MODIFY | Add support library target |

**Implementation details**:

1. **Shared fixture**: `E2ETestEnvironment` extends `::testing::Environment` (GoogleTest global setup/teardown):
   - `SetUp()`: reads env vars, creates temp dir, calls `odai_initialize_sdk()`, registers all test models
   - `TearDown()`: calls `odai_shutdown()`, cleans up temp dir
   - Provides accessors for model names, temp paths, and common configs

2. **This is global setup, not per-test setup** — the SDK init + model registration is expensive and should happen once per test binary, not per test case.

#### Step 4.2: E2E feature tests

**Deliverables**:

| File | Action | What |
|---|---|---|
| `tests/e2e/CMakeLists.txt` | NEW | Test target `odai_e2e_tests`, labeled `e2e`, gated by `ODAI_BUILD_E2E_TESTS` |
| `tests/e2e/feature_e2e_test.cpp` | NEW | GoogleTest file covering all E2E scenarios from §4 E2E table |
| `tests/CMakeLists.txt` | MODIFY | Add E2E-gated `add_subdirectory(e2e)` |

**Implementation details**:

1. **Migrate `tests/main.cpp` scenarios** one-for-one into GoogleTest:
   - `test_registration()` → `TEST(E2EFeature, ModelRegistration_AllTypes_Succeed)`
   - `test_semantic_spaces()` → `TEST(E2EFeature, SemanticSpaceCRUD_FullLifecycle)`
   - `test_streaming_text()` → `TEST(E2EFeature, StreamingText_GeneratesOutput)`
   - `test_streaming_image()` → `TEST(E2EFeature, StreamingImage_GeneratesOutput)`
   - `test_streaming_audio()` → `TEST(E2EFeature, StreamingAudio_GeneratesOutput)`
   - `test_chat_multimodal()` → `TEST(E2EFeature, ChatMultimodal_MultiTurnHistory)`
   - `test_shutdown_reinitialize()` → `TEST(E2EFeature, ShutdownReinitialize_PreservesState)`

2. **Replace `std::cout` assertions with `ASSERT_*`/`EXPECT_*`**:
   - `res == ODAI_SUCCESS` → `ASSERT_EQ(res, ODAI_SUCCESS)`
   - `res != ODAI_SUCCESS` → `ASSERT_NE(res, ODAI_SUCCESS)` with `GTEST_SKIP()` if model not found
   - Token count > 0 → `EXPECT_GT(token_count, 0)`

3. **Non-assertions rule**: Never `EXPECT_EQ(output_text, "Paris")`. Instead: `EXPECT_FALSE(output_text.empty())`.

#### Step 4.3: Retire legacy test infrastructure

**Deliverables**:

| File | Action | What |
|---|---|---|
| `tests/main.cpp` | DELETE | Replaced by GoogleTest E2E suite |
| `tests/run_tests.sh` | DELETE | Replaced by CTest |

**Only do this** after confirming all E2E GoogleTest equivalents pass successfully.

**Acceptance criteria**:
- [ ] `ctest --test-dir build -L e2e --output-on-failure` passes with models present
- [ ] Every scenario from `tests/main.cpp` has a GoogleTest equivalent
- [ ] `tests/main.cpp` and `tests/run_tests.sh` are deleted
- [ ] `ctest --test-dir build --output-on-failure` runs the full suite (all phases)

#### Phase 4 verification

```bash
# Full suite
cmake --preset linux-default-release -DODAI_BUILD_TESTS=ON -DODAI_BUILD_E2E_TESTS=ON
cmake --build --preset linux-default-release
ODAI_E2E_MODEL_ROOT=/path/to/models ctest --test-dir build --output-on-failure

# Fast suite only (CI without models)
cmake --preset linux-default-debug -DODAI_BUILD_TESTS=ON
cmake --build --preset linux-default-debug
ctest --test-dir build -L "db|image|audio|api" --output-on-failure
```

---

## 7. Test Guidelines

1. **Test the contract, not the implementation.** Assert on return values, output shapes, and observable state.
2. **Use real implementations.** Real SQLite with temp dir, real decoders with test files, real llama.cpp with GGUF models.
3. **Name tests descriptively.** `RegisterModel_DuplicateName_ReturnsAlreadyExists` not `test_register_2`.
4. **Clean up after yourself.** Use RAII or GoogleTest fixtures to manage temp files and directories.
5. **One concern per test.** If a test name needs "and" in it, split it.
6. **Skip gracefully.** Model-backed tests should `GTEST_SKIP()` with a clear message when model files aren't found.
7. **Keep layer tests isolated.** DB tests don't depend on backend engine. Backend tests don't depend on DB. Only E2E combines layers.

---

## 8. Sanitizer Integration (Planned)

Passing without crashing does not prove memory correctness. The `api` and `e2e` layers cross the C/C++ ownership boundary — memory is allocated on one side and freed on the other — making them prime candidates for silent leaks, use-after-free, and buffer overruns.

Sanitizer-specific test workflow is not current infrastructure. Add it as a dedicated follow-up once the API boundary target exists, and keep `docs/architecture/testing.md` plus the testing guide in sync when the CMake option is introduced.

### What to enable

| Sanitizer | Flag | What it catches |
|---|---|---|
| AddressSanitizer (ASan) | `-fsanitize=address` | Heap/stack buffer overflows, use-after-free, double-free |
| LeakSanitizer (LSan) | Included with ASan by default on Linux | Memory leaks — allocated but never freed |

### CMake integration

Add a CMake option that layers the sanitizer flags onto the build:

```cmake
option(ODAI_ENABLE_SANITIZERS "Enable ASan + LSan for test builds" OFF)

if(ODAI_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()
```

### When to add sanitizer support

Sanitizer CMake integration should be added after the API boundary suite exists. It becomes most valuable starting from **Phase 2** because that is where the C/C++ ownership boundary is actively exercised.

### Running tests with sanitizers

```bash
cmake --preset linux-default-debug -DODAI_BUILD_TESTS=ON -DODAI_ENABLE_SANITIZERS=ON
cmake --build --preset linux-default-debug
ctest --test-dir build -L "api|e2e" --output-on-failure
```

This command is directional until `ODAI_ENABLE_SANITIZERS` and the `api`/`e2e` targets exist.

### Which layers benefit most

- **C API (`api`)** — Every `odai_free_*` call is a manual deallocation of memory allocated by `to_c()`. ASan will flag any missed free or double-free immediately.
- **E2E (`e2e`)** — Full lifecycle init → operate → shutdown exercises every allocation/deallocation path across all layers.
- **DB (`db`)** and decoders — Less critical (SQLite and stb/miniaudio manage their own memory), but running them under ASan is cheap and can catch bugs in the wrapper glue code.

> [!NOTE]
> ASan adds ~2× runtime overhead, which is acceptable for test suites but should not be enabled in release builds. LSan runs at process exit and adds negligible overhead.

---

## 9. Future Scope

### Micro-models for automated CI

When model-backed tests (`backend`, `e2e`) are added, they will initially require manually providing model files via `ODAI_E2E_MODEL_ROOT`. For CI/CD pipelines and new-developer onboarding, this creates friction.

**Future work:** Host and auto-fetch a sub-100MB "micro-model" (tiny LLaMA + tiny embedding model) specifically for test runs. This would let model-backed tests run in CI without manual downloads while still exercising the real GGML execution paths.

For hardware-specific validation (GPU offloading, multi-GPU placement, VRAM pressure), larger models on dedicated runners would remain necessary — the micro-model covers functional correctness, not performance or offloading behavior.
