# Phase 3 Backend Test Breakdown

> [!NOTE]
> Companion to [`testing-strategy-plan.md`](./testing-strategy-plan.md). This file is the executable Phase 3 plan for backend-engine tests; the main roadmap stays high level and points here for the detailed sequence.

## Scope

- Real backend-engine integration tests for `OdaiLlamaEngine`
- Model-backed tests gated behind `ODAI_BUILD_E2E_TESTS`
- Model discovery through environment variables instead of hard-coded fixture paths
- Clean skips when model files are absent
- Assertions based on observable behavior: result codes, shapes, device metadata, token counts, cancellation, and reload behavior
- Prefer ODAI-owned test observability over upstream log parsing when a test needs to inspect backend placement, reload state, or other internal-but-meaningful runtime choices
- Keep any such observability backend-local and gated so it helps multiple subphases without becoming part of the public interface or C ABI

## Execution Order

Work through these subphases in order. Do not move on until the current subphase is passing.

1. 3A - Harness and model discovery
2. 3B - Engine smoke, init, and validation
3. 3C - Text-generation baseline
4. 3D - Placement, cache, and reload behavior
5. 3E - Multimodal projector path
6. 3F - Documentation and nuance sync

## Verification Rule

At the end of each subphase, run a focused `ctest --test-dir build -L backend --output-on-failure` pass, or a narrower label-filtered run once backend labels are split further.

---

### 3A - Harness and model discovery

**Goal**: Add the backend test scaffolding that can locate real model fixtures, pick a preferred device, and skip cleanly when model files are not available.

**Tasks**:
- add `tests/backendEngine/CMakeLists.txt` behind `ODAI_BUILD_E2E_TESTS`
- add a backend test executable labeled `backend` and `integration`
- define a reusable fixture/helper layer for model-root resolution
- resolve model roots from `ODAI_E2E_MODEL_ROOT`
- accept a preferred device override from `ODAI_E2E_DEVICE`, defaulting to `AUTO`
- call `GTEST_SKIP()` when expected model files are absent or invalid
- keep the helper reusable for later smoke, generation, and multimodal tests

**Acceptance**:
- backend tests configure cleanly with the backend gate enabled
- backend tests skip cleanly when `ODAI_E2E_MODEL_ROOT` is unset or points at missing fixtures
- the fixture exposes the resolved model paths and selected device without duplicating path logic in individual tests

**Verification**:
- `ctest --test-dir build -L backend --output-on-failure`

---

### 3B - Engine smoke, init, and validation

**Goal**: Prove the engine starts, discovers devices, and validates model files using minimal real fixtures.

**Tasks**:
- cover `initialize_engine()`
- cover `get_candidate_devices()`
- cover `validate_model_files()` for LLM, LLM + mmproj, and embedding model cases
- keep assertions to result codes, device metadata, and shape/structure checks
- do not assert on exact log text or exact model output text

**Acceptance**:
- engine initialization passes on CPU-only hosts and on accelerated hosts when the fixtures are valid
- `get_candidate_devices()` returns observable device metadata that matches the host
- model validation rejects missing or mismatched files and accepts valid fixture layouts

**Verification**:
- `ctest --test-dir build -L backend --output-on-failure`

---

### 3C - Text-generation baseline

**Goal**: Establish the first generation-path coverage for text-only prompts.

**Tasks**:
- cover `generate_streaming_response()`
- cover `generate_streaming_chat_response()`
- collect streamed chunks through a simple callback accumulator
- verify UTF-8 chunking across callback boundaries
- verify token counts are positive and reasonable
- verify cancellation by returning `false` from the callback

**Acceptance**:
- text-only generation succeeds and produces positive streamed output
- streamed chunks concatenate into valid UTF-8
- cancellation stops generation deterministically and marks the request as cancelled

**Verification**:
- `ctest --test-dir build -L backend --output-on-failure`

---

### 3D - Placement, cache, and reload behavior

**Goal**: Exercise backend placement policy and reload behavior through observable public API outcomes.

**Tasks**:
- add or extend backend-only placement/reload snapshot accessors behind `ODAI_BUILD_E2E_TESTS` so tests can read the chosen load plan without depending on log text
- keep the observability backend-local rather than on `IOdaiBackendEngine` or the C API
- verify CPU, GPU, IGPU, and AUTO placement outcomes
- verify candidate ordering and memory reporting from device discovery
- verify same-model cache reuse with unchanged model files and config
- verify a context-window change forces a reload
- verify shutdown cleans up backend state after model use

**Acceptance**:
- Phase 3 can verify placement and reload behavior using ODAI-owned debug snapshots or equivalent backend-local observability, not llama.cpp log wording
- observable device selection matches the requested placement policy
- repeated loads with the same model/config remain stable
- config changes that alter runtime identity force a reload instead of silently reusing stale state
- shutdown leaves the backend ready for a later fresh initialization

**Verification**:
- `ctest --test-dir build -L backend --output-on-failure`

---

### 3E - Multimodal projector path

**Goal**: Cover the multimodal projector path on real backend fixtures.

**Tasks**:
- cover `mmproj_model_path` handling
- cover image prompt inputs on a multimodal model
- cover audio prompt inputs on a multimodal model if the fixture set includes them
- cover text-only prompts on the same multimodal model
- keep the tests focused on the real decoder pipeline rather than on internal helper behavior

**Acceptance**:
- multimodal requests exercise the real decoder pipeline successfully
- text-only requests continue to succeed on the same multimodal model
- projector handling remains optional per request and does not block plain text generation

**Verification**:
- `ctest --test-dir build -L backend --output-on-failure`

---

### 3F - Documentation and nuance sync

**Goal**: Keep the testing docs aligned with the backend test suite as the suite is implemented.

**Tasks**:
- update `docs/architecture/testing.md` if the backend test structure or labels change
- update `test_nuances.md` for any non-obvious assertion boundaries or environment requirements discovered while building the suite
- explicitly check whether `dev_nuances.md` should be updated when a build quirk, runtime workaround, or toolchain-specific behavior is uncovered
- keep the phase-3 breakdown aligned with the testing architecture and the actual CMake layout

**Acceptance**:
- the phase-3 breakdown and the testing architecture agree on labels, gating, and skip behavior
- test-specific rationale lives in `test_nuances.md` instead of being duplicated in the roadmap
- any backend-specific workaround that matters for future work is captured in `dev_nuances.md`

**Verification**:
- `ctest --test-dir build -L backend --output-on-failure`
