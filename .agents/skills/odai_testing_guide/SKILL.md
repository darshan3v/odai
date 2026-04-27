---
name: ODAI Testing Guide
description: Test patterns, per-layer build/run commands, and assertion conventions for the ODAI SDK.
---

# ODAI SDK Testing Guide

This document covers how to **write, build, and run** tests. For the testing framework design and architecture, see [`docs/architecture/testing.md`](../../../docs/architecture/testing.md). For non-obvious assertion rationale, see [`test_nuances.md`](../../../test_nuances.md).

## 1. Structure

Tests live in `tests/` and mirror `src/impl/`. Each layer has its own subdirectory with a `CMakeLists.txt` containing:
- A feature-gate guard (e.g. `if(NOT ODAI_ENABLE_<FEATURE>) return() endif()`)
- CTest labels for per-layer execution (e.g. `db`, `image`, `audio`)
- `TEST_DATA_DIR` compile definition where test assets are needed

When a layer tests a swappable interface, prefer a layer-local split between reusable interface contract coverage and implementation-specific coverage. Keep behavior promised by the interface in a contract suite, and keep concrete backend details in the implementation suite. For example, `tests/db/` can own reusable `IOdaiDb` contract tests while the SQLite-specific suite covers SQLite filesystem and validation details.

Reusable interface-suite targets keep the normal layer and `integration` labels and also add `contract` when CMake can label them cleanly.

Implementation-specific targets must be guarded by the implementation's build flag and should add an implementation label when useful. The matching contract target for that implementation should be created under the same guard, so a build only registers contract and implementation tests for implementations actually compiled into `odai`.

Shared test assets are generated at build time by `scripts/generate_test_data.py`. Real sample files (JPEG, MP3) live in `tests/data/` and are copied to the build output alongside the generated ones. The test-data target declares both generated and copied fixtures as outputs and clears `build/tests/data` before regeneration, so removed or renamed generated fixtures do not linger in the build tree. When fixture names change, update the CMake output list at the same time.

## 2. Building

For non-backend layers, skip the expensive llama.cpp build:

```bash
cmake --preset linux-default-debug -DODAI_BUILD_TESTS=ON -DODAI_ENABLE_LLAMA_BACKEND=OFF
cmake --build --preset linux-default-debug
```

Backend, C API, E2E, and sanitizer-specific test workflows are planned/deferred. Do not advertise `ODAI_BUILD_E2E_TESTS`, `api`, `backend`, or `e2e` as available until their CMake targets exist.

## 3. Running

Use CTest labels to run only the layer you're changing:

```bash
ctest --test-dir build -L <label> --output-on-failure
```

Available labels: `db`, `image`, `audio`, `integration`, `contract`, `sqlite`, `stb`, `miniaudio`.

Run all fast (no-model) tests at once:

```bash
ctest --test-dir build -L "db|image|audio" --output-on-failure
```

Run reusable interface contract suites:

```bash
ctest --test-dir build -L contract --output-on-failure
```

Run all tests for one concrete implementation with its implementation label:

```bash
ctest --test-dir build -L sqlite --output-on-failure
ctest --test-dir build -L stb --output-on-failure
ctest --test-dir build -L miniaudio --output-on-failure
```

Combine the implementation label with `contract`, `integration`, or a layer label only when intentionally narrowing the subset:

```bash
ctest --test-dir build -L contract -L sqlite --output-on-failure
ctest --test-dir build -L image -L stb --output-on-failure
ctest --test-dir build -L audio -L miniaudio --output-on-failure
```

## 4. Conventions

- **Test the behavior, not the brittle implementation details.** Assert on observable state, return values, output shapes, and correct functional logic (e.g., proper device selection). We want to verify that the system behaves correctly, not just that we get final output tokens, while continuing to avoid brittle assertions on exact values tied to third-party library internals.
- **Keep contract and implementation-specific assertions separated.** Put behavior promised by a swappable interface in the layer-local contract suite. Put concrete backend details, dependency-specific support matrices, filesystem layouts, and similar implementation facts in implementation-specific suites.
- **One behavior per `TEST_F`**: Each `TEST_F` should verify one distinct behavior or contract. If a scenario needs multiple independent expectations, split it into separate tests with names that describe the behavior under test.
- **`ASSERT_TRUE` before dereferencing**: Always `ASSERT_TRUE(result.has_value())` before accessing `.value()` on `OdaiResult<T>`.
- **No LLM output assertions**: Model output is non-deterministic. Assert on result codes, non-emptiness, and token counts.
- **Verify error codes**: Always check the specific `OdaiResultEnum` on error paths.
- **Skip gracefully**: Model-backed tests should `GTEST_SKIP()` with a clear message when model files aren't available.
- **Document intentional test limits**: When a test intentionally avoids an exact assertion because it would be brittle, environment-dependent, or dominated by third-party behavior, record the rationale in `test_nuances.md` and keep its index in sync.
- **Track manual-only verification explicitly**: If some coverage is intentionally left to manual checking, add or update an entry in `test_nuances.md` describing what must be checked, why it is manual, and when it should be revisited.

### Code Style

- Test files **are** covered by `clang-format` (`scripts/format.sh`).
- Test files are **not** covered by `clang-tidy` (`scripts/lint.sh`).

## 5. Adding a New Test Layer

1. Create `tests/<layerName>/CMakeLists.txt` — feature guard, test executable, labels.
2. Add `add_subdirectory(<layerName>)` to `tests/CMakeLists.txt`. If the layer is model-backed, first add the missing model-backed test option and document the new workflow.
3. Follow the conventions above.
