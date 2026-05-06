# Inference Benchmark Stats Collection Plan

## Objective

Define a minimal path for collecting inference benchmark stats without letting benchmark-specific code spread into
production request handling.

The immediate goal is not a full performance framework. It is to create a small, reusable stats collection shape that:

- captures common request metrics for any backend engine
- allows backend-specific stats only where the active backend can provide them
- keeps timing and result emission in a benchmark harness instead of production hot paths
- introduces at most one minimal shared timing utility for consistent measurement

## Current Situation

The SDK already exposes the real streaming request path through `OdaiSdk`, `OdaiRagEngine`, and
`IOdaiBackendEngine`, and the existing `tests/main.cpp` file already exercises text, image, audio, and chat requests.

That makes it a reasonable base for a first benchmark runner. At the same time:

- production APIs currently do not expose rich benchmark telemetry
- the C ABI streaming calls collapse results down to generated token count
- backend-specific runtime details differ by implementation
- full benchmark code does not belong in production request logic

## Design Goals

- Keep benchmark orchestration outside production code.
- Define one generic stats layer that all backend engines can participate in.
- Allow optional backend-specific stats without forcing all engines to implement the same details.
- Keep the first iteration serial and simple.
- Introduce only a minimal timing utility that can be reused by the benchmark runner and any small internal timing site
  that later proves broadly useful.

## Non-Goals

- Building a complete regression dashboard in this pass
- Extending the stable C ABI for benchmark-only telemetry
- Adding benchmark-specific hooks throughout SDK and backend request code
- Forcing every backend engine to report the same low-level runtime details
- Solving concurrent benchmarking while the current llama backend is effectively single in-flight per loaded LLM

## Product Decisions

### 1. Benchmark Code Stays Out of Production Paths

The benchmark runner owns:

- iterations
- warmup policy
- total request timing
- first-token timing
- output file writing
- baseline comparison later

Production code should only expose small runtime facts that are useful even outside benchmarking.

### 2. Stats Are Split Into Generic and Backend-Specific Layers

Every benchmark record should contain a common section plus an optional backend-specific section.

The common section is the contract we can compare across backends. The backend-specific section is an extension point for
details such as llama-specific placement or other implementation-owned diagnostics.

### 3. Device Capture Follows the Same Split

Device information should also be split into:

- generic device inventory fields that every backend can report consistently
- optional backend-specific device details when available

### 4. Timing Should Use One Minimal Utility Class

Timing collection should not be open-coded throughout the benchmark harness or backend code.

Add one small utility class that wraps a monotonic clock and exposes only the operations needed for benchmark
measurement, such as:

- capture a start point
- read elapsed milliseconds from that start point
- optionally capture named checkpoints in a small local scope

The first use should be in the benchmark runner for total latency and time-to-first-token measurement. It should not
become a broad instrumentation framework.

## Target Stats Shape

### Generic Request Stats

These fields should be collected for every benchmark run:

- timestamp
- backend engine type
- model name
- request type: text, image, audio, chat
- requested device type
- requested context window
- sampler config
- success or failure
- cancelled or not cancelled
- generated token count
- total latency
- time to first token
- total stream duration
- derived tokens per second

### Generic Device Stats

These fields should be collected once per benchmark session, and optionally attached per run if useful:

- device name
- device type
- total memory when exposed by the backend

### Backend-Specific Stats

This section is optional and owned by the active backend implementation.

For the first llama.cpp-oriented version, examples may include:

- discovered backend family or selected device names
- actual placement summary
- accelerated vs CPU execution summary
- load or reuse state summary

These should stay optional. The generic benchmark schema must remain valid when this section is empty.

## Minimal Implementation Plan

### Phase 1. Define Benchmark Record Types

Add benchmark-side data structures for:

- generic request stats
- generic device stats
- optional backend-specific stats payload
- one benchmark session or run record

These types should live with the benchmark harness, not in the stable public API.

### Phase 2. Add a Minimal Timing Utility

Add a small utility class under the shared utility area for monotonic elapsed-time measurement.

Constraints:

- monotonic clock only
- no backend-specific behavior
- no logging or file output responsibility
- simple enough to use both in tests and in narrow internal timing sites later

This utility should be the only shared timing primitive introduced by this work.

### Phase 3. Create a Small Benchmark Runner

Add a dedicated benchmark executable or benchmark-oriented test target that:

- initializes the SDK
- registers models
- enumerates backend devices
- runs a small fixed set of serial requests
- collects generic stats
- attaches backend-specific stats when supported
- writes JSON or JSONL output

The first scenario set should stay small:

- one warm text request
- one warm image request
- one warm chat request

### Phase 4. Add Optional Backend Diagnostics for llama.cpp

If needed, add a narrow backend-owned diagnostic surface that the benchmark runner can query after a request or load.

This should remain optional and should not force unrelated backends to implement llama-specific details.

Keep this pull-based where possible instead of adding benchmark callbacks into the request path.

### Phase 5. Establish Baseline Artifacts

Once the runner emits stable records, save one baseline result per machine or environment so later work can compare
performance changes.

This comparison step is out of scope for the first implementation, but the output format should already support it.

## Expected File Touch Points

The implementation should likely touch only a small set of areas:

- a new plan or benchmark runner file under `tests/` or a dedicated benchmark directory
- a new minimal timing utility under the shared utility area
- small benchmark-side record types
- optional backend-specific diagnostic access points for the active backend if needed

Avoid touching:

- the stable C ABI unless a later non-benchmark use case justifies it
- generic production request code with benchmark-only branching

## Open Questions

- Should the first benchmark runner live in `tests/` or in a dedicated `benchmarks/` directory?
- Do we want backend-specific diagnostics only through C++ internals at first, or should the SDK layer expose a narrow
  debug-only snapshot later?
- Is the existing device enumeration surface already sufficient for the first generic device inventory pass?

## Recommended First Pass

The smallest useful first pass is:

1. Add the minimal timing utility.
2. Add a benchmark runner that wraps the existing streaming flows.
3. Collect generic request stats and generic device inventory.
4. Leave backend-specific stats empty at first unless one llama-specific field is trivial to expose cleanly.

That gives ODAI a real benchmark artifact path without polluting production inference code.
