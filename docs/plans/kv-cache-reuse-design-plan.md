# KV Cache Reuse Design Plan

## Objective

Define a backend-internal design for request-time KV cache reuse in ODAI without leaking llama.cpp-specific cache
details into `IOdaiBackendEngine`, the SDK layer, or the C ABI.

The immediate goal is not to build a general cache-management API. It is to establish:

- the right behavioral contract for reuse
- the right identity model for deciding when reuse is safe
- the minimal llama backend changes needed for a first useful implementation
- the boundaries where DB/chat-history rules materially affect reuse correctness

## Current Situation

The current llama backend already has one useful prerequisite for KV reuse:

- one loaded LLM state owns one preallocated reusable `llama_context`
- every request clears that context before prompt/chat replay
- chat generation rebuilds the prefix by replaying all prior messages into that same context

That means ODAI already has a clear place where KV reuse could fit: the request-preparation path around
`prepare_reusable_llm_context_for_request()` and `load_chat_messages_into_context()`.

At the same time:

- `IOdaiBackendEngine` does not expose any KV-cache concepts today
- the current documented backend input contract says media items are `FILE_PATH` and text is `MEMORY_BUFFER`
- the DB/history contract is stricter for chat history than the backend implementation actually enforces
- chat history replay currently reprocesses multimodal items directly, and does not verify that file paths came from
  DB-cached immutable artifacts

Those last two points matter because KV reuse depends on stable prefix identity, especially for multimodal history.

## Design Goals

- Keep KV cache reuse implementation-specific to the active backend.
- Do not add KV-cache concepts to `IOdaiBackendEngine` unless a real cross-backend need appears later.
- Define correctness in terms of equivalent prepared request prefixes, not in terms of object identity or reuse of raw
  chat-history structs.
- Keep reuse opportunistic: correctness must not depend on whether the backend hit or missed a cache entry.
- Make multimodal history identity explicit enough that future reuse decisions are trustworthy.
- Start with an exact-prefix reuse design that fits the current single reusable-context ownership model.

## Non-Goals

- Exposing explicit KV-cache handles or cache-management APIs in the C ABI or C++ SDK
- Guaranteeing reuse across process restarts
- Building a multi-context pool or concurrent multi-request execution design in this pass
- Solving every future cache-eviction or memory-pressure policy upfront
- Treating arbitrary external file paths as stable multimodal identity

## Core Product Decisions

### 1. KV Cache Reuse Stays Behind The Backend Interface

KV cache reuse is an internal optimization of a concrete backend implementation.

The stable contract should remain:

- callers submit prompt or chat requests
- the backend may reuse internal prepared state when it can prove the effective request prefix is equivalent
- the request must produce the same semantic result whether reuse happens or not

This keeps `IOdaiBackendEngine` focused on behavior, not llama.cpp storage mechanics.

### 2. The Correct Reuse Unit Is An Equivalent Prepared Prefix

The real question is not "is this the same chat history object?" It is:

"Would the backend feed the exact same prefix into the model before continuing generation?"

That prefix equivalence must account for:

- model identity
- relevant runtime config
- chat-template formatting
- message ordering and contents
- multimodal placeholder layout
- multimodal content identity

This is the contract to reason about. A hash is only one implementation mechanism for checking it.

### 3. Hash The Canonical Prefix Fingerprint, Not Raw History Structs

A full-history hash is a reasonable first implementation strategy, but it should hash a canonical
`PrefixFingerprint`, not raw `ChatMessage` memory layout and not arbitrary file-path strings.

At minimum, the fingerprint must include:

- loaded model identity
- relevant `LLMModelConfig` fields, especially requested context window
- backend formatting identity where it affects prompt materialization
- the ordered replayed prefix content
- multimodal content identity

For append-only chat, hashing the complete history prefix is fine as a first practical form of exact-prefix reuse.

### 4. Multimodal Reuse Needs Stable Content Identity

Multimodal history is where a naive hash becomes dangerous.

If the cache key uses only file paths, then two bad cases appear:

- the same file content moved to a different path looks like a false miss
- the same path with mutated on-disk content looks like a false hit

That means file paths alone are not a trustworthy reuse key for multimodal history.

The preferred long-term direction is:

- chat history media used for replay should carry stable content identity
- DB-cached artifacts are the cleanest way to provide that identity
- if history media is not known to be immutable and content-addressed, reuse should degrade conservatively

### 5. The First Implementation Should Be Exact-Prefix And Single-Context

ODAI's current llama backend owns one reusable context per loaded LLM state.

That makes the right first implementation:

- exact-prefix reuse only
- one cached prepared-prefix record per loaded LLM state, or a very small backend-local set later
- no interface churn
- no concurrency redesign

This gets useful wins for the common append-only chat path without forcing a larger runtime ownership rewrite.

## Recommended Contract Split

### Interface-Level Contract

`IOdaiBackendEngine` should continue to say nothing KV-specific.

The intended behavior should be:

- backends may reuse prepared internal state across requests
- callers must not depend on reuse
- reuse eligibility is backend-defined
- request results must remain valid regardless of reuse hit or miss

This can live in architecture docs or implementation docs without changing the C++ virtual interface.

### RAG / Chat-History Contract

The history replay path should have a clear behavioral expectation:

- chat history is treated as a replayable prefix for future generation
- message order and content must be stable for reuse to be considered valid
- text history is stable as serialized bytes
- multimodal history should have stable content identity, ideally DB-cached immutable artifacts or stored checksums

If history contains mutable external file references with no stable content identity, the backend should treat that
history as replayable but not safely reusable.

### Backend-Implementation Contract

The llama backend should define one internal invariant:

- a prepared context may be reused only when a newly computed `PrefixFingerprint` exactly matches the fingerprint of
  the cached prepared prefix for the current loaded LLM state

On mismatch, the backend falls back to the existing clear-and-replay path.

## Prefix Fingerprint Design

### What The Fingerprint Must Represent

The fingerprint should represent the exact prepared prefix that would exist in the reusable context immediately before
the new request suffix is evaluated.

For chat generation, that means the replayed chat history prefix, not the new user message currently being generated.

For a future prompt-only reuse path, it would mean the prepared prompt prefix.

### Suggested Internal Shape

```cpp
struct PrefixFingerprint {
  std::string m_modelIdentity;
  std::uint32_t m_requestedContextLength = 0;
  std::string m_templateIdentity;
  std::vector<std::uint8_t> m_digest;
};
```

The digest input should be canonical serialized data, not pointer values or container memory layout.

### Fingerprint Inputs

The serialized fingerprint payload should include:

- model registration identity or file checksum identity
- backend engine type
- relevant `LLMModelConfig` fields that affect the loaded runtime contract
- a stable identifier for the formatting/chat-template behavior used by the backend
- each replayed message role in order
- each replayed text item's bytes
- each replayed media item's stable content identity

For media items, preferred identity sources in descending order:

1. DB-stored content checksum or cache key
2. immutable DB-cached absolute path only if the DB guarantees content-addressed immutability
3. otherwise: mark the prefix as not safely reusable

The important design choice is that "unsafe to fingerprint reliably" should not be treated as a fatal error. It should
only disable reuse for that request.

## Cache Eligibility Rules

Reuse should be allowed only when all of the following hold:

- the requested LLM is already loaded and matches the current loaded state
- the loaded runtime contract still matches the requested config
- the new request is using the same backend formatting behavior as the cached prefix
- the computed prefix fingerprint exactly matches the cached prepared prefix
- the backend still owns a reusable context populated for that fingerprint

Reuse should be skipped when any of the following hold:

- model reload happened since the cached prefix was prepared
- requested context length changed
- chat history changed in any way
- multimodal history identity is not stable enough to fingerprint safely
- the backend had to clear or invalidate the reusable context due to an earlier failure

## Proposed llama.cpp Runtime Flow

### Existing Flow

Today the chat path is conceptually:

1. validate prompt input support
2. load or reuse the requested LLM state
3. clear reusable context
4. replay full chat history into context
5. process current prompt items
6. generate

### New Flow

With exact-prefix reuse, the chat path should become:

1. validate prompt input support
2. load or reuse the requested LLM state
3. compute `PrefixFingerprint` for the replayable chat history prefix
4. if the cached prepared prefix exactly matches, reuse the populated context as-is
5. otherwise clear the reusable context, replay the chat history, and store the new fingerprint as the prepared prefix
6. process current prompt items
7. generate

The prompt-only generation path can continue using the current clear-and-load behavior in the first iteration.

## Suggested Internal Types

```cpp
struct PreparedPrefixState {
  bool m_isValid = false;
  PrefixFingerprint m_fingerprint;
};
```

This state belongs inside the loaded llama runtime state, alongside the reusable context it describes.

If the backend later grows beyond one reusable context, this can evolve into a tiny LRU keyed by `PrefixFingerprint`.

## Multimodal Design Guidance

### Preferred Direction

For long-term correctness, the history replay path should move toward one of these shapes:

- only DB-cached multimodal history is eligible for reuse
- or chat-history media carries explicit content checksums independent of paths

Both are better than treating any file path as cache-stable identity.

### Transitional Rule

For the first implementation, reuse should be conservative:

- if a multimodal history item has trusted stable identity, include it in the fingerprint
- if it does not, skip reuse for that request

That lets the feature land without first redesigning all chat-history media contracts.

## Failure And Invalidation Rules

- Any model reload invalidates prepared-prefix state.
- Any context-clear failure invalidates prepared-prefix state.
- Any failed replay of chat history invalidates prepared-prefix state.
- Any generation-time failure that leaves the reusable context in an uncertain state should invalidate prepared-prefix
  state before returning.
- Cache misses are normal control flow, not warnings.

## Minimal Implementation Plan

### Phase 1. Define The Internal Fingerprint And Prepared-Prefix State

Add backend-local types for:

- `PrefixFingerprint`
- `PreparedPrefixState`
- serialization / hashing helpers for replayable prefixes

Keep these in the llama backend only.

### Phase 2. Add Conservative Chat-History Fingerprinting

Implement fingerprint construction for text history first, then add multimodal support only when stable media identity
is available.

If multimodal identity is not reliable, return "reuse not eligible" instead of failing the request.

### Phase 3. Integrate Reuse Into Request Preparation

Update the shared request-preparation path so chat generation:

- computes the fingerprint
- checks the prepared-prefix state
- reuses or rebuilds accordingly
- updates prepared-prefix state only after successful replay

### Phase 4. Add Tests For Correctness Before Performance Tuning

The first tests should prove:

- exact same history prefix reuses successfully
- changed history forces replay
- model/config change forces replay
- multimodal history with unstable identity disables reuse rather than producing a false hit
- any replay failure invalidates prepared-prefix state

Performance benchmarking can come after correctness coverage exists.

### Phase 5. Revisit Stronger History Identity Contracts

Once the first version works, decide whether to strengthen the history contract by:

- requiring DB-cached media for replayable chat history
- or carrying explicit media checksums in message content metadata

That decision should be made before attempting broader multimodal reuse optimization.

## Expected File Touch Points

The first implementation should likely stay narrow:

- llama backend header and implementation for fingerprint state and request-preparation changes
- tests covering exact-prefix reuse and invalidation
- architecture docs for the backend implementation behavior if the runtime shape changes materially

Avoid in the first pass:

- C ABI changes
- `IOdaiBackendEngine` changes
- DB schema changes unless we explicitly choose checksum-backed media identity now

## Open Questions

- Do we want the first reusable fingerprint to be chat-only, or do we also want prompt-prefix reuse immediately?
- Should multimodal history become "reuse-ineligible unless DB-cached" as an explicit product rule?
- Is model identity best represented by DB model registration checksum data, backend-local file checksums, or current
  loaded-state equality inputs?
- Do we want only one prepared prefix per loaded LLM state at first, or a tiny LRU from the start?

## Recommended First Pass

The smallest defensible first pass is:

1. Keep KV reuse entirely inside `OdaiLlamaEngine`.
2. Define reuse in terms of exact prepared-prefix equivalence.
3. Add one backend-local fingerprint for the replayed chat-history prefix.
4. Reuse only when the fingerprint exactly matches.
5. Treat multimodal history without trusted stable identity as replayable but not reusable.

That gives ODAI a correct and extensible foundation for KV reuse without prematurely expanding the public contract.
