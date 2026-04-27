---
name: ODAI Planning Guide
description: Steps AI agents should follow before creating implementation plans for the ODAI SDK.
---

# Planning Guidelines for ODAI SDK

Before creating an implementation plan for non-trivial work, follow these steps to produce a plan that is architecturally sound and consistent with existing patterns.

## 1. Check Architecture First

Read `docs/architecture/README.md` to understand:
- Which layer(s) your change touches (C API, C++ SDK, RagEngine, interfaces, implementations)
- Whether your change involves a swappable interface boundary
- How data flows through the system

If your change touches a specific interface or implementation, also read its dedicated doc under `docs/architecture/interfaces/` or `docs/architecture/implementations/`.

## 2. Understand the Layers

Each layer has different rules:

| Layer | Key Constraint |
|---|---|
| **C ABI** (`odai_public.h`) | Must remain binary-stable. Use C-compatible types only. Must have matching `free` functions for allocated memory. |
| **C++ SDK** (`odai_sdk.h`) | Must stay in sync with the C API (feature parity). Catches all exceptions. |
| **RagEngine** | Concrete orchestrator — owns backend engine and DB. Merges model details before forwarding to DB. |
| **Interfaces** | Define contracts only. Don't leak implementation details. Methods are non-const by convention. |
| **Implementations** | Guarded by `#ifdef` flags. Only one impl per interface at compile time. |

## 3. Respect Swappable Boundaries

When adding functionality:
- **Is this a capability all backends need?** → Add to the interface, then implement in each backend.
- **Is this specific to one backend?** → Keep it in the implementation only. Don't pollute the interface.
- **Does this introduce a new swappable dimension?** → Design a new interface, document it.

## 4. Break Work into Phases

Order your work by dependency:
1. **Types & Errors** — New structs, enums, configs in `odai_types.h` / `odai_ctypes.h`, and any new `OdaiResultEnum` values.
2. **Interface changes** — New or modified virtual methods.
3. **Implementation** — Concrete logic in the chosen backend.
4. **Wiring** — Factory/instantiation changes in central orchestrators (e.g., `odai_sdk.cpp` or `odai_rag_engine.cpp`).
5. **Public API** — Expose via `odai_public.h` / `odai_public.cpp` with sanitizers and `toCpp`/`toC` converters.
6. **Tests** — Verify at each layer.

## 5. Check Existing Patterns

Before writing code:
- Read the `odai_development_guide` skill for coding patterns (sanitize → convert → forward, `is_sane()`, `OdaiResult<T>`, naming conventions, interface design rules, and documentation).
- Read `dev_nuances.md` for known build quirks, especially when integrating new third-party libraries or header-only libraries.
- Search the codebase for similar features already implemented — follow the established patterns rather than inventing new ones.

## 6. Verify Impact

Before executing your plan:
- List which existing tests might break.
- Identify which build presets (Linux, Android) are affected.
- Note if the change affects the C ABI surface (modifying structs in `odai_ctypes.h` or signatures in `odai_public.h`). This is a breaking change for consumers and should be highlighted.

## 7. Update Docs

If your plan touches architecture, include explicit steps in the plan to:
- Update the relevant `docs/architecture/` files
- Update `dev_nuances.md` if build quirks are involved
- Update the development guide skill if new patterns are introduced
