---
name: ODAI Nuances Tracking
description: Instructions for tracking and documenting quirky behaviors, build nuances, and specific technical reasoning in the ODAI SDK.
---

# ODAI Nuances Tracking

When working on the ODAI SDK, you may encounter quirky behaviors, specific build system requirements (like CMake quirks), or non-obvious technical reasoning. It is crucial to document these "nuances" so that future agents and developers can understand the "why" behind certain implementations.

## How to Track Nuances

All such nuances MUST be documented in the `nuances.md` file located at the root of the project (`/home/darshanv3v/Projects/odaisdk/nuances.md`).

### Process for Adding a Nuance:

1.  **Identify the Nuance:** Recognize when a technical decision is not immediately obvious, involves a workaround, or relies on specific tool/library behavior (e.g., header-only library integration in CMake).
2.  **Update `nuances.md`:**
    *   Add a clear, concise heading for the topic.
    *   Explain the *what* and, most importantly, the *why*.
    *   Provide code examples if applicable.
3.  **Update the Index:** The `nuances.md` file contains an index at the top. **You MUST update this index** whenever you add a new section, so that an LLM can easily scan the document and jump to the relevant section. Use markdown links to link the index item to its corresponding header.

### Example `nuances.md` Structure:

```markdown
# Development Nuances

## Index
- [Build System (CMake)](#build-system-cmake)
- [New Nuance Topic](#new-nuance-topic)

## Build System (CMake)
...

## New Nuance Topic
...
```

By maintaining this document, we ensure that hard-won knowledge about the project's quirks is preserved and easily accessible.
