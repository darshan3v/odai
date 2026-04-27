# AGENTS.md

Always check the relevant skill files in `.agents/skills/` before thinking or making changes. 

If a task introduces or reveals a workaround, build quirk, toolchain-specific behavior, non-obvious constraint, or important technical reasoning, update `dev_nuances.md` and keep its index in sync.

If a task writes or changes tests, test strategy, or test-specific rationale, check whether `test_nuances.md` should be updated and keep its index in sync.

If an existing nuance is no longer relevant because the code, build system, or dependency behavior changed, remove or replace that nuance so `dev_nuances.md` stays accurate.

If an existing test nuance or manual verification entry is no longer relevant because the tests, fixtures, dependencies, or workflow changed, remove or replace it so `test_nuances.md` stays accurate.

Before finishing, explicitly check whether `dev_nuances.md` should be updated.
Before finishing test-related work, explicitly check whether `test_nuances.md` should be updated.

## Architecture Documentation

Before making structural changes — adding/modifying interfaces, swappable components, new layers, or changing data flow — review `docs/architecture/README.md` and the relevant sub-documents. After completing such changes, update the affected architecture files so documentation stays accurate.

Keep stable behavior, ownership, data flow, and current runtime structure in `docs/architecture/`.

Keep only non-obvious rationale, workarounds, dependency/toolchain quirks, and failure-mode reasoning in `dev_nuances.md`.

If a topic appears in both places, avoid duplicating the same descriptive content:
- `docs/architecture/` should describe what the system does
- `dev_nuances.md` should explain why an unusual rule exists

When both documents need touching, prefer one canonical description and let the other side link or refer briefly instead of restating it.

## Planning

Before formulating an implementation plan for non-trivial work, check the `odai_planning_guide` skill.

## Testing

Before writing or modifying tests, check the `odai_testing_guide` skill for test patterns, naming conventions, and per-layer build/run commands.

If a task changes the testing infrastructure — adding a new test layer, changing fixture patterns, modifying the test data pipeline, or adding new CTest labels — update `docs/architecture/testing.md` so the testing architecture stays accurate.
