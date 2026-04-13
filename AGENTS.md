# AGENTS.md

Always check the relevant skill files in `.agents/skills/` before thinking or making changes.

If a task introduces or reveals a workaround, build quirk, toolchain-specific behavior, non-obvious constraint, or important technical reasoning, update `nuances.md` and keep its index in sync.

If an existing nuance is no longer relevant because the code, build system, or dependency behavior changed, remove or replace that nuance so `nuances.md` stays accurate.

Before finishing, explicitly check whether `nuances.md` should be updated.

## Architecture Documentation

Before making structural changes — adding/modifying interfaces, swappable components, new layers, or changing data flow — review `docs/architecture/README.md` and the relevant sub-documents. After completing such changes, update the affected architecture files so documentation stays accurate.

Keep stable behavior, ownership, data flow, and current runtime structure in `docs/architecture/`.

Keep only non-obvious rationale, workarounds, dependency/toolchain quirks, and failure-mode reasoning in `nuances.md`.

If a topic appears in both places, avoid duplicating the same descriptive content:
- `docs/architecture/` should describe what the system does
- `nuances.md` should explain why an unusual rule exists

When both documents need touching, prefer one canonical description and let the other side link or refer briefly instead of restating it.

## Planning

Before formulating an implementation plan for non-trivial work, check the `.agents/skills/odai_planning_guide/SKILL.md` skill.
