---
name: understand
description: Interprets a /summarize handoff into a concise working model (constraints, blockers, next focus) without inventing tensor shapes or repo facts. Use when the user invokes /understand, pastes a summarize report, asks what the summarize means for the current task, or wants to onboard from a session handoff.
disable-model-invocation: true
---

# /understand (interpret summarize context)

When the user requests **`/understand`**, treat the **most recent `/summarize` output in the thread** as the primary source. If none is present, say so and either ask for a pasted handoff or offer to produce one using [summarize/SKILL.md](../summarize/SKILL.md).

## What to extract from a summarize handoff

A valid handoff has **exactly** these five sections (same headings as summarize):

1. **Component Map** — who owns what (`ExperimentRunner`, `PatchTarget`, etc.).
2. **Current State** — what already works; do not re-implement this unless asked.
3. **Input/Output Shapes** — **treat as authoritative for the session** unless the user says the code changed; do not substitute shapes from memory.
4. **Active Blockers** — assumptions to avoid; open bugs/TODOs to respect.
5. **Next Goal** — default priority unless the user overrides in the same message.

## Output template (use this structure)

**Interpretation:** 2–4 sentences: what the project slice is, where work left off, and what “done” looks like next.

**Non-negotiables:** Bullet list copied or paraphrased only from **Input/Output Shapes**, **Active Blockers**, and any explicit constraints in the handoff (e.g., identical clean/corrupt token shapes). If a constraint is missing, say **unknown** rather than guessing.

**Safe assumptions vs verify:** Bullets — what you may rely on from the handoff vs what must be confirmed in code (`causal_patcher/runner.py`, `causal_patcher/targets.py`, tests) before editing.

**Implied task focus:** One short paragraph tying **Next Goal** to the user’s latest instruction; flag conflicts between Next Goal and the user ask.

## Anti-hallucination rules

- Do **not** invent tensor ranks, hook names, or APIs that are not in the handoff or repo.
- If shapes in the handoff conflict with something you see in code, **prefer the repo** and state the discrepancy.
- Do not dismiss **Active Blockers** as “minor”; surface them in **Non-negotiables** or **verify**.

## Optional deep link

For the canonical summarize format and default shape table, see [summarize/SKILL.md](../summarize/SKILL.md).
